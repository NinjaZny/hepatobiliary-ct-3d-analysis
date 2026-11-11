#!/usr/bin/env python3
"""
Liver tubular structure trainers — self-contained single file.

Trainers
--------
nnUNetTrainerDiceCEClDice035          — DiceCE + 0.35·clDice  (bile duct baseline)
nnUNetTrainerDiceCEClDice035InvDT050  — DiceCE + 0.35·clDice + 0.50·InvDT-MSE  (configuration of the delivered checkpoint)
nnUNetTrainerVesselDiceCEClDice035InvDT050  — same loss, dmax=10 for portal/hepatic veins

Loss components (inlined — no extra files needed)
-----------------
- DiceCE     : standard nnU-Net segmentation loss with deep supervision
- clDice     : soft centreline Dice, promotes tubular connectivity
- InvDT-MSE  : inverted distance-transform regression
                (boundary voxels = 1, lumen interior < 1, background = 0)
- SoftDT     : differentiable distance transform (utility, not used in training loop)

Installation
------------
Copy this single file to:
    nnunetv2/training/nnUNetTrainer/variants/loss/trainer_bile_duct.py
No other files are required.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from scipy.ndimage import distance_transform_edt

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import dummy_context


# ==============================================================================
# Soft Distance Transform (differentiable utility — Pham et al. DAGM 2021)
# kernel_size=11 for thin bile ducts; increase for larger structures.
# Not used in the training loop; available for custom experiments.
# ==============================================================================

def _soft_binarize(prob: torch.Tensor, sharpness: float = 10.0) -> torch.Tensor:
    return torch.sigmoid(sharpness * (prob - 0.5))


def _soft_boundary(bin_map: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool3d(bin_map, kernel_size=3, stride=1, padding=1)
    return dilated - bin_map


_sdt_kernel_cache: dict = {}


def _get_sdt_kernel(kernel_size: int, device: torch.device) -> torch.Tensor:
    key = (kernel_size, str(device))
    if key not in _sdt_kernel_cache:
        c = kernel_size // 2
        coords = torch.arange(kernel_size, device=device).float() - c
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        K = torch.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
        _sdt_kernel_cache[key] = K.unsqueeze(0).unsqueeze(0)
    return _sdt_kernel_cache[key]


def soft_distance_transform(
    seg_prob: torch.Tensor,
    kernel_size: int = 11,
    lam: float = 0.1,
) -> torch.Tensor:
    """Differentiable approximation of the distance transform.

    Args:
        seg_prob:    [B, 1, D, H, W] foreground probability (softmax ch 1)
        kernel_size: convolution kernel side length (11 for bile ducts)
        lam:         temperature; smaller = sharper approximation
    Returns:
        [B, 1, D, H, W], values in [0, 1], per-sample normalised
    """
    assert lam > 0
    seg_prob = seg_prob.float()
    bin_map  = _soft_binarize(seg_prob)
    boundary = _soft_boundary(bin_map)
    K          = _get_sdt_kernel(kernel_size, seg_prob.device)
    exp_kernel = torch.exp(-K / lam).float()
    conv_out   = F.conv3d(boundary, exp_kernel, padding=kernel_size // 2)
    sdt = -lam * torch.log(conv_out.clamp(min=1e-6))
    sdt = sdt * bin_map
    sdt_max = sdt.amax(dim=[2, 3, 4], keepdim=True).clamp(min=1e-6)
    return sdt / sdt_max


# ==============================================================================
# Soft clDice loss (Shit et al. CVPR 2021)
# ==============================================================================

def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    p1 = -F.max_pool3d(-x, (3, 1, 1), stride=1, padding=(1, 0, 0))
    p2 = -F.max_pool3d(-x, (1, 3, 1), stride=1, padding=(0, 1, 0))
    p3 = -F.max_pool3d(-x, (1, 1, 3), stride=1, padding=(0, 0, 1))
    return torch.min(torch.min(p1, p2), p3)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool3d(x, 3, stride=1, padding=1)


def _soft_skeleton(x: torch.Tensor, num_iter: int = 5) -> torch.Tensor:
    skel = torch.relu(x - _soft_dilate(_soft_erode(x)))
    for _ in range(num_iter - 1):
        x     = _soft_erode(x)
        delta = torch.relu(x - _soft_dilate(_soft_erode(x)))
        skel  = skel + torch.relu(delta - skel * delta)
    return skel


class SoftClDiceForegroundLoss(nn.Module):
    """clDice on all foreground classes combined.

    Accepts raw logits; softmax computed internally in fp32 (AMP-safe).
    Skips samples with no foreground voxels.

    Args:
        iter_num:     soft-skeleton iterations (5 is standard)
        smooth:       numerical smoothing
        ignore_label: voxels with this label are excluded from gt_fg
    """

    def __init__(self, iter_num: int = 5, smooth: float = 1e-5, ignore_label=None):
        super().__init__()
        self.iter_num     = iter_num
        self.smooth       = smooth
        self.ignore_label = ignore_label

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred_logits: [B, C, D, H, W]
        target:      [B, 1, D, H, W] or [B, D, H, W] integer labels
        """
        probs   = torch.softmax(pred_logits.float(), dim=1)
        pred_fg = probs[:, 1:].sum(dim=1, keepdim=True)

        if target.ndim == 4:
            target = target[:, None]
        gt_fg = (target[:, 0:1] > 0).float()
        if self.ignore_label is not None:
            gt_fg = gt_fg * (target[:, 0:1] != self.ignore_label).float()

        has_fg = gt_fg.flatten(1).sum(dim=1) > 0
        if not has_fg.any():
            return pred_logits.sum() * 0.0

        p = pred_fg[has_fg]
        g = gt_fg[has_fg]

        skel_p = _soft_skeleton(p, self.iter_num)
        skel_g = _soft_skeleton(g, self.iter_num)

        tprec  = (skel_p * g).sum(dim=(1, 2, 3, 4)) / (skel_p.sum(dim=(1, 2, 3, 4)) + self.smooth)
        tsens  = (skel_g * p).sum(dim=(1, 2, 3, 4)) / (skel_g.sum(dim=(1, 2, 3, 4)) + self.smooth)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + self.smooth)
        return 1.0 - cldice.mean()


class _DiceCEPlusClDiceLoss(nn.Module):
    """Wraps DiceCE (with deep supervision) and adds clDice at full resolution."""

    def __init__(self, dicece_loss: nn.Module, cldice_weight: float,
                 iter_num: int = 5, ignore_label=None):
        super().__init__()
        self.dicece_loss   = dicece_loss
        self.cldice_weight = cldice_weight
        self.cldice_loss   = SoftClDiceForegroundLoss(iter_num=iter_num,
                                                      ignore_label=ignore_label)

    def forward(self, net_output, target):
        l_dicece = self.dicece_loss(net_output, target)
        logits   = net_output[0] if isinstance(net_output, (list, tuple)) else net_output
        tgt      = target[0]    if isinstance(target,     (list, tuple)) else target
        return l_dicece + self.cldice_weight * self.cldice_loss(logits, tgt)


# ==============================================================================
# Distance Transform loss utilities
# ==============================================================================

def make_dt_target_batch(target, fg_label=1, dmax=5.0, log_norm=False, invert_dt=False):
    """Compute DT targets on-the-fly from GT segmentation.

    Args:
        target:    [B, 1, D, H, W] or [B, D, H, W] integer labels
        fg_label:  foreground label, or None for all non-zero labels
        dmax:      clip distance then normalise to [0, 1]
        log_norm:  use log(1+d)/log(1+dmax) instead of d/dmax
        invert_dt: False → centre=1, boundary≈0, bg=0  (centreline mode)
                   True  → boundary=1, centre<1,  bg=0  (boundary mode)
    Returns:
        torch.Tensor [B, 1, D, H, W], values in [0, 1]
    """
    target_np = target[:, 0].detach().cpu().numpy() if target.ndim == 5 \
                else target.detach().cpu().numpy()

    dt_list = []
    for b in range(target_np.shape[0]):
        mask = (target_np[b] > 0 if fg_label is None
                else target_np[b] == fg_label).astype(np.uint8)
        if mask.sum() == 0:
            dt_list.append(np.zeros_like(mask, dtype=np.float32)[None])
            continue
        raw = np.clip(distance_transform_edt(mask).astype(np.float32), 0, dmax)
        if log_norm:
            dt_norm = np.log1p(raw) / np.log1p(dmax)
        else:
            dt_norm = raw / dmax
        dt = (np.where(mask > 0, 1.0 - dt_norm, 0.0) if invert_dt
              else dt_norm).astype(np.float32)
        dt_list.append(dt[None])

    return torch.from_numpy(np.stack(dt_list, axis=0)).to(target.device)


def weighted_dt_mse(dt_pred, dt_gt, target, fg_label=1, bg_weight=0.1):
    """Foreground-weighted MSE for DT regression.

    Foreground and background are normalised independently so thin structures
    are not drowned out by the background mass.

    Args:
        dt_pred:   [B, 1, D, H, W] sigmoid output
        dt_gt:     [B, 1, D, H, W] DT ground truth in [0, 1]
        target:    [B, 1, D, H, W] or [B, D, H, W] integer labels
        fg_label:  foreground label, or None for all non-zero
        bg_weight: weight for background MSE (default 0.1)
    """
    if target.ndim == 4:
        target = target[:, None]
    fg = (target > 0 if fg_label is None else target == fg_label).float()
    bg = 1.0 - fg
    loss_fg = ((dt_pred - dt_gt) ** 2 * fg).sum() / (fg.sum() + 1e-6)
    loss_bg = ((dt_pred - dt_gt) ** 2 * bg).sum() / (bg.sum() + 1e-6)
    return loss_fg + bg_weight * loss_bg


def split_seg_dt_outputs(output, num_seg_classes=2):
    """Split network output into segmentation channels and DT channel.

    Works with both a single tensor and a deep-supervision list.
    DT loss is only computed on the full-resolution output (index 0).

    Args:
        output:          [B, num_seg_classes+1, ...] or list thereof
        num_seg_classes: number of segmentation channels (2 for binary)
    Returns:
        seg_outputs: list of [B, num_seg_classes, ...] or single tensor
        dt_raw:      [B, 1, D, H, W] full-resolution (not yet sigmoid)
    """
    if isinstance(output, (list, tuple)):
        seg_outputs = [o[:, :num_seg_classes] for o in output]
        dt_raw      = output[0][:, num_seg_classes:num_seg_classes + 1]
    else:
        seg_outputs = output[:, :num_seg_classes]
        dt_raw      = output[:, num_seg_classes:num_seg_classes + 1]
    return seg_outputs, dt_raw


# ==============================================================================
# Network wrapper
# ==============================================================================

class _DTInferenceWrapper(nn.Module):
    """Strips the DT regression channel in eval/inference mode.

    train: forward returns full output [B, seg+1, D, H, W]
    eval : forward returns only seg channels [B, seg, D, H, W]
           so nnUNetPredictor and validation work without modification.
    """

    def __init__(self, network: nn.Module, num_seg_classes: int = 2):
        super().__init__()
        self.network = network
        self.num_seg_classes = num_seg_classes

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == 'network':
                raise
            return getattr(self.network, name)

    def forward(self, x):
        output = self.network(x)
        if self.training:
            return output
        if isinstance(output, (list, tuple)):
            return [o[:, :self.num_seg_classes] for o in output]
        return output[:, :self.num_seg_classes]


# ==============================================================================
# Base trainers
# ==============================================================================

class _DTTrainerBase(nnUNetTrainer):
    """Base trainer adding a DT regression head.

    Network outputs [B, num_seg_classes + 1, D, H, W]:
      channels 0..num_seg_classes-1 : segmentation logits
      channel  num_seg_classes      : DT regression (pre-sigmoid)
    """

    dt_weight: float = 0.50
    dt_dmax: float = 5.0
    dt_log_norm: bool = False
    dt_invert: bool = False
    num_seg_classes: int = 2
    dt_fg_label: Optional[int] = 1

    @staticmethod
    def build_network_architecture(architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        inner = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels + 1,
            allow_init=True,
            deep_supervision=enable_deep_supervision,
        )
        return _DTInferenceWrapper(inner, num_seg_classes=num_output_channels)

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == 'cuda' \
              else dummy_context()
        with ctx:
            output = self.network(data)
            seg_outputs, dt_raw = split_seg_dt_outputs(output, self.network.num_seg_classes)

            l_seg = self.loss(seg_outputs, target)

            target_full = target[0] if isinstance(target, list) else target
            dt_pred = torch.sigmoid(dt_raw)
            dt_gt   = make_dt_target_batch(target_full,
                                           fg_label=self.dt_fg_label,
                                           dmax=self.dt_dmax,
                                           log_norm=self.dt_log_norm,
                                           invert_dt=self.dt_invert)
            l_dt = weighted_dt_mse(dt_pred, dt_gt, target_full,
                                   fg_label=self.dt_fg_label, bg_weight=0.1)

            loss = l_seg + self.dt_weight * l_dt

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}


class _DTClDiceTrainerBase(_DTTrainerBase):
    """Adds clDice on top of DT trainer."""

    cldice_weight: float = 0.35
    cldice_num_iter: int = 5

    def _build_loss(self):
        loss = super()._build_loss()
        self.cldice_loss_fn = SoftClDiceForegroundLoss(
            iter_num=self.cldice_num_iter,
            ignore_label=getattr(self.label_manager, 'ignore_label', None),
        )
        return loss

    def train_step(self, batch: dict) -> dict:
        data   = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [t.to(self.device, non_blocking=True) for t in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        ctx = autocast(self.device.type, enabled=True) if self.device.type == 'cuda' \
              else dummy_context()
        with ctx:
            output = self.network(data)
            seg_outputs, dt_raw = split_seg_dt_outputs(output, self.network.num_seg_classes)

            l_seg = self.loss(seg_outputs, target)

            target_full = target[0] if isinstance(target, list) else target

            dt_pred = torch.sigmoid(dt_raw)
            dt_gt   = make_dt_target_batch(target_full,
                                           fg_label=self.dt_fg_label,
                                           dmax=self.dt_dmax,
                                           log_norm=self.dt_log_norm,
                                           invert_dt=self.dt_invert)
            l_dt = weighted_dt_mse(dt_pred, dt_gt, target_full,
                                   fg_label=self.dt_fg_label, bg_weight=0.1)

            seg_logits_full = seg_outputs[0] if isinstance(seg_outputs, list) else seg_outputs
            l_cldice = self.cldice_loss_fn(seg_logits_full, target_full)

            loss = l_seg + self.dt_weight * l_dt + self.cldice_weight * l_cldice

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {'loss': loss.detach().cpu().numpy()}


# ==============================================================================
# Public trainer classes
# ==============================================================================

class nnUNetTrainerDiceCEClDice035(nnUNetTrainer):
    """Standalone clDice trainer — DiceCE + 0.35·clDice (bile duct baseline E1).

    No train_step() override: AMP, grad scaler, and deep supervision are
    all handled by the standard nnUNetTrainer machinery.
    """

    cldice_weight: float = 0.35
    cldice_num_iter: int = 5

    def _build_loss(self):
        dicece       = super()._build_loss()
        ignore_label = getattr(self.label_manager, 'ignore_label', None)
        return _DiceCEPlusClDiceLoss(
            dicece_loss=dicece,
            cldice_weight=self.cldice_weight,
            iter_num=self.cldice_num_iter,
            ignore_label=ignore_label,
        )


class nnUNetTrainerDiceCEClDice035InvDT050(_DTClDiceTrainerBase):
    """Trainer used for the delivered bile duct model (not uniformly better than the baseline; see README).

    Loss = DiceCE + 0.35·clDice + 0.50·InvDT-MSE
    InvDT target: 1 - min(d, dmax)/dmax inside the duct (boundary voxels ~0.8 for dmax=5), 0 outside.
    dmax = 5 voxels (suited for 1–3 mm bile ducts).
    """
    cldice_weight = 0.35
    dt_weight     = 0.50
    dt_dmax       = 5.0
    dt_invert     = True


class nnUNetTrainerVesselDiceCEClDice035InvDT050(_DTClDiceTrainerBase):
    """Trainer for binary vessel segmentation (portal vein or hepatic vein).

    Identical loss to the bile duct trainer but dmax=10 to match the larger
    vessel diameter (5–15 mm vs 1–3 mm for bile ducts): with dmax=5 the
    inverted-DT gradient would be flat across most of the vessel lumen.

    Loss = DiceCE + 0.35·clDice + 0.50·InvDT-MSE
    Train one model per vessel type (pv / hv) on a binary mask.
    """
    cldice_weight = 0.35
    dt_weight     = 0.50
    dt_dmax       = 10.0
    dt_invert     = True
