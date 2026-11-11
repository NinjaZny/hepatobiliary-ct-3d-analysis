#!/usr/bin/env python3
"""
Evaluate bile duct segmentation predictions against ground truth.

Metrics per case: Dice, HD95, clDice, LCC_ratio, CC_count, Topo_split_extra, Topo_merge_extra

Mode A — all cases in gt_dir:
    python evaluate.py \
        --gt_dir  labelsTr/ \
        --out_csv results/eval.csv \
        --pred    ModelA=preds_A/  ModelB=preds_B/

Mode B — validation fold from nnU-Net splits:
    python evaluate.py \
        --gt_dir      labelsTr/ \
        --splits_json splits_final.json \
        --out_csv     results/eval.csv \
        --pred        ModelA=preds_A/ \
        [--fold 0] [--fg_labels 1]
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from scipy.ndimage import (
    label as cc_label,
    generate_binary_structure,
    distance_transform_edt,
    binary_erosion,
)
from skimage.morphology import skeletonize


STRUCT3D = generate_binary_structure(rank=3, connectivity=1)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nii(path):
    img = nib.load(path)
    arr = np.round(img.get_fdata()).astype(np.int16)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    return arr, spacing


# ── Metrics ───────────────────────────────────────────────────────────────────

def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    den = pred.sum() + gt.sum()
    return 1.0 if den == 0 else 2.0 * inter / den


def hd95(pred, gt, spacing):
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    def surf(mask):
        return mask & ~binary_erosion(mask, structure=STRUCT3D, border_value=0)

    pred_s, gt_s = surf(pred), surf(gt)
    d_p2g = distance_transform_edt(~gt_s,   sampling=spacing)[pred_s]
    d_g2p = distance_transform_edt(~pred_s, sampling=spacing)[gt_s]
    return float(np.percentile(np.concatenate([d_p2g, d_g2p]), 95))


def cldice_score(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    sk_pred = skeletonize(pred)
    sk_gt   = skeletonize(gt)

    tprec = np.logical_and(sk_pred, gt).sum() / (sk_pred.sum() + 1e-8)
    tsens = np.logical_and(sk_gt, pred).sum() / (sk_gt.sum()  + 1e-8)
    denom = tprec + tsens
    return float(2.0 * tprec * tsens / denom) if denom > 0 else 0.0


def split_merge_metrics(gt_mask, pred_mask):
    gt_lab,   n_gt   = cc_label(gt_mask,   structure=STRUCT3D)
    pred_lab, n_pred = cc_label(pred_mask, structure=STRUCT3D)

    gt_sizes   = np.bincount(gt_lab.ravel())
    pred_sizes = np.bincount(pred_lab.ravel())

    # count overlaps between each (gt_i, pred_j) pair
    matched = np.zeros((n_gt + 1, n_pred + 1), dtype=bool)
    valid = (gt_lab > 0) & (pred_lab > 0)
    if valid.any():
        pairs, counts = np.unique(
            gt_lab[valid].astype(np.int64) * (n_pred + 1) + pred_lab[valid],
            return_counts=True,
        )
        for code, ov in zip(pairs, counts):
            gi, pj = int(code // (n_pred + 1)), int(code % (n_pred + 1))
            if ov >= 2 or ov / (min(gt_sizes[gi], pred_sizes[pj]) + 1e-8) >= 0.01:
                matched[gi, pj] = True

    block = matched[1:, 1:]
    split_extra = int(np.sum(np.maximum(block.sum(axis=1) - 1, 0)))
    merge_extra = int(np.sum(np.maximum(block.sum(axis=0) - 1, 0)))

    lcc_ratio = float(pred_sizes[1:].max() / pred_mask.sum()) if n_pred > 0 and pred_mask.sum() > 0 else 0.0

    return split_extra, merge_extra, n_pred, lcc_ratio, n_gt


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt_dir",      required=True)
    ap.add_argument("--splits_json", default=None,
                    help="nnU-Net splits_final.json (omit to evaluate all cases in gt_dir)")
    ap.add_argument("--out_csv",     required=True)
    ap.add_argument("--pred", nargs="+", required=True,
                    help="model_name=pred_dir  (repeat for multiple models)")
    ap.add_argument("--fold",       type=int,   default=0)
    ap.add_argument("--fg_labels",  nargs="+",  type=int, default=[1])
    args = ap.parse_args()

    if args.splits_json:
        with open(args.splits_json) as f:
            val_cases = json.load(f)[args.fold]["val"]
    else:
        val_cases = [f.replace(".nii.gz", "").replace(".nii", "")
                     for f in sorted(os.listdir(args.gt_dir))
                     if f.endswith((".nii.gz", ".nii"))]

    rows = []
    for item in args.pred:
        model_name, pred_dir = item.split("=", 1)
        for case in val_cases:
            gt_path   = os.path.join(args.gt_dir,  case + ".nii.gz")
            pred_path = os.path.join(pred_dir, case + ".nii.gz")
            if not os.path.exists(gt_path):
                print(f"[WARN] missing GT:   {gt_path}");   continue
            if not os.path.exists(pred_path):
                print(f"[WARN] missing pred: {pred_path}"); continue

            gt_arr,   spacing = load_nii(gt_path)
            pred_arr, _       = load_nii(pred_path)
            gt   = np.isin(gt_arr,   args.fg_labels)
            pred = np.isin(pred_arr, args.fg_labels)

            split, merge, cc_count, lcc_ratio, n_gt = split_merge_metrics(gt, pred)
            rows.append({
                "model":            model_name,
                "case":             case,
                "Dice":             dice_score(pred, gt),
                "HD95":             hd95(pred, gt, spacing),
                "clDice":           cldice_score(pred, gt),
                "LCC_ratio":        lcc_ratio,
                "CC_count":         cc_count,
                "GT_CC_count":      n_gt,
                "Topo_split_extra": split,
                "Topo_merge_extra": merge,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    METRICS = ["Dice", "HD95", "clDice", "LCC_ratio", "CC_count",
               "Topo_split_extra", "Topo_merge_extra"]
    summary = (
        df.groupby("model")[METRICS]
        .agg(lambda x: f"{x.mean():.4f} ± {x.std():.4f}")
        .reset_index()
    )
    summary_path = args.out_csv.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print("\nPer-case :", args.out_csv)
    print("Summary  :", summary_path)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
