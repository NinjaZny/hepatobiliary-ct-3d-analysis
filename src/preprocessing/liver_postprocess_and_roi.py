#!/usr/bin/env python3
"""
Post-processing for nnU-Net liver segmentation predictions.

Step 1 — Clean liver mask : keep largest connected component + fill 3D holes.
Step 2 — Crop CT to ROI   : crop CT images to the liver bounding box with a configurable margin.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import label as scipy_label


# ── CONFIGURATION ────────────────────────────────────────────────────────────

# Default locations (override with the command-line flags below).
LIVER_PRED_DIR = Path("data/liver_predictions")          # raw nnU-Net liver masks
LIVER_POSTPROC_DIR = Path("data/liver_masks_clean")      # Step 1 output
IMG_DIR = Path("data/ct_images")                         # *_0000.nii.gz CT volumes
CROPPED_IMG_DIR = Path("data/ct_liver_roi")              # Step 2 output

TARGET_LABELS: Optional[List[int]] = [1]  # None = all non-zero
OUTPUT_LABEL = 1
KEEP_LCC   = True
FILL_HOLES = True
MARGIN_MM  = 20.0
HU_MIN     = -200
HU_MAX     =  200

# ─────────────────────────────────────────────────────────────────────────────


def get_foreground_mask(arr: np.ndarray, target_labels: Optional[List[int]]) -> np.ndarray:
    if target_labels is None:
        return arr > 0
    mask = np.zeros(arr.shape, dtype=bool)
    for lv in target_labels:
        mask |= arr == lv
    return mask


def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = scipy_label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=np.uint8)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(np.uint8)


def fill_3d_holes(mask: np.ndarray) -> np.ndarray:
    return binary_fill_holes(mask).astype(np.uint8)


def compute_bbox(mask: np.ndarray) -> Optional[tuple]:
    coords = np.where(mask > 0)
    if len(coords[0]) == 0:
        return None
    return (
        int(coords[0].min()), int(coords[0].max()),
        int(coords[1].min()), int(coords[1].max()),
        int(coords[2].min()), int(coords[2].max()),
    )


def crop_sitk_with_margin(img_sitk: sitk.Image, bbox: tuple, margin_mm: float):
    # bbox is in numpy z-y-x order; SimpleITK uses x-y-z
    spacing  = img_sitk.GetSpacing()
    vol_size = img_sitk.GetSize()
    zmin, zmax, ymin, ymax, xmin, xmax = bbox
    mx = int(round(margin_mm / spacing[0]))
    my = int(round(margin_mm / spacing[1]))
    mz = int(round(margin_mm / spacing[2]))
    xmin = max(0, xmin - mx);  xmax = min(vol_size[0] - 1, xmax + mx)
    ymin = max(0, ymin - my);  ymax = min(vol_size[1] - 1, ymax + my)
    zmin = max(0, zmin - mz);  zmax = min(vol_size[2] - 1, zmax + mz)
    index     = [int(xmin), int(ymin), int(zmin)]
    crop_size = [int(xmax - xmin + 1), int(ymax - ymin + 1), int(zmax - zmin + 1)]
    return sitk.RegionOfInterest(img_sitk, crop_size, index), index, crop_size


def collect_nifti(directory: Path) -> List[Path]:
    return sorted(p for p in directory.iterdir()
                  if p.name.endswith(".nii") or p.name.endswith(".nii.gz"))


def case_id_from_name(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]; break
    if name.endswith("_0000"):
        name = name[:-5]
    return name


def run_step1(
    pred_dir: Path,
    out_dir: Path,
    target_labels: Optional[List[int]],
    output_label: int,
    keep_lcc: bool,
    fill_holes: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = collect_nifti(pred_dir)
    if not files:
        raise RuntimeError(f"No NIfTI files found in: {pred_dir}")

    print(f"[Step 1] {len(files)} files  {pred_dir} → {out_dir}")

    for i, path in enumerate(files, 1):
        img_sitk = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)

        mask = get_foreground_mask(arr, target_labels)
        n_orig = int(mask.sum())

        if keep_lcc:
            mask = keep_largest_connected_component(mask)
        n_lcc = int(mask.sum())

        if fill_holes:
            mask = fill_3d_holes(mask)
        n_fill = int(mask.sum())

        out_arr = (mask > 0).astype(np.uint8) * output_label
        out_sitk = sitk.GetImageFromArray(out_arr)
        out_sitk.CopyInformation(img_sitk)
        sitk.WriteImage(out_sitk, str(out_dir / path.name))

        print(f"  [{i:03d}/{len(files):03d}] {path.name} | orig={n_orig} lcc={n_lcc} filled={n_fill}")

    print(f"[Step 1] Done.\n")


def run_step2(
    img_dir: Path,
    mask_dir: Path,
    out_dir: Path,
    margin_mm: float,
    keep_lcc: bool,
    hu_min: float,
    hu_max: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    img_files = sorted(img_dir.glob("*_0000.nii.gz"))
    if not img_files:
        raise RuntimeError(f"No *_0000.nii.gz files found in: {img_dir}")

    print(f"[Step 2] {len(img_files)} images  margin={margin_mm}mm  HU=[{hu_min}, {hu_max}]  {img_dir} → {out_dir}")

    failed = []
    for img_path in img_files:
        case_id = case_id_from_name(img_path.name)
        mask_path = mask_dir / f"{case_id}.nii.gz"

        if not mask_path.exists():
            print(f"  [WARN]  No mask found for {case_id}: {mask_path}")
            failed.append(case_id)
            continue

        img_sitk = sitk.ReadImage(str(img_path))
        mask_sitk = sitk.ReadImage(str(mask_path))

        img_arr = sitk.GetArrayFromImage(img_sitk)
        mask_arr = sitk.GetArrayFromImage(mask_sitk)

        if img_arr.shape != mask_arr.shape:
            print(f"  [ERROR] Shape mismatch for {case_id}: "
                  f"image {img_arr.shape} vs mask {mask_arr.shape}")
            failed.append(case_id)
            continue

        liver_mask = mask_arr > 0
        if keep_lcc:
            liver_mask = keep_largest_connected_component(liver_mask)

        bbox = compute_bbox(liver_mask)
        if bbox is None:
            print(f"  [ERROR] Empty liver mask for {case_id}")
            failed.append(case_id)
            continue

        cropped, index, crop_size = crop_sitk_with_margin(img_sitk, bbox, margin_mm)

        arr = sitk.GetArrayFromImage(cropped).astype(np.float32)
        arr = np.clip(arr, hu_min, hu_max)
        out_sitk = sitk.GetImageFromArray(arr)
        out_sitk.CopyInformation(cropped)
        sitk.WriteImage(out_sitk, str(out_dir / img_path.name))

        print(f"  {case_id}: {img_sitk.GetSize()} → crop@{index}  size={crop_size}")

    n_ok = len(img_files) - len(failed)
    print(f"[Step 2] Done. {n_ok}/{len(img_files)} cases saved.")
    if failed:
        print(f"  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", nargs="+", type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument("--liver_pred_dir",     default=str(LIVER_PRED_DIR))
    parser.add_argument("--liver_postproc_dir", default=str(LIVER_POSTPROC_DIR))
    parser.add_argument("--img_dir",            default=str(IMG_DIR))
    parser.add_argument("--cropped_img_dir",    default=str(CROPPED_IMG_DIR))
    parser.add_argument("--margin_mm",          type=float, default=MARGIN_MM)
    parser.add_argument("--hu_min",             type=float, default=HU_MIN)
    parser.add_argument("--hu_max",             type=float, default=HU_MAX)
    parser.add_argument("--all_nonzero",        action="store_true")
    args = parser.parse_args()

    target_labels = None if args.all_nonzero else TARGET_LABELS

    if 1 in args.steps:
        run_step1(
            pred_dir=Path(args.liver_pred_dir),
            out_dir=Path(args.liver_postproc_dir),
            target_labels=target_labels,
            output_label=OUTPUT_LABEL,
            keep_lcc=KEEP_LCC,
            fill_holes=FILL_HOLES,
        )

    if 2 in args.steps:
        run_step2(
            img_dir=Path(args.img_dir),
            mask_dir=Path(args.liver_postproc_dir),
            out_dir=Path(args.cropped_img_dir),
            margin_mm=args.margin_mm,
            keep_lcc=KEEP_LCC,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
        )


if __name__ == "__main__":
    main()
