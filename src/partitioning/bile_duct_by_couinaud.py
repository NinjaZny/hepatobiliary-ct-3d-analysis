#!/usr/bin/env python3
"""
Couinaud-segment labelling of bile duct segmentation.

Each bile duct voxel is labelled with its Couinaud segment (1-8) using a
pre-computed Couinaud mask. If the two masks have different grids, the
Couinaud mask is resampled to the bile duct grid (nearest-neighbour).

Outputs per case (in --out_dir):
  <stem>_couinaud.nii.gz   -- bile duct voxels labelled 1..8
  <stem>_couinaud.csv      -- per-segment: voxels, vol_mm3, vol_mL, pct_of_bile

Usage:
    python couinaud_analysis.py \\
        --bile bile.nii.gz --couinaud couinaud_mask.nii.gz --out_dir ./output

    # Batch: folders matched by stem name (or couinaud-<stem> prefix)
    python couinaud_analysis.py \\
        --bile /bile/ --couinaud /couinaud/ --out_dir ./output
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

SEGMENT_NAMES = {
    1: "Segment I",   2: "Segment II",  3: "Segment III", 4: "Segment IV",
    5: "Segment V",   6: "Segment VI",  7: "Segment VII", 8: "Segment VIII",
}


def voxel_volume_mm3(nii: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(nii.affine[:3, :3])))


def resample_nn(src_arr: np.ndarray, src_nii: nib.Nifti1Image,
                ref_nii: nib.Nifti1Image) -> np.ndarray:
    """Resample src_arr from src_nii space into ref_nii space (nearest-neighbour).
    Uses full affine transform so origin differences are handled correctly.
    """
    src_inv = np.linalg.inv(src_nii.affine)
    M = src_inv[:3, :3] @ ref_nii.affine[:3, :3]
    b = src_inv[:3, :3] @ ref_nii.affine[:3, 3] + src_inv[:3, 3]
    return affine_transform(
        src_arr.astype(np.float32), M, offset=b,
        output_shape=ref_nii.shape[:3], order=0, cval=0, prefilter=False,
    ).astype(src_arr.dtype)


def save_nii(arr: np.ndarray, ref_nii: nib.Nifti1Image, path: Path):
    img = nib.Nifti1Image(arr, affine=ref_nii.affine, header=ref_nii.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, path)


def run(bile_path: Path, couinaud_path: Path, out_dir: Path,
        out_stem: str | None = None, liver_path: Path | None = None):
    stem = out_stem or bile_path.name.replace(".nii.gz", "").replace(".nii", "")

    bile_nii = nib.load(bile_path)
    coui_nii = nib.load(couinaud_path)

    bile_arr = (np.asarray(bile_nii.dataobj) > 0).astype(np.uint8)
    coui_arr = np.round(np.asarray(coui_nii.dataobj)).astype(np.uint8)

    if coui_arr.shape != bile_arr.shape:
        print(f"  [resample] Couinaud {coui_arr.shape} -> bile {bile_arr.shape}")
        coui_arr = resample_nn(coui_arr, coui_nii, bile_nii)

    labelled = np.where(bile_arr > 0, coui_arr, 0).astype(np.uint8)
    save_nii(labelled, bile_nii, out_dir / f"{stem}_couinaud.nii.gz")

    vox_vol = voxel_volume_mm3(bile_nii)
    total   = int(bile_arr.sum())
    rows = []
    for seg in range(1, 9):
        vox = int((labelled == seg).sum())
        vol = vox * vox_vol
        rows.append({"segment": seg, "name": SEGMENT_NAMES[seg],
                     "voxels": vox, "vol_mm3": round(vol, 2),
                     "vol_mL": round(vol * 1e-3, 3),
                     "pct_of_bile": round(vox / total * 100, 2) if total else 0.0})

    csv_path = out_dir / f"{stem}_couinaud.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["segment", "name", "voxels", "vol_mm3", "vol_mL", "pct_of_bile"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {stem}: {total:,} bile vox  ({total * vox_vol * 1e-3:.2f} mL)")
    for r in rows:
        if r["voxels"] > 0:
            print(f"    Seg {r['segment']} {r['name']}: {r['vol_mL']:.3f} mL  ({r['pct_of_bile']:.1f}%)")

    # Liver Couinaud partition (optional)
    if liver_path is not None and liver_path.exists():
        liver_nii = nib.load(liver_path)
        liver_arr = (np.asarray(liver_nii.dataobj) > 0).astype(np.uint8)
        coui_liver = np.round(np.asarray(coui_nii.dataobj)).astype(np.uint8)
        if coui_liver.shape != liver_arr.shape:
            coui_liver = resample_nn(coui_liver, coui_nii, liver_nii)
        liver_labelled = np.where(liver_arr > 0, coui_liver, 0).astype(np.uint8)
        save_nii(liver_labelled, liver_nii, out_dir / f"{stem}_liver_couinaud.nii.gz")
        liver_vv = voxel_volume_mm3(liver_nii)
        print(f"  liver: {int(liver_arr.sum()):,} vox  ({int(liver_arr.sum()) * liver_vv * 1e-3:.1f} mL)")


def find_couinaud(bile_path: Path, couinaud_dir: Path) -> Path | None:
    stem = bile_path.name.replace(".nii.gz", "").replace(".nii", "")
    for suffix in [".nii.gz", ".nii"]:
        for prefix in ["", "couinaud-"]:
            p = couinaud_dir / f"{prefix}{stem}{suffix}"
            if p.exists():
                return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bile",     required=True)
    parser.add_argument("--couinaud", required=True)
    parser.add_argument("--out_dir",  required=True)
    args = parser.parse_args()

    bile_p  = Path(args.bile)
    coui_p  = Path(args.couinaud)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bile_p.is_dir():
        files = sorted(bile_p.glob("*.nii.gz")) + sorted(bile_p.glob("*.nii"))
        for bf in files:
            cf = find_couinaud(bf, coui_p) if coui_p.is_dir() else coui_p
            if cf is None:
                print(f"  [WARN] no Couinaud mask for {bf.name}, skipping")
                continue
            run(bf, cf, out_dir)
    else:
        run(bile_p, coui_p, out_dir)


if __name__ == "__main__":
    main()
