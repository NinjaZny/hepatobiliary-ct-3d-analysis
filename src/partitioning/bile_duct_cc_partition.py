#!/usr/bin/env python3
"""
NOTE (showcase copy): distances below are computed on voxel indices (no voxel
spacing), so the partition is a geometric, exploratory analysis in voxel space.

CC-based region partition of bile duct segmentation.

Algorithm:
  1. Find all CCs in the intra-hepatic bile duct mask (6-connectivity).
  2. Among the N_SEEDS (default 3) largest CCs, keep only those whose volume
     >= MIN_SEED_RATIO (default 5%) of total bile duct volume as seeds.
  3. Each remaining small CC is assigned as a whole to the nearest seed,
     measured from the CC centroid to the nearest seed voxel (KD-tree).
  4. If a liver mask is provided: bile duct is first clipped to the liver,
     and every liver voxel then inherits the label of its nearest bile voxel
     (EDT nearest-neighbour propagation).

Outputs per case (in --out_dir):
  <stem>_bile_parts.nii.gz   — bile duct labelled 1..K
  <stem>_bile_parts.csv      — per-part volumes
  <stem>_liver_parts.nii.gz  — liver labelled 1..K  (only if --liver given)

Usage:
    python cc_partition.py --bile bile.nii.gz --out_dir ./output
    python cc_partition.py --bile bile.nii.gz --liver liver.nii.gz --out_dir ./output

    # Batch: folders matched by stem name
    python cc_partition.py --bile /bile/ --liver /liver/ --out_dir ./output
"""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (label as scipy_label,
                           generate_binary_structure,
                           distance_transform_edt,
                           zoom)
from scipy.spatial import cKDTree

N_SEEDS        = 3
MIN_SEED_RATIO = 0.05


def voxel_volume_mm3(nii: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(nii.affine[:3, :3])))


def resample_nn(src_arr: np.ndarray, src_nii: nib.Nifti1Image,
                ref_nii: nib.Nifti1Image) -> np.ndarray:
    src_sp = np.abs(np.diag(src_nii.affine[:3, :3]))
    ref_sp = np.abs(np.diag(ref_nii.affine[:3, :3]))
    resampled = zoom(src_arr, src_sp / ref_sp, order=0)
    ref_shape = ref_nii.shape[:3]
    out = np.zeros(ref_shape, dtype=src_arr.dtype)
    sl = tuple(slice(0, min(s, r)) for s, r in zip(resampled.shape, ref_shape))
    out[sl] = resampled[sl]
    return out


def cc_partition(binary: np.ndarray,
                 n_seeds: int = N_SEEDS,
                 min_seed_ratio: float = MIN_SEED_RATIO):
    """Returns: part_map (uint8), n_cc (int), k (int, actual seeds used)."""
    struct = generate_binary_structure(3, 1)
    labeled, n_cc = scipy_label(binary, structure=struct)
    if n_cc == 0:
        return np.zeros_like(binary, dtype=np.uint8), 0, 0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    order = order[sizes[order] > 0]

    total_vox = int(binary.sum())
    candidates = order[:min(n_seeds, n_cc)]
    seeds = [lv for lv in candidates
             if sizes[lv] / total_vox >= min_seed_ratio]
    if not seeds:
        seeds = [order[0]]

    k = len(seeds)
    non_seeds = [lv for lv in order if lv not in seeds]

    part_map = np.zeros_like(binary, dtype=np.uint8)
    for pid, lv in enumerate(seeds, 1):
        part_map[labeled == lv] = pid

    if non_seeds:
        seed_trees = {pid: cKDTree(np.column_stack(np.where(part_map == pid)))
                      for pid in range(1, k + 1)}
        for lv in non_seeds:
            centroid = np.column_stack(np.where(labeled == lv)).mean(axis=0)
            nearest = min(range(1, k + 1),
                          key=lambda p: seed_trees[p].query(centroid)[0])
            part_map[labeled == lv] = nearest

    return part_map, n_cc, k


def propagate_to_liver(part_map: np.ndarray,
                       liver_mask: np.ndarray) -> np.ndarray:
    _, indices = distance_transform_edt(part_map == 0, return_indices=True)
    liver_pts = liver_mask.nonzero()
    nearest = tuple(indices[d][liver_pts] for d in range(3))
    liver_cc = np.zeros_like(part_map, dtype=np.uint8)
    liver_cc[liver_pts] = part_map[nearest]
    return liver_cc


def save_nii(arr: np.ndarray, ref_nii: nib.Nifti1Image, path: Path):
    img = nib.Nifti1Image(arr, affine=ref_nii.affine, header=ref_nii.header)
    img.set_data_dtype(np.uint8)
    nib.save(img, path)


def run(bile_path: Path, liver_path: Path | None, out_dir: Path,
        n_seeds: int = N_SEEDS, min_seed_ratio: float = MIN_SEED_RATIO):
    stem = bile_path.name.replace(".nii.gz", "").replace(".nii", "")

    bile_nii = nib.load(bile_path)
    bile_arr = (np.asarray(bile_nii.dataobj) > 0).astype(np.uint8)
    vox_vol  = voxel_volume_mm3(bile_nii)

    target, liver_mask = bile_arr, None

    if liver_path and liver_path.exists():
        liver_nii = nib.load(liver_path)
        liver_arr = (np.asarray(liver_nii.dataobj) > 0).astype(np.uint8)
        if liver_arr.shape != bile_arr.shape:
            liver_arr = resample_nn(liver_arr, liver_nii, bile_nii)
        target, liver_mask = bile_arr & liver_arr, liver_arr > 0

    part_map, n_cc, k = cc_partition(target, n_seeds, min_seed_ratio)
    save_nii(part_map, bile_nii, out_dir / f"{stem}_bile_parts.nii.gz")

    if liver_mask is not None:
        save_nii(propagate_to_liver(part_map, liver_mask),
                 bile_nii, out_dir / f"{stem}_liver_parts.nii.gz")

    rows = [{"part": pid,
             "voxels":  (vox := int((part_map == pid).sum())),
             "vol_mm3": round(vox * vox_vol, 2),
             "vol_mL":  round(vox * vox_vol * 1e-3, 3)}
            for pid in range(1, k + 1)]
    csv_path = out_dir / f"{stem}_bile_parts.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["part", "voxels", "vol_mm3", "vol_mL"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  {stem}: {n_cc} CCs -> {k} parts  "
          + "  ".join(f"P{r['part']}={r['vol_mL']:.2f}mL" for r in rows))


def find_match(folder: Path, stem: str) -> Path | None:
    for suffix in [".nii.gz", ".nii"]:
        p = folder / f"{stem}{suffix}"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bile",           required=True)
    parser.add_argument("--liver",          default=None)
    parser.add_argument("--out_dir",        required=True)
    parser.add_argument("--n_seeds",        type=int,   default=N_SEEDS)
    parser.add_argument("--min_seed_ratio", type=float, default=MIN_SEED_RATIO)
    args = parser.parse_args()

    bile_p  = Path(args.bile)
    liver_p = Path(args.liver) if args.liver else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if bile_p.is_dir():
        files = sorted(bile_p.glob("*.nii.gz")) + sorted(bile_p.glob("*.nii"))
        for bf in files:
            stem = bf.name.replace(".nii.gz", "").replace(".nii", "")
            lf = find_match(liver_p, stem) if liver_p and liver_p.is_dir() else liver_p
            run(bf, lf, out_dir, args.n_seeds, args.min_seed_ratio)
    else:
        run(bile_p, liver_p, out_dir, args.n_seeds, args.min_seed_ratio)


if __name__ == "__main__":
    main()
