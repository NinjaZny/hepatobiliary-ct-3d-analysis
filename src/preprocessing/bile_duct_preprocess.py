"""
Preprocessing pipeline for bile duct segmentation — Dataset334/337 protocol.

Steps applied to each case:
  1. Largest connected component of liver mask
  2. Liver bounding box crop + 5-voxel margin
  3. HU clip [-160, 240]
  4. Linear normalization -> [0, 1]

Single case:
    python preprocess.py --ct <ct.nii.gz> --liver <liver_mask.nii.gz> --out <output_0000.nii.gz>

Batch (expects paired files in two directories, matched by filename stem):
    python preprocess.py --ct_dir <ct_dir/> --liver_dir <liver_dir/> --out_dir <out_dir/>
"""

import argparse
import os
import sys
import numpy as np
import nibabel as nib
from scipy.ndimage import label as cc_label

HU_MIN   = -160.0
HU_MAX   =  240.0
MARGIN   =  5       # voxels


def largest_cc(mask: np.ndarray) -> np.ndarray:
    labeled, n = cc_label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    return (labeled == sizes.argmax()).astype(mask.dtype)


def liver_bbox(mask: np.ndarray, margin: int, shape: tuple) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    bbox = []
    for ax, dim in enumerate(shape):
        lo = max(0,       int(coords[:, ax].min()) - margin)
        hi = min(dim - 1, int(coords[:, ax].max()) + margin)
        bbox.append((lo, hi))
    return bbox


def update_affine(affine: np.ndarray, start: list[int]) -> np.ndarray:
    a = affine.copy()
    a[:3, 3] = affine[:3, :3] @ np.array(start, dtype=float) + affine[:3, 3]
    return a


def resample_mask_to_ct(liver_nib, ct_nib):
    """Return liver mask resampled into CT voxel grid (nearest-neighbour)."""
    if liver_nib.shape == ct_nib.shape:
        return (liver_nib.get_fdata() > 0.5)
    from nibabel.processing import resample_from_to
    liver_rs = resample_from_to(liver_nib, ct_nib, order=0, mode="constant", cval=0)
    return (liver_rs.get_fdata() > 0.5)


def preprocess_one(ct_path: str, liver_path: str, out_path: str) -> None:
    ct_nib    = nib.load(ct_path)
    liver_nib = nib.load(liver_path)

    ct_data   = ct_nib.get_fdata(dtype=np.float32)
    liver     = resample_mask_to_ct(liver_nib, ct_nib)

    # step 1 — largest connected component
    liver = largest_cc(liver)
    if not liver.any():
        sys.exit(f"[ERROR] Empty liver mask: {liver_path}")

    # step 2 — bounding box crop
    bbox = liver_bbox(liver, MARGIN, ct_data.shape)
    sl   = tuple(slice(lo, hi + 1) for lo, hi in bbox)
    crop = ct_data[sl]

    # step 3 — HU clip
    clipped = np.clip(crop, HU_MIN, HU_MAX).astype(np.float32)

    # step 4 — normalize to [0, 1]
    normed = (clipped - HU_MIN) / (HU_MAX - HU_MIN)

    # save
    starts     = [lo for lo, _ in bbox]
    new_affine = update_affine(ct_nib.affine, starts)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    nib.save(nib.Nifti1Image(normed, new_affine), out_path)
    print(f"  saved  shape={normed.shape}  val=[{normed.min():.3f},{normed.max():.3f}]  -> {out_path}")


def _stem(fname: str) -> str:
    """Remove .nii.gz or .nii suffix."""
    for ext in (".nii.gz", ".nii"):
        if fname.endswith(ext):
            return fname[: -len(ext)]
    return fname


def run_batch(ct_dir: str, liver_dir: str, out_dir: str) -> None:
    ct_files = sorted(f for f in os.listdir(ct_dir) if f.endswith((".nii", ".nii.gz")))
    if not ct_files:
        sys.exit(f"[ERROR] No .nii/.nii.gz files found in {ct_dir}")

    os.makedirs(out_dir, exist_ok=True)
    errors = []

    for ct_fname in ct_files:
        stem = _stem(ct_fname)
        ct_path    = os.path.join(ct_dir, ct_fname)

        # try exact match first, then stem without _0000
        liver_fname = ct_fname
        if not os.path.exists(os.path.join(liver_dir, liver_fname)):
            # strip _0000 channel suffix if present
            bare = stem.removesuffix("_0000")
            for candidate in (f"{bare}.nii.gz", f"{bare}.nii", f"{stem}.nii.gz"):
                if os.path.exists(os.path.join(liver_dir, candidate)):
                    liver_fname = candidate
                    break
            else:
                print(f"  [SKIP] no liver mask found for {ct_fname}")
                errors.append(ct_fname)
                continue

        liver_path = os.path.join(liver_dir, liver_fname)
        out_fname  = f"{stem}_0000.nii.gz" if not stem.endswith("_0000") else f"{stem}.nii.gz"
        out_path   = os.path.join(out_dir, out_fname)

        print(f"  {ct_fname}")
        try:
            preprocess_one(ct_path, liver_path, out_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            errors.append(ct_fname)

    if errors:
        print(f"\nFailed ({len(errors)}): {errors}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Bile duct preprocessing — Dataset334/337 protocol.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g_single = ap.add_argument_group("single case")
    g_single.add_argument("--ct",    help="Input CT (.nii/.nii.gz)")
    g_single.add_argument("--liver", help="Liver mask (.nii/.nii.gz)")
    g_single.add_argument("--out",   help="Output path (.nii.gz, nnU-Net _0000 suffix recommended)")

    g_batch = ap.add_argument_group("batch mode")
    g_batch.add_argument("--ct_dir",    help="Directory of CT images")
    g_batch.add_argument("--liver_dir", help="Directory of liver masks (paired by filename)")
    g_batch.add_argument("--out_dir",   help="Output directory")

    args = ap.parse_args()

    if args.ct and args.liver and args.out:
        print(f"[bile_duct preprocess]  {args.ct}")
        preprocess_one(args.ct, args.liver, args.out)
    elif args.ct_dir and args.liver_dir and args.out_dir:
        print(f"[bile_duct preprocess batch]  {args.ct_dir}")
        run_batch(args.ct_dir, args.liver_dir, args.out_dir)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
