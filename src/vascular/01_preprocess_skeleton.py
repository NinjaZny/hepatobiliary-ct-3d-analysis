"""
Script 01 — Preprocessing & Skeletonization
============================================

What this script does
---------------------
Step 1 : Load raw vessel segmentation mask (NIfTI) from LIRCAD labels
         or a precomputed full-mask file.
Step 2 : Keep the N largest connected components — removes small noise blobs.
Step 3 : Crop to bounding box — all heavy operations run on a tight sub-volume
         to save memory and time; results are pasted back to full size for saving.
Step 4 : Morphological preprocessing — binary closing (fills small holes in
         the vessel wall) followed by optional opening (removes hair-thin spurs).
Step 5 : Lee-algorithm 3-D skeletonization — reduces the binary mask to a
         1-voxel-wide centreline that preserves topology.
Step 6 : EDT radius map — distance_transform_edt gives the Euclidean distance
         from each voxel to the nearest background voxel, i.e. the local vessel
         radius in mm when evaluated on the mask.
Step 7 : Iterative leaf pruning — repeatedly removes terminal (degree-1)
         branches shorter than `leaf_min_length_mm`, leaving only meaningful
         branches. Stops when no more short stubs remain.
Step 8 : Branch labelling — bifurcation voxels (degree ≥ 3) are removed;
         the remaining connected segments are labelled with unique integer IDs.
         Endpoint and bifurcation voxel maps are also saved.

Inputs
------
  CLI arguments:
    --case   CASE_ID      e.g. 001  (default: 001)
    --vessel VESSEL       pv | hv | both  (default: both)
    --all                 process all 20 cases in sequence
    --no-vis              skip visualisation (useful in batch mode)
    --keep-components N   override default keep_components
    --closing-radius  R   override morphological closing radius
    --opening-radius  R   override morphological opening radius
    --leaf-min-length L   override minimum leaf branch length (mm)

  Data on disk (auto-selected):
    SKEL_FULL_DIR / case_{id}_{vessel}_full.nii.gz   (if available)
    LIRCAD_LABELS / image{num}.nii.gz                (fallback)

Outputs  (saved to OUTPUT_DIR / case_{id}/)
-----------
  {vessel}_clean_mask.nii.gz       binary mask after component filter + morphology
  {vessel}_skeleton_raw.nii.gz     skeleton BEFORE leaf pruning  (binary uint8)
  {vessel}_skeleton_pruned.nii.gz  skeleton AFTER leaf pruning   (binary uint8)
  {vessel}_radius_map.nii.gz       EDT local radius per voxel    (float32, mm)
  {vessel}_branch_labels.nii.gz    branch segment IDs on skeleton (uint16)
  {vessel}_endpoints.nii.gz        degree-1 endpoint voxels      (binary uint8)
  {vessel}_bifurcations.nii.gz     degree-≥3 bifurcation voxels  (binary uint8)

Visualisation  (saved to OUTPUT_DIR / case_{id}/)
--------------
  {vessel}_preprocess_vis.png
    Row 1 (2-D axial slices of the representative z-plane):
      Panel 1 : raw mask (red) vs clean mask (blue)
      Panel 2 : skeleton before (red) and after pruning (green) on mask
      Panel 3 : EDT radius map heatmap + pruned skeleton overlay (white)
      Panel 4 : branch labels (tab20) + endpoints (red) + bifurcations (cyan)
    Row 2 (3-D scatter):
      Panel 5 : pruned skeleton coloured by local radius
      Panel 6 : pruned skeleton coloured by branch ID + EP/BIF highlighted
      Panel 7 : pruned (white) vs removed stubs (red) — pruning effect
      Panel 8 : branch length histogram with median line

Usage
-----
  python 01_preprocess_skeleton.py --case 001 --vessel pv
  python 01_preprocess_skeleton.py --case 001 --vessel both
  python 01_preprocess_skeleton.py --all --no-vis
"""

import argparse
from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.morphology import skeletonize_3d, ball

try:
    from tqdm import tqdm as _tqdm
    def tqdm(it, **kw): return _tqdm(it, **kw)
    def tqdm_write(s):  _tqdm.write(s)
except ImportError:
    def tqdm(it, **kw): return it
    def tqdm_write(s):  print(s)


# ── paths ──────────────────────────────────────────────────────────────────────
BASE          = Path(__file__).parent
LIRCAD_LABELS = BASE / "LIRCAD" / "Labels"
SKEL_FULL_DIR = BASE / "skeletons_full"
OUTPUT_DIR    = BASE / "output"

TRAIN_IDS       = [f"{i:03d}" for i in range(1, 17)]
TEST_IDS        = [f"{i:03d}" for i in range(17, 21)]
LIRCAD_OFFSET   = 53
Z_CORRECTION_MM = +1.6

# ── per-vessel default config ──────────────────────────────────────────────────
PV_CFG = dict(keep_components=1, closing_radius=1, opening_radius=0,
              leaf_min_length_mm=3.0, smooth_sigma=0.5)
HV_CFG = dict(keep_components=5, closing_radius=1, opening_radius=0,
              leaf_min_length_mm=4.0, smooth_sigma=0.5)

# 26-connectivity kernel (centre excluded) used for degree computation
CONN26 = np.ones((3, 3, 3), dtype=np.uint8)
CONN26[1, 1, 1] = 0

MAX_SCATTER = 60_000   # max points in 3-D scatter plots


# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_full_mask(case_id, vessel):
    """Load binary vessel mask.  Prefers a cached full-mask NIfTI; falls back
    to extracting the relevant label from the raw LIRCAD annotation."""
    p = SKEL_FULL_DIR / f"case_{case_id}_{vessel}_full.nii.gz"
    if p.exists():
        img = sitk.ReadImage(str(p))
        return sitk.GetArrayFromImage(img).astype(np.uint8), img

    label_val  = 1 if vessel == "pv" else 2
    lircad_num = int(case_id) + LIRCAD_OFFSET
    raw  = sitk.ReadImage(str(LIRCAD_LABELS / f"image{lircad_num:03d}.nii.gz"))
    arr  = sitk.GetArrayFromImage(raw)
    origin = list(raw.GetOrigin())
    origin[2] += Z_CORRECTION_MM
    img  = sitk.GetImageFromArray((arr == label_val).astype(np.uint8))
    img.SetOrigin(origin)
    img.SetSpacing(raw.GetSpacing())
    img.SetDirection(raw.GetDirection())
    return sitk.GetArrayFromImage(img).astype(np.uint8), img


def save_nifti(arr, ref, path, dtype=np.uint8):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(arr.astype(dtype))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path), useCompression=True)


def get_bbox(mask, pad=8):
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return tuple(slice(0, s) for s in mask.shape)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))


def uncrop(arr_small, full_shape, bbox, fill=0):
    out = np.full(full_shape, fill, dtype=arr_small.dtype)
    out[bbox] = arr_small
    return out


# ── processing helpers ─────────────────────────────────────────────────────────

def keep_largest_cc(mask, n, min_size=20):
    """Keep the n largest connected components; discard the rest."""
    lab, _ = ndi.label(mask, structure=np.ones((3, 3, 3)))
    sizes  = np.bincount(lab.ravel())
    sizes[0] = 0
    top = [i for i in np.argsort(sizes)[::-1] if sizes[i] >= min_size][:n]
    return np.isin(lab, top).astype(np.uint8)


def preprocess_mask(mask, closing_r=1, opening_r=0):
    result = mask.astype(bool)
    if closing_r > 0:
        result = ndi.binary_closing(result, structure=ball(closing_r))
    if opening_r > 0:
        result = ndi.binary_opening(result, structure=ball(opening_r))
    return result.astype(np.uint8)


def compute_degree(skel):
    """Count 26-connected skeleton neighbours for each skeleton voxel."""
    deg = ndi.convolve(skel.astype(np.uint8), CONN26, mode="constant", cval=0)
    return deg * skel




def split_branches(skel, bif_mask):
    """Remove bifurcation voxels and label each remaining connected segment."""
    lab, _ = ndi.label(
        skel & ~bif_mask.astype(bool),
        structure=np.ones((3, 3, 3)))
    return lab.astype(np.uint16)


def iterative_leaf_prune(skel, sp_zyx, lml, max_iters=50):
    """Remove terminal branches shorter than lml mm, iterating until stable."""
    vox_diag = float(np.mean(sp_zyx))
    current  = skel.astype(bool)
    for it in range(max_iters):
        deg  = compute_degree(current.astype(np.uint8))
        bif  = (current & (deg >= 3)).astype(np.uint8)
        ep   = current & (deg == 1)
        blab = split_branches(current.astype(np.uint8), bif)
        to_rm   = np.zeros_like(current, dtype=bool)
        removed = 0
        for bid in range(1, int(blab.max()) + 1):
            seg = blab == bid
            if not seg.any() or not ep[seg].any():
                continue
            if int(seg.sum()) * vox_diag < lml:
                to_rm  |= seg
                removed += 1
        if removed == 0:
            break
        kept     = current & ~to_rm
        bif_keep = ndi.binary_dilation(kept, structure=np.ones((3, 3, 3))) & bif.astype(bool)
        current  = kept | bif_keep
    return current.astype(np.uint8), it + 1


# ── main processing function ───────────────────────────────────────────────────

def process_vessel(case_id, vessel, cfg, out_dir):
    tag = f"case_{case_id} [{vessel.upper()}]"
    tqdm_write(f"  {tag}")

    # ── load ──────────────────────────────────────────────────────────────────
    mask_arr, ref_img = load_full_mask(case_id, vessel)
    sp_xyz     = ref_img.GetSpacing()           # (x, y, z) in mm
    sp_zyx     = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
    full_shape = mask_arr.shape
    tqdm_write(f"    {tag}  loaded: {full_shape}  spacing={sp_zyx} mm")

    # ── step 1: component filter ───────────────────────────────────────────────
    tqdm_write(f"    {tag}  1/6  component filter  (keep={cfg['keep_components']}) ...")
    clean_raw = keep_largest_cc(mask_arr, cfg["keep_components"])

    # ── step 1b: bounding-box crop ────────────────────────────────────────────
    bbox        = get_bbox(clean_raw, pad=8)
    crop_shape  = tuple(s.stop - s.start for s in bbox)
    crop_origin = np.array([s.start for s in bbox], dtype=int)
    tqdm_write(f"    {tag}       bbox crop: {full_shape} → {crop_shape}")

    # ── step 2: morphological preprocessing ───────────────────────────────────
    tqdm_write(f"    {tag}  2/6  preprocessing  "
               f"(closing={cfg['closing_radius']}, opening={cfg['opening_radius']}) ...")
    clean_c = preprocess_mask(clean_raw[bbox],
                              closing_r=cfg["closing_radius"],
                              opening_r=cfg["opening_radius"])

    # ── step 3: Gaussian smoothing before skeletonization ─────────────────────
    # Blurs the mask boundary to reduce staircase artifacts on diagonal vessels.
    # sigma=0 skips smoothing entirely.
    sigma = cfg.get("smooth_sigma", 0.5)
    if sigma > 0:
        tqdm_write(f"    {tag}  3/7  Gaussian smooth  (sigma={sigma}) ...")
        smooth_c = (gaussian_filter(clean_c.astype(float), sigma=sigma) > 0.5
                    ).astype(np.uint8)
    else:
        smooth_c = clean_c
    tqdm_write(f"    {tag}       mask vox: {clean_c.sum():,} → {smooth_c.sum():,}")

    # ── step 4: Lee skeletonization ───────────────────────────────────────────
    tqdm_write(f"    {tag}  4/7  skeletonize ({smooth_c.sum():,} vox) ...")
    skel_raw_c = (skeletonize_3d(smooth_c.astype(bool)) > 0).astype(np.uint8)
    tqdm_write(f"    {tag}       raw skeleton: {skel_raw_c.sum():,} vox")

    # ── step 5: EDT radius map (on original clean mask, not smoothed) ─────────
    tqdm_write(f"    {tag}  5/7  EDT radius map ...")
    radius_map_c = distance_transform_edt(clean_c.astype(bool),
                                          sampling=sp_zyx).astype(np.float32)

    # ── step 6: iterative leaf pruning ────────────────────────────────────────
    tqdm_write(f"    {tag}  6/7  leaf pruning  "
               f"(leaf_min_length={cfg['leaf_min_length_mm']} mm) ...")
    skel_pruned_c, n_iters = iterative_leaf_prune(
        skel_raw_c, sp_zyx, cfg["leaf_min_length_mm"])
    tqdm_write(f"    {tag}       {skel_raw_c.sum():,} → {skel_pruned_c.sum():,} vox  "
               f"({n_iters} iter)")

    # ── step 7: branch labelling ──────────────────────────────────────────────
    tqdm_write(f"    {tag}  7/7  branch labelling ...")
    deg_c     = ndi.convolve(skel_pruned_c, CONN26, mode="constant", cval=0) * skel_pruned_c
    bif_c     = (deg_c >= 3).astype(np.uint8)
    ep_c      = (deg_c == 1).astype(np.uint8)
    lab_c, n_branches = ndi.label(skel_pruned_c & ~bif_c.astype(bool),
                                   structure=np.ones((3, 3, 3)))
    lab_c = lab_c.astype(np.uint16)
    n_ep  = int(ep_c.sum())
    n_bif = int(ndi.label(bif_c, structure=np.ones((3,3,3)))[1])
    tqdm_write(f"    {tag}       branches={n_branches}  endpoints={n_ep}  bif_clusters={n_bif}")

    # ── uncrop and save ────────────────────────────────────────────────────────
    clean_full   = uncrop(clean_c,        full_shape, bbox)
    skel_raw     = uncrop(skel_raw_c,     full_shape, bbox)
    skel_pruned  = uncrop(skel_pruned_c,  full_shape, bbox)
    radius_map   = uncrop(radius_map_c,   full_shape, bbox, fill=np.float32(0))
    branch_lab   = uncrop(lab_c,          full_shape, bbox)
    endpoints    = uncrop(ep_c,           full_shape, bbox)
    bifurcations = uncrop(bif_c,          full_shape, bbox)

    save_nifti(clean_full,   ref_img, out_dir / f"{vessel}_clean_mask.nii.gz")
    save_nifti(skel_raw,     ref_img, out_dir / f"{vessel}_skeleton_raw.nii.gz")
    save_nifti(skel_pruned,  ref_img, out_dir / f"{vessel}_skeleton_pruned.nii.gz")
    save_nifti(radius_map,   ref_img, out_dir / f"{vessel}_radius_map.nii.gz",
               dtype=np.float32)
    save_nifti(branch_lab,   ref_img, out_dir / f"{vessel}_branch_labels.nii.gz",
               dtype=np.uint16)
    save_nifti(endpoints,    ref_img, out_dir / f"{vessel}_endpoints.nii.gz")
    save_nifti(bifurcations, ref_img, out_dir / f"{vessel}_bifurcations.nii.gz")
    tqdm_write(f"    {tag}  saved 7 NIfTI files")

    return dict(
        mask_raw=mask_arr,
        clean=clean_full,
        skel_raw=skel_raw,
        skel_pruned=skel_pruned,
        radius_map=radius_map,
        branch_lab=branch_lab,
        endpoints=endpoints,
        bifurcations=bifurcations,
        n_branches=n_branches,
        n_ep=n_ep,
        n_bif=n_bif,
        n_bif_raw=int(bif_c.sum()),
        ref_img=ref_img,
        sp_zyx=sp_zyx,
    )


# ── visualisation ──────────────────────────────────────────────────────────────

def _best_slice(arr):
    """Return the z-index with the most non-zero voxels."""
    counts = arr.sum(axis=(1, 2))
    return int(np.argmax(counts))



def visualize(case_id, vessel, data, out_dir, cfg):
    sp        = data["sp_zyx"]          # (z_mm, y_mm, x_mm)
    clean     = data["clean"]
    skel_r    = data["skel_raw"]
    skel_p    = data["skel_pruned"]
    rmap      = data["radius_map"]
    blab      = data["branch_lab"]
    ep        = data["endpoints"]
    bif       = data["bifurcations"]
    mask_raw  = data["mask_raw"]

    z0 = _best_slice(skel_p)
    # Panel 4 needs a slice that actually contains bifurcation voxels.
    # Find the z with the most bifurcation voxels; fall back to z0 if none.
    bif_counts = bif.sum(axis=(1, 2))
    z4 = int(np.argmax(bif_counts)) if bif_counts.max() > 0 else z0

    fig = plt.figure(figsize=(24, 11))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"case_{case_id}  {vessel.upper()}  —  Script 01: Preprocessing & Skeletonization\n"
        f"branches={data['n_branches']}   endpoints={data['n_ep']}   "
        f"bif clusters={data['n_bif']} (raw vox={data['n_bif_raw']})   "
        f"smooth σ={cfg.get('smooth_sigma', 0.5)}   slice z={z0}",
        fontsize=11, fontweight="bold", color="white")

    common_ax = dict(facecolor="#111111")

    # ── row 1: 2-D slice panels ────────────────────────────────────────────────
    # Tight crop around the vessel for all 2-D panels so skeleton is visible.
    ys, xs = np.where(clean[z0] > 0)
    if len(ys):
        pad2d = 12
        r0, r1 = max(ys.min()-pad2d, 0), min(ys.max()+pad2d+1, clean.shape[1])
        c0, c1 = max(xs.min()-pad2d, 0), min(xs.max()+pad2d+1, clean.shape[2])
    else:
        r0, r1 = 0, clean.shape[1]
        c0, c1 = 0, clean.shape[2]

    def _crop(arr2d):
        return arr2d[r0:r1, c0:c1]

    def _show_mask_bg(ax, alpha=0.30):
        """Semi-transparent clean mask as anatomical reference."""
        ax.imshow(_crop(clean[z0]), cmap="gray", vmin=0, vmax=1,
                  alpha=alpha, interpolation="nearest")

    def _overlay_crop(ax, binary2d, color, alpha=1.0, s=6):
        ys2, xs2 = np.where(_crop(binary2d) > 0)
        if len(xs2):
            ax.scatter(xs2, ys2, c=color, s=s, alpha=alpha, linewidths=0)

    # Panel 1 — raw mask (semi-transparent red) vs clean mask (semi-transparent blue)
    #            показываем diff: красное = только в raw, синее = только в clean,
    #            фиолетовое = общее
    ax1 = fig.add_subplot(2, 4, 1, **common_ax)
    ax1.imshow(_crop(mask_raw[z0]), cmap="Reds",  alpha=0.55, vmin=0, vmax=1,
               interpolation="nearest")
    ax1.imshow(_crop(clean[z0]),    cmap="Blues", alpha=0.45, vmin=0, vmax=1,
               interpolation="nearest")
    # Highlight voxels removed by component filter in bright orange
    removed_cc = mask_raw[z0].astype(bool) & ~clean[z0].astype(bool)
    _overlay_crop(ax1, removed_cc, "#ff8800", alpha=0.9, s=3)
    ax1.set_title("mask: raw (red) / clean (blue)\norange = removed by CC filter",
                  fontsize=7, color="white")
    ax1.axis("off")

    # Panel 2 — clean mask (semi-transparent) + skeleton before/after pruning
    #            pruned stubs shown in orange to make the effect obvious
    ax2 = fig.add_subplot(2, 4, 2, **common_ax)
    _show_mask_bg(ax2, alpha=0.25)
    pruned_stubs = skel_r[z0].astype(bool) & ~skel_p[z0].astype(bool)
    _overlay_crop(ax2, skel_p[z0],    "#00ff88", alpha=1.0, s=8)   # kept — green
    _overlay_crop(ax2, pruned_stubs,  "#ff4400", alpha=1.0, s=8)   # removed — orange-red
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#00ff88",
               markersize=5, label="kept skeleton", linewidth=0),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#ff4400",
               markersize=5, label="pruned stubs",  linewidth=0),
    ], loc="lower right", fontsize=6, facecolor="#222222", labelcolor="white",
       edgecolor="gray", framealpha=0.8)
    ax2.set_title("skeleton after pruning (green)\npruned stubs (orange-red)",
                  fontsize=7, color="white")
    ax2.axis("off")

    # Panel 3 — EDT radius heatmap (masked to vessel) + pruned skeleton (white dots)
    ax3 = fig.add_subplot(2, 4, 3, **common_ax)
    rslice = _crop(rmap[z0]).astype(float)
    rslice[_crop(clean[z0]) == 0] = np.nan
    im3 = ax3.imshow(rslice, cmap="plasma", interpolation="nearest")
    cb3 = fig.colorbar(im3, ax=ax3, shrink=0.75, pad=0.02)
    cb3.set_label("radius (mm)", fontsize=7, color="white")
    cb3.ax.yaxis.set_tick_params(color="white", labelsize=6)
    plt.setp(cb3.ax.yaxis.get_ticklabels(), color="white")
    _overlay_crop(ax3, skel_p[z0], "white", alpha=0.9, s=6)
    ax3.set_title("EDT radius map\n+ pruned skeleton (white)",
                  fontsize=7, color="white")
    ax3.axis("off")

    # Panel 4 — slice chosen to maximise visible bifurcations (z4 ≠ z0 in general)
    #            semi-transparent mask + branch IDs coloured by tab20
    #            endpoints (★) and bifurcations (◆) drawn on top as large markers
    ys4, xs4 = np.where(clean[z4] > 0)
    if len(ys4):
        r40, r41 = max(ys4.min()-pad2d, 0), min(ys4.max()+pad2d+1, clean.shape[1])
        c40, c41 = max(xs4.min()-pad2d, 0), min(xs4.max()+pad2d+1, clean.shape[2])
    else:
        r40, r41, c40, c41 = r0, r1, c0, c1

    def _crop4(arr2d):
        return arr2d[r40:r41, c40:c41]

    ax4 = fig.add_subplot(2, 4, 4, **common_ax)
    ax4.imshow(_crop4(clean[z4]), cmap="gray", vmin=0, vmax=1,
               alpha=0.20, interpolation="nearest")
    bslice4 = _crop4(blab[z4]).astype(float)
    bslice4[bslice4 == 0] = np.nan
    if not np.all(np.isnan(bslice4)):
        ax4.imshow(bslice4 % 20, cmap="tab20", alpha=0.85,
                   interpolation="nearest", vmin=0, vmax=19)
    for node_mask, col, mk, sz, lbl in [
            (ep[z4],  "#ff2222", "*", 80, "endpoint"),
            (bif[z4], "#00eeff", "D", 30, "bifurcation")]:
        ny, nx = np.where(_crop4(node_mask) > 0)
        if len(nx):
            ax4.scatter(nx, ny, c=col, marker=mk, s=sz,
                        alpha=1.0, linewidths=0, zorder=5, label=lbl)
    ax4.legend(loc="lower right", fontsize=6, facecolor="#222222",
               labelcolor="white", edgecolor="gray", framealpha=0.8)
    ax4.set_title(f"branch IDs (tab20) + mask (gray)  z={z4}\n★ endpoint  ◆ bifurcation",
                  fontsize=7, color="white")
    ax4.axis("off")

    # ── row 2: 3-D scatter panels ──────────────────────────────────────────────
    elev, azim = 25, -55

    def _3d_setup(ax, title):
        ax.set_facecolor("#111111")
        ax.tick_params(labelsize=5, colors="white")
        ax.set_xlabel("x mm", fontsize=6, color="white")
        ax.set_ylabel("y mm", fontsize=6, color="white")
        ax.set_zlabel("z mm", fontsize=6, color="white")
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_title(title, fontsize=8, color="white")
        ax.view_init(elev, azim)

    rng = np.random.default_rng(0)

    # Panel 5 — pruned skeleton coloured by radius
    ax5 = fig.add_subplot(2, 4, 5, projection="3d")
    idx = np.argwhere(skel_p > 0)
    if len(idx) > MAX_SCATTER:
        idx = idx[rng.choice(len(idx), MAX_SCATTER, replace=False)]
    if len(idx):
        rv  = rmap[idx[:, 0], idx[:, 1], idx[:, 2]]
        sc5 = ax5.scatter(idx[:, 2]*sp[2], idx[:, 1]*sp[1], idx[:, 0]*sp[0],
                          c=rv, cmap="plasma", s=3, alpha=0.9,
                          linewidths=0, depthshade=False,
                          vmin=0, vmax=float(rv.max()) if rv.max() > 0 else 1)
        cb5 = fig.colorbar(sc5, ax=ax5, shrink=0.5, pad=0.05)
        cb5.set_label("radius mm", fontsize=7, color="white")
        plt.setp(cb5.ax.yaxis.get_ticklabels(), color="white")
    _3d_setup(ax5, "3D skeleton: coloured by radius")

    # Panel 6 — pruned skeleton coloured by branch ID + EP/BIF nodes
    ax6 = fig.add_subplot(2, 4, 6, projection="3d")
    idx = np.argwhere(skel_p > 0)
    if len(idx) > MAX_SCATTER:
        idx = idx[rng.choice(len(idx), MAX_SCATTER, replace=False)]
    if len(idx):
        bv = (blab[idx[:, 0], idx[:, 1], idx[:, 2]] % 20).astype(float)
        ax6.scatter(idx[:, 2]*sp[2], idx[:, 1]*sp[1], idx[:, 0]*sp[0],
                    c=bv, cmap="tab20", s=3, alpha=0.9,
                    linewidths=0, depthshade=False, vmin=0, vmax=19)
    for node_arr, col, sz in [(ep, "red", 25), (bif, "cyan", 12)]:
        ni = np.argwhere(node_arr > 0)
        if len(ni):
            ax6.scatter(ni[:, 2]*sp[2], ni[:, 1]*sp[1], ni[:, 0]*sp[0],
                        c=col, s=sz, alpha=1.0, linewidths=0, depthshade=False)
    _3d_setup(ax6, "3D skeleton: branch IDs + EP (red) / BIF (cyan)")

    # Panel 7 — pruning effect: kept (white) vs removed stubs (red)
    ax7 = fig.add_subplot(2, 4, 7, projection="3d")
    removed = skel_r.astype(bool) & ~skel_p.astype(bool)
    for arr_pt, col, sz, lbl in [
            (skel_p, "white", 2,  "kept"),
            (removed, "red",   5,  "pruned stubs")]:
        ni = np.argwhere(arr_pt > 0)
        if len(ni) > MAX_SCATTER:
            ni = ni[rng.choice(len(ni), MAX_SCATTER, replace=False)]
        if len(ni):
            ax7.scatter(ni[:, 2]*sp[2], ni[:, 1]*sp[1], ni[:, 0]*sp[0],
                        c=col, s=sz, alpha=0.9 if col == "white" else 1.0,
                        linewidths=0, depthshade=False, label=lbl)
    ax7.legend(loc="upper left", fontsize=6, labelcolor="white",
               facecolor="#222222", edgecolor="gray")
    ax7.set_facecolor("black")
    _3d_setup(ax7, "Pruning effect: kept (white) vs removed stubs (red)")

    # Panel 8 — branch length distribution histogram
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.set_facecolor("#111111")
    vox_diag = float(np.mean(sp))
    br_lengths = []
    for bid in range(1, data["n_branches"] + 1):
        n = int((blab == bid).sum())
        if n > 0:
            br_lengths.append(n * vox_diag)
    if br_lengths:
        ax8.hist(br_lengths, bins=40, color="steelblue", edgecolor="none")
        med = float(np.median(br_lengths))
        ax8.axvline(med, color="orange", lw=1.5,
                    label=f"median {med:.1f} mm")
        ax8.axvline(cfg["leaf_min_length_mm"], color="red", lw=1.2,
                    linestyle="--", label=f"prune threshold {cfg['leaf_min_length_mm']} mm")
        ax8.legend(fontsize=7, labelcolor="white",
                   facecolor="#222222", edgecolor="gray")
    ax8.set_xlabel("branch length (mm)", fontsize=8, color="white")
    ax8.set_ylabel("count",              fontsize=8, color="white")
    ax8.set_title("Branch length distribution", fontsize=8, color="white")
    ax8.tick_params(labelsize=7, colors="white")
    for spine in ax8.spines.values():
        spine.set_edgecolor("#444444")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    png = out_dir / f"{vessel}_preprocess_vis.png"
    fig.savefig(str(png), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    tqdm_write(f"    figure saved → {png.name}  (interactive window opening ...)")
    plt.show()
    plt.close(fig)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Script 01 — Preprocessing & Skeletonization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case",            default="001",
                    help="case ID, e.g. 001")
    ap.add_argument("--vessel",          default="both",
                    choices=["pv", "hv", "both"])
    ap.add_argument("--all",             action="store_true",
                    help="process all 20 cases")
    ap.add_argument("--no-vis",          action="store_true",
                    help="skip visualisation (batch mode)")
    ap.add_argument("--keep-components", type=int,   default=None,
                    help="number of connected components to keep")
    ap.add_argument("--closing-radius",  type=int,   default=None,
                    help="morphological closing ball radius (vox)")
    ap.add_argument("--opening-radius",  type=int,   default=None,
                    help="morphological opening ball radius (vox)")
    ap.add_argument("--leaf-min-length", type=float, default=None,
                    help="minimum leaf branch length to keep (mm)")
    ap.add_argument("--smooth-sigma",    type=float, default=None,
                    help="Gaussian blur sigma before skeletonization (0=off, default 0.5)")
    ap.add_argument("--merge-radius",    type=float, default=None,
                    help="merge bifurcation clusters within this distance mm (default 2.0)")
    ap.add_argument("--node-ball-radius", type=float, default=None,
                    help="node ball radius mm used to cut branches (default 1.0)")
    ap.add_argument("--output-dir", default=None,
                    help="override output directory (default: <script_dir>/output)")
    ap.add_argument("--skel-dir",   default=None,
                    help="override skeletons_full directory for input masks")
    ap.add_argument("--ids",        default=None,
                    help="comma-separated list of case IDs to process (e.g. 001,002,003)")
    args = ap.parse_args()

    global OUTPUT_DIR, SKEL_FULL_DIR
    if args.output_dir:
        OUTPUT_DIR    = Path(args.output_dir)
    if args.skel_dir:
        SKEL_FULL_DIR = Path(args.skel_dir)

    pv_cfg = dict(PV_CFG)
    hv_cfg = dict(HV_CFG)
    if args.keep_components is not None:
        pv_cfg["keep_components"]    = args.keep_components
        hv_cfg["keep_components"]    = args.keep_components
    if args.closing_radius is not None:
        pv_cfg["closing_radius"]     = args.closing_radius
        hv_cfg["closing_radius"]     = args.closing_radius
    if args.opening_radius is not None:
        pv_cfg["opening_radius"]     = args.opening_radius
        hv_cfg["opening_radius"]     = args.opening_radius
    if args.leaf_min_length is not None:
        pv_cfg["leaf_min_length_mm"] = args.leaf_min_length
        hv_cfg["leaf_min_length_mm"] = args.leaf_min_length
    if args.smooth_sigma is not None:
        pv_cfg["smooth_sigma"]       = args.smooth_sigma
        hv_cfg["smooth_sigma"]       = args.smooth_sigma
    if args.merge_radius is not None:
        pv_cfg["merge_radius_mm"]    = args.merge_radius
        hv_cfg["merge_radius_mm"]    = args.merge_radius
    if args.node_ball_radius is not None:
        pv_cfg["node_ball_radius_mm"] = args.node_ball_radius
        hv_cfg["node_ball_radius_mm"] = args.node_ball_radius

    vessels = ["pv", "hv"] if args.vessel == "both" else [args.vessel]
    cfgs    = {"pv": pv_cfg, "hv": hv_cfg}

    case_list = ([c.strip() for c in args.ids.split(",")] if args.ids
                 else TRAIN_IDS + TEST_IDS)

    if args.all or args.ids:
        for cid in tqdm(case_list, desc="cases", unit="case"):
            print(f"\n=== case_{cid} ===")
            out = OUTPUT_DIR / f"case_{cid}"
            out.mkdir(parents=True, exist_ok=True)
            for v in vessels:
                process_vessel(cid, v, cfgs[v], out)
        print(f"\nAll done.  Output → {OUTPUT_DIR}")
    else:
        print(f"\n=== case_{args.case} ===")
        out = OUTPUT_DIR / f"case_{args.case}"
        out.mkdir(parents=True, exist_ok=True)
        for v in vessels:
            data = process_vessel(args.case, v, cfgs[v], out)
            if not args.no_vis:
                visualize(args.case, v, data, out, cfgs[v])
        print(f"\nOutput → {out}")


if __name__ == "__main__":
    main()
