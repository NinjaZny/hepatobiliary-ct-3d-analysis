"""
Script 03 — Backbone Extraction & Level Assignment
===================================================

Inputs  (from Script 01 + 02 outputs in OUTPUT_DIR / case_{id}/)
-------
  {vessel}_graph.pkl                  Script 02 networkx Graph
  {vessel}_skeleton_pruned.nii.gz     Script 01 skeleton
  {vessel}_branch_labels.nii.gz       Script 01 branch IDs on skeleton
  {vessel}_clean_mask.nii.gz          Script 01 vessel mask

Processing
----------
1. Load Script 02 graph + Script 01 NIfTI files.
2. Identify root node (PV: thickest node, HV: extreme-z node).
3. Score every root-to-endpoint path  Σ(length × radius²).
4. Greedy diverse backbone selection: top-k paths with length-based overlap control.
   For HV, k=4 is selected first; then final 3 trunks are chosen by the rule:
   remove IVC-like trunk if angle to z-axis < threshold, otherwise keep top-3 by added score.
5. Trunk-aware BFS levelling: backbone = Level 0, neighbours = Level 1, …
6. Map graph branch IDs → NIfTI voxels via original_bids, build output arrays.
7. Save NIfTI outputs + branch_table.csv.
8. Visualise: 3-D line graph coloured by level + stat panels.

Outputs  (saved to OUTPUT_DIR / case_{id}/)
-----------
  {vessel}_backbone_centerline.nii.gz   backbone skeleton voxels (binary)
  {vessel}_backbone_mask.nii.gz         backbone vessel mask (binary)
  {vessel}_trunk_levels_centerline.nii.gz  level on skeleton (int16)
  {vessel}_trunk_levels_mask.nii.gz        level on full mask (int16)
  {vessel}_branch_table.csv             per-branch attributes + level + backbone flag

Usage
-----
  python 03_extract_backbone.py --case 001 --vessel pv
  python 03_extract_backbone.py --all --no-vis
"""

import argparse
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path

try:
    import vtk
    from vtk.util import numpy_support
    HAS_VTK = True
except ImportError:
    HAS_VTK = False

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARNING] networkx not found: pip install networkx")

try:
    from tqdm import tqdm as _tqdm
    def tqdm(it, **kw): return _tqdm(it, **kw)
    def tqdm_write(s):  _tqdm.write(s)
except ImportError:
    def tqdm(it, **kw): return it
    def tqdm_write(s):  print(s)


# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent
OUTPUT_DIR = BASE / "output"
TRAIN_IDS  = [f"{i:03d}" for i in range(1, 17)]
TEST_IDS   = [f"{i:03d}" for i in range(17, 21)]
MAX_SCATTER = 60_000

# ── per-vessel config ──────────────────────────────────────────────────────────
PV_CFG = dict(backbone_k=5,
              backbone_max_overlap=0.30,
              coverage_decay=0.5,
              coverage_max_hops=3)

HV_CFG = dict(backbone_k=3,
              ivc_angle_max_deg=20.0,           # only consider paths with root→leaf angle < this as IVC candidates
              ivc_min_length_mm=30.0,           # if identified IVC is shorter than this, use full path without thinning
              ivc_thinning_ratio=0.50,          # stop IVC walk when next branch radius < ratio × current IVC avg radius
              ivc_overlap_max=0.30,            # reject candidates sharing >30% length with identified IVC
              backbone_max_overlap=0.30,
              min_added_score=1e-6,            # reject added=0.0 trunks
              coverage_decay=0.5,
              coverage_max_hops=3,
              backbone_straighten_angle=0.0,   # post-proc: 0=off
              backbone_ref_frac=0.4)           # fraction from root for ref direction


# ── I/O ───────────────────────────────────────────────────────────────────────

def _load_nifti(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def save_nifti(arr, ref, path, dtype=np.uint8):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = sitk.GetImageFromArray(arr.astype(dtype))
    out.CopyInformation(ref)
    sitk.WriteImage(out, str(path), useCompression=True)


def _write_vtp(vtk_pts, vtk_cells, cell_arrays, vtp_path):
    """Write a vtkPolyData with polylines and cell scalar arrays."""
    if not HAS_VTK:
        tqdm_write("      [warn] VTK not available — skipping VTP")
        return
    pd_vtk = vtk.vtkPolyData()
    pd_vtk.SetPoints(vtk_pts)
    pd_vtk.SetLines(vtk_cells)
    for arr_np, name, first in cell_arrays:
        vtk_arr = numpy_support.numpy_to_vtk(arr_np, deep=True)
        vtk_arr.SetName(name)
        pd_vtk.GetCellData().AddArray(vtk_arr)
        if first:
            pd_vtk.GetCellData().SetActiveScalars(name)
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(vtp_path))
    w.SetInputData(pd_vtk)
    w.Write()
    tqdm_write(f"      → {vtp_path.name}  ({vtk_pts.GetNumberOfPoints()} pts)")


def _save_levels_vtp(branch_data, levels, backbone_bids, vtp_path):
    """All branches coloured by level (0=backbone). Load as Model in Slicer,
    set Scalar Visibility → colour by 'level'."""
    vtk_pts   = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    lv_list, bb_list, rad_list = [], [], []
    for bid, bd in branch_data.items():
        poly = bd.get("smooth_polyline", bd.get("polyline"))
        if poly is None or len(poly) < 2: continue
        start = vtk_pts.GetNumberOfPoints()
        for pt in poly:
            vtk_pts.InsertNextPoint(-float(pt[0]), -float(pt[1]), float(pt[2]))
        n = len(poly)
        vtk_cells.InsertNextCell(n)
        for j in range(n): vtk_cells.InsertCellPoint(start + j)
        lv_list.append(int(levels.get(bid, -1)))
        bb_list.append(int(bid in backbone_bids))
        rad_list.append(float(bd.get("mean_radius_mm", 0.0)))
    _write_vtp(vtk_pts, vtk_cells,
               [(np.array(lv_list,  np.int32),   "level",         True),
                (np.array(bb_list,  np.int32),   "is_backbone",   False),
                (np.array(rad_list, np.float32), "mean_radius_mm",False)],
               vtp_path)


def _save_backbone_vtp(branch_data, backbone_bids, vtp_path):
    """Only backbone branches. Load as Model in Slicer → always white/solid."""
    vtk_pts   = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    rad_list  = []
    for bid, bd in branch_data.items():
        if bid not in backbone_bids: continue
        poly = bd.get("smooth_polyline", bd.get("polyline"))
        if poly is None or len(poly) < 2: continue
        start = vtk_pts.GetNumberOfPoints()
        for pt in poly:
            vtk_pts.InsertNextPoint(-float(pt[0]), -float(pt[1]), float(pt[2]))
        n = len(poly)
        vtk_cells.InsertNextCell(n)
        for j in range(n): vtk_cells.InsertCellPoint(start + j)
        rad_list.append(float(bd.get("mean_radius_mm", 0.0)))
    _write_vtp(vtk_pts, vtk_cells,
               [(np.array(rad_list, np.float32), "mean_radius_mm", True)],
               vtp_path)


def load_script02(vessel, out_dir):
    pkl = out_dir / f"{vessel}_graph.pkl"
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return (data["G"], data["node_data"],
            data["branch_data"], np.array(data["sp_zyx"]))


def load_script01_niftis(vessel, out_dir):
    def p(s): return out_dir / f"{vessel}_{s}"
    skel, ref = _load_nifti(p("skeleton_pruned.nii.gz"))
    blab, _   = _load_nifti(p("branch_labels.nii.gz"))
    mask, _   = _load_nifti(p("clean_mask.nii.gz"))
    return (skel.astype(np.uint8), blab.astype(np.uint16),
            mask.astype(np.uint8), ref)


# ── coordinate helpers ─────────────────────────────────────────────────────────

def _fmt_xyz_mm(pt):
    if pt is None:
        return "(nan, nan, nan)"
    return f"({float(pt[0]):.1f}, {float(pt[1]):.1f}, {float(pt[2]):.1f})"


def _node_xyz_mm(node, node_data):
    nd = node_data.get(node, {})
    pt = nd.get("pos_mm", None)
    if pt is None:
        return None
    return np.asarray(pt, dtype=float)


def _node_coord_string(node, node_data):
    """Return both node/voxel coordinate and physical coordinate for logs."""
    pt = _node_xyz_mm(node, node_data)
    return f"zyx={node}  xyz_mm={_fmt_xyz_mm(pt)}"


def voxel_zyx_to_xyz_mm(idx_zyx, ref_img):
    """
    Convert array indices in z,y,x order to SimpleITK physical points in x,y,z mm.

    This keeps the mask scatter in the same coordinate system as node_data['pos_mm']
    and branch_data['polyline'] / ['smooth_polyline'].
    """
    idx_zyx = np.asarray(idx_zyx, dtype=float)
    if idx_zyx.size == 0:
        return np.empty((0, 3), dtype=float)

    # Convert z,y,x array indices to x,y,z image indices.
    idx_xyz = idx_zyx[:, [2, 1, 0]]

    spacing = np.asarray(ref_img.GetSpacing(), dtype=float)      # x,y,z
    origin  = np.asarray(ref_img.GetOrigin(), dtype=float)       # x,y,z
    direction = np.asarray(ref_img.GetDirection(), dtype=float).reshape(3, 3)

    # SimpleITK physical point: origin + direction @ (index_xyz * spacing).
    scaled = idx_xyz * spacing[None, :]
    pts = origin[None, :] + scaled @ direction.T
    return pts.astype(float)


# ── backbone selection ─────────────────────────────────────────────────────────

def graph_to_tree(G):
    """
    Convert a potentially cyclic graph to a Maximum Spanning Tree.

    Uses score_weight = length × radius² as edge weight so that thick, long
    branches (anatomical main trunks) are preferentially kept.  Cycles are
    broken by discarding the lowest-weight edge in each cycle.

    Works per connected component so isolated fragments are preserved.
    Returns a new nx.Graph that is guaranteed to be a forest (one tree per CC).
    """
    if G.number_of_edges() == 0:
        return G.copy()
    comps = list(nx.connected_components(G))
    T = nx.Graph()
    T.add_nodes_from(G.nodes(data=True))
    for comp in comps:
        sub = G.subgraph(comp)
        mst = nx.maximum_spanning_tree(sub, weight="score_weight")
        T.add_edges_from(mst.edges(data=True))
    return T


def _select_root(T, node_data, vessel="pv"):
    """
    Select root node of the vessel tree.

    HV : node with maximum z voxel coordinate (IVC entrance).
    PV : iterative leaf-pruning convergence point, then pick thickest survivor.
    """
    try:
        main_comp = max(nx.connected_components(T), key=len)
    except Exception:
        main_comp = set(T.nodes())

    all_nodes = [n for n in main_comp if n in node_data]
    if not all_nodes:
        all_nodes = list(main_comp)

    if vessel == "hv":
        # Use z-max node directly as root (IVC entry point).
        # IVC identification is handled downstream by _identify_ivc_bids.
        root = max(all_nodes, key=lambda n: node_data[n]["pos_zyx"][0])
        nd   = node_data.get(root, {})
        tqdm_write(f"        [root] HV z-max  {_node_coord_string(root, node_data)}  "
                   f"type={nd.get('type','?')}  r={nd.get('radius_mm',0):.2f} mm  "
                   f"z_index={nd.get('pos_zyx',[0])[0]}")
        return root, set()

    remaining = set(main_comp)
    degree    = {n: sum(1 for nb in T.neighbors(n) if nb in remaining)
                 for n in remaining}
    while True:
        leaves = [n for n in remaining if degree[n] <= 1]
        if len(leaves) >= len(remaining):
            break
        for n in leaves:
            remaining.discard(n)
            for nb in T.neighbors(n):
                if nb in remaining:
                    degree[nb] -= 1

    candidates = [n for n in remaining if n in node_data]
    if not candidates:
        candidates = [n for n in main_comp if n in node_data]

    root = max(candidates, key=lambda n: node_data[n]["radius_mm"])
    nd   = node_data[root]
    tqdm_write(f"        [root] convergence  {_node_coord_string(root, node_data)}  "
               f"type={nd['type']}  "
               f"r={nd['radius_mm']:.2f} mm  "
               f"z_index={nd['pos_zyx'][0]}  "
               f"survivors={len(remaining)}")
    return root, set()



def _ensure_zmax_is_node(G, node_data, branch_data, ref_img):
    """
    Ensure the globally highest z-mm point of the skeleton is a graph node.

    Scans all branch polylines. If the highest point is already within 1 mm of
    an existing node, returns unchanged. Otherwise splits the branch at that
    point, inserts a bifurcation node, and returns modified copies of G,
    node_data, and branch_data.
    """
    best_z_mm, best_bid, best_idx = -float("inf"), None, None
    for bid, bd in branch_data.items():
        poly = bd.get("smooth_polyline", bd.get("polyline"))
        if poly is None or len(poly) == 0:
            continue
        poly = np.asarray(poly)
        li = int(np.argmax(poly[:, 2]))
        lz = float(poly[li, 2])
        if lz > best_z_mm:
            best_z_mm, best_bid, best_idx = lz, bid, li

    if best_bid is None:
        return G, node_data, branch_data

    # Already at or near an existing node?
    for nd in node_data.values():
        pm = nd.get("pos_mm")
        if pm is not None and abs(float(np.asarray(pm)[2]) - best_z_mm) < 1.0:
            return G, node_data, branch_data

    bd_orig = branch_data[best_bid]
    poly    = np.asarray(bd_orig.get("smooth_polyline", bd_orig.get("polyline")))

    if best_idx == 0 or best_idx == len(poly) - 1:
        return G, node_data, branch_data

    # Find the graph edge for this branch
    edge = next(((u, v, d) for u, v, d in G.edges(data=True)
                 if d.get("branch_id") == best_bid), None)
    if edge is None:
        return G, node_data, branch_data
    u, v, _ = edge

    # Convert mm → voxel zyx
    best_pt   = poly[best_idx].astype(float)
    spacing   = np.asarray(ref_img.GetSpacing(), dtype=float)
    origin    = np.asarray(ref_img.GetOrigin(),  dtype=float)
    direction = np.asarray(ref_img.GetDirection(), dtype=float).reshape(3, 3)
    idx_xyz   = np.linalg.solve(direction, (best_pt - origin)) / spacing
    new_zyx   = (int(round(idx_xyz[2])), int(round(idx_xyz[1])), int(round(idx_xyz[0])))

    if new_zyx in node_data:
        return G, node_data, branch_data

    mean_r  = float(bd_orig.get("mean_radius_mm", 5.0))
    max_r   = float(bd_orig.get("max_radius_mm",  mean_r))
    new_nd  = {"pos_zyx": list(new_zyx),
               "pos_mm":  [float(best_pt[0]), float(best_pt[1]), float(best_pt[2])],
               "radius_mm": mean_r, "type": "bif"}

    max_bid = max(branch_data.keys())
    bid_a, bid_b = max_bid + 1, max_bid + 2

    def _sub(poly_sub):
        diffs  = np.diff(poly_sub, axis=0) if len(poly_sub) > 1 else np.zeros((1, 3))
        length = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return {"smooth_polyline": poly_sub, "polyline": poly_sub,
                "length_mm": length, "mean_radius_mm": mean_r,
                "max_radius_mm": max_r, "score_weight": length * mean_r,
                "original_bids": bd_orig.get("original_bids", [best_bid]),
                "n_merged": bd_orig.get("n_merged", 1)}

    bd_a = _sub(poly[:best_idx + 1])
    bd_b = _sub(poly[best_idx:])

    G_new = G.copy()
    G_new.remove_edge(u, v)
    G_new.add_node(new_zyx, **new_nd)
    for nb, bd_new, bid_new in ((new_zyx, bd_a, bid_a), (v, bd_b, bid_b)):
        G_new.add_edge(u if nb == new_zyx else new_zyx, nb,
                       branch_id=bid_new,
                       length_mm=bd_new["length_mm"],
                       mean_radius_mm=bd_new["mean_radius_mm"],
                       score_weight=bd_new["score_weight"])

    branch_data_new = {k: bv for k, bv in branch_data.items() if k != best_bid}
    branch_data_new[bid_a] = bd_a
    branch_data_new[bid_b] = bd_b


    return G_new, {**node_data, new_zyx: new_nd}, branch_data_new


def _path_score(path, G):
    return sum(G[a][b]["length_mm"] * G[a][b]["mean_radius_mm"]
               for a, b in zip(path[:-1], path[1:])
               if G.has_edge(a, b))


def _path_bids(path, G):
    return [G[a][b]["branch_id"]
            for a, b in zip(path[:-1], path[1:])
            if G.has_edge(a, b)]


def _path_length_from_bids(bids, branch_data):
    """Total physical length of a set/list of graph branch IDs."""
    return sum(float(branch_data.get(b, {}).get("length_mm", 0.0))
               for b in set(bids))


def _overlap_candidate_ratio(candidate_bids, previous_bids, branch_data):
    """
    How much of the candidate trunk is already contained in a previous trunk.

    This is stricter than Jaccard for short candidate paths:
        shared_length / candidate_length
    """
    cand = set(candidate_bids)
    prev = set(previous_bids)
    shared_len = _path_length_from_bids(cand & prev, branch_data)
    cand_len   = _path_length_from_bids(cand, branch_data)
    return shared_len / (cand_len + 1e-9)


def _overlap_shorter_ratio(bids_a, bids_b, branch_data):
    """
    Pairwise trunk overlap measured on the shorter path:
        shared_length / min(length_a, length_b)
    This prevents a small trunk from being hidden inside a long trunk.
    """
    a = set(bids_a)
    b = set(bids_b)
    shared_len = _path_length_from_bids(a & b, branch_data)
    len_a      = _path_length_from_bids(a, branch_data)
    len_b      = _path_length_from_bids(b, branch_data)
    return shared_len / (min(len_a, len_b) + 1e-9)


def _global_overlap_ratio(candidate_bids, backbone_bids, branch_data):
    """How much of the candidate trunk is already covered by all selected trunks."""
    cand = set(candidate_bids)
    bb   = set(backbone_bids)
    shared_len = _path_length_from_bids(cand & bb, branch_data)
    cand_len   = _path_length_from_bids(cand, branch_data)
    return shared_len / (cand_len + 1e-9)


def _passes_overlap_filters(candidate_bids, previous_trunks, backbone_bids,
                            branch_data, max_overlap):
    """
    Length-based overlap filters for backbone selection.

    Checks both:
    1) candidate vs each previous trunk;
    2) candidate vs the union of all selected backbone branches.
    """
    cand = set(candidate_bids)

    # Global check: candidate should not be mostly already covered.
    if backbone_bids:
        gov = _global_overlap_ratio(cand, backbone_bids, branch_data)
        if gov > max_overlap:
            return False, f"global={gov:.1%}"

    # Pairwise check against each already selected trunk.
    for prev in previous_trunks:
        ov_cand  = _overlap_candidate_ratio(cand, prev, branch_data)
        ov_short = _overlap_shorter_ratio(cand, prev, branch_data)
        if ov_cand > max_overlap or ov_short > max_overlap:
            return False, f"pairwise cand={ov_cand:.1%}, short={ov_short:.1%}"

    return True, ""


def _angle_to_z_axis_deg(root_node, leaf_node, node_data):
    """
    Angle between the root→leaf vector and the physical z-axis in xyz-mm space.

    Returns:
        0°   = almost parallel to z-axis, i.e. very vertical / IVC-like
        90°  = almost horizontal
    """
    if root_node not in node_data or leaf_node not in node_data:
        return float("nan")
    p0 = np.array(node_data[root_node].get("pos_mm"), dtype=float)
    p1 = np.array(node_data[leaf_node].get("pos_mm"), dtype=float)
    v = p1 - p0
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        return float("nan")
    dz = abs(float(v[2]))
    cos_val = np.clip(dz / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_val)))


# ── backbone straightening post-process ───────────────────────────────────────

def _branch_dir_from(bid, from_node, node_data, branch_data):
    """Unit direction of branch bid oriented away from from_node."""
    bd = branch_data.get(bid, {})
    poly = bd.get("smooth_polyline", bd.get("polyline"))
    if poly is None or len(poly) < 2:
        return None
    p0 = np.array(poly[0])
    p1 = np.array(poly[-1])
    nd = node_data.get(from_node, {})
    if not nd:
        return None
    from_pos = np.array(nd["pos_mm"])
    v = (p1 - p0) if np.linalg.norm(p0 - from_pos) <= np.linalg.norm(p1 - from_pos) \
                  else (p0 - p1)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else None


def _straighten_trunk(node_path, bids, branch_data, node_data, T,
                      ref_frac=0.4, angle_threshold_deg=45.0):
    """
    Post-process one backbone trunk (node_path, bids ordered root→leaf).

    1. Compute reference direction from root-proximal branches, weighted
       exponentially so branches closer to root have more influence.
    2. Walk root→leaf; when a branch deviates > angle_threshold_deg from the
       reference, switch to a greedy extension: at each fork pick the neighbour
       branch best aligned with the reference direction.

    Returns (new_node_path, new_bids).
    """
    if len(bids) < 2 or len(node_path) < 3:
        return node_path, bids

    n_ref   = max(1, int(len(bids) * ref_frac))
    cos_thr = np.cos(np.radians(angle_threshold_deg))

    # ── reference direction (exponential decay from root) ──────────────────
    weighted = []
    for i, bid in enumerate(bids[:n_ref]):
        d = _branch_dir_from(bid, node_path[i], node_data, branch_data)
        if d is None:
            continue
        length = branch_data.get(bid, {}).get("length_mm", 1.0)
        weight = np.exp(-3.0 * i / n_ref) * length
        weighted.append(d * weight)

    if not weighted:
        return node_path, bids

    ref = np.array(weighted).sum(axis=0)
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-9:
        return node_path, bids
    ref /= ref_norm

    # ── find first deviating branch (only check AFTER the ref portion) ────
    # Root-proximal branches (first n_ref) are always kept unchanged.
    deviate_at = None
    for i, bid in enumerate(bids):
        if i < n_ref:            # protected zone — never replace
            continue
        d = _branch_dir_from(bid, node_path[i], node_data, branch_data)
        if d is not None and np.dot(d, ref) < cos_thr:
            deviate_at = i
            break

    if deviate_at is None:
        return node_path, bids   # already straight enough

    # ── greedy straight extension from fork node ───────────────────────────
    new_node_path = list(node_path[:deviate_at + 1])
    new_bids      = list(bids[:deviate_at])
    visited       = set(new_node_path)
    cur           = node_path[deviate_at]

    for _ in range(60):
        best_align, best_bid, best_nb = -2.0, None, None
        for nb in T.neighbors(cur):
            if nb in visited:
                continue
            edata = T[cur][nb]
            bid   = edata.get("branch_id")
            if bid is None:
                continue
            d = _branch_dir_from(bid, cur, node_data, branch_data)
            if d is None:
                continue
            a = np.dot(d, ref)
            if a > best_align:
                best_align, best_bid, best_nb = a, bid, nb

        if best_bid is None or best_align < 0.0:
            break

        new_bids.append(best_bid)
        new_node_path.append(best_nb)
        visited.add(best_nb)
        if node_data.get(best_nb, {}).get("type") == "ep":
            break
        cur = best_nb

    tqdm_write(f"          [straighten] deviate@branch {deviate_at}  "
               f"ref_dir=[{ref[0]:.2f},{ref[1]:.2f},{ref[2]:.2f}]  "
               f"kept={len(new_bids)} branches")
    return new_node_path, new_bids


def _identify_ivc_bids(root, T_search, node_data, branch_data, cfg):
    """
    1. Among all root-to-leaf paths with angle < ivc_angle_max_deg, pick the
       one with the highest Σ(length × radius) score.
    2. Truncate at first thinning point (next_r < avg_so_far × ivc_thinning_ratio).
    3. If the result is shorter than ivc_min_length_mm, use the full path.
    4. Fallback: if no leaf path qualifies, find the longest root-to-bifurcation
       path with angle < ivc_angle_max_deg.
    """
    thinning      = cfg.get("ivc_thinning_ratio", 0.50)
    ivc_angle_max = cfg.get("ivc_angle_max_deg", 20.0)
    ivc_min_len   = cfg.get("ivc_min_length_mm", 30.0)

    root_comp  = nx.node_connected_component(T_search, root)

    def _path_bids(path):
        return [T_search[a][b].get("branch_id")
                for a, b in zip(path[:-1], path[1:])
                if T_search.has_edge(a, b)
                and T_search[a][b].get("branch_id") is not None]

    def _score(bids):
        return sum(float(branch_data.get(b, {}).get("length_mm", 0.0)) *
                   float(branch_data.get(b, {}).get("mean_radius_mm", 0.0))
                   for b in bids)

    def _length(bids):
        return sum(float(branch_data.get(b, {}).get("length_mm", 0.0)) for b in bids)

    def _truncate(bids):
        kept, radii = [], []
        for bid in bids:
            r = float(branch_data.get(bid, {}).get("mean_radius_mm", 0.0))
            if radii and r < (sum(radii) / len(radii)) * thinning:
                break
            kept.append(bid)
            radii.append(r)
        return kept

    # ── Step 1: search root-to-leaf paths ──────────────────────────────────
    leaf_nodes = [n for n in root_comp
                  if n in node_data and node_data[n].get("type") == "ep" and n != root]

    best_score, best_path_bids = -1.0, []
    for leaf in leaf_nodes:
        angle = _angle_to_z_axis_deg(root, leaf, node_data)
        if not np.isfinite(angle) or angle >= ivc_angle_max:
            continue
        try:
            path = nx.shortest_path(T_search, root, leaf)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        bids = _path_bids(path)
        if not bids:
            continue
        sc = _score(bids)
        if sc > best_score:
            best_score, best_path_bids = sc, bids

    if best_path_bids:
        ivc_bids = _truncate(best_path_bids)
        if _length(ivc_bids) < ivc_min_len:
            ivc_bids = best_path_bids
        return set(ivc_bids)

    # ── Step 2: fallback — root-to-bifurcation, angle < ivc_angle_max ──────
    bif_nodes = [n for n in root_comp
                 if n != root and T_search.degree(n) >= 3]

    best_score_bif, best_bids = -1.0, []
    for bif in bif_nodes:
        angle = _angle_to_z_axis_deg(root, bif, node_data)
        if not np.isfinite(angle) or angle >= ivc_angle_max:
            continue
        try:
            path = nx.shortest_path(T_search, root, bif)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        bids = _path_bids(path)
        sc = _score(bids)
        if sc > best_score_bif:
            best_score_bif, best_bids = sc, bids

    if best_bids:
        return set(best_bids)

    # ── Step 3: last-resort fallback — no angle constraint ───────────────
    best_score_bif, best_bids = -1.0, []
    for bif in bif_nodes:
        try:
            path = nx.shortest_path(T_search, root, bif)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        bids = _path_bids(path)
        sc = _score(bids)
        if sc > best_score_bif:
            best_score_bif, best_bids = sc, bids

    if best_bids:
        return set(best_bids)

    return set()


def _apply_straightening(selected_trunk_paths, branch_data, node_data, T, cfg):
    """
    Post-process all selected backbone trunks.

    Important: return per-trunk paths, not a single union set.
    Otherwise we cannot re-check overlap after straightening.
    """
    angle_thr = cfg.get("backbone_straighten_angle", 0.0)
    if angle_thr <= 0.0 or T is None:
        return [(list(node_path), list(bids))
                for node_path, bids in selected_trunk_paths]

    ref_frac = cfg.get("backbone_ref_frac", 0.4)
    new_trunks = []
    for node_path, bids in selected_trunk_paths:
        new_node_path, straightened = _straighten_trunk(
            node_path, bids, branch_data, node_data, T,
            ref_frac=ref_frac, angle_threshold_deg=angle_thr)
        if straightened:
            new_trunks.append((list(new_node_path), list(straightened)))
    return new_trunks


def _filter_trunks_after_straightening(trunk_paths, branch_data, cfg):
    """
    Re-apply length-based overlap filters after straightening.

    Straightening can change trunk geometry/path membership, so overlap must be
    checked again here.
    """
    max_overlap = cfg.get("backbone_max_overlap", 0.20)

    kept = []
    kept_bids = set()

    for idx, (node_path, bids) in enumerate(trunk_paths, start=1):
        bids_set = set(bids)

        ok, reason = _passes_overlap_filters(
            bids_set, [set(b) for _, b in kept], kept_bids,
            branch_data, max_overlap)
        if not ok:
            tqdm_write(f"          [post-filter] trunk {idx} rejected: overlap {reason}")
            continue

        kept.append((node_path, list(bids)))
        kept_bids.update(bids_set)

    tqdm_write(f"          [post-filter] kept {len(kept)}/{len(trunk_paths)} trunks "
               f"after straightening")
    return kept, kept_bids


def find_backbone(G, node_data, branch_data, cfg, vessel="pv"):
    """
    Coverage-maximizing backbone selection (original algorithm).

    Scores every root→leaf path by Σ(length × radius²), then greedily selects
    k paths that maximise L0+L1 branch coverage with minimal overlap.
    """
    if not HAS_NX or G is None:
        return set(), 0, [], [], None, []

    T = graph_to_tree(G)

    root, ivc_bids = _select_root(T, node_data, vessel=vessel)

    # Strip IVC branches from search tree so no path can traverse them.
    if ivc_bids:
        T_search = T.copy()
        T_search.remove_edges_from([(u, v) for u, v, d in T.edges(data=True)
                                    if d.get("branch_id") in ivc_bids])
        tqdm_write(f"        [ivc] stripped {len(ivc_bids)} IVC branches from search tree")
    else:
        T_search = T

    root_comp = nx.node_connected_component(T_search, root)
    leaf_nodes = [n for n in root_comp
                  if n in node_data
                  and node_data[n]["type"] == "ep"
                  and n != root]

    scored_paths = []
    for leaf in leaf_nodes:
        try:
            path = nx.shortest_path(T_search, root, leaf)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        bids  = _path_bids(path, T_search)
        score = _path_score(path, T_search)
        if bids:
            scored_paths.append((score, bids, leaf, path))   # path = node list

    if not scored_paths:
        return set(), root, [], [], None, []
    scored_paths.sort(key=lambda x: x[0], reverse=True)

    ivc_overlap_max   = cfg.get("ivc_overlap_max", 1.0)
    ivc_excluded_bids = _identify_ivc_bids(root, T_search, node_data, branch_data, cfg)

    k                 = cfg.get("backbone_k", 3)
    max_overlap       = cfg.get("backbone_max_overlap", 0.20)
    min_added_score   = cfg.get("min_added_score", -float("inf"))
    coverage_decay    = cfg.get("coverage_decay", 0.5)
    coverage_max_hops = cfg.get("coverage_max_hops", 3)

    # Branch adjacency (shared graph node → adjacent branches)
    node_to_bids = defaultdict(set)
    for na, nb, edata in G.edges(data=True):
        bid = edata.get("branch_id")
        if bid is not None:
            node_to_bids[na].add(bid)
            node_to_bids[nb].add(bid)
    branch_adj = defaultdict(set)
    for bids_set in node_to_bids.values():
        for a in bids_set:
            for b in bids_set:
                if a != b:
                    branch_adj[a].add(b)

    backbone_bids       = set()
    trunk_all_bids      = []   # full bids_set per trunk (for overlap check)
    selected_trunk_paths = []  # (node_path, bids) per selected trunk
    selected_trunk_meta  = []  # dict per selected trunk: leaf, added score, angle_to_z
    n_selected          = 0

    for _ in range(k):
        best_path  = None
        best_added = -1.0

        for score, bids, leaf, node_path in scored_paths:
            bids_set = set(bids)
            new_bb   = bids_set - backbone_bids
            if not new_bb:
                continue

            # Length-based overlap, not Jaccard.
            # This rejects a candidate if too much of its own physical length
            # is already covered by one previous trunk or by all selected trunks.
            ok, reason = _passes_overlap_filters(
                bids_set, trunk_all_bids, backbone_bids, branch_data, max_overlap)
            if not ok:
                continue

            # Reject candidates that overlap too much with IVC-excluded paths.
            if ivc_excluded_bids:
                ivc_ov = _overlap_candidate_ratio(bids_set, ivc_excluded_bids, branch_data)
                if ivc_ov > ivc_overlap_max:
                    continue

            def _w(bid):
                bd = branch_data.get(bid, {})
                return bd.get("length_mm", 1.0) * bd.get("mean_radius_mm", 1.0)

            # Step 1: BFS from backbone (stopping at new_bb as barrier) to mark all
            # territory already claimed by other selected trunks and their side branches.
            visited_bfs = set(new_bb)
            bfs_q = deque()
            for bid in backbone_bids:
                if bid not in visited_bfs:
                    visited_bfs.add(bid)
                    bfs_q.append(bid)
            while bfs_q:
                cur = bfs_q.popleft()
                for adj in branch_adj[cur]:
                    if adj not in visited_bfs:
                        visited_bfs.add(adj)
                        bfs_q.append(adj)

            # Step 2: new_bb_with_sides uses visited_bfs so backbone side branches
            # are correctly excluded from the "has side branches" check.
            new_bb_with_sides = {bid for bid in new_bb
                                 if any(adj not in visited_bfs
                                        for adj in branch_adj[bid])}

            # Step 3: L0=full weight; each hop multiplied by coverage_decay.
            added = sum(_w(b) for b in new_bb_with_sides)
            frontier = set()
            for bid in new_bb:
                for adj in branch_adj[bid]:
                    if adj not in visited_bfs:
                        frontier.add(adj)
                        visited_bfs.add(adj)
            coeff = coverage_decay
            n_hops = coverage_max_hops if coverage_max_hops > 0 else 999
            for _ in range(n_hops):
                if not frontier:
                    break
                added += coeff * sum(_w(b) for b in frontier)
                next_frontier = set()
                for bid in frontier:
                    for adj in branch_adj[bid]:
                        if adj not in visited_bfs:
                            next_frontier.add(adj)
                            visited_bfs.add(adj)
                frontier = next_frontier
                coeff *= coverage_decay

            # Continuous overlap penalty: the more of the candidate is already
            # covered by existing backbone, the lower the score.
            if backbone_bids:
                ov = _global_overlap_ratio(bids_set, backbone_bids, branch_data)
                added *= (1.0 - ov)

            if added > best_added:
                best_added = added
                best_path  = (score, bids, leaf, node_path)

        if best_path is None or best_added <= min_added_score:
            if best_path is not None:
                tqdm_write(f"        [stop] best_added={best_added:.3f} "
                           f"<= min_added_score={min_added_score:.3f}")
            break

        _, bids, leaf, node_path = best_path
        bids_set     = set(bids)
        new_bb_added = bids_set - backbone_bids


        backbone_bids.update(bids_set)
        trunk_all_bids.append(bids_set)
        angle_to_z = _angle_to_z_axis_deg(root, leaf, node_data)
        selected_trunk_paths.append((list(node_path), list(bids)))
        selected_trunk_meta.append(dict(leaf=leaf,
                                        added=float(best_added),
                                        angle_to_z_deg=float(angle_to_z)))
        n_selected += 1
        tqdm_write(f"        [trunk {n_selected}] leaf {_node_coord_string(leaf, node_data)}  "
                   f"angle_to_z={angle_to_z:.1f}°  "
                   f"new_L0={len(new_bb_added)}  added={best_added:.1f}  "
                   f"covered={len(backbone_bids)}/{len(branch_data)}")

    tqdm_write(f"        backbone: {len(backbone_bids)} branches  "
               f"({n_selected}/{k} trunks)  overlap≤{max_overlap:.0%}  "
               f"covered={len(backbone_bids)}/{len(branch_data)}")
    return (backbone_bids, root,
            [(s, b, l) for s, b, l, _ in scored_paths],
            selected_trunk_paths, T_search, selected_trunk_meta,
            ivc_excluded_bids)


# ── level assignment ───────────────────────────────────────────────────────────

def trunk_aware_levels(backbone_bids, G, branch_data):
    """
    BFS outward from backbone branches.
    Level 0 = backbone.  Level n = n hops away.
    Disconnected branches → max_level + 1.
    """
    # Build branch adjacency via shared graph nodes
    node_to_bids = defaultdict(set)
    for na, nb, edata in G.edges(data=True):
        bid = edata.get("branch_id")
        if bid is not None:
            node_to_bids[na].add(bid)
            node_to_bids[nb].add(bid)

    branch_adj = defaultdict(set)
    for bids in node_to_bids.values():
        for a in bids:
            for b in bids:
                if a != b:
                    branch_adj[a].add(b)

    levels = {}
    queue  = deque()
    for bid in backbone_bids:
        levels[bid] = 0
        queue.append(bid)

    while queue:
        cur = queue.popleft()
        for nb in branch_adj[cur]:
            if nb not in levels:
                levels[nb] = levels[cur] + 1
                queue.append(nb)

    max_lv = max(levels.values()) if levels else 0
    for bid in branch_data:
        if bid not in levels:
            levels[bid] = max_lv + 1

    return levels


# ── NIfTI output generation ────────────────────────────────────────────────────

def _propagate_to_mask(mask, branch_lab):
    """Nearest-skeleton-voxel propagation of branch labels to full mask."""
    _, inds = distance_transform_edt(branch_lab == 0, return_indices=True)
    return np.where(mask, branch_lab[tuple(inds)], 0).astype(np.uint16)


def build_nifti_outputs(backbone_bids, levels, branch_data,
                         skel, branch_lab, mask):
    """
    Generate four NIfTI arrays from graph-level information.

    Uses branch_data["original_bids"] to map graph branch IDs back to the
    NIfTI branch_label IDs produced by Script 01 (the voxel regions).
    Falls back to [branch_id] if the key is missing (no collinear merge).
    """
    # ── build lookup tables ────────────────────────────────────────────────────
    nifti_to_level = {}      # NIfTI label ID → level
    bb_nifti_ids   = set()   # NIfTI label IDs that belong to backbone branches

    for bid, bd in branch_data.items():
        lv        = levels.get(bid, -1)
        orig_bids = bd.get("original_bids", [bid])
        for ob in orig_bids:
            nifti_to_level[ob] = lv
        if bid in backbone_bids:
            bb_nifti_ids.update(orig_bids)

    # ── skeleton-level arrays ──────────────────────────────────────────────────
    # Stored as level+1 so Slicer label 0 = background (transparent).
    # 0 = unassigned, 1 = L0 backbone, 2 = L1, 3 = L2, …
    skel_level    = np.zeros(skel.shape, dtype=np.uint8)
    backbone_skel = np.zeros(skel.shape, dtype=np.uint8)

    for nid, lv in nifti_to_level.items():
        region = branch_lab == nid
        if lv >= 0:
            skel_level[region] = lv + 1
        if nid in bb_nifti_ids:
            backbone_skel[region] = 1

    # ── propagate to full mask via EDT ─────────────────────────────────────────
    mask_branches = _propagate_to_mask(mask, branch_lab)

    mask_level    = np.zeros(mask.shape, dtype=np.uint8)
    backbone_mask = np.zeros(mask.shape, dtype=np.uint8)

    for nid in set(np.unique(mask_branches)) - {0}:
        region = mask_branches == nid
        lv     = nifti_to_level.get(int(nid), -1)
        if lv >= 0:
            mask_level[region] = lv + 1
        if int(nid) in bb_nifti_ids:
            backbone_mask[region] = 1

    return dict(backbone_skel=backbone_skel,
                backbone_mask=backbone_mask,
                skel_level=skel_level,
                mask_level=mask_level)


# ── branch table ───────────────────────────────────────────────────────────────

def build_branch_table(backbone_bids, levels, branch_data):
    rows = []
    for bid, bd in branch_data.items():
        rows.append(dict(
            branch_id=bid,
            length_mm=round(bd["length_mm"], 2),
            mean_radius_mm=round(bd["mean_radius_mm"], 3),
            max_radius_mm=round(bd["max_radius_mm"], 3),
            is_backbone=int(bid in backbone_bids),
            level=levels.get(bid, -1),
            path_score=round(bd["length_mm"] * bd["mean_radius_mm"], 3),
            n_merged=bd.get("n_merged", 1)))
    return pd.DataFrame(rows).sort_values("level")


# ── visualisation ──────────────────────────────────────────────────────────────

def visualize(case_id, vessel, result, out_dir):
    G           = result["G"]
    node_data   = result["node_data"]
    branch_data = result["branch_data"]
    levels      = result["levels"]
    backbone_bids = result["backbone_bids"]
    root_node   = result["root_node"]
    sp          = result["sp_zyx"]
    mask        = result["mask"]

    if not HAS_NX or G is None:
        return

    max_lv = max(levels.values()) if levels else 1
    cmap   = plt.colormaps["plasma_r"]

    def lv_color(lv):
        return cmap(lv / max(max_lv, 1))

    elev, azim = 28, -50

    def _axis3d(fig, pos, title):
        ax = fig.add_subplot(*pos, projection="3d")
        ax.set_facecolor("#111111")
        ax.tick_params(labelsize=5, colors="white")
        ax.set_xlabel("x mm", fontsize=6, color="white")
        ax.set_ylabel("y mm", fontsize=6, color="white")
        ax.set_zlabel("z mm", fontsize=6, color="white")
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.set_title(title, fontsize=8, color="white")
        ax.view_init(elev, azim)
        return ax

    def _draw_branches(ax, highlight_backbone=True):
        for bid, bd in branch_data.items():
            key  = "smooth_polyline" if "smooth_polyline" in bd else "polyline"
            poly = bd.get(key)
            if poly is None or len(poly) < 2:
                continue
            lv  = levels.get(bid, max_lv)
            col = "white" if (highlight_backbone and bid in backbone_bids) else lv_color(lv)
            lw  = 2.5     if (highlight_backbone and bid in backbone_bids) else 1.0
            # poly is LPS (x, y, z) mm
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                    color=col, linewidth=lw, alpha=0.9)

    def _draw_nodes(ax):
        for nid, nd in node_data.items():
            px, py, pz = nd["pos_mm"]   # pos_mm is LPS (x, y, z)
            col  = "#22ff88" if nd["type"] == "ep" else "#ffdd00"
            size = 30        if nd["type"] == "ep" else 60
            mk   = "o"       if nd["type"] == "ep" else "D"
            ax.scatter([px], [py], [pz], c=col, s=size, marker=mk,
                       linewidths=0, depthshade=False, zorder=5)
        # Root star
        if root_node in node_data:
            nd = node_data[root_node]
            px, py, pz = nd["pos_mm"]
            ax.scatter([px], [py], [pz], c="red", s=300, marker="*",
                       linewidths=0, depthshade=False, zorder=10)

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor("#1a1a1a")

    n_bb  = len(backbone_bids)
    lv_counts = pd.Series(levels).value_counts().sort_index()
    lv_str    = "  ".join(f"L{lv}:{cnt}" for lv, cnt in lv_counts.items())

    fig.suptitle(
        f"case_{case_id}  {vessel.upper()}  —  Script 03: Backbone Extraction\n"
        f"backbone={n_bb} branches  max_level={max_lv}  root=★  |  {lv_str}",
        fontsize=11, fontweight="bold", color="white")

    # Panel 1: backbone (white thick) + side branches by level
    ax1 = _axis3d(fig, (2, 3, 1),
                  "Backbone (white) + side branches by level\n"
                  "plasma_r: L0=bright  ★=root  ●=ep  ◆=bif")
    _draw_branches(ax1, highlight_backbone=True)
    _draw_nodes(ax1)
    sm1 = plt.cm.ScalarMappable(cmap="plasma_r",
                                 norm=plt.Normalize(vmin=0, vmax=max_lv))
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, ax=ax1, shrink=0.45, pad=0.05)
    cb1.set_label("level", fontsize=7, color="white")
    plt.setp(cb1.ax.yaxis.get_ticklabels(), color="white")

    # Panel 2: all branches coloured by level (no backbone highlight)
    ax2 = _axis3d(fig, (2, 3, 2), "All branches coloured by level")
    _draw_branches(ax2, highlight_backbone=False)
    _draw_nodes(ax2)

    # Panel 3: mask scatter coloured by level
    ax3 = _axis3d(fig, (2, 3, 3), "Mask voxels by level (xyz physical mm)")
    mask_level = result["nifti_maps"]["mask_level"]
    idx = np.argwhere(mask > 0)
    rng = np.random.default_rng(0)
    if len(idx) > MAX_SCATTER:
        idx = idx[rng.choice(len(idx), MAX_SCATTER, replace=False)]
    if len(idx):
        lvs = mask_level[idx[:, 0], idx[:, 1], idx[:, 2]].astype(float)
        lvs_norm = np.clip(lvs / max(max_lv, 1), 0, 1)
        colors = cmap(lvs_norm)
        ref_img = result.get("ref_img", None)
        if ref_img is not None:
            pts = voxel_zyx_to_xyz_mm(idx, ref_img)
            ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                        c=colors, s=1, alpha=0.15, linewidths=0, depthshade=False)
        else:
            # Fallback only: spacing-based display, without origin/direction.
            ax3.scatter(idx[:, 2]*sp[2], idx[:, 1]*sp[1], idx[:, 0]*sp[0],
                        c=colors, s=1, alpha=0.15, linewidths=0, depthshade=False)

    # Panel 4: level distribution bar chart
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#111111")
    if lv_counts.any():
        bars = ax4.bar(lv_counts.index.astype(str), lv_counts.values,
                       color=[lv_color(lv) for lv in lv_counts.index],
                       edgecolor="none")
        for bar, lv in zip(bars, lv_counts.index):
            if lv == 0:
                bar.set_edgecolor("white")
                bar.set_linewidth(1.5)
    ax4.set_xlabel("level",  fontsize=8, color="white")
    ax4.set_ylabel("branches", fontsize=8, color="white")
    ax4.set_title("Branch count per level", fontsize=8, color="white")
    ax4.tick_params(colors="white", labelsize=7)
    for sp_ in ax4.spines.values():
        sp_.set_edgecolor("#444444")

    # Panel 5: path score distribution (top 20 paths)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#111111")
    scored = result.get("scored_paths", [])
    if scored:
        top_scores = [s for s, _, _ in scored[:min(30, len(scored))]]
        ax5.barh(range(len(top_scores)), top_scores[::-1],
                 color="#4488ff", edgecolor="none")
        ax5.set_xlabel("path score (Σ len×r²)", fontsize=8, color="white")
        ax5.set_ylabel("path rank",             fontsize=8, color="white")
        ax5.set_title("Top path scores", fontsize=8, color="white")
        ax5.tick_params(colors="white", labelsize=7)
    for sp_ in ax5.spines.values():
        sp_.set_edgecolor("#444444")

    # Panel 6: summary text
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#111111")
    ax6.axis("off")
    df = result.get("branch_table")
    lines = [
        f"Root node:      {root_node}",
        f"Root type:      {node_data.get(root_node, {}).get('type', '?')}",
        f"Root radius:    {node_data.get(root_node, {}).get('radius_mm', 0):.2f} mm",
        "",
        f"Backbone branches: {n_bb}",
        f"Side branches:     {len(branch_data) - n_bb}",
        f"Max level:         {max_lv}",
        "",
    ]
    if df is not None and len(df):
        bb_len  = df[df["is_backbone"] == 1]["length_mm"].sum()
        all_len = df["length_mm"].sum()
        lines += [
            f"Backbone length: {bb_len:.1f} mm",
            f"Total length:    {all_len:.1f} mm",
            f"Backbone ratio:  {bb_len / (all_len + 1e-9):.1%}",
        ]
    for i, line in enumerate(lines):
        ax6.text(0.05, 0.95 - i * 0.085, line, transform=ax6.transAxes,
                 fontsize=8, color="white", va="top", family="monospace")
    ax6.set_title("Summary", fontsize=8, color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png = out_dir / f"{vessel}_backbone_vis.png"
    fig.savefig(str(png), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    tqdm_write(f"      figure saved → {png.name}  (interactive window ...)")
    plt.show()
    plt.close(fig)


# ── per-vessel pipeline ────────────────────────────────────────────────────────

def process_vessel(case_id, vessel, cfg, out_dir):
    tag = f"case_{case_id} [{vessel.upper()}]"
    tqdm_write(f"  {tag}")

    # ── 1. load ────────────────────────────────────────────────────────────────
    tqdm_write(f"    {tag}  1/5  loading ...")
    G, node_data, branch_data, sp_zyx = load_script02(vessel, out_dir)
    skel, branch_lab, mask, ref_img   = load_script01_niftis(vessel, out_dir)
    tqdm_write(f"    {tag}       graph: {len(node_data)} nodes  "
               f"{len(branch_data)} branches")

    if vessel == "hv":
        G, node_data, branch_data = _ensure_zmax_is_node(
            G, node_data, branch_data, ref_img)

    # ── 2. backbone selection ──────────────────────────────────────────────────
    tqdm_write(f"    {tag}  2/5  backbone selection  (k={cfg['backbone_k']}) ...")
    backbone_bids, root_node, scored_paths, trunk_paths, T_search, _, ivc_bids = find_backbone(
        G, node_data, branch_data, cfg, vessel=vessel)

    # Straightening may change the selected paths, so re-filter afterwards.
    straight_trunks = _apply_straightening(
        trunk_paths, branch_data, node_data, T_search, cfg)
    trunk_paths, backbone_bids = _filter_trunks_after_straightening(
        straight_trunks, branch_data, cfg)

    # Filter branch_data to only branches present in the search tree (MST minus IVC).
    # Used for VTP exports and visualisation; NIfTI outputs still use the full set.
    tree_bids = {d.get("branch_id") for _, _, d in T_search.edges(data=True)
                 if d.get("branch_id") is not None}
    tree_branch_data = {bid: bd for bid, bd in branch_data.items() if bid in tree_bids}
    tqdm_write(f"    {tag}       tree: {len(tree_branch_data)}/{len(branch_data)} branches "
               f"after MST (dropped {len(branch_data) - len(tree_branch_data)} cycle edges)")

    # ── 3. level assignment ────────────────────────────────────────────────────
    tqdm_write(f"    {tag}  3/5  trunk-aware BFS levelling ...")
    levels  = trunk_aware_levels(backbone_bids, G, branch_data)
    max_lv  = max(levels.values()) if levels else 0
    lv_cnt  = pd.Series(levels).value_counts().sort_index()
    tqdm_write(f"    {tag}       max_level={max_lv}  " +
               "  ".join(f"L{lv}:{cnt}" for lv, cnt in lv_cnt.items()))

    # ── 4. build NIfTI arrays ──────────────────────────────────────────────────
    tqdm_write(f"    {tag}  4/5  building NIfTI outputs ...")
    nifti_maps = build_nifti_outputs(
        backbone_bids, levels, branch_data, skel, branch_lab, mask)

    # ── 5. save ────────────────────────────────────────────────────────────────
    tqdm_write(f"    {tag}  5/5  saving ...")
    save_nifti(nifti_maps["backbone_skel"], ref_img,
               out_dir / f"{vessel}_backbone_centerline.nii.gz")
    save_nifti(nifti_maps["backbone_mask"], ref_img,
               out_dir / f"{vessel}_backbone_mask.nii.gz")
    save_nifti(nifti_maps["skel_level"], ref_img,
               out_dir / f"{vessel}_trunk_levels_centerline.nii.gz",
               dtype=np.uint8)
    save_nifti(nifti_maps["mask_level"], ref_img,
               out_dir / f"{vessel}_trunk_levels_mask.nii.gz",
               dtype=np.uint8)

    branch_table = build_branch_table(backbone_bids, levels, branch_data)
    branch_table.to_csv(out_dir / f"{vessel}_branch_table.csv", index=False)
    meta = {"root_node": [int(x) for x in root_node] if isinstance(root_node, tuple) else int(root_node),
            "backbone_k_initial": cfg.get("backbone_k", 3),
            "final_backbone_k": cfg.get("final_backbone_k", cfg.get("backbone_k", 3)),
            "ivc_z_angle_deg": cfg.get("ivc_z_angle_deg", None)}
    (out_dir / f"{vessel}_backbone_meta.json").write_text(json.dumps(meta))
    tqdm_write(f"    {tag}       saved 4 NIfTI + branch_table.csv + backbone_meta.json")

    # Slicer VTP exports (tree-filtered: no cycle edges)
    _save_levels_vtp(tree_branch_data, levels, backbone_bids,
                     out_dir / f"{vessel}_levels.vtp")
    _save_backbone_vtp(tree_branch_data, backbone_bids,
                       out_dir / f"{vessel}_backbone.vtp")
    if ivc_bids:
        _save_backbone_vtp(tree_branch_data, ivc_bids,
                           out_dir / f"{vessel}_ivc.vtp")

    return dict(G=G, node_data=node_data, branch_data=tree_branch_data,
                levels=levels, backbone_bids=backbone_bids,
                root_node=root_node, scored_paths=scored_paths,
                nifti_maps=nifti_maps, branch_table=branch_table,
                sp_zyx=sp_zyx, mask=mask, ref_img=ref_img)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Script 03 — Backbone Extraction & Level Assignment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case",        default="001")
    ap.add_argument("--vessel",      default="both", choices=["pv", "hv", "both"])
    ap.add_argument("--all",         action="store_true")
    ap.add_argument("--no-vis",      action="store_true")
    ap.add_argument("--backbone-k",  type=int,   default=None,
                    help="number of backbone trunks (default PV=5, HV=4)")
    ap.add_argument("--max-overlap", type=float, default=None,
                    help="max length-based overlap fraction between trunks")
    ap.add_argument("--min-added-score", type=float, default=None,
                    help="minimum added score for accepting a trunk; use >0 to reject added=0 trunks")
    ap.add_argument("--output-dir", default=None,
                    help="override output directory")
    ap.add_argument("--ids",        default=None,
                    help="comma-separated case IDs to process")
    args = ap.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    if not HAS_NX:
        print("ERROR: networkx required.  pip install networkx")
        return

    pv_cfg = dict(PV_CFG)
    hv_cfg = dict(HV_CFG)
    if args.backbone_k  is not None:
        pv_cfg["backbone_k"] = hv_cfg["backbone_k"] = args.backbone_k
    if args.max_overlap is not None:
        pv_cfg["backbone_max_overlap"] = hv_cfg["backbone_max_overlap"] = args.max_overlap
    if args.min_added_score is not None:
        pv_cfg["min_added_score"] = hv_cfg["min_added_score"] = args.min_added_score
    vessels = ["pv", "hv"] if args.vessel == "both" else [args.vessel]
    cfgs    = {"pv": pv_cfg, "hv": hv_cfg}

    case_list = ([c.strip() for c in args.ids.split(",")] if args.ids
                 else TRAIN_IDS + TEST_IDS)

    if args.all or args.ids:
        for cid in tqdm(case_list, desc="cases", unit="case"):
            print(f"\n=== case_{cid} ===")
            out = OUTPUT_DIR / f"case_{cid}"
            for v in vessels:
                process_vessel(cid, v, cfgs[v], out)
        print(f"\nAll done.  Output → {OUTPUT_DIR}")
    else:
        print(f"\n=== case_{args.case} ===")
        out = OUTPUT_DIR / f"case_{args.case}"
        for v in vessels:
            result = process_vessel(args.case, v, cfgs[v], out)
            if not args.no_vis:
                visualize(args.case, v, result, out)
        print(f"\nOutput → {out}")


if __name__ == "__main__":
    main()
