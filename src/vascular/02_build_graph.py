"""
Script 02 — Graph Construction, Pruning & Smoothing
=====================================================

Algorithm mirrors test_graph_from_vtp.py (LPS physical coordinates throughout).

Inputs  (Script 01 outputs in OUTPUT_DIR / case_{id}/)
-------
  {vessel}_skeleton_pruned.nii.gz
  {vessel}_radius_map.nii.gz
  {vessel}_branch_labels.nii.gz
  {vessel}_clean_mask.nii.gz        (visualisation only)

Processing
----------
1.  Load Script 01 NIfTI outputs.
2.  Build 26-connected voxel graph; assign LPS physical coord + radius per node.
3.  Classify: degree-1 → endpoint, degree≥3 → bifurcation candidate.
4.  Cluster adjacent bifurcation voxels → keep highest-radius representative.
5.  Contract degree-2 chains → branch-level graph (path_lps per edge).
6.  Fix path_lps endpoints (snap to node coord → no Slicer gaps).
7.  Remove local loops (perimeter < loop_max_length_mm).
8.  Prune short/thin terminal branches.
9.  Merge degree-2 pass-through nodes created by pruning.
10. Savitzky-Golay smoothing in LPS space; re-snap endpoints.
11. Save graph.pkl + node_data.csv + branch_data.csv (for Scripts 03/04).
12. Save Slicer files: smooth skeleton .vtp + endpoints/bifurcations .fcsv.
13. Visualise: 3D line graph + stat panels.

Outputs  (saved to OUTPUT_DIR / case_{id}/)
-----------
  {vessel}_graph.pkl
  {vessel}_node_data.csv
  {vessel}_branch_data.csv
  {vessel}_graph_vis.png
  {vessel}_graph_smooth.vtp          ← load as Model in 3D Slicer
  {vessel}_graph_endpoints.fcsv      ← load as Markups in 3D Slicer
  {vessel}_graph_bifurcations.fcsv   ← load as Markups in 3D Slicer

Usage
-----
  python 02_build_graph.py --case 001 --vessel hv
  python 02_build_graph.py --case 001 --vessel both
  python 02_build_graph.py --all --no-vis
"""

import argparse
import pickle
from pathlib import Path

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.signal import savgol_filter

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARNING] networkx not found: pip install networkx")

try:
    import vtk
    from vtk.util import numpy_support
    HAS_VTK = True
except ImportError:
    HAS_VTK = False
    print("[WARNING] vtk not found — Slicer VTP export disabled")

try:
    from tqdm import tqdm as _tqdm
    def tqdm(it, **kw): return _tqdm(it, **kw)
    def tqdm_write(s):  _tqdm.write(s)
except ImportError:
    def tqdm(it, **kw): return it
    def tqdm_write(s):  print(s)


# ── paths ──────────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent          # .../utile/1005/0705/
OUTPUT_DIR = BASE / "output"

TRAIN_IDS = [f"{i:03d}" for i in range(1, 17)]
TEST_IDS  = [f"{i:03d}" for i in range(17, 21)]

FCSV_HEADER = ("# Markups fiducial file version = 4.11\n"
               "# CoordinateSystem = LPS\n"
               "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,"
               "associatedNodeID\n")

# ── per-vessel config ──────────────────────────────────────────────────────────
PV_CFG = dict(loop_max_length_mm=8.0,
              prune_len_factor=2.5,
              prune_radius_factor=0.6,
              savgol_window=11,
              savgol_polyorder=3)
HV_CFG = dict(loop_max_length_mm=8.0,
              prune_len_factor=2.5,
              prune_radius_factor=0.6,
              savgol_window=11,
              savgol_polyorder=3)


# ── I/O ───────────────────────────────────────────────────────────────────────

def load_nifti(path):
    img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img), img


def load_script01(case_id, vessel, data_dir):
    def p(suffix): return data_dir / f"{vessel}_{suffix}"
    skel, ref = load_nifti(p("skeleton_pruned.nii.gz"))
    rmap, _   = load_nifti(p("radius_map.nii.gz"))
    blab, _   = load_nifti(p("branch_labels.nii.gz"))
    mask, _   = load_nifti(p("clean_mask.nii.gz"))
    sp_xyz    = ref.GetSpacing()
    sp_zyx    = np.array([sp_xyz[2], sp_xyz[1], sp_xyz[0]], dtype=float)
    return dict(skel=skel.astype(np.uint8),
                radius_map=rmap.astype(np.float32),
                branch_lab=blab.astype(np.uint16),
                mask=mask.astype(np.uint8),
                sp_zyx=sp_zyx,
                ref_img=ref)


# ── graph construction (test_graph_from_vtp algorithm, LPS coords) ────────────

def build_voxel_graph(skel, rmap, ref_img):
    """26-connected voxel graph; each node gets coord_lps (physical mm)."""
    voxels = set(map(tuple, np.argwhere(skel > 0)))
    G = nx.Graph()
    for v in voxels:
        z, y, x = v
        lps = ref_img.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
        G.add_node(v, pos_zyx=v, coord_lps=lps, radius=float(rmap[z, y, x]))
    for v in voxels:
        z, y, x = v
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if (dz, dy, dx) == (0, 0, 0): continue
                    nb = (z+dz, y+dy, x+dx)
                    if nb in voxels and not G.has_edge(v, nb):
                        d = float(np.linalg.norm(
                            np.array(G.nodes[v]['coord_lps']) -
                            np.array(G.nodes[nb]['coord_lps'])))
                        G.add_edge(v, nb, length_mm=d)
    return G


def classify_nodes(G):
    eps  = [n for n in G.nodes() if G.degree(n) == 1]
    bifs = [n for n in G.nodes() if G.degree(n) >= 3]
    return eps, bifs


def cluster_bifurcations(G, bif_cands):
    sub      = G.subgraph(bif_cands)
    clusters = list(nx.connected_components(sub))
    node_to_rep = {}
    for cluster in clusters:
        rep = max(cluster, key=lambda n: G.nodes[n]['radius'])
        for node in cluster:
            node_to_rep[node] = rep
    return node_to_rep


def _is_crit(node, node_to_rep, eps_set):
    return node in eps_set or node in node_to_rep

def _canon(node, node_to_rep):
    return node_to_rep.get(node, node)

def _trace_chain(G, start, entry, node_to_rep, eps_set):
    path = [start, entry]; visited = {start, entry}
    while not _is_crit(path[-1], node_to_rep, eps_set):
        nbrs = [n for n in G.neighbors(path[-1]) if n not in visited]
        if not nbrs: break
        path.append(nbrs[0]); visited.add(nbrs[0])
    return path


def _add_edge(G_b, u, v, path, G_vox, branch_lab, bid_counter, is_loop=False):
    pts_lps = [G_vox.nodes[p]['coord_lps'] for p in path]
    radii   = [G_vox.nodes[p]['radius']    for p in path]
    length  = sum(G_vox[path[i]][path[i+1]]['length_mm']
                  for i in range(len(path)-1))
    euc     = float(np.linalg.norm(
        np.array(pts_lps[-1]) - np.array(pts_lps[0])))
    orig    = list(set(int(branch_lab[p]) for p in path if branch_lab[p] > 0))
    bid_counter[0] += 1
    G_b.add_edge(u, v,
                 branch_id=bid_counter[0],
                 path_lps=pts_lps,
                 length_mm=length,
                 mean_radius=float(np.mean(radii)),
                 max_radius=float(np.max(radii)),
                 min_radius=float(np.min(radii)),
                 tortuosity=length / (euc + 1e-9),
                 original_bids=orig,
                 is_loop=is_loop)


def contract_to_branches(G, node_to_rep, endpoints, branch_lab):
    eps_set  = set(endpoints)
    critical = eps_set | set(node_to_rep.keys())
    G_b = nx.Graph()
    for node in critical:
        rep = _canon(node, node_to_rep)
        if not G_b.has_node(rep):
            nd = G.nodes[rep]
            G_b.add_node(rep,
                         pos_zyx=nd['pos_zyx'],
                         coord_lps=nd['coord_lps'],
                         radius=nd['radius'],
                         is_endpoint=(rep in eps_set))
    seen = set(); bid_counter = [0]
    for node in critical:
        for nb in G.neighbors(node):
            key = (min(node, nb, key=lambda v: v), max(node, nb, key=lambda v: v))
            if key in seen: continue
            seen.add(key)
            path = _trace_chain(G, node, nb, node_to_rep, eps_set)
            end  = path[-1]
            if not _is_crit(end, node_to_rep, eps_set): continue
            u = _canon(node, node_to_rep)
            v = _canon(end,  node_to_rep)
            if G_b.has_edge(u, v): continue
            _add_edge(G_b, u, v, path, G, branch_lab, bid_counter,
                      is_loop=(u == v))
    return G_b


def fix_branch_endpoints(G_b):
    """Snap path endpoints to node coord_lps — prevents visual gaps in Slicer.

    Applies to both path_lps (raw) and smooth_path_lps (smoothed), so the
    VTP export is always gap-free regardless of which key is used.
    """
    for u, v, data in G_b.edges(data=True):
        u_lps = G_b.nodes[u]['coord_lps']
        v_lps = G_b.nodes[v]['coord_lps']
        for key in ('path_lps', 'smooth_path_lps'):
            pts = data.get(key)
            if not pts or len(pts) < 2: continue
            d_start = np.linalg.norm(np.array(pts[0])  - np.array(u_lps))
            d_end   = np.linalg.norm(np.array(pts[-1]) - np.array(u_lps))
            interior = list(pts[1:-1])
            if d_start <= d_end:
                data[key] = [u_lps] + interior + [v_lps]
            else:
                data[key] = [v_lps] + interior + [u_lps]
    return G_b


def contract_local_loops(G_b, cfg):
    """Remove small cycles (skeleton artifacts near bifurcations)."""
    max_len = cfg.get('loop_max_length_mm', 8.0)
    removed = 0
    for u, v, d in list(G_b.edges(data=True)):
        if u == v or d.get('is_loop', False):
            G_b.remove_edge(u, v); removed += 1
    changed = True
    while changed:
        changed = False
        try: cycles = nx.minimum_cycle_basis(G_b)
        except Exception: break
        for cycle in cycles:
            total = 0.0; valid = True
            for i in range(len(cycle)):
                a, b = cycle[i], cycle[(i+1) % len(cycle)]
                if G_b.has_edge(a, b): total += G_b[a][b]['length_mm']
                else: valid = False; break
            if not valid or total > max_len: continue
            reps = [n for n in cycle if n in G_b]
            if not reps: continue
            keep     = max(reps, key=lambda n: G_b.nodes[n].get('radius', 0))
            keep_lps = np.array(G_b.nodes[keep]['coord_lps'])
            for node in reps:
                if node == keep or node not in G_b: continue
                node_lps = np.array(G_b.nodes[node]['coord_lps'])
                for nb in list(G_b.neighbors(node)):
                    if nb in (keep, node) or G_b.has_edge(keep, nb): continue
                    edata = dict(G_b[node][nb])
                    pts   = list(edata.get('path_lps', []))
                    if pts:
                        d_s = np.linalg.norm(np.array(pts[0])  - node_lps)
                        d_e = np.linalg.norm(np.array(pts[-1]) - node_lps)
                        if d_s <= d_e: edata['path_lps'] = [tuple(keep_lps)] + pts[1:]
                        else:          edata['path_lps'] = pts[:-1] + [tuple(keep_lps)]
                    G_b.add_edge(keep, nb, **edata)
                G_b.remove_node(node); removed += 1; changed = True
            break
    return G_b


def prune_terminal_branches(G_b, cfg):
    """Remove short/thin terminal branches iteratively."""
    lf = cfg.get('prune_len_factor', 2.5)
    rf = cfg.get('prune_radius_factor', 0.6)
    total = 0; changed = True
    while changed:
        changed = False
        for u, v, data in list(G_b.edges(data=True)):
            du, dv = G_b.degree(u), G_b.degree(v)
            if du == 1:   leaf, parent = u, v
            elif dv == 1: leaf, parent = v, u
            else: continue
            pr = G_b.nodes[parent].get('radius', 1.0)
            if data['length_mm'] < lf * pr and data['mean_radius'] < rf * pr:
                G_b.remove_edge(u, v)
                if G_b.degree(leaf) == 0: G_b.remove_node(leaf)
                total += 1; changed = True; break
    return G_b


def contract_degree2_nodes(G_b):
    """Merge degree-2 pass-through nodes (false junctions after pruning)."""
    merged = 0; changed = True
    while changed:
        changed = False
        for node in list(G_b.nodes()):
            if node not in G_b or G_b.degree(node) != 2: continue
            if G_b.nodes[node].get('is_endpoint', False): continue
            a, b = list(G_b.neighbors(node))
            if a == b: continue
            d_an = dict(G_b[a][node]); d_nb = dict(G_b[node][b])
            node_lps = np.array(G_b.nodes[node]['coord_lps'])
            pts_an = list(d_an.get('path_lps', []))
            if pts_an:
                if np.linalg.norm(np.array(pts_an[0]) - node_lps) < \
                   np.linalg.norm(np.array(pts_an[-1]) - node_lps):
                    pts_an = pts_an[::-1]
            pts_nb = list(d_nb.get('path_lps', []))
            if pts_nb:
                if np.linalg.norm(np.array(pts_nb[-1]) - node_lps) < \
                   np.linalg.norm(np.array(pts_nb[0]) - node_lps):
                    pts_nb = pts_nb[::-1]
            merged_pts  = pts_an + (pts_nb[1:] if pts_nb else [])
            merged_len  = d_an.get('length_mm', 0) + d_nb.get('length_mm', 0)
            merged_r    = (d_an.get('mean_radius', 0) * d_an.get('length_mm', 1) +
                           d_nb.get('mean_radius', 0) * d_nb.get('length_mm', 1)
                           ) / (merged_len + 1e-9)
            merged_orig = list(set(d_an.get('original_bids', []) +
                                   d_nb.get('original_bids', [])))
            if merged_pts:
                euc  = float(np.linalg.norm(
                    np.array(merged_pts[-1]) - np.array(merged_pts[0])))
                tort = merged_len / (euc + 1e-9)
            else:
                tort = 1.0
            if not G_b.has_edge(a, b):
                G_b.add_edge(a, b,
                             branch_id=max((e.get('branch_id', 0)
                                            for _, _, e in G_b.edges(data=True)),
                                           default=0) + 1,
                             path_lps=merged_pts,
                             length_mm=merged_len,
                             mean_radius=merged_r,
                             max_radius=max(d_an.get('max_radius', 0),
                                            d_nb.get('max_radius', 0)),
                             min_radius=min(d_an.get('min_radius', 0),
                                            d_nb.get('min_radius', 0)),
                             tortuosity=tort,
                             original_bids=merged_orig)
            G_b.remove_node(node); merged += 1; changed = True; break
    return G_b


def smooth_branch_paths(G_b, cfg):
    """Savitzky-Golay smoothing in LPS space; endpoints re-snapped."""
    window  = cfg.get('savgol_window', 11)
    polyord = cfg.get('savgol_polyorder', 3)
    for _, _, data in G_b.edges(data=True):
        pts = data.get('path_lps', [])
        if len(pts) < 4: continue
        arr = np.array(pts, dtype=float)
        n   = len(arr)
        w   = min(window, n) if min(window, n) % 2 == 1 else min(window, n) - 1
        w   = max(w, 3)
        po  = min(polyord, w - 1)
        smoothed = arr.copy()
        for dim in range(3):
            smoothed[:, dim] = savgol_filter(arr[:, dim], w, po)
        data['smooth_path_lps'] = [tuple(p) for p in smoothed]
    fix_branch_endpoints(G_b)
    return G_b


def build_graph_dicts(G_b, sp_zyx):
    """Build node_data and branch_data dicts for Scripts 03/04.

    pos_mm  = LPS (x, y, z) in mm   — used for visualization & distance calc
    polyline / smooth_polyline = LPS (x, y, z) numpy arrays
    """
    node_data = {}
    for node, nd in G_b.nodes(data=True):
        z, y, x = nd.get('pos_zyx', node)
        node_data[node] = dict(
            type="ep" if nd.get('is_endpoint', False) else "bif",
            pos_zyx=(int(z), int(y), int(x)),
            pos_mm=nd['coord_lps'],      # LPS (x, y, z)
            radius_mm=nd.get('radius', 0.0))

    branch_data = {}
    for u, v, edata in G_b.edges(data=True):
        bid  = edata.get('branch_id', 0)
        pts  = edata.get('path_lps', [])
        spts = edata.get('smooth_path_lps', pts)
        poly        = np.array(pts,  dtype=float) if pts  else np.zeros((0, 3))
        smooth_poly = np.array(spts, dtype=float) if spts else poly
        branch_data[bid] = dict(
            length_mm      = edata.get('length_mm', 0.0),
            mean_radius_mm = edata.get('mean_radius', 0.0),
            max_radius_mm  = edata.get('max_radius', 0.0),
            node_a=u, node_b=v,
            polyline=poly,
            smooth_polyline=smooth_poly,
            original_bids=edata.get('original_bids', []),
            n_merged=1)
        G_b[u][v]['mean_radius_mm'] = edata.get('mean_radius', 0.0)
        G_b[u][v]['score_weight']   = (edata.get('length_mm', 0.0) *
                                        edata.get('mean_radius', 0.0) ** 2)
        G_b[u][v]['smooth_polyline'] = smooth_poly

    return node_data, branch_data


# ── save ──────────────────────────────────────────────────────────────────────

def save_outputs(G, node_data, branch_data, sp_zyx, ref_img, out_dir, vessel):
    """Save graph.pkl + CSVs (for Scripts 03/04) + Slicer VTP/FCSV."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # graph.pkl
    with open(out_dir / f"{vessel}_graph.pkl", "wb") as f:
        pickle.dump({"G": G, "node_data": node_data,
                     "branch_data": branch_data, "sp_zyx": sp_zyx}, f)

    # node_data.csv
    rows_n = []
    for nid, nd in node_data.items():
        px, py, pz = nd["pos_mm"]
        rows_n.append(dict(node_id=str(nid), type=nd["type"],
                           lps_x=round(px, 2), lps_y=round(py, 2),
                           lps_z=round(pz, 2),
                           radius_mm=round(nd["radius_mm"], 3)))
    pd.DataFrame(rows_n).to_csv(out_dir / f"{vessel}_node_data.csv", index=False)

    # branch_data.csv
    rows_b = []
    for bid, bd in branch_data.items():
        rows_b.append(dict(branch_id=bid,
                           node_a=str(bd["node_a"]), node_b=str(bd["node_b"]),
                           length_mm=round(bd["length_mm"], 2),
                           mean_radius_mm=round(bd["mean_radius_mm"], 3),
                           max_radius_mm=round(bd["max_radius_mm"], 3),
                           n_points=len(bd["polyline"])))
    pd.DataFrame(rows_b).sort_values("branch_id").to_csv(
        out_dir / f"{vessel}_branch_data.csv", index=False)

    tqdm_write(f"      saved: graph.pkl + 2 CSVs  "
               f"({len(node_data)} nodes, {len(branch_data)} branches)")

    # Slicer exports
    _save_vtp(branch_data, out_dir / f"{vessel}_graph_smooth.vtp")
    _save_fcsv(node_data, out_dir / f"{vessel}_graph_endpoints.fcsv",   "ep",  "EP")
    _save_fcsv(node_data, out_dir / f"{vessel}_graph_bifurcations.fcsv","bif", "BIF")


def _save_vtp(branch_data, vtp_path):
    """Export smooth polylines as VTK PolyData (.vtp) for 3D Slicer."""
    if not HAS_VTK:
        tqdm_write("      [warn] VTK not available — skipping VTP export")
        return

    vtk_pts   = vtk.vtkPoints()
    vtk_cells = vtk.vtkCellArray()
    bid_list, len_list, rad_list = [], [], []

    for bid, bd in branch_data.items():
        poly = bd.get("smooth_polyline", bd["polyline"])
        if poly is None or len(poly) < 2: continue
        start = vtk_pts.GetNumberOfPoints()
        for pt in poly:
            vtk_pts.InsertNextPoint(-float(pt[0]), -float(pt[1]), float(pt[2]))
        n = len(poly)
        vtk_cells.InsertNextCell(n)
        for j in range(n):
            vtk_cells.InsertCellPoint(start + j)
        bid_list.append(int(bid))
        len_list.append(float(bd["length_mm"]))
        rad_list.append(float(bd["mean_radius_mm"]))

    pd_vtk = vtk.vtkPolyData()
    pd_vtk.SetPoints(vtk_pts)
    pd_vtk.SetLines(vtk_cells)

    for arr_np, name in [(np.array(bid_list, np.int32),   "branch_id"),
                          (np.array(len_list, np.float32), "length_mm"),
                          (np.array(rad_list, np.float32), "mean_radius_mm")]:
        vtk_arr = numpy_support.numpy_to_vtk(arr_np, deep=True)
        vtk_arr.SetName(name)
        pd_vtk.GetCellData().AddArray(vtk_arr)
    pd_vtk.GetCellData().SetActiveScalars("mean_radius_mm")

    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(str(vtp_path))
    w.SetInputData(pd_vtk)
    w.Write()
    tqdm_write(f"      → {vtp_path.name}  "
               f"({len(bid_list)} branches, {vtk_pts.GetNumberOfPoints()} pts)")


def _save_fcsv(node_data, fcsv_path, node_type, label_prefix):
    """Export nodes of a given type as Slicer Markup Fiducials (.fcsv)."""
    entries = [(nid, nd) for nid, nd in node_data.items()
               if nd["type"] == node_type]
    rows = []
    for i, (_, nd) in enumerate(entries):
        lx, ly, lz = nd["pos_mm"]    # LPS (x, y, z)
        label = f"{label_prefix}-{i}"
        rows.append(f"vtkMRMLMarkupsFiducialNode_{i},"
                    f"{lx:.4f},{ly:.4f},{lz:.4f},"
                    f"0,0,0,1,1,1,0,{label},,")
    with open(fcsv_path, "w") as f:
        f.write(FCSV_HEADER)
        f.write("\n".join(rows) + "\n")
    tqdm_write(f"      → {fcsv_path.name}  ({len(rows)} points)")


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize(case_id, vessel, G, node_data, branch_data, out_dir):
    if not HAS_NX or G is None:
        return

    all_r  = [bd["mean_radius_mm"] for bd in branch_data.values()]
    r_min  = min(all_r) if all_r else 0
    r_max  = max(all_r) if all_r else 1
    cmap_r = plt.colormaps["plasma"]

    def r_to_color(r):
        return cmap_r((r - r_min) / (r_max - r_min + 1e-9))

    def _3d_axis(fig, pos, title):
        ax = fig.add_subplot(*pos, projection="3d")
        ax.set_facecolor("#111111")
        ax.tick_params(labelsize=5, colors="white")
        ax.set_xlabel("x mm", fontsize=6, color="white")
        ax.set_ylabel("y mm", fontsize=6, color="white")
        ax.set_zlabel("z mm", fontsize=6, color="white")
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.set_title(title, fontsize=8, color="white")
        ax.view_init(28, -50)
        return ax

    def _draw_polylines(ax, use_smooth=True):
        for _, bd in branch_data.items():
            key  = "smooth_polyline" if use_smooth else "polyline"
            poly = bd.get(key, bd["polyline"])
            if poly is None or len(poly) < 2: continue
            col = r_to_color(bd["mean_radius_mm"])
            # poly is LPS (x, y, z)
            ax.plot(poly[:, 0], poly[:, 1], poly[:, 2],
                    color=col, linewidth=1.2, alpha=0.85)

    def _draw_nodes(ax):
        for nd in node_data.values():
            px, py, pz = nd["pos_mm"]   # LPS (x, y, z)
            col  = "#22ff88" if nd["type"] == "ep" else "#ffdd00"
            size = 40        if nd["type"] == "ep" else 80
            mk   = "o"       if nd["type"] == "ep" else "D"
            ax.scatter([px], [py], [pz], c=col, s=size, marker=mk,
                       linewidths=0, depthshade=False, zorder=5)

    n_ep  = sum(1 for nd in node_data.values() if nd["type"] == "ep")
    n_bif = sum(1 for nd in node_data.values() if nd["type"] == "bif")

    fig = plt.figure(figsize=(22, 10))
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(
        f"case_{case_id}  {vessel.upper()}  —  Script 02: Graph Construction\n"
        f"nodes: {n_ep} endpoints (●)  +  {n_bif} bifurcations (◆)  |  "
        f"branches: {len(branch_data)}",
        fontsize=11, fontweight="bold", color="white")

    ax1 = _3d_axis(fig, (2, 3, 1), "Smooth polylines (colour = radius)")
    _draw_polylines(ax1, use_smooth=True); _draw_nodes(ax1)
    sm = plt.cm.ScalarMappable(cmap="plasma",
                                norm=plt.Normalize(vmin=r_min, vmax=r_max))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax1, shrink=0.5, pad=0.05)
    cb.set_label("mean radius mm", fontsize=7, color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax2 = _3d_axis(fig, (2, 3, 2), "Raw polylines (pre-smoothing)")
    _draw_polylines(ax2, use_smooth=False); _draw_nodes(ax2)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#111111")
    degrees = [deg for _, deg in G.degree()]
    if degrees:
        ax3.hist(degrees, bins=range(1, max(degrees)+2), color="steelblue",
                 edgecolor="none", align="left")
    ax3.set_xlabel("node degree", fontsize=8, color="white")
    ax3.set_ylabel("count",       fontsize=8, color="white")
    ax3.set_title("Node degree distribution", fontsize=8, color="white")
    ax3.tick_params(colors="white", labelsize=7)
    for sp in ax3.spines.values(): sp.set_edgecolor("#444444")

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#111111")
    lengths = [bd["length_mm"] for bd in branch_data.values()]
    if lengths:
        ax4.hist(lengths, bins=30, color="#4488ff", edgecolor="none")
        med = float(np.median(lengths))
        ax4.axvline(med, color="orange", lw=1.5, label=f"median {med:.1f} mm")
        ax4.legend(fontsize=7, labelcolor="white", facecolor="#222222", edgecolor="gray")
    ax4.set_xlabel("branch length mm", fontsize=8, color="white")
    ax4.set_ylabel("count",            fontsize=8, color="white")
    ax4.set_title("Branch length distribution", fontsize=8, color="white")
    ax4.tick_params(colors="white", labelsize=7)
    for sp in ax4.spines.values(): sp.set_edgecolor("#444444")

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#111111")
    radii = [bd["mean_radius_mm"] for bd in branch_data.values()]
    if radii:
        ax5.hist(radii, bins=25, color="#ff6644", edgecolor="none")
        med_r = float(np.median(radii))
        ax5.axvline(med_r, color="orange", lw=1.5, label=f"median {med_r:.2f} mm")
        ax5.legend(fontsize=7, labelcolor="white", facecolor="#222222", edgecolor="gray")
    ax5.set_xlabel("mean radius mm", fontsize=8, color="white")
    ax5.set_ylabel("count",          fontsize=8, color="white")
    ax5.set_title("Branch radius distribution", fontsize=8, color="white")
    ax5.tick_params(colors="white", labelsize=7)
    for sp in ax5.spines.values(): sp.set_edgecolor("#444444")

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#111111"); ax6.axis("off")
    comps = list(nx.connected_components(G))
    lines = [
        f"Connected components: {len(comps)}",
        f"Largest: {max(len(c) for c in comps)} nodes" if comps else "",
        "",
        f"Total nodes:    {len(node_data)}",
        f"  endpoints:    {n_ep}",
        f"  bifurcations: {n_bif}",
        "",
        f"Total branches: {len(branch_data)}",
        "",
        f"Avg branch length: {np.mean(lengths):.1f} mm" if lengths else "",
        f"Avg radius:        {np.mean(radii):.2f} mm"  if radii  else "",
    ]
    for i, line in enumerate(lines):
        ax6.text(0.05, 0.92 - i*0.08, line, transform=ax6.transAxes,
                 fontsize=8, color="white", va="top", family="monospace")
    ax6.set_title("Graph summary", fontsize=8, color="white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png = out_dir / f"{vessel}_graph_vis.png"
    fig.savefig(str(png), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    tqdm_write(f"      figure saved → {png.name}")
    plt.show()
    plt.close(fig)


# ── per-vessel pipeline ────────────────────────────────────────────────────────

def process_vessel(case_id, vessel, cfg, out_dir):
    tag = f"case_{case_id} [{vessel.upper()}]"
    tqdm_write(f"  {tag}")

    tqdm_write(f"    {tag}  1/7  loading Script 01 outputs ...")
    d = load_script01(case_id, vessel, out_dir)
    tqdm_write(f"    {tag}       skeleton={d['skel'].sum():,} vox  "
               f"branches={int(d['branch_lab'].max())}")

    tqdm_write(f"    {tag}  2/7  building voxel graph ...")
    G_vox = build_voxel_graph(d["skel"], d["radius_map"], d["ref_img"])
    tqdm_write(f"    {tag}       {G_vox.number_of_nodes():,} nodes  "
               f"{G_vox.number_of_edges():,} edges")

    tqdm_write(f"    {tag}  3/7  contracting to branch graph ...")
    eps, bifs   = classify_nodes(G_vox)
    node_to_rep = cluster_bifurcations(G_vox, bifs)
    G_b         = contract_to_branches(G_vox, node_to_rep, eps, d["branch_lab"])
    G_b         = fix_branch_endpoints(G_b)

    tqdm_write(f"    {tag}  4/7  removing local loops "
               f"(max={cfg['loop_max_length_mm']} mm) ...")
    G_b = contract_local_loops(G_b, cfg)
    G_b = fix_branch_endpoints(G_b)

    tqdm_write(f"    {tag}  5/7  pruning terminal branches ...")
    G_b = prune_terminal_branches(G_b, cfg)
    G_b = fix_branch_endpoints(G_b)

    tqdm_write(f"    {tag}  6/7  merging degree-2 nodes ...")
    G_b = contract_degree2_nodes(G_b)
    G_b = fix_branch_endpoints(G_b)

    tqdm_write(f"    {tag}  7/7  Savitzky-Golay smoothing "
               f"(window={cfg['savgol_window']}, order={cfg['savgol_polyorder']}) ...")
    G_b = smooth_branch_paths(G_b, cfg)

    node_data, branch_data = build_graph_dicts(G_b, d["sp_zyx"])
    save_outputs(G_b, node_data, branch_data, d["sp_zyx"], d["ref_img"],
                 out_dir, vessel)

    return dict(G=G_b, node_data=node_data, branch_data=branch_data,
                sp_zyx=d["sp_zyx"], mask=d["mask"])


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Script 02 — Graph Construction & Branch Smoothing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--case",                default="001")
    ap.add_argument("--vessel",              default="both",
                    choices=["pv", "hv", "both"])
    ap.add_argument("--all",                 action="store_true")
    ap.add_argument("--no-vis",              action="store_true")
    ap.add_argument("--loop-max-length",     type=float, default=None)
    ap.add_argument("--prune-len-factor",    type=float, default=None)
    ap.add_argument("--prune-radius-factor", type=float, default=None)
    ap.add_argument("--savgol-window",       type=int,   default=None)
    ap.add_argument("--savgol-order",  type=int,   default=None)
    ap.add_argument("--output-dir",    default=None,
                    help="override output directory")
    ap.add_argument("--ids",           default=None,
                    help="comma-separated case IDs to process")
    args = ap.parse_args()

    global OUTPUT_DIR
    if args.output_dir:
        OUTPUT_DIR = Path(args.output_dir)

    if not HAS_NX:
        print("ERROR: networkx required.  pip install networkx")
        return

    pv_cfg = dict(PV_CFG); hv_cfg = dict(HV_CFG)
    for cfg in (pv_cfg, hv_cfg):
        if args.loop_max_length     is not None: cfg["loop_max_length_mm"]   = args.loop_max_length
        if args.prune_len_factor    is not None: cfg["prune_len_factor"]     = args.prune_len_factor
        if args.prune_radius_factor is not None: cfg["prune_radius_factor"]  = args.prune_radius_factor
        if args.savgol_window       is not None: cfg["savgol_window"]        = args.savgol_window
        if args.savgol_order        is not None: cfg["savgol_polyorder"]     = args.savgol_order

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
            if result and not args.no_vis:
                visualize(args.case, v,
                          result["G"], result["node_data"],
                          result["branch_data"], out)
        print(f"\nOutput → {out}")


if __name__ == "__main__":
    main()
