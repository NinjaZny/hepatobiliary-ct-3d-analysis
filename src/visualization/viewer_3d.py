#!/usr/bin/env python3
"""
Liver Anatomy 3D Viewer

Views:
  Anatomy      -- Liver, Bile Duct, Hepatic/Portal veins + global volumes
  CC Partition -- Liver + bile duct coloured by CC region (max 3) + per-part volumes
  Couinaud     -- Bile duct coloured by Couinaud segment (I-VIII) + per-segment volumes

Volume numbers are only shown in the panel when the corresponding partition
view (CC or Couinaud) is active.

Run:
    python viewer_3d.py
"""

import sys
import os
import numpy as np

for _pkg in ["nibabel", "pyvista", "pyvistaqt", "PyQt5"]:
    try:
        __import__(_pkg)
    except ImportError:
        print(f"pip install {_pkg}"); sys.exit(1)

import nibabel as nib
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QStackedWidget

# ==============================================================================
# Paths — configure before running
# ==============================================================================

_BASE = os.environ.get("VIEWER_DATA", os.path.join("data", "viewer"))
_DIRS = {
    "bile":     os.path.join(_BASE, "bile_duct"),        # {n}.nii(.gz), n = case_id
    "liver":    os.path.join(_BASE, "liver"),            # case_{n-1:03d}.nii.gz
    "vessel":   os.path.join(_BASE, "vessels"),          # case_{n-1:03d}.nii.gz, HV=1, PV=2
    "cc":       os.path.join(_BASE, "cc_partition"),     # output of bile_duct_cc_partition.py
    "couinaud": os.path.join(_BASE, "couinaud"),         # output of bile_duct_by_couinaud.py
}
MAX_CASE = 11

# ==============================================================================
# Structure / partition colour tables
# ==============================================================================

STRUCTURES = [
    # (key, label, hex_color, default_opacity, dir_key, label_range)
    ("liver",   "Liver",         "#D4A574", 0.30, "liver",  None),
    ("bile",    "Bile Duct",     "#E8D44D", 1.00, "bile",   None),
    ("hepatic", "Hepatic Veins", "#4878C0", 1.00, "vessel", (1, 1)),
    ("portal",  "Portal Veins",  "#C55A11", 1.00, "vessel", (2, 2)),
]
_MESH_CFG = {
    "liver":   dict(smooth=30, decimate=0.85),
    "bile":    dict(smooth=28, decimate=0.60),
    "hepatic": dict(smooth=20, decimate=0.72),
    "portal":  dict(smooth=20, decimate=0.72),
    "part":    dict(smooth=22, decimate=0.65),
    "liver_part": dict(smooth=15, decimate=0.85),
}
CC_PARTS = [
    (1, "Part 1", "#E74C3C"),
    (2, "Part 2", "#2ECC71"),
    (3, "Part 3", "#3498DB"),
]
COUINAUD_SEGS = [
    (1, "Seg I",    "#9B59B6"),
    (2, "Seg II",   "#F39C12"),
    (3, "Seg III",  "#1ABC9C"),
    (4, "Seg IV",   "#E74C3C"),
    (5, "Seg V",    "#3498DB"),
    (6, "Seg VI",   "#2ECC71"),
    (7, "Seg VII",  "#E67E22"),
    (8, "Seg VIII", "#C0392B"),
]

# ==============================================================================
# File discovery
# ==============================================================================

def _find_file(case_id: int, dir_key: str) -> str | None:
    d = _DIRS.get(dir_key, "")
    if dir_key == "bile":
        for ext in ["", ".gz"]:
            p = os.path.join(d, f"{case_id}.nii{ext}")
            if os.path.exists(p): return p
    else:
        n = case_id - 1
        if n < 1: return None
        for ext in [".nii.gz", ".nii"]:
            p = os.path.join(d, f"case_{n:03d}{ext}")
            if os.path.exists(p): return p
    return None


def _find_part_file(case_id: int, kind: str) -> str | None:
    """kind: 'bile_parts' | 'liver_parts' | 'couinaud' | 'liver_couinaud'
    Tries both deliverable naming (*_bile_parts) and validation script naming (*_bile_cc).
    """
    n = case_id - 1
    if n < 1: return None
    dir_key = "couinaud" if kind in ("couinaud", "liver_couinaud") else "cc"
    d = _DIRS.get(dir_key, "")
    aliases = {
        "bile_parts":     ["bile_parts",  "bile_cc"],
        "liver_parts":    ["liver_parts", "liver_cc"],
        "couinaud":       ["couinaud"],
        "liver_couinaud": ["liver_couinaud"],
    }
    for stem in aliases.get(kind, [kind]):
        for ext in [".nii.gz", ".nii"]:
            p = os.path.join(d, f"case_{n:03d}_{stem}{ext}")
            if os.path.exists(p): return p
    return None

# ==============================================================================
# NIfTI loading
# ==============================================================================

def _load_labeled(path: str):
    """Load a label NIfTI; returns (data_uint8, spacing, origin) with flipped axes corrected."""
    img  = nib.load(path)
    data = np.round(np.asarray(img.dataobj)).astype(np.uint8)
    aff  = img.affine
    origin = [float(aff[i, 3]) for i in range(3)]
    for ax in range(3):
        if aff[ax, ax] < 0:
            data       = np.flip(data, axis=ax)
            origin[ax] = float(aff[ax, 3] + aff[ax, ax] * (data.shape[ax] - 1))
    spacing = tuple(abs(float(aff[i, i])) for i in range(3))
    return np.ascontiguousarray(data), spacing, tuple(origin)


def _load_mask(path: str, label_range=None):
    data, spacing, origin = _load_labeled(path)
    if label_range:
        lo, hi = label_range
        mask = ((data >= lo) & (data <= hi)).astype(np.uint8)
    else:
        mask = (data > 0).astype(np.uint8)
    return mask, spacing, origin

# ==============================================================================
# Mesh building + cache
# ==============================================================================

def _build_mesh(mask: np.ndarray, spacing, origin, smooth=20, decimate=0.80):
    grid = pv.ImageData(dimensions=mask.shape, spacing=spacing, origin=origin)
    grid.point_data["v"] = mask.ravel(order="F").astype(np.float32)
    mesh = grid.contour([0.5], scalars="v", method="flying_edges")
    if not mesh.n_points: return None
    mesh = (mesh
            .smooth(n_iter=smooth, relaxation_factor=0.15, boundary_smoothing=False)
            .decimate(target_reduction=decimate)
            .compute_normals(consistent_normals=True, split_vertices=False))
    return mesh if mesh.n_points else None


_CACHE: dict = {}


def _get_mesh(case_id, key, dir_key, label_range=None, smooth=20, decimate=0.8):
    ck = (case_id, key)
    if ck in _CACHE: return _CACHE[ck]
    path = _find_file(case_id, dir_key)
    if not path: _CACHE[ck] = None; return None
    try:
        mask, sp, org = _load_mask(path, label_range)
        _CACHE[ck] = _build_mesh(mask, sp, org, smooth, decimate) if mask.sum() else None
    except Exception as e:
        print(f"[WARN] {key} case {case_id}: {e}"); _CACHE[ck] = None
    return _CACHE[ck]


def _get_partition_meshes(case_id: int, mode: str) -> list:
    """Returns [(part_id, mesh, color, name), ...] for 'cc' (bile) or 'couinaud'."""
    ck = (case_id, mode)
    if ck in _CACHE: return _CACHE[ck]
    kind = "bile_parts" if mode == "cc" else "couinaud"
    path = _find_part_file(case_id, kind)
    defs = CC_PARTS if mode == "cc" else COUINAUD_SEGS
    if not path: _CACHE[ck] = []; return []
    try:
        data, sp, org = _load_labeled(path)
        cfg = _MESH_CFG["part"]
        results = []
        for pid, name, color in defs:
            mask = (data == pid).astype(np.uint8)
            if not mask.sum(): continue
            mesh = _build_mesh(mask, sp, org, **cfg)
            if mesh: results.append((pid, mesh, color, name))
        _CACHE[ck] = results
    except Exception as e:
        print(f"[WARN] {mode} meshes case {case_id}: {e}"); _CACHE[ck] = []
    return _CACHE[ck]


def _get_liver_part_meshes(case_id: int) -> list:
    """Returns [(part_id, mesh, color), ...] for the CC-partitioned liver."""
    ck = (case_id, "cc_liver")
    if ck in _CACHE: return _CACHE[ck]
    path = _find_part_file(case_id, "liver_parts")
    if not path: _CACHE[ck] = []; return []
    try:
        data, sp, org = _load_labeled(path)
        cfg = _MESH_CFG["liver_part"]
        results = []
        for pid, _, color in CC_PARTS:
            mask = (data == pid).astype(np.uint8)
            if not mask.sum(): continue
            mesh = _build_mesh(mask, sp, org, **cfg)
            if mesh: results.append((pid, mesh, color))
        _CACHE[ck] = results
    except Exception as e:
        print(f"[WARN] cc_liver case {case_id}: {e}"); _CACHE[ck] = []
    return _CACHE[ck]


def _get_liver_couinaud_meshes(case_id: int) -> list:
    """Returns [(seg_id, mesh, color, name), ...] for the Couinaud-partitioned liver."""
    ck = (case_id, "couinaud_liver")
    if ck in _CACHE: return _CACHE[ck]
    path = _find_part_file(case_id, "liver_couinaud")
    if not path: _CACHE[ck] = []; return []
    try:
        data, sp, org = _load_labeled(path)
        cfg = _MESH_CFG["liver_part"]
        results = []
        for pid, name, color in COUINAUD_SEGS:
            mask = (data == pid).astype(np.uint8)
            if not mask.sum(): continue
            mesh = _build_mesh(mask, sp, org, **cfg)
            if mesh: results.append((pid, mesh, color, name))
        _CACHE[ck] = results
    except Exception as e:
        print(f"[WARN] couinaud_liver case {case_id}: {e}"); _CACHE[ck] = []
    return _CACHE[ck]

# ==============================================================================
# Volume computation (inline — no external module)
# ==============================================================================

def _vox_mm3(nii: nib.Nifti1Image) -> float:
    return abs(float(np.linalg.det(nii.affine[:3, :3])))


def _compute_anatomy_vols(case_id: int) -> dict:
    MM, r = 1e-3, {}
    liver_arr = None

    p = _find_file(case_id, "liver")
    if p:
        nii = nib.load(p); arr = np.asarray(nii.dataobj) > 0; vv = _vox_mm3(nii)
        liver_arr = arr; r["liver_vol"] = float(arr.sum()) * vv * MM

    p = _find_file(case_id, "bile")
    if p:
        nii = nib.load(p); arr = np.asarray(nii.dataobj) > 0; vv = _vox_mm3(nii)
        if liver_arr is not None and liver_arr.shape == arr.shape: arr = arr & liver_arr
        r["bile_vol"] = float(arr.sum()) * vv * MM
        if r.get("liver_vol"): r["bile_pct"] = r["bile_vol"] / r["liver_vol"] * 100

    p = _find_file(case_id, "vessel")
    if p:
        nii = nib.load(p); data = np.asarray(nii.dataobj); vv = _vox_mm3(nii)
        for key, lbl in [("hepatic_vol", 1), ("portal_vol", 2)]:
            a = np.round(data) == lbl
            if liver_arr is not None and liver_arr.shape == a.shape: a = a & liver_arr
            r[key] = float(a.sum()) * vv * MM
        r["vessel_vol"] = r.get("hepatic_vol", 0) + r.get("portal_vol", 0)
        if r.get("liver_vol"): r["vessel_pct"] = r["vessel_vol"] / r["liver_vol"] * 100

    if r.get("liver_vol"):
        para = r["liver_vol"] - r.get("vessel_vol", 0) - r.get("bile_vol", 0)
        r["parenchyma_vol"] = max(0.0, para)
        r["parenchyma_pct"] = r["parenchyma_vol"] / r["liver_vol"] * 100
    return r


def _compute_partition_vols(case_id: int, mode: str) -> dict:
    """Returns {part_id: {'bile_mL': float, 'liver_mL': float|None, 'liver_pct': float|None}}.
    liver_mL / liver_pct are filled for CC (from liver_cc file); None for Couinaud.
    """
    kind = "bile_parts" if mode == "cc" else "couinaud"
    defs = CC_PARTS if mode == "cc" else COUINAUD_SEGS
    p = _find_part_file(case_id, kind)
    if not p: return {}
    try:
        nii  = nib.load(p)
        data = np.round(np.asarray(nii.dataobj)).astype(np.uint8)
        vv   = _vox_mm3(nii) * 1e-3
        bile_vols = {pid: float((data == pid).sum()) * vv for pid, *_ in defs}
    except Exception:
        return {}

    # Liver territory per part (CC: liver_parts file; Couinaud: liver_couinaud file)
    liver_vols, liver_total = {}, None
    liver_kind = "liver_parts" if mode == "cc" else "liver_couinaud"
    lp = _find_part_file(case_id, liver_kind)
    if lp:
        try:
            l_nii  = nib.load(lp)
            l_data = np.round(np.asarray(l_nii.dataobj)).astype(np.uint8)
            l_vv   = _vox_mm3(l_nii) * 1e-3
            liver_total = float((l_data > 0).sum()) * l_vv
            liver_vols  = {pid: float((l_data == pid).sum()) * l_vv for pid, *_ in defs}
        except Exception:
            pass

    result = {}
    for pid, *_ in defs:
        bile_mL   = bile_vols.get(pid, 0.0)
        liver_mL  = liver_vols.get(pid) if liver_vols else None
        liver_pct = (liver_mL / liver_total * 100
                     if (liver_mL is not None and liver_total) else None)
        result[pid] = {'bile_mL': bile_mL, 'liver_mL': liver_mL, 'liver_pct': liver_pct}
    return result

# ==============================================================================
# Qt stylesheet
# ==============================================================================

_QSS = """
QWidget#panel { background-color: #F5F5F5; }
QGroupBox {
    color: #333333; border: 1px solid #CCCCCC; border-radius: 5px;
    margin-top: 14px; font-weight: bold; font-size: 12px;
    padding: 6px 4px 4px 4px; background-color: #FAFAFA;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #1A4A7A; background-color: #F5F5F5;
}
QPushButton {
    background-color: #E8E8E8; color: #2C2C2C;
    border: 1px solid #B8B8B8; border-radius: 4px;
    padding: 5px 8px; font-size: 11px;
}
QPushButton:hover   { background-color: #D6E8F8; border-color: #4878C0; color: #1A3A5A; }
QPushButton:pressed { background-color: #B8D0E8; }
QPushButton:checked { background-color: #4878C0; border-color: #3060A8; color: #FFFFFF; }
QPushButton:disabled { color: #AAAAAA; border-color: #DDDDDD; background: #F0F0F0; }
QComboBox {
    background-color: #FFFFFF; color: #2C2C2C;
    border: 1px solid #B8B8B8; border-radius: 4px; padding: 4px 8px; font-size: 12px;
}
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background-color: #FFFFFF; color: #2C2C2C;
    selection-background-color: #4878C0; selection-color: #FFFFFF;
    border: 1px solid #B8B8B8;
}
QSlider::groove:horizontal { height: 4px; background: #D0D0D0; border-radius: 2px; }
QSlider::handle:horizontal {
    width: 14px; height: 14px; background: #4878C0;
    border-radius: 7px; margin: -5px 0; border: 1px solid #3060A8;
}
QSlider::sub-page:horizontal { background: #7AAAD8; border-radius: 2px; }
QCheckBox { color: #2C2C2C; spacing: 6px; font-size: 12px; }
QCheckBox::indicator {
    width: 16px; height: 16px; border: 1px solid #B8B8B8;
    border-radius: 3px; background: #FFFFFF;
}
QCheckBox::indicator:checked  { background: #4878C0; border-color: #3060A8; }
QCheckBox::indicator:disabled { background: #EEEEEE; border-color: #CCCCCC; }
QCheckBox:disabled { color: #AAAAAA; }
QLabel { color: #2C2C2C; font-size: 12px; }
QScrollBar:vertical { background: #F0F0F0; width: 8px; border: none; }
QScrollBar::handle:vertical { background: #C0C0C0; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #A0A0A0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #DDDDDD; }
"""

# ==============================================================================
# Main Window
# ==============================================================================

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Liver Anatomy 3D Viewer")
        self.resize(1540, 900)
        self.setStyleSheet("QMainWindow { background: #1A1A1A; }")

        self._case_id   = 1
        self._actors    = {}         # anatomy actors: key -> actor
        self._part_acts = {}         # partition actors: (mode, pid) -> actor
        self._visible   = {s[0]: True for s in STRUCTURES}
        self._opacity   = {s[0]: s[3] for s in STRUCTURES}
        self._sw        = {}         # structure row widgets
        self._view_mode = "anatomy"  # "anatomy" | "cc" | "couinaud"

        # Volume label dicts (populated during panel build)
        self._anat_vol_lbls = {}   # key -> (QLabel, pct_key)
        self._cc_vol_lbls   = {}   # pid -> QLabel
        self._coui_vol_lbls = {}   # pid -> QLabel

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_panel())

        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet("color: #3A3A3A;")
        root.addWidget(sep)

        self.plotter = QtInteractor(central, auto_update=False)
        self.plotter.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        root.addWidget(self.plotter, stretch=1)

        self._setup_plotter()
        self._rot_timer = QTimer(self)
        self._rot_timer.setInterval(33)
        self._rot_timer.timeout.connect(self._rotate_step)
        QTimer.singleShot(100, lambda: self._load_case(1))

    # ── plotter ───────────────────────────────────────────────────────────────

    def _setup_plotter(self):
        self.plotter.set_background("#111111", top="#050505")
        self.plotter.enable_trackball_style()
        self.plotter.enable_lightkit()
        self.plotter.add_axes(line_width=3, labels_off=False,
                              xlabel="R", ylabel="A", zlabel="S",
                              interactive=False, viewport=(0.0, 0.0, 0.22, 0.22),
                              color="white")

    # ── panel ─────────────────────────────────────────────────────────────────

    def _build_panel(self):
        panel = QtWidgets.QWidget()
        panel.setObjectName("panel"); panel.setFixedWidth(290)
        panel.setStyleSheet(_QSS)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #F5F5F5; border: none; }")

        inner = QtWidgets.QWidget(); inner.setStyleSheet("background: #F5F5F5;")
        vbox = QtWidgets.QVBoxLayout(inner)
        vbox.setContentsMargins(10, 10, 10, 10); vbox.setSpacing(10)

        title = QtWidgets.QLabel("Liver Anatomy 3D Viewer")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size:15px; font-weight:bold; color:#1A4A7A; padding:8px 0 4px;")
        vbox.addWidget(title); vbox.addWidget(self._hline())

        # Case selector
        cg = QtWidgets.QGroupBox("Patient Case")
        cgl = QtWidgets.QVBoxLayout(cg); cgl.setContentsMargins(8,16,8,8); cgl.setSpacing(5)
        self._case_combo = QtWidgets.QComboBox()
        for i in range(1, MAX_CASE + 1):
            self._case_combo.addItem(f"Case {i:02d}", i)
        self._case_combo.currentIndexChanged.connect(self._on_case_changed)
        cgl.addWidget(self._case_combo)
        btn = QtWidgets.QPushButton("Force Reload (clear cache)")
        btn.clicked.connect(self._force_reload); cgl.addWidget(btn)
        vbox.addWidget(cg)

        # Anatomy structures
        sg = QtWidgets.QGroupBox("Structures")
        sgl = QtWidgets.QVBoxLayout(sg); sgl.setContentsMargins(8,16,8,8); sgl.setSpacing(14)
        for key, label, color, opacity, *_ in STRUCTURES:
            sgl.addWidget(self._struct_row(key, label, color, opacity))
        vbox.addWidget(sg)

        # Quick views
        vg = QtWidgets.QGroupBox("Quick Views")
        vgl = QtWidgets.QGridLayout(vg); vgl.setContentsMargins(8,16,8,8); vgl.setSpacing(4)
        presets = [
            ("All",          self._show_all),
            ("Liver",        lambda: self._activate_anatomy("liver")),
            ("Bile Duct",    lambda: self._activate_anatomy("bile")),
            ("Hepatic V.",   lambda: self._activate_anatomy("hepatic")),
            ("Portal V.",    lambda: self._activate_anatomy("portal")),
            ("Vessels",      lambda: self._activate_anatomy("hepatic", "portal")),
            ("CC Partition", lambda: self._activate_mode("cc")),
            ("Couinaud",     lambda: self._activate_mode("couinaud")),
        ]
        for i, (name, fn) in enumerate(presets):
            b = QtWidgets.QPushButton(name)
            b.clicked.connect(fn)
            vgl.addWidget(b, i // 2, i % 2)
        vbox.addWidget(vg)

        # Camera
        cam = QtWidgets.QGroupBox("Camera")
        caml = QtWidgets.QVBoxLayout(cam); caml.setContentsMargins(8,16,8,8); caml.setSpacing(6)
        self._rot_btn = QtWidgets.QPushButton("Auto Rotate")
        self._rot_btn.setCheckable(True)
        self._rot_btn.toggled.connect(self._toggle_rotation)
        caml.addWidget(self._rot_btn)
        spd = QtWidgets.QHBoxLayout()
        spd.addWidget(QtWidgets.QLabel("Speed:"))
        self._spd_sl = QtWidgets.QSlider(Qt.Horizontal)
        self._spd_sl.setRange(1, 10); self._spd_sl.setValue(3)
        spd.addWidget(self._spd_sl); caml.addLayout(spd)
        caml.addWidget(self._hline())
        dr = QtWidgets.QHBoxLayout()
        for txt, fn in [("Front", self._view_front),
                        ("Top",   self._view_top),
                        ("Right", self._view_right)]:
            b = QtWidgets.QPushButton(txt); b.clicked.connect(fn); dr.addWidget(b)
        caml.addLayout(dr)
        rb = QtWidgets.QPushButton("Reset Camera"); rb.clicked.connect(self._reset_camera)
        caml.addWidget(rb)
        vbox.addWidget(cam)

        # Volume panel
        vbox.addWidget(self._build_vol_panel())

        vbox.addStretch(); vbox.addWidget(self._hline())
        self._status_lbl = QtWidgets.QLabel("Ready")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setWordWrap(True)
        self._status_lbl.setStyleSheet("color: #888888; font-size: 10px; padding: 4px;")
        vbox.addWidget(self._status_lbl)

        scroll.setWidget(inner)
        outer = QtWidgets.QWidget(); outer.setFixedWidth(290)
        ol = QtWidgets.QVBoxLayout(outer)
        ol.setContentsMargins(0, 0, 0, 0); ol.addWidget(scroll)
        return outer

    def _build_vol_panel(self):
        grp = QtWidgets.QGroupBox("Volume Analysis")
        lay = QtWidgets.QVBoxLayout(grp); lay.setContentsMargins(8,16,8,8); lay.setSpacing(4)

        self._vol_stack = QStackedWidget()

        # ── Page 0: anatomy volumes ───────────────────────────────────────────
        p0 = QtWidgets.QWidget()
        p0l = QtWidgets.QVBoxLayout(p0); p0l.setContentsMargins(0,0,0,0); p0l.setSpacing(3)
        for key, label, pct_key in [
            ("liver_vol",      "Liver total",   None),
            ("hepatic_vol",    "Hepatic Veins", None),
            ("portal_vol",     "Portal Veins",  None),
            ("vessel_vol",     "Total Vessels", "vessel_pct"),
            ("bile_vol",       "Bile Duct",     "bile_pct"),
            ("parenchyma_vol", "Parenchyma",    "parenchyma_pct"),
        ]:
            row = QtWidgets.QHBoxLayout()
            n = QtWidgets.QLabel(label + ":"); n.setFixedWidth(110)
            n.setStyleSheet("color: #555555; font-size: 11px;")
            v = QtWidgets.QLabel("—")
            v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            v.setStyleSheet(
                "color: #1A3A6A; font-size: 11px; font-family: Consolas, monospace;")
            row.addWidget(n); row.addWidget(v, stretch=1)
            p0l.addLayout(row)
            self._anat_vol_lbls[key] = (v, pct_key)
        p0l.addStretch()
        self._vol_stack.addWidget(p0)

        # ── Page 1: CC part volumes ───────────────────────────────────────────
        p1 = QtWidgets.QWidget()
        p1l = QtWidgets.QVBoxLayout(p1); p1l.setContentsMargins(0,0,0,0); p1l.setSpacing(2)
        hdr1 = QtWidgets.QLabel("Bile duct CC partition:")
        hdr1.setStyleSheet("color: #555555; font-size: 11px; font-weight: bold;")
        p1l.addWidget(hdr1)
        for pid, name, color in CC_PARTS:
            # header row with colour dot
            hrow = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel(); dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{color}; border-radius:2px;")
            hrow.addWidget(dot)
            hrow.addWidget(QtWidgets.QLabel(name)); hrow.addStretch()
            p1l.addLayout(hrow)
            # bile sub-row
            for sub_label, sub_key in [("  Bile duct", "bile"), ("  Liver", "liver")]:
                srow = QtWidgets.QHBoxLayout()
                ln = QtWidgets.QLabel(sub_label + ":"); ln.setFixedWidth(72)
                ln.setStyleSheet("color: #777777; font-size: 10px;")
                lv = QtWidgets.QLabel("—")
                lv.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                lv.setStyleSheet(
                    "color: #1A3A6A; font-size: 10px; font-family: Consolas, monospace;")
                srow.addWidget(ln); srow.addWidget(lv, stretch=1)
                p1l.addLayout(srow)
                self._cc_vol_lbls.setdefault(pid, {})[sub_key] = lv
        p1l.addStretch()
        self._vol_stack.addWidget(p1)

        # ── Page 2: Couinaud segment volumes ──────────────────────────────────
        p2 = QtWidgets.QWidget()
        p2l = QtWidgets.QVBoxLayout(p2); p2l.setContentsMargins(0,0,0,0); p2l.setSpacing(2)
        hdr2 = QtWidgets.QLabel("Bile duct by Couinaud segment:")
        hdr2.setStyleSheet("color: #555555; font-size: 11px; font-weight: bold;")
        p2l.addWidget(hdr2)
        for pid, name, color in COUINAUD_SEGS:
            hrow = QtWidgets.QHBoxLayout()
            dot = QtWidgets.QLabel(); dot.setFixedSize(10, 10)
            dot.setStyleSheet(f"background:{color}; border-radius:2px;")
            hrow.addWidget(dot)
            hrow.addWidget(QtWidgets.QLabel(name)); hrow.addStretch()
            p2l.addLayout(hrow)
            for sub_label, sub_key in [("  Bile duct", "bile"), ("  Liver", "liver")]:
                srow = QtWidgets.QHBoxLayout()
                ln = QtWidgets.QLabel(sub_label + ":"); ln.setFixedWidth(72)
                ln.setStyleSheet("color: #777777; font-size: 10px;")
                lv = QtWidgets.QLabel("—")
                lv.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                lv.setStyleSheet(
                    "color: #1A3A6A; font-size: 10px; font-family: Consolas, monospace;")
                srow.addWidget(ln); srow.addWidget(lv, stretch=1)
                p2l.addLayout(srow)
                self._coui_vol_lbls.setdefault(pid, {})[sub_key] = lv
        p2l.addStretch()
        self._vol_stack.addWidget(p2)

        lay.addWidget(self._vol_stack)
        hint = QtWidgets.QLabel("Partition volumes shown in CC / Couinaud views only")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #AAAAAA; font-size: 10px;")
        lay.addWidget(hint)
        return grp

    def _struct_row(self, key, label, hex_color, opacity):
        w = QtWidgets.QWidget()
        vl = QtWidgets.QVBoxLayout(w); vl.setContentsMargins(0,0,0,0); vl.setSpacing(3)
        hdr = QtWidgets.QHBoxLayout(); hdr.setContentsMargins(0,0,0,0); hdr.setSpacing(5)
        swatch = QtWidgets.QLabel(); swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(
            f"background: {hex_color}; border-radius: 2px; border: 1px solid #999;")
        hdr.addWidget(swatch)
        cb = QtWidgets.QCheckBox(label); cb.setChecked(self._visible.get(key, True))
        cb.setStyleSheet("font-size: 11px;")
        cb.toggled.connect(lambda v, k=key: self._on_visibility(k, v))
        hdr.addWidget(cb, stretch=1)
        pct = QtWidgets.QLabel(f"{int(opacity * 100)}%"); pct.setFixedWidth(34)
        pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct.setStyleSheet("color: #888888; font-size: 11px;")
        hdr.addWidget(pct)
        vl.addLayout(hdr)
        sl = QtWidgets.QSlider(Qt.Horizontal); sl.setRange(0, 100)
        sl.setValue(int(opacity * 100)); sl.setFixedHeight(16)
        sl.valueChanged.connect(lambda v, k=key, p=pct: self._on_opacity(k, v, p))
        vl.addWidget(sl)
        self._sw[key] = {"cb": cb, "slider": sl, "pct": pct}
        return w

    def _hline(self):
        f = QtWidgets.QFrame(); f.setFrameShape(QtWidgets.QFrame.HLine)
        f.setStyleSheet("color: #DDDDDD;"); return f

    # ── case loading ──────────────────────────────────────────────────────────

    def _load_case(self, case_id: int):
        self._case_id = case_id
        self._set_status(f"Loading case {case_id}...")

        # Remove all existing actors
        for key in list(self._actors): self.plotter.remove_actor(key, render=False)
        for aname in [f"part_{m}_{p}" for (m, p) in self._part_acts]:
            self.plotter.remove_actor(aname, render=False)
        self._actors.clear(); self._part_acts.clear()

        # Load anatomy structures
        loaded = 0
        for key, label, color, def_op, dir_key, lr in STRUCTURES:
            self._set_status(f"Loading {label}...")
            cfg  = _MESH_CFG.get(key, dict(smooth=20, decimate=0.80))
            mesh = _get_mesh(case_id, key, dir_key, lr, **cfg)
            if mesh:
                op    = self._opacity.get(key, def_op)
                actor = self.plotter.add_mesh(
                    mesh, name=key, color=color, opacity=op,
                    smooth_shading=True, specular=0.1, specular_power=10.0,
                    ambient=0.20, diffuse=0.85, show_scalar_bar=False,
                    reset_camera=False, render=False)
                actor.SetVisibility(int(self._visible.get(key, True)))
                self._actors[key] = actor; loaded += 1
                self._sw[key]["cb"].setEnabled(True)
                self._sw[key]["cb"].setToolTip("")
            else:
                self._actors[key] = None
                self._sw[key]["cb"].setEnabled(False)
                self._sw[key]["cb"].setToolTip("Not available for this case")

        # Reload partition actors if a partition mode is active
        if self._view_mode in ("cc", "couinaud"):
            self._load_partition_actors(case_id, self._view_mode)
            self._apply_mode_visibility(self._view_mode)

        self.plotter.reset_camera(); self.plotter.render()
        self._set_status(f"Case {case_id} — {loaded}/{len(STRUCTURES)} structures")
        self._update_vol_display()

    def _load_partition_actors(self, case_id: int, mode: str):
        """Build meshes for the given partition mode and add them as actors."""
        self._set_status(f"Loading {mode} partition meshes...")

        # Bile / Couinaud parts
        for pid, mesh, color, name in _get_partition_meshes(case_id, mode):
            aname = f"part_{mode}_{pid}"
            actor = self.plotter.add_mesh(
                mesh, name=aname, color=color, opacity=1.0,
                smooth_shading=True, specular=0.1, specular_power=10.0,
                ambient=0.20, diffuse=0.85, show_scalar_bar=False,
                reset_camera=False, render=False)
            self._part_acts[(mode, pid)] = actor

        # For CC: also add coloured liver parts (semi-transparent)
        if mode == "cc":
            for pid, mesh, color in _get_liver_part_meshes(case_id):
                aname = f"part_cc_liver_{pid}"
                actor = self.plotter.add_mesh(
                    mesh, name=aname, color=color, opacity=0.25,
                    smooth_shading=True, specular=0.05, specular_power=5.0,
                    ambient=0.20, diffuse=0.85, show_scalar_bar=False,
                    reset_camera=False, render=False)
                self._part_acts[("cc_liver", pid)] = actor

        # For Couinaud: also add Couinaud-coloured liver (semi-transparent)
        if mode == "couinaud":
            for pid, mesh, color, name in _get_liver_couinaud_meshes(case_id):
                aname = f"part_couinaud_liver_{pid}"
                actor = self.plotter.add_mesh(
                    mesh, name=aname, color=color, opacity=0.25,
                    smooth_shading=True, specular=0.05, specular_power=5.0,
                    ambient=0.20, diffuse=0.85, show_scalar_bar=False,
                    reset_camera=False, render=False)
                self._part_acts[("couinaud_liver", pid)] = actor

    def _apply_mode_visibility(self, mode: str):
        """Show/hide anatomy and partition actors for the given mode."""
        # In CC and Couinaud modes: hide plain anatomy (liver shown via coloured parts)
        for key, actor in self._actors.items():
            if not actor: continue
            actor.SetVisibility(0) if mode in ("cc", "couinaud") else None

        # Partition actors: show only for the active mode
        for (m, pid), actor in self._part_acts.items():
            show = (m == mode) or \
                   (mode == "cc" and m == "cc_liver") or \
                   (mode == "couinaud" and m == "couinaud_liver")
            actor.SetVisibility(int(show))

    # ── view switching ────────────────────────────────────────────────────────

    def _activate_mode(self, mode: str):
        """Switch to a partition view (CC or Couinaud)."""
        # Check data availability before switching
        kind = "bile_parts" if mode == "cc" else "couinaud"
        if _find_part_file(self._case_id, kind) is None:
            if mode == "couinaud":
                QtWidgets.QMessageBox.information(
                    self, "Couinaud — data not found",
                    "No Couinaud partition files found for this case.\n\n"
                    "To generate them:\n"
                    "  1. Run couinaud_analysis.py with your Couinaud atlas masks.\n"
                    "  2. Set _DIRS['couinaud'] in viewer_3d.py to the output folder.")
            else:
                QtWidgets.QMessageBox.information(
                    self, "CC Partition — data not found",
                    "No CC partition files found for this case.\n\n"
                    "Run cc_partition.py first to generate them.")
            return

        self._view_mode = mode

        if not any(k[0] == mode for k in self._part_acts):
            self._load_partition_actors(self._case_id, mode)

        self._apply_mode_visibility(mode)
        self._vol_stack.setCurrentIndex(1 if mode == "cc" else 2)
        self.plotter.render()
        self._update_vol_display()

    def _activate_anatomy(self, *only_keys):
        """Switch back to anatomy view, optionally showing only specified structures."""
        self._view_mode = "anatomy"

        # Hide all partition actors
        for actor in self._part_acts.values():
            actor.SetVisibility(0)

        # Restore liver opacity (may have been dimmed in Couinaud mode)
        liver_actor = self._actors.get("liver")
        if liver_actor:
            liver_actor.GetProperty().SetOpacity(self._opacity.get("liver", 0.30))

        # Show anatomy structures
        for key, actor in self._actors.items():
            if not actor: continue
            vis = (key in only_keys) if only_keys else self._visible.get(key, True)
            actor.SetVisibility(int(vis))
            if only_keys:
                self._visible[key] = vis
                if key in self._sw: self._sw[key]["cb"].setChecked(vis)

        self._vol_stack.setCurrentIndex(0)
        self.plotter.render()
        self._update_vol_display()

    def _show_all(self):
        self._activate_anatomy()
        for key, w in self._sw.items():
            if w["cb"].isEnabled(): w["cb"].setChecked(True)

    def _force_reload(self):
        for key, *_ in STRUCTURES: _CACHE.pop((self._case_id, key), None)
        for mode in ("cc", "cc_liver", "couinaud"):
            _CACHE.pop((self._case_id, mode), None)
        self._load_case(self._case_id)

    # ── volume display ────────────────────────────────────────────────────────

    def _update_vol_display(self):
        if self._view_mode == "anatomy":
            try:
                r = _compute_anatomy_vols(self._case_id)
            except Exception as e:
                print(f"[WARN] anatomy vol: {e}"); return
            for key, (lbl, pct_key) in self._anat_vol_lbls.items():
                val = r.get(key)
                if val is None:
                    lbl.setText("—")
                else:
                    pct = r.get(pct_key) if pct_key else None
                    lbl.setText(f"{val:.1f} mL" + (f"  ({pct:.1f}%)" if pct else ""))
        else:
            vols = _compute_partition_vols(self._case_id, self._view_mode)
            lbls = (self._cc_vol_lbls if self._view_mode == "cc"
                    else self._coui_vol_lbls)
            for pid, sub in lbls.items():
                v = vols.get(pid, {})
                # bile
                bile = v.get('bile_mL', 0.0)
                sub['bile'].setText(f"{bile:.2f} mL" if bile else "—")
                # liver + pct
                liver_mL  = v.get('liver_mL')
                liver_pct = v.get('liver_pct')
                if liver_mL is not None:
                    txt = f"{liver_mL:.1f} mL"
                    if liver_pct is not None:
                        txt += f"  ({liver_pct:.1f}%)"
                    sub['liver'].setText(txt)
                else:
                    sub['liver'].setText("—")

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_case_changed(self, _):
        self._load_case(self._case_combo.currentData())

    def _on_visibility(self, key: str, visible: bool):
        self._visible[key] = visible
        actor = self._actors.get(key)
        if actor: actor.SetVisibility(int(visible)); self.plotter.render()

    def _on_opacity(self, key: str, value: int, pct_lbl):
        pct_lbl.setText(f"{value}%")
        self._opacity[key] = value / 100.0
        actor = self._actors.get(key)
        if actor: actor.GetProperty().SetOpacity(value / 100.0); self.plotter.render()

    def _set_status(self, msg: str):
        self._status_lbl.setText(msg); QtWidgets.QApplication.processEvents()

    # ── camera ────────────────────────────────────────────────────────────────

    def _reset_camera(self): self.plotter.reset_camera(); self.plotter.render()

    def _orient(self, pos_dir, up):
        cam = self.plotter.camera; fp = cam.focal_point; d = cam.distance
        cam.position = (fp[0]+pos_dir[0]*d, fp[1]+pos_dir[1]*d, fp[2]+pos_dir[2]*d)
        cam.up = up; self.plotter.renderer.ResetCameraClippingRange(); self.plotter.render()

    def _view_front(self): self._orient((0, -1, 0), (0, 0, 1))
    def _view_top(self):   self._orient((0,  0, 1), (0, -1, 0))
    def _view_right(self): self._orient((1,  0, 0), (0, 0, 1))

    def _toggle_rotation(self, on: bool):
        if on: self._rot_btn.setText("Stop Rotation"); self._rot_timer.start()
        else:  self._rot_btn.setText("Auto Rotate");   self._rot_timer.stop()

    def _rotate_step(self):
        self.plotter.camera.Azimuth(self._spd_sl.value() * 0.3)
        self.plotter.renderer.ResetCameraClippingRange(); self.plotter.render()

    def closeEvent(self, event):
        self._rot_timer.stop(); self.plotter.close(); super().closeEvent(event)


# ==============================================================================
# Entry point
# ==============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    pal = QtGui.QPalette()
    for role, rgb in [
        (QtGui.QPalette.Window,          (245,245,245)),
        (QtGui.QPalette.WindowText,      (44, 44, 44 )),
        (QtGui.QPalette.Base,            (255,255,255)),
        (QtGui.QPalette.AlternateBase,   (240,240,240)),
        (QtGui.QPalette.Text,            (44, 44, 44 )),
        (QtGui.QPalette.Button,          (232,232,232)),
        (QtGui.QPalette.ButtonText,      (44, 44, 44 )),
        (QtGui.QPalette.Highlight,       (72, 120,192)),
        (QtGui.QPalette.HighlightedText, (255,255,255)),
        (QtGui.QPalette.ToolTipBase,     (255,255,220)),
        (QtGui.QPalette.ToolTipText,     (44, 44, 44 )),
    ]:
        pal.setColor(role, QtGui.QColor(*rgb))
    app.setPalette(pal)
    win = MainWindow(); win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
