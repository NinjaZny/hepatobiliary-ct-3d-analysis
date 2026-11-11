# Source code (curated subset)

This folder holds the core files of the internship code, chosen to show the methods. It is not a complete, runnable reproduction: patient data, trained weights and the training environment are not included.

| File | What it does |
|---|---|
| `preprocessing/liver_postprocess_and_roi.py` | Cleans nnU-Net liver masks (largest component + 3D hole filling), then crops the CT to the liver bounding box with a mm margin and applies HU windowing |
| `preprocessing/bile_duct_preprocess.py` | Liver-ROI crop (5-voxel margin) and HU windowing to [−160, 240] → [0, 1] for bile duct inference |
| `segmentation/trainer_bile_duct.py` | Custom nnU-Net v2 trainers: soft-clDice loss, and an auxiliary DT / inverted-DT regression head with an inference wrapper that removes the extra channel |
| `evaluation/topology_metrics.py` | Per-case Dice, HD95, clDice, connected components, largest-component ratio, split / merge errors; mean ± std summary per model |
| `vascular/01_preprocess_skeleton.py` | Component filtering, closing, 3D Lee thinning, EDT radius map, iterative leaf pruning, branch labelling |
| `vascular/02_build_graph.py` | Voxel skeleton → branch graph (bifurcation clustering, chain contraction, loop removal, pruning, smoothing), Slicer exports |
| `vascular/03_extract_backbone.py` | Maximum spanning tree, root selection, greedy main-trunk selection, hierarchical levels propagated back to the mask |
| `partitioning/bile_duct_cc_partition.py` | Bile-duct-component seeds → liver territories (nearest-seed assignment) with per-territory volumes |
| `partitioning/bile_duct_by_couinaud.py` | Labels bile duct voxels by Couinaud segment and computes per-segment volumes |
| `visualization/viewer_3d.py` | PyQt5 + PyVista multi-structure viewer: opacity, visibility, partition views, camera presets, volume panel |

## Changes made for publication

The algorithms are unchanged. Only the following was edited:

- Hard-coded server and workstation paths were replaced by relative defaults (`data/...`) or the `VIEWER_DATA` environment variable.
- Comments were corrected where they overstated results ("best" trainer) or described the inverted-DT target inaccurately.
- A note was added to `bile_duct_cc_partition.py` stating that distances are computed in voxel index space.

## Requirements

See `../requirements.txt`. The trainer file must be placed inside an nnU-Net v2 installation (`nnunetv2/training/nnUNetTrainer/variants/loss/`). It was developed with nnU-Net v2.6.4 and PyTorch 2.7.
