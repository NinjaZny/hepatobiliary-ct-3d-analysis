# Results and their sources

Every number on the main page comes from one of the sources below. "Archived per-case file" means the value was recomputed from per-case CSVs kept on the lab server. "As reported" means the value is taken from the internship report and its per-case files are not part of this showcase.

## Bile duct segmentation (Dataset337: 10 clinical + 5 public CTs)

Setup: nnU-Net v2 3d_fullres, 300 epochs, 5-fold split of the 15 manually annotated volumes (12 train / 3 validation per fold). Folds 0, 1 and 2 were trained for the two configurations below. Source: archived per-case file `D337_3fold_top2_all_percase.csv` (n = 9).

| Configuration | Dice | HD95 (mm) | clDice | LCC ratio | CC count | Split | Merge |
|---|---|---|---|---|---|---|---|
| nnU-Net baseline (DiceCE) | 0.656 ± 0.165 | 8.45 ± 6.81 | 0.707 ± 0.160 | 0.706 ± 0.175 | 29.2 ± 19.5 | 11.9 ± 11.5 | 6.0 ± 8.2 |
| DiceCE + 0.35 clDice + 0.50 InvDT | 0.657 ± 0.160 | 8.61 ± 7.05 | 0.718 ± 0.153 | 0.719 ± 0.169 | 23.2 ± 13.0 | 8.0 ± 5.8 | 7.2 ± 10.5 |

Values are mean ± sample standard deviation over 9 validation cases.

Other variants (clDice 1.0, InvDT 1.0, consistency-regularised semi-supervised training, SVLS) were run on **fold 0 only (3 validation cases)**. The internship report's Table 6.4 corresponds to that single fold. Those numbers are too small a sample to rank methods, so they are not shown on the main page.

## Vessel benchmark (Dataset331: MSD Task08, VSNet re-annotation, thin-slice subset)

48 training / 12 held-out test CTs, labels hepatic vein = 1, portal vein = 2, union metrics. Source: archived per-case file `D331_all_best_topology_eval_per_case.csv` (n = 12 per configuration). These values match the internship report's Table 6.3.

| Configuration | Dice | HD95 (mm) | clDice | Split | LCC ratio |
|---|---|---|---|---|---|
| nnU-Net baseline | 0.922 | 2.83 | 0.921 | 1.42 | 0.954 |
| + clDice 0.35 | 0.924 | 2.44 | 0.940 | 1.58 | 0.981 |
| + clDice 0.50 | 0.921 | 2.70 | 0.937 | 1.08 | 0.982 |
| + clDice 0.35 + DT 1.0 | 0.925 | 2.49 | 0.939 | 1.58 | 0.981 |
| + clDice 0.35 + InvDT 0.5 | 0.925 | 2.31 | 0.941 | 0.92 | 0.982 |
| + clDice 0.35 + InvDT 1.0 | 0.925 | 2.14 | 0.944 | 1.58 | 0.998 |
| + clDice 1.0 + DT 1.0 | 0.922 | 2.16 | 0.949 | 0.92 | 0.998 |
| + clDice 1.0 + InvDT 1.0 | 0.922 | 2.23 | 0.948 | 0.67 | 0.997 |

No merge errors were observed for any configuration.

## Vascular landmarks (3D-IRCADb-01, 20 cases), as reported

A case counts as correct when the selected structure matches the expected anatomical landmark on the reference mask (qualitative check).

| Landmark | Correct |
|---|---|
| Inferior vena cava | 18 / 20 |
| Left hepatic trunk | 17 / 20 |
| Middle hepatic trunk | 20 / 20 |
| Right hepatic trunk | 20 / 20 |

## Couinaud segmentation with vascular priors, as reported

The data split and annotations come from the public resource of Zhang et al., and the same split is used for every input configuration. The scores are Dice, accuracy inside the liver, HD95 and ASSD (see main page).

## Liver segmentation, as reported

External test set: SLIVER07 (30 cases). The model trained on LiTS + CRLM (nnU-Net, 600 epochs) produced the liver masks used downstream.

## Notes on protocol

- The base nnU-Net trainer on the server was set to 300 epochs instead of the default 1000. Trainers without their own setting inherit 300.
- The teacher–student pseudo-labelling variant (Dataset338) used a teacher ensemble that had seen the student's validation cases. It is therefore excluded from the reported results.
