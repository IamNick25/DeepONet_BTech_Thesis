# Experiment index

One row per major experiment, with the chain from research question to figure. Use this to find which
notebook produced which result.

## E1 — 1D operator feasibility

| | |
|---|---|
| **Question** | Can a branch–trunk network represent `Q(x) ↦ T(x)` for steady 1D conduction? |
| **Setup** | `x ∈ [0,1]`, `T = 0` at both ends, sources from an RBF Gaussian process |
| **Data generation** | 500 source–temperature pairs (code not in archive) |
| **Model** | Branch and trunk each 2 × 128, ReLU; inner product + bias |
| **Training** | Adam 1e-3, batch 512, 200 epochs, MSE |
| **Result** | Converged loss, accurate held-out profiles across input scales spanning four orders of magnitude |
| **Figures** | `1D_pointwise_deeponet/figures/` — all three reproduced from the BTP report |
| **Code** | **not present in the archive** |

---

## E2 — 2D pointwise temperature surrogate

| | |
|---|---|
| **Question** | Can the operator idea carry to transient 2D melting with configurable boundaries? |
| **Setup** | 50 mm × 50 mm PCM–foam, 40 × 40, 9 snapshots, 200 GP-source cases |
| **Data generation** | `2D/01_pointwise_deeponet/code/data_generation/pointwise_dataset_generation_2d.ipynb` → `deeponet_temp_dataset.npz` |
| **Model** | 4-network DeepONet, `D = 256`, additive + multiplicative fusion |
| **Training** | `code/training/pointwise_deeponet_temperature_2d.ipynb`; Adam 1e-3, 60 epochs, 16,384 points/batch |
| **Checkpoint** | `deeponet_temp_model_4net.pt` (not included) |
| **Result** | Final RMSE 3–4 K; melt-front location and evolution reproduced; slow inference, smoothed interfaces |
| **Figures** | `2D/01_pointwise_deeponet/figures/training_{mse,rmse}_vs_epoch_pointwise_temperature.png`, `temperature_prediction_vs_reference_sample01–06.png` |

## E3 — 2D pointwise liquid-fraction surrogate

Same dataset and protocol as E2 with `TARGET = "f"`.
Code: `code/training/pointwise_deeponet_liquid_fraction_2d.ipynb`; checkpoint `deeponet_frac_model_5net.pt`.
Figures: `liquid_fraction_prediction_vs_reference_sample01–04.png`.
Result: the interface smoothing that motivated the U-DeepONet redesign is most visible here.

---

## E4 — 2D U-DeepONet temperature surrogate

| | |
|---|---|
| **Question** | Does grid-to-grid operator learning fix the cost and the front-smearing of E2? |
| **Setup** | 120 × 120, 24 snapshots, 500 cases, GP / battery / hybrid sources, 13 input channels |
| **Data generation** | `2D/02_u_deeponet/code/data_generation/udeeponet_dataset_generation_2d.ipynb` → `udeeponet_dataset.npz` (≈ 230 MB) |
| **Dataset metadata** | `2D/02_u_deeponet/data/dataset_meta.json`, `dataset_splits.json`, `case_statistics.csv` |
| **Model** | U-Net branch (`f = 64`, 3 blocks) + 10-layer sinusoidal time trunk |
| **Training** | `code/training/udeeponet_temperature_2d.ipynb`; Adam 1e-4, wd 1e-4, batch 16, 60 epochs |
| **Log** | `results/training_log_temperature_500cases.txt` |
| **Checkpoint** | `checkpoints_udeeponet_T/best.pt` (not included) |
| **Result** | Temperature MAE 1.7–8.7 K across six heating configurations; inference 0.069–0.300 s |
| **Figures** | `figures/predictions/temperature_prediction_error_*.png` |

## E5 — 2D U-DeepONet liquid-fraction surrogate

Same as E4 with `TARGET = "f"`.
Code: `code/training/udeeponet_liquid_fraction_2d.ipynb`; log `results/training_log_liquid_fraction_500cases.txt`;
checkpoint `checkpoints_udeeponet_f/best.pt`.
Result: liquid-fraction MAE 0.025–0.044; maximum error concentrated on the melt front.
Figures: `figures/predictions/liquid_fraction_prediction_error_*.png`.

## E6 — Six prediction case studies

| | |
|---|---|
| **Question** | How does accuracy vary with heat-source complexity? |
| **Setup** | A dedicated case-study dataset, cases of increasing complexity |
| **Code** | `2D/02_u_deeponet/code/evaluation/prediction_case_studies_2d.ipynb`, plus the inference cells of the two training notebooks |
| **Metrics** | `2D/02_u_deeponet/results/inference_metrics_prediction_cases.txt` |
| **Result** | Case 1 GP 1.73 K · Case 2 GP+BC 3.10 K · Case 3 single prismatic 8.75 K · Case 4 multi prismatic 3.32 K · Case 5 generalised 2.75 K · Case 6 generalised+BC 3.35 K |
| **Figures** | `figures/predictions/` |

---

## E7 — Data-sensitivity sweep

| | |
|---|---|
| **Question** | How many simulations does the U-DeepONet need? |
| **Setup** | Nine training-set sizes (80 → 500), identical hyper-parameters, liquid-fraction target |
| **Code** | The E5 notebook, re-run per size (no separate notebook survives) |
| **Results** | `2D/03_data_sensitivity/results/udeeponet_data_sensitivity_summary.csv` and `liquid_fraction_epochwise_rmse_all_training_set_sizes.csv` |
| **Result** | Best validation RMSE 0.2550 → 0.1500; three regimes; saturation beyond ~400 cases |
| **Figures** | `2D/03_data_sensitivity/figures/` (5 plots) |

---

## E8 — Heat-source intensity study (reference solver)

`q_scale` ∈ {5e4, 8e4, 1e5, 3e5} W/m³ at `T_bound = 330` K, all walls constant.
Figures: `2D/04_boundary_and_physics_studies/figures/heat_source_intensity_study_Q_*.png`.
Result: establishes 1e5 W/m³ as the dataset scale and shows how strongly melt topology depends on it.

## E9 — Boundary-only melting study (reference solver)

No volumetric source, walls at 350 K, constant and Gaussian variants.
Figures: `boundary_heating_only_*.png`, `boundary_temperature_profiles_gaussian_case.png`.
Result: the opposite melt topology to E8 — inward-collapsing solid core.

## E10 — U-DeepONet with prescribed boundary profiles

Code: `2D/04_boundary_and_physics_studies/code/udeeponet_liquid_fraction_2d_with_boundary_heating.ipynb`.
Figures: `liquid_fraction_prediction_error_with_boundary_heating_01–06.png`.
Result: superimposed boundary heating is handled without a separate model (Cases 2 and 6 of E6).

---

## E11 — Cross-geometry generalisation, circular sources

| | |
|---|---|
| **Question** | Does a prismatic-trained operator transfer to unseen circular sources, and does a generalised training set help? |
| **Code** | `2D/05_cross_geometry_generalization/code/cross_geometry_circular_temperature.ipynb` (one cell per model) |
| **Metrics** | `results/cross_geometry_inference_metrics.txt` |
| **Result** | Prismatic-only: T MAE 3.95 K, R² 0.9918. Generalised: T MAE 2.96 K, R² 0.9952. Liquid fraction slightly favours the prismatic-only model |
| **Figures** | `figures/circular_sources_{generalized_dataset,prismatic_only}_model_01–06.png`, matched case for case |

## E12 — Cross-geometry generalisation, elliptical sources

Same design with elliptical footprints of differing aspect ratio and orientation.
Code: `cross_geometry_elliptic_temperature.ipynb`.
Result: prismatic-only T MAE 3.48 K / R² 0.9927; generalised 2.82 K / R² 0.9946.
Figures: `elliptical_sources_*_01–04.png`.

---

## E13 — Slice-based 3D liquid-fraction surrogate

| | |
|---|---|
| **Question** | Can 3D transient melting be learned without Conv3D? |
| **Setup** | 24³ grid, 50 mm cube, one prismatic battery (lumped mode), 500 cases, 15 slices per axis, 18 channels per slice |
| **Data generation** | `3D/code/data_generation/slice_dataset_generation_3d.ipynb` → `slice_dataset_3d.npz` (0.3–1.7 GB) |
| **Splits** | `3D/data/dataset_splits.json` |
| **Model** | Unified slice U-Net, 13-slice window, Fourier time features, interface-weighted loss |
| **Training** | `3D/code/training/slice_udeeponet_liquid_fraction_3d.ipynb`; Adam 1e-3, batch 128, 60 epochs configured |
| **Log** | `3D/results/training_log_liquid_fraction_500cases_window13.txt` (reaches epoch 34; best at epoch 28) |
| **Checkpoint** | `checkpoints_unified_slice_unet_f/best.pt` (not included) |
| **Result** | Validation RMSE 0.370 → 0.128 normalised (0.155 → 0.054 physical) |
| **Figures** | `3D/figures/liquid_fraction_prediction_error_{x,y,z}_slice_*.png` |

## E14 — Slice-based 3D temperature surrogate

Same framework with `TARGET = "T"`.
Code: `3D/code/training/slice_udeeponet_temperature_3d.ipynb`.
Figures: `3D/figures/temperature_prediction_error_slice_01–06.png`, plus the volumetric animations
`animation_3d_*.gif`.

---

## Chain summary

```
research question
   └─ physical setup            docs/methodology/physical_model_and_solver.md
        └─ data generation      <stage>/code/data_generation/
             └─ dataset          dataset_run_*/  (regenerate; metadata in <stage>/data/)
                  └─ DeepONet    docs/methodology/deeponet_architectures.md
                       └─ training    <stage>/code/training/   →  <stage>/results/*.txt
                            └─ checkpoint  checkpoints_*/best.pt  (regenerate)
                                 └─ inference   evaluation notebook / inference cells
                                      └─ metrics    <stage>/results/*.txt, *.csv
                                           └─ figures   <stage>/figures/
```

Every figure in the repository is listed with its description and its origin in
[`../figure_index.csv`](../figure_index.csv).
