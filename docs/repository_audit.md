# Repository audit

How this repository was reconstructed from the working archive, what was kept, what was left out, and
what you should verify yourself.

---

## 1. Research overview — what was reconstructed

The source material was a **working research archive of roughly 1,900 PNG figures, ~35 Jupyter
notebooks (many of them near-duplicates), several multi-hundred-megabyte `.npz` datasets and `.pt`
checkpoints, the BTech thesis in several revisions, three presentation decks, and one unpublished
journal manuscript**, spread across six sibling folders (`1/`, `2/`, `3/`, `4/`, `5/`, `journal/`) whose
names carried no information about research status.

Reconstruction was done by reading notebook **source cells** (not filenames), correlating them with the
final hard-bound BTech thesis report, and cross-checking against the surviving training logs, inference
metric files and figures. The result is the 1D → 2D → 3D progression documented in the root
[`README.md`](../README.md), which matches the thesis narrative and, independently, the code timeline.

### What each source folder turned out to be

| Source folder | Character | Latest modification | Fate |
|---|---|---|---|
| `1/BTP_DeepONet/` | 3D-focused snapshot: 3D dataset generation and liquid-fraction notebooks, two 1.1 GB slice datasets, the four volumetric animation GIFs, thesis `.docx` drafts, midsem deck | May 2026 | 3D temperature notebook and the four GIFs taken; rest superseded |
| `2/BTP work/` | Thin snapshot: one 3D liquid-fraction notebook copy and the endsem deck | — | Superseded by `4/` |
| `3/BTP_DEEPONET/` | The most complete **earlier** snapshot: the original pointwise DeepONet pipeline (Nov 2025), first U-Net prototype, early prismatic dataset generator, the final hard-bound thesis PDF, and the data-sensitivity plot archive | Aug 2026 | Source of all stage-2a code, the sensitivity plots, and the thesis cross-reference |
| `4/BTP work/` | The **most recent and most complete** snapshot (Sept 2026): final 2D U-DeepONet pipeline at 500 cases, the six prediction case studies, the with-boundary-conditions variant, both cross-geometry studies, the per-training-size sweep folders, and the final 3D slice pipeline | 10 Sept 2026 | Primary source for stages 2b–2e and 3 |
| `5/` | Curated *results*: physics parameter studies (heat-source intensity sweep, boundary-only melting) and 20 finished U-DeepONet prediction figures with error maps | Apr 2026 | Source of stage 2d figures and 20 prediction figures |
| `journal/` | One PDF: the unpublished manuscript | Sept 2026 | **Reference only — never copied** (§8) |

The decisive finding was that `4/BTP work/` supersedes `3/BTP_DEEPONET/` for everything the final
results depend on, and that `3/` is nevertheless indispensable because it is the **only** place the
pointwise-DeepONet stage survives.

---

## 2. The 1D → 2D → 3D progression as reconstructed

```
1D                 2D pointwise           2D U-DeepONet            2D studies              3D slice-based
feasibility   →    transient melting  →   grid-to-grid        →    sensitivity,       →    24³ + prismatic
Q(x)→T(x)          40×40, 200 cases       120×120, 500 cases       boundary, cross-         battery, 15 slices
2×128 MLP          4-net, D=256           U-Net f=64 + trunk       geometry                 per axis, window 13
(code missing)     Nov 2025               Jan–Sep 2026             Apr–Sep 2026             Mar–Sep 2026
```

Two things are worth stating explicitly, because the folder structure actively obscures them:

1. **The dimensional ladder is not the only axis of progress.** The single most consequential change in
   the project happened *within* 2D — replacing pointwise sampling with grid-to-grid learning. Without
   it, neither the data-sensitivity sweep nor the 3D extension would have been affordable. The
   repository therefore splits `2D/` into five sub-stages rather than treating 2D as one block.
2. **The 3D stage is 2D machinery applied cleverly**, not a new architecture. It reuses the same U-Net
   block, adds a 13-slice window and Fourier time features, and never performs a 3D convolution.

### Evidence used

| Claim | Evidence |
|---|---|
| 1D stage existed | Thesis §3.3.1, §3.3.2, §4.3, Ch. 5; Figs. 2(a), 20, 25(a) |
| Pointwise stage preceded U-DeepONet | `MetalFoamPCM_vscode_Temp_field_Arch.ipynb` (Nov 2025) implements `DeepONet4`; thesis §3.4 explains the motivation to abandon it |
| U-DeepONet is the final 2D model | `MetalFoamPCM_prismatic_batt_UNet_Arch_*_testing.ipynb` in `4/BTP work/Sim_500/` (Sept 2026), matching `INFERENCE TEXT METRICS.txt` |
| Data-sensitivity sweep | `udeeponet_sensitivity_plots_updated.zip` + `liquid_fraction_epochwise_rmse_all_8_datasets.csv` + the nine `Sim_*` folders |
| Cross-geometry study | `cross_circular/` and `cross_elliptic/` notebooks + `CROSS GEOMETRY INFERENCE.txt` |
| 3D is slice-based | `MetalFoamPCM_3D_UNet_Arch_Liqfrac-3*.ipynb` defines `Unified3DSliceDataset` and `UnifiedSliceUNet`; thesis §3.5.3 and Table 2 |

---

## 3. Major experiments

Fourteen experiments are catalogued in [`experiments/experiment_index.md`](experiments/experiment_index.md),
each with its question, setup, data-generation notebook, model, training notebook, checkpoint name,
metrics file and figures. Summary:

| ID | Experiment | Stage |
|---|---|---|
| E1 | 1D operator feasibility | 1D |
| E2, E3 | 2D pointwise temperature / liquid fraction | 2a |
| E4, E5 | 2D U-DeepONet temperature / liquid fraction | 2b |
| E6 | Six prediction case studies | 2b |
| E7 | Data-sensitivity sweep (80 → 500 cases) | 2c |
| E8, E9 | Heat-source intensity / boundary-only reference studies | 2d |
| E10 | U-DeepONet with prescribed boundary profiles | 2d |
| E11, E12 | Cross-geometry generalisation, circular / elliptical | 2e |
| E13, E14 | Slice-based 3D liquid fraction / temperature | 3 |

---

## 4. Experiment → code → data → result → figure mapping

| Experiment | Code in repository | Dataset (regenerate) | Results | Figures |
|---|---|---|---|---|
| E2/E3 | `2D/01_pointwise_deeponet/code/` | `deeponet_{temp,frac}_dataset.npz` | — | `2D/01_pointwise_deeponet/figures/` |
| E4/E5/E6 | `2D/02_u_deeponet/code/` | `udeeponet_dataset.npz` (≈ 230 MB) | `2D/02_u_deeponet/results/` | `2D/02_u_deeponet/figures/` |
| E7 | E5 notebook, re-run per size | same | `2D/03_data_sensitivity/results/` | `2D/03_data_sensitivity/figures/` |
| E8/E9/E10 | `2D/04_boundary_and_physics_studies/code/` | regenerate with modified `q_scale` / `T_bound` | — | `2D/04_boundary_and_physics_studies/figures/` |
| E11/E12 | `2D/05_cross_geometry_generalization/code/` | three datasets: prismatic-only, generalised, circular/elliptical test | `.../results/cross_geometry_inference_metrics.txt` | `.../figures/` |
| E13/E14 | `3D/code/` | `slice_dataset_3d.npz` (0.3–1.7 GB) | `3D/results/` | `3D/figures/` |

Dataset **metadata** (channel lists, sampling ranges, splits) is preserved in each stage's `data/`
folder so regenerated datasets can be verified against the ones used for the reported results.

---

## 5. Duplicate handling

The archive contained many notebooks with near-identical names across folders — `- Copy`,
`- Copy - Copy`, `-2`, `-3`, `400`, `500`, `_testing`, `_Final09thMay`. Classification was done on
**content**, never on filename.

### Kept as distinct research iterations

| Kept as | From | Why |
|---|---|---|
| `2D/02_u_deeponet/code/training/udeeponet_temperature_2d_first_unet_prototype.ipynb` | `3/BTP_DEEPONET/MetalFoamPCM_vscode_UNet_Arch_Temp.ipynb` (Jan 2026) | The first U-Net-branch experiment — the transition point between stage 2a and 2b. Scientifically meaningful even though superseded |
| `2D/02_u_deeponet/code/data_generation/udeeponet_dataset_generation_2d_earlier_version.ipynb` | `3/BTP_DEEPONET/MetalFoamPCM_Prismatic_Battery_Dataset_Gen.ipynb` (Feb 2026) | The prismatic-only generator that produced the first battery-source datasets, before generalised geometries were introduced |
| `2D/01_pointwise_deeponet/code/pointwise_pipeline_all_in_one_2d.ipynb` | `3/BTP_DEEPONET/MetalFoamPCM_vscode_All_in_one.ipynb` | The most complete surviving single-notebook record of the pointwise stage (solver + dataset + analysis together) |
| `2D/04_boundary_and_physics_studies/code/udeeponet_liquid_fraction_2d_with_boundary_heating.ipynb` | `4/BTP work/Sim_500/With BCs/` | Same architecture, different dataset family — a genuine experimental variable, not a duplicate |

### Chosen as the final implementation

| Kept | Chosen over | Reason |
|---|---|---|
| `4/BTP work/Sim_500/MetalFoamPCM_prismatic_batt_UNet_Arch_{Temp,Liqfrac}_testing.ipynb` | ~10 same-named copies under `3/`, `4/.../Sim_80…Sim_500/`, `prismatic_phys_dim_test2_N=120…/` | Most recent (Sept 2026), 500 cases, contains the case-study and test-split inference cells whose outputs match `INFERENCE TEXT METRICS.txt` |
| `4/BTP work/24_GRID-3DML/3D tests deeponet battery/*` | `1/BTP_DeepONet/`, `2/BTP work/`, `3/.../3D_LiqFracTrialForLogs/` copies | Most recent (10 Sept 2026); the only version whose dataset generator includes the updated prismatic-battery model |
| `1/BTP_DeepONet/3D_LiqFracTrialForLogs/MetalFoamPCM_3D_UNet_Arch_Temp-3_Final_09thMay.ipynb` | the `3/` copy of the same file | Byte-identical in the archive listing; one copy kept |

### Excluded as redundant or superseded

- **The nine `Sim_80` … `Sim_500` folders** under `4/BTP work/prismatic_phys_dim_test2_N=120 and 500Cases_withoutbc/`.
  Each contains the *same* notebook re-run at a different training-set size. Rather than ship nine
  near-identical 15–27 MB notebooks, the **results** of that sweep are preserved
  (`2D/03_data_sensitivity/results/*.csv`) together with the single notebook that produces them.
  This is the one place where the repository deliberately keeps the *output* of a set of runs instead of
  nine copies of the code — the difference between the copies is one configuration value.
- `MetalFoamPCM_prismatic_batt_UNet_Arch_Liqfrac.ipynb`, `...Liqfrac400.ipynb`,
  `...Liqfractest (2).ipynb`, `MetalFoamPCM_prismatic_batt_UNet_Arch_Liqfrac500.ipynb` — earlier
  training-size variants of the kept notebook.
- `MetalFoamPCM_3D_UNet_Arch_Liqfrac.ipynb`, `-2`, `-3`, and their `- Copy` / `- Copy - Copy` variants
  across `1/`, `2/`, `3/` — superseded by the Sept 2026 3D notebook.
- `MetalFoamPCM_vscode (1).ipynb`, `MetalFoamPCM_vscode_All_in_one_testing.ipynb`,
  `MetalFoamPCM_vscode_UNet_Liq.ipynb`, `WINDOW3/` variants — exploratory or debugging copies with no
  distinct scientific content.
- `1/BTP_DeepONet/WINDOW9/` — **empty directory** in the archive.

---

## 6. Important figures

136 figures were curated from roughly 1,900 in the archive. Every one is listed with a description and
its origin in [`figure_index.csv`](figure_index.csv). Headline items:

| Figure | Location |
|---|---|
| Data-sensitivity saturation curve | `2D/03_data_sensitivity/figures/best_validation_rmse_vs_training_set_size.png` |
| 20 liquid-fraction predictions with error maps | `2D/02_u_deeponet/figures/predictions/liquid_fraction_prediction_error_test_case_01–20.png` |
| Temperature predictions, four source families | `2D/02_u_deeponet/figures/predictions/temperature_prediction_error_*.png` |
| Cross-geometry, matched model pairs | `2D/05_cross_geometry_generalization/figures/` |
| 3D slice predictions, all three axes | `3D/figures/liquid_fraction_prediction_error_{x,y,z}_slice_*.png` |
| Volumetric animations | `3D/figures/animation_3d_*.gif` |
| Reference-solver physics studies | `2D/04_boundary_and_physics_studies/figures/` |

### Where the figures came from

- **Loose PNGs in the archive** (`5/Predictions without bc and with error maps/1–20.png`,
  `5/No_*_contribution/`, the sensitivity-plot ZIP) — copied and renamed.
- **Extracted from notebook cell outputs.** Many key figures existed *only* as base64 PNGs embedded in
  notebook outputs (this is why some notebooks were 15–32 MB). These were decoded and saved as files
  before the outputs were stripped, so nothing was lost.
- **Extracted from the BTech thesis PDF** — the three 1D figures and the two 2D pointwise training
  curves, because their original image files could not be located anywhere in the archive. Each is
  labelled with its thesis figure number in the corresponding README.

### Figure naming

Names describe content, established by inspecting the images, not by trusting cell labels. Two
specific decisions:

- One cell in the final temperature notebook is commented `### CASE-1 AND CASE-2`, but inspection of
  all 15 figures it produced showed that **every one is a prismatic-battery source** (one, two or three
  rectangles), not the smooth GP source that Case 1 uses. The comment appears to be stale. Those figures
  were therefore named by what they show (`single_prismatic_source`, `multiple_prismatic_sources`) and
  **not** by case number.
- Figures whose exact case identity could not be established are named by source family and index
  rather than given a spuriously precise name.

---

## 7. Excluded files and categories

| Category | Examples | Why |
|---|---|---|
| **Datasets** | `slice_dataset_3d.npz` (1.68 GB and 1.14 GB), `udeeponet_dataset.npz` (229 MB), `deeponet_{temp,frac}_dataset.npz` | Far beyond what belongs in Git; fully regenerable. Metadata and splits are preserved in each `data/` folder |
| **Model checkpoints** | `deeponet_frac_model_5net.pt` (36 MB), `deeponet_temp_model_4net.pt` (20 MB), all `checkpoints_*/best.pt` | Large, regenerable, and tied to specific dataset runs. Git LFS was considered and rejected: the checkpoints are only meaningful alongside their 230 MB–1.7 GB datasets, so storing them alone would not make any result reproducible |
| **Per-case diagnostic plots** | ~1,850 `case_NN_{temp_and_liquid_profiles,boundary_profiles,time_stats,variance_stats}.png` across six `dataset_run_*` folders | Generated automatically, one set per simulated case; representative examples are kept in `figures/dataset/` and `04_boundary_and_physics_studies/figures/` |
| **Thesis and presentation binaries** | `BTP_2nd_endsem report_final_hardbind.pdf`, `BTP_2nd_midsem report_.docx`, `BTP_midsem report_asmit_improved.docx`, `Cover page.docx`, three `.pptx` decks | Institutional deliverables, not research artefacts; several are 10–20 MB. The report was read extensively as a cross-reference, and five of its figures were extracted, but the documents themselves are not committed |
| **Third-party PDF** | `Sanjeet-2025.pdf` (14.7 MB) | A reference paper belonging to someone else — redistributing it would be a copyright problem |
| **Journal manuscript** | `journal/Deeponet_Asmit (1).pdf` | See §8 |
| **Duplicate notebooks** | see §5 | |
| **Empty directories** | `1/BTP_DeepONet/WINDOW9/` | Nothing to preserve |

---

## 8. Confidentiality

```
Unpublished journal manuscript:
NOT INCLUDED.

Journal source files (.tex, .bib, submission or reviewer material):
NOT INCLUDED.

Journal manuscript remains outside this repository.
```

The manuscript (`E:\BTP_Work_All Files\journal\Deeponet_Asmit (1).pdf`, 30 pages) was opened **once, as
confidential reference only**, to confirm terminology, the definitions of the six prediction cases, and
the structure of the cross-geometry study. Its text was not copied, paraphrased at length, or used as
the basis of any document here.

No figure in this repository was extracted from the manuscript. Every figure came from the archive
itself or, for five figures, from the BTech thesis report — the manuscript was never needed as a figure
source, so §11 of the curation brief (extract from the manuscript only if the original cannot be found)
was never invoked.

The numerical results quoted in the READMEs — the six prediction cases and the cross-geometry table —
come from **`INFERENCE TEXT METRICS.txt` and `CROSS GEOMETRY INFERENCE.txt` in the source archive**,
which are raw notebook outputs, not manuscript content. Both files are included in the repository so
every quoted number can be traced to its origin.

### Automated scan of the finished repository

Run over all 177 files:

| Check | Result |
|---|---|
| `*.tex`, `*.bib`, `*.bbl`, `*.aux` | none |
| filenames containing `journal`, `manuscript`, `submission`, `reviewer`, `supplementary` | none |
| `*.pdf`, `*.docx`, `*.pptx` anywhere in the repository | none |
| text content matching `manuscript`, `submission`, `reviewer`, `supplementary`, `Deeponet_Asmit`, `\begin{document}`, `elsarticle`, `\cite{` | one hit — the sentence in the root `README.md` stating that the manuscript is *not* included |
| `.gitignore` safety net for all of the above | present |

---

## 9. Reproducibility

### Reproducible today from this repository

- **All dataset generation**, 2D and 3D. The generators are complete and self-contained, including the
  solver, the GP heat-source sampler, the battery-footprint renderer and every sampling range.
- **All training**, 2D and 3D, once a dataset has been generated.
- **All inference and plotting**, once a model has been trained.
- **The data-sensitivity study**, by re-running the liquid-fraction training at each of the nine
  training-set sizes.
- **The cross-geometry study**, by generating the three required datasets and running the two notebooks.

### Requires additional work

- **The 1D stage cannot be re-run** — its code is absent from the archive. The thesis specifies the
  architecture and training protocol completely enough to re-implement it (see
  the *Stage 1* section of the root [`README.md`](../README.md#stage-1--1d-pointwise-deeponet-feasibility)), but that would be a
  re-implementation, not the original.
- **Exact numerical reproduction of the reported metrics is not guaranteed.** The datasets were seeded
  (`default_rng(2024)` for splits and mode selection, `10_000 + i` per case for boundaries), but the
  original checkpoints are not included, so training must be repeated and will land at a different
  local optimum. Expect the *trends* and the approximate magnitudes to reproduce, not the exact digits.
- **Compute.** The archived 3D run logged 1,000–3,900 s per epoch on CPU and reached epoch 34 of a
  configured 60. A GPU is effectively required to repeat it in reasonable time.
- **Disk.** Budget ~250 MB for a 2D run and up to 1.7 GB for a 3D run.

### Paths

Every machine-specific absolute path (`C:\Users\AsmitBTP\...`, `D:\BTP work\...`, `F:\BTP work\...`,
`E:\BTP_DEEPONET\...`) in the curated notebooks was rewritten to a repository-relative `runs/...` path —
46 distinct paths across 15 of the 16 curated notebooks. The rewrite preserved the `dataset_run_*` folder name, so
`D:\BTP work\Sim_500\dataset_run_20260817-133248\udeeponet_dataset.npz` became
`runs/dataset_run_20260817-133248/udeeponet_dataset.npz`. A verification pass confirms **zero**
absolute paths remain in any code cell. No username or personal directory structure is exposed.

---

## 10. Potential issues to verify manually

These are things found during curation that a reader should check rather than take on trust. **None of
them were fixed**, in line with the instruction not to silently alter scientific code.

1. **`NUM_CASES` is written as a float.** The configuration cells say `NUM_CASES = 500.0` and are cast
   with `int()` inside the generator. Harmless, but it looks like a typo and would break if the value
   were ever used unconverted.

2. **`n_plot = min(10e8, max_slices)` in the 3D inference loop.** `10e8` is a float, so `n_plot` is a
   float and `range(n_plot)` would raise a `TypeError` unless `max_slices` (an `int`) wins the `min`,
   which it always does in practice. Fragile, and clearly intended as "no limit".

3. **A stale cell comment.** `### CASE-1 AND CASE-2` labels a cell whose 15 output figures are all
   prismatic-battery sources, while Cases 1 and 2 in the metrics file are Gaussian-process sources. The
   figures were named from their content instead (§6). Worth confirming which dataset run corresponds
   to which case before reusing that cell.

4. **Duplicated function definitions inside single notebooks.** `plot_heat_trueT_predT` is defined three
   times in the final temperature notebook and `infer_and_plot_from_ckpt` twice, at different cell
   positions, with the later definition winning at runtime. The earlier definitions are dead code that
   looks live. Nothing was removed, but if you edit one of these, make sure you are editing the last one.

5. **Effective vs. explicit foam properties.** The thesis writes the governing equations with the
   porosity `φ` and separate PCM/metal specific heats explicit; the implemented solver uses single
   effective `ρ`, `c_p`, `k` and `L`. These are the same model with the volume averaging folded into the
   constants, but the porosity and foam material corresponding to the effective values are not recorded
   anywhere in the code. If those need to be traced, they must come from the thesis or from
   Dinesh & Bhattacharya (2019).

6. **`save_times` differs between the dataset generator and the training notebooks.** The generator
   configuration cell lists 24 snapshots; the training notebooks' configuration cells list 9. The
   training code reads the actual times from the dataset `.npz`, so the notebook constant is not used
   for anything — but it is misleading, and someone reading the training notebook alone would draw the
   wrong conclusion about the temporal resolution.

7. **`LR` also differs between configuration cells.** Dataset-generation notebooks carry a stale
   `LR = 1e-3` in their copied configuration block; the 2D training notebooks use `LR = 1e-4`, which is
   what the thesis reports and what the results reflect. The 3D training notebooks genuinely use
   `LR = 1e-3`. The training notebook is authoritative in each case.

8. **The 3D liquid-fraction training log ends at epoch 34** of a configured 60, with the best checkpoint
   at epoch 28. The reported 3D numbers come from that incomplete run. Whether it was stopped
   deliberately or interrupted is not recorded.

9. **Epoch time grows from 1,045 s to 3,896 s** over that same 3D run. This looks like resource
   contention on the training machine rather than anything in the model, but it is unexplained.

10. **Single-seed results.** No experiment in the archive was repeated with multiple random seeds. The
    data-sensitivity differences between adjacent sizes (e.g. 400 vs 450 cases) and the small
    liquid-fraction reversal in the cross-geometry study are within plausible run-to-run variance.

11. **Liquid fraction maximum error reaches 1.0** in several reported cases. This is expected for a
    near-binary field — it means the melt front is locally displaced by at least one cell — but it is
    worth stating explicitly in any write-up, since a maximum error equal to the full range of the
    variable reads alarmingly out of context.

---

## 11. What was changed, and what was not

**Changed** (organisation and hygiene only):

- Folder structure, rebuilt around the research progression rather than the archive's `1/`…`5/`.
- Filenames, made descriptive.
- Notebook **outputs cleared** (1,046 output objects across the 16 curated notebooks), reducing them from 15–32 MB
  each to 31–370 KB. Every figure those outputs contained was extracted to `figures/` first.
- Machine-specific absolute paths rewritten to `runs/...` (§9).
- One explanatory markdown cell prepended to each curated notebook, stating what was and was not
  changed.
- Documentation, environment files and `.gitignore` written from scratch. All stage-level
  documentation is consolidated into the single root `README.md` at the author's request; the only
  other prose documents are the four under `docs/`.

**Not changed:**

- No equation, physical assumption, numerical scheme, architecture, hyper-parameter, dataset definition
  or result was modified.
- No bug was fixed — the issues in §10 are reported, not patched.
- No cell was reordered or deleted.
- `common/physics/*.py` are **verbatim extractions** of notebook cells, provided as a readable reference
  copy. The notebooks remain self-contained and do not import them, so the two cannot silently diverge
  in behaviour — but if you edit the notebook cells, the reference copy will need updating by hand.

---

## 11b. Known artefact of the file transfer: C2PA metadata on images

Verified after the files were written to disk, by reading several of them back and comparing them
byte-for-byte against the versions built in this session:

- **Every text file is byte-identical** (notebooks, `.md`, `.csv`, `.json`, `.py`, `.txt` — MD5 match).
- **Every image file is pixel-identical but not byte-identical.** Each of the 136 `.png` / `.gif`
  files gained roughly **5.8 KB** during the copy to disk: PNGs carry an extra `caBX` chunk of 5,758
  bytes (5,770 with the chunk header) inserted immediately after `IHDR`, and the GIFs carry an
  equivalent 5,796-byte block. `caBX` is a **C2PA content-credentials / provenance manifest**, injected
  by the file-transfer path, not by anything in this repository.

The images themselves are sound: chunk structure is valid, the `IDAT` image data is byte-identical to
the originals, and a decoded pixel comparison matches exactly. They will render correctly everywhere.
Total overhead across the repository is about 0.8 MB.

**Why this is worth knowing.** These figures are matplotlib plots of the author's own numerical
simulations. A C2PA manifest attached to them is provenance metadata about the *file transfer*, and
should not be read as a statement about how the figures were produced. If that metadata is unwanted —
for instance if the figures will be submitted somewhere that inspects content credentials — it can be
removed without touching the image content:

```python
# strip the injected C2PA chunk from every PNG in the repository
import struct, glob
for f in glob.glob("**/*.png", recursive=True):
    d = open(f, "rb").read()
    out, i = d[:8], 8
    while i < len(d):
        ln = struct.unpack(">I", d[i:i+4])[0]
        if d[i+4:i+8] != b"caBX":
            out += d[i:i+12+ln]
        i += 12 + ln
    if len(out) != len(d):
        open(f, "wb").write(out)
```

Alternatively, unzipping the archive copies delivered in the chat over this folder restores the
original, unmodified image files.

---

## 11c. Folder names changed after curation

The three stage folders were renamed by the author after this repository was built:

| As built | As it now stands |
|---|---|
| `01_1D_pointwise_deeponet/` | `1D_pointwise_deeponet/` |
| `02_2D/` | `2D/` |
| `03_3D/` | `3D/` |

All internal links, embedded image paths and the `repository_path` column of `figure_index.csv` were
updated to match. The sub-structure inside each stage folder is unchanged. Note that this audit and the
documents in `docs/` refer to the source archive's own folder names (`1/`, `3/`, `4/`, `5/`) separately —
those are paths inside `E:\BTP_Work_All Files` and are unaffected.

---

## 12. Source archive integrity

```
The original source archive at E:\BTP_Work_All Files was treated as read-only.

No file in it was deleted, renamed, moved, overwritten or modified.
Files were opened for reading and copied out; nothing was written back.
```

---

## 13. Repository status

```
This folder is prepared as a repository root
for manual upload to a PRIVATE GitHub repository.

No GitHub push was performed.
```

No `git init` was run, no commit was created, no remote was added, no GitHub repository was created or
modified. The folder contains **177 files, approximately 39 MB**, with no file larger than GitHub's
recommended limits.

Suggested first steps after upload:

```bash
cd DeepONet-BTech-Thesis-GitHub
git init
git add .
git commit -m "Curated BTech thesis research: DeepONet surrogates for PCM-metal foam systems"
git branch -M main
git remote add origin <your private repository URL>
git push -u origin main
```

Keep the repository **private** until the journal manuscript is published — the `LICENSE` file reserves
all rights and states this explicitly.
