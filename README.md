# Detection of Liver Steatosis from Microscopy Images

An end-to-end pipeline for the automated detection of **liver steatosis (hepatic steatosis)** from whole-slide histopathology (WSI) images.

> Istanbul Technical University - Faculty of Computer and Informatics
> Department of Computer Engineering - Graduation Project (June 2026)
>
> **Student:** İsmet Umuralp Şen
> **Advisor:** Prof. Dr. Behçet Uğur Töreyin

---

## Project Overview

Hepatic steatosis, defined as the excessive accumulation of lipid (fat) droplets within hepatocytes, is the primary histopathological feature of non-alcoholic fatty liver disease (NAFLD). In current clinical practice, steatosis grading is performed by pathologists examining H&E-stained biopsy slides under a microscope; this process is time-consuming and subject to inter-observer variability.

This project presents an automated classification system that processes raw whole-slide images in SVS format and distinguishes steatotic from normal liver tissue **without any manual annotation**: every training label is the output of a morphological blob counter run at a fixed physical scale. That makes the labels reproducible, and it also bounds what they can mean — the model is measured against the counter, not against a pathologist's grade (see [Limitations](#limitations)).

The system is designed to run entirely on CPU hardware.

## Pipeline

The pipeline is a chain of `bin/` entry points. Each one reads its parameters from `configs/default.yaml` and its input from the artifact the previous stage wrote, so a run is a sequence of commands with a single configuration behind them.

| Order | Entry point | What it does |
|-------|-------------|--------------|
| 1 | `bin/extract_patches.py` | Tiles each specimen into 256 x 256 patches at a fixed 0.5 µm/px — a 128 µm field of view whatever the scanner's native magnification — and drops background tiles with an adaptive tissue filter. Writes the patch tree and `patches.csv`. |
| 2 | `bin/label_patches.py` | Labels every patch by counting morphological blobs in square microns at the patch's own achieved resolution; at least `labeling.blob_threshold` (5) blobs is steatosis. Writes the class tree and `labels.csv`. |
| 3 | `bin/ablate_threshold.py` | Sweeps `ablation.thresholds` over the same index and reports the class distribution **per specimen** at every threshold. Writes `results/label_analysis/`. |
| 4 | `bin/train.py` | Trains leave-one-specimen-out over the cohort the manifest admits, selecting each fold's epoch on validation F1. Writes `results/fold_summary.csv`, `results/training_summary.json`, a log and a checkpoint per fold. |
| 5 | `bin/evaluate.py` | Scores a checkpoint over the specimens it never trained on, with bootstrap intervals. Writes `results/evaluation/`. |

Two further entry points run after training: `bin/predict.py` (inference over an image or a directory, writing `results/predictions.csv`) and `bin/gradcam.py` (Grad-CAM figures for one specimen, writing `results/gradcam/`).

### Reproduction

`make all` runs the five stages in that order and is the single command that reproduces every number reported in this README. The Makefile reads its paths and the training cohort from `configs/default.yaml` and `data/manifest.yaml` through the project's own code; it names no directory and no specimen itself.

```bash
make all        # extract -> label -> ablate -> train -> evaluate
```

The individual stages are `make extract`, `make label`, `make ablate`, `make train`, `make evaluate`, plus `make predict`, `make gradcam`, `make install` and `make test`. `make clean` removes `results/` and the generated label directory and nothing else — the `.svs` slides are never touched, by the same path guard the extraction stage uses. Training runs on CPU.

## Folder Structure

```
.
├── Makefile                      # `make all` reproduces every reported number in one command
├── pyproject.toml                # package metadata, dependencies, pytest and ruff config
├── requirements.txt              # the same dependencies, for a non-editable install
├── configs/default.yaml          # the single source of runtime parameters
├── data/                         # the six .svs slides (git-ignored) and manifest.yaml (tracked)
├── blobcount/                    # the package
│   ├── __init__.py               # package marker, version
│   ├── config.py                 # load, validate, override, seed control, device selection
│   ├── registry.py               # load and validate the manifest, gate specimen usability
│   ├── slides.py                 # slide metadata, resolution planning, patch reading, adaptive tissue threshold
│   ├── blobs.py                  # blob counting in physical units, adaptive gray threshold, Reinhard stain normalization
│   ├── extract.py                # patch extraction orchestration, index CSV emission, path guards
│   ├── labeling.py               # patch-to-label assignment, output directory lifecycle
│   ├── ablation.py               # threshold sweep with per-specimen breakdown
│   ├── splits.py                 # leave-one-specimen-out fold construction
│   ├── data.py                   # dataset, transforms, per-fold class weights, sampler
│   ├── models.py                 # model construction from a name
│   ├── train.py                  # cross-validation training loop, epoch selection, summary
│   ├── evaluate.py               # metrics, integrity guard, bootstrap confidence intervals
│   └── interpret.py              # Grad-CAM
├── bin/                          # the CLI entry points, in run order
│   ├── extract_patches.py
│   ├── label_patches.py
│   ├── ablate_threshold.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── gradcam.py
├── results/                      # generated artifacts: metrics, figures and summaries are tracked, *.pth are not
├── tests/                        # one test module per package module
│   ├── conftest.py               # shared fixtures: synthetic patch, temporary output tree
│   ├── test_config.py            # config tests
│   ├── test_registry.py          # registry tests
│   ├── test_slides.py            # resolution planning and metadata tests
│   ├── test_blobs.py             # blob counter and scale-invariance tests
│   ├── test_extract.py           # extraction tests
│   ├── test_labeling.py          # labeling tests
│   ├── test_ablation.py          # ablation tests
│   ├── test_splits.py            # LOSO tests
│   ├── test_data.py              # dataset and sampler tests
│   ├── test_models.py            # model construction tests
│   ├── test_train.py             # training summary tests
│   ├── test_evaluate.py          # metric and guard tests
│   └── test_interpret.py         # Grad-CAM tests
├── LICENSE
└── README.md
```

The extracted patches and the labelled tree are generated data and live outside the repository, at the `paths.patches` and `paths.labels` of `configs/default.yaml` (`%LOCALAPPDATA%\UmurOS\dataset` on this machine). A fresh clone that runs `make all` writes them there, not into the checkout.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

make install     # python -m pip install -e ".[dev]"
```

**OpenSlide note:** `bin/extract_patches.py` requires the OpenSlide binary library installed on the system to read SVS files. On Windows it can be installed with `pip install openslide-bin`, on Linux with `apt-get install openslide-tools`. (Only needed for the patch extraction stage.)

## Labels and the Threshold Ablation

Every label is the output of a morphological blob counter, not a pathologist's judgement. A patch is labelled steatosis when it holds at least `labeling.blob_threshold` (**5**) blobs, where a blob is a connected component whose area is between 3.1 and 125 µm² and whose circularity is at least 0.6, counted below a per-slide adaptive gray threshold. The counter works in physical units — square microns against each patch's own achieved µm/px — so the same structure is counted the same way at every slide's native magnification. The labeling rule and its threshold are configured values (`blobs.*`, `labeling.blob_threshold` in `configs/default.yaml`); they are not derived from the data.

The ablation sweep exists because a pooled class ratio hides exactly the thing that decides whether a threshold means anything: how differently the slides label. All numbers in this section are from `results/label_analysis/threshold_ablation.json`, which `make ablate` regenerates.

**Per-specimen class distribution at the selected threshold 5** (the three specimens the manifest clears for training; patches and steatosis counts are over the artifact's `blob_counts`):

| Specimen | Patches | Steatosis | Steatosis ratio | Gray threshold | Achieved µm/px |
|----------|---------|-----------|-----------------|---------------|----------------|
| 4 | 1835 | 321 | 0.175 | 233.3 | 0.5005 |
| 5 | 1976 | 1307 | 0.661 | 127.8 | 0.5001 |
| 6 | 2191 | 1874 | 0.855 | 126.8 | 0.4999 |

Pooled over the three specimens the ratio is **0.583** (3502 of 6002), with a per-specimen spread of **0.680**. No pooled ratio at any swept threshold makes the per-specimen spread close, and the balance the pooled number appears to show at threshold 10 (0.475) is an average over one slide labelling 9% steatotic and another labelling 81% — a spread of 0.717, wider than at the selected 5 — not a property of any slide.

**The sweep**, one row per threshold (`n_steatosis` counts patches at or above the threshold):

| Threshold | n_steatosis | n_normal | Pooled ratio | Per-specimen spread | Selected |
|-----------|-------------|----------|--------------|---------------------|----------|
| 3 | 3853 | 2149 | 0.642 | 0.651 | |
| 5 | 3502 | 2500 | 0.583 | 0.680 | yes |
| 7 | 3201 | 2801 | 0.533 | 0.695 | |
| 10 | 2852 | 3150 | 0.475 | 0.717 | |
| 15 | 2403 | 3599 | 0.400 | 0.724 | |
| 20 | 2087 | 3915 | 0.348 | 0.706 | |
| 25 | 1839 | 4163 | 0.306 | 0.687 | |
| 30 | 1627 | 4375 | 0.271 | 0.664 | |

The configured range does **not** bracket the measured count distribution: 3741 of the 6002 patches carry a blob count outside the swept range 3–30 (2149 below it, 1592 above it; the counts run min 0, p50 8.0, mean 21.99, max 150), so the sweep describes the knee of the distribution and not its tail. The artifact records this as `threshold_range_coverage.brackets_distribution: false`.

<p align="center">
  <img src="results/label_analysis/threshold_ablation.png" width="85%" alt="Threshold ablation: per-specimen steatosis ratio at every swept threshold"/>
</p>

## Method Details

- **Patches:** 256 x 256 at a fixed 0.5 µm/px, a 128 µm field of view for every slide; an adaptive tissue filter drops background tiles before anything is labelled.
- **Labels:** morphological blob counting in square microns (area 3.1–125 µm², circularity >= 0.6, per-slide adaptive gray threshold); at least 5 blobs is steatosis.
- **Architecture:** ResNet18 (ImageNet pretrained), final layer changed from 512 -> 2.
- **Input:** 224x224x3 RGB.
- **Loss:** class-weighted cross entropy, the weights from each fold's own training labels.
- **Sampling:** balanced batches via WeightedRandomSampler.
- **Optimization:** Adam (lr = 1e-4), ReduceLROnPlateau (factor=0.5, patience=3).
- **Augmentation:** horizontal/vertical flip, ±15° rotation, color jitter, affine translation.
- **Cross-validation:** leave-one-specimen-out over the specimens the manifest admits; each fold holds out every patch of one specimen and trains on the rest. The unit of splitting is the specimen.
- **Model selection:** per fold, the epoch with the highest validation F1. Accuracy is reported but never selects.
- **Early stopping:** a fold stops after 3 consecutive epochs without a new best validation F1.
- **Reported metrics:** from each fold's selected epoch. Final-epoch metrics are never reported.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.
