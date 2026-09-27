# TÜBİTAK Rebuild: Design Specification

- **Date:** 2026-09-27
- **Branch:** `feature/tubitak-rebuild`
- **Status:** Draft for review
- **Author:** İsmet Umuralp Şen

---

## 1. Context

The current repository is the code produced for a June 2026 undergraduate
graduation project: an end-to-end pipeline that detects liver steatosis from
H&E-stained whole-slide images (WSI) using an ImageNet-pretrained ResNet18.

The intent is to evolve this into a TÜBİTAK-funded research project. That
changes the acceptance criteria fundamentally. A graduation project needs a
working pipeline and plausible numbers. A funded project needs numbers that
survive review: correct data provenance, defensible labels, honest evaluation,
and a stated contribution that is not already published.

### 1.1 What the existing results actually measure

A review of the code and the committed result artifacts found that the three
headline results of the graduation project are artifacts of a single bug, not
properties of the data or the method.

**Root cause.** The six source slides were not scanned at a uniform
magnification:

| Slide | Dimensions (px) | `aperio.MPP` | Objective | Patches | Mean blob count | Labeled steatosis |
|-------|-----------------|--------------|-----------|---------|-----------------|-------------------|
| 1     | 22000 x 26046   | 0.4990       | 20x       | 2,413   | 12.99           | 90.9%             |
| 2     | 62350 x 24247   | 0.2520       | 40x       | 10,434  | 4.75            | 42.2%             |
| 3     | 27818 x 37154   | 0.2520       | 40x       | 7,568   | 7.78            | 75.1%             |
| 4     | 39839 x 45988   | 0.2527       | 40x       | 5,735   | 5.07            | 46.9%             |
| 5     | 43824 x 30054   | 0.2525       | 40x       | 7,218   | 4.72            | 48.7%             |
| 6     | 33515 x 35918   | 0.2485       | 40x       | 8,108   | 3.16            | 24.7%             |

`patch.py` reads every slide at pyramid level 0 with a fixed 256 x 256 pixel
window. This yields a physical field of view of 128 µm for slide 1 and 64 µm
for slides 2-6. The blob area filter in `label.py` is expressed in fixed pixels
(`50 < area < 2000`), so the same physical structure covers four times the
pixel area at 20x as at 40x and passes the filter with much higher probability.

**Consequence chain:**

1. Slide 1 receives 90.9% "steatosis" labels and slide 6 receives 24.7%
   "normal" labels, largely for instrumental rather than biological reasons.
2. Pooling the six slides yields a steatosis ratio of 49.39%, which lands
   almost exactly on the 50% line that `validate_labels.py` plots as the
   "ideal balance" criterion. Threshold 5 therefore appears to be the
   empirically optimal choice, and the README presents it as such. The
   apparent balance is an artifact of averaging over two different
   magnifications.
3. `GroupKFold(n_splits=3)` distributes the six slides by patch count, which
   produces the following validation folds:

   | Fold | Validation slides | Patches | Steatosis ratio |
   |------|-------------------|---------|-----------------|
   | 1    | 1, 2              | 12,847  | 51.3%           |
   | 2    | 4, 6              | 13,843  | 33.9%           |
   | 3    | 3, 5              | 14,786  | 62.2%           |

   Fold 2 is steatosis-poor, so the fold-2 model under-predicts steatosis and
   its recall collapses to 0.326.
4. The README attributes the fold-2 collapse to the validation patients having
   "a disproportionately low steatosis density". This is a misattribution. The
   cause is the labeling bias introduced by the magnification mismatch.

**Two further evaluation defects:**

- `evaluate.py` evaluates on `dataset/train`, which is the full labeled
  dataset, not a held-out split. The reported accuracy of 86.30% and AUC of
  0.9476 are resubstitution figures and are not valid performance estimates.
- `train.py` reports the best-epoch accuracy alongside the final-epoch
  precision, recall, and F1. The cross-validation summary table therefore
  mixes metrics from two different epochs, and the reported mean F1 of
  0.737 is not a meaningful quantity.

**Definitional limit.** Patch labels are produced by a hand-tuned
morphological blob counter. The model is therefore distilling that counter.
No amount of additional data changes this: the measured quantity is agreement
with a heuristic, not diagnostic accuracy. This must be stated explicitly and
is the single largest gap between the current work and a fundable project.

### 1.2 Position relative to published work

The current approach is below the published state of the art. Reviewers will
know the following:

| Work | Contribution | Relevant numbers |
|------|--------------|------------------|
| Boehringer Ingelheim / *Sci Rep* 2022 | 296 train / 171 test, pathologist ground-truth Kleiner scores | Quadratic weighted Cohen's κ = 0.66 for steatosis |
| MMD-Net, 2025 | Weakly supervised MIL, multi-task NAS scoring, 282 cases | κ = 0.766 for steatosis, mean κ = 0.845 |
| Koga et al., 2025 | Foundation models (Prov-GigaPath, UNI), 68 donors | 96.4% accuracy, statistically indistinguishable from surgical pathologists |
| Hao et al., *BPE* 2026 | Tile screening + UNet++ fat-droplet segmentation | Dice 0.97, WSI-level fat fraction MAE 0.44% |
| DLiPath, 2025 | 636 WSIs from 304 donors, expert-annotated, MIL benchmark | Six graded donor-liver criteria |

Every one of these uses pathologist-derived ground truth. None of the current
work's results are comparable to these numbers, and none currently would be
considered a contribution.

---

## 2. Goals and non-goals

### 2.1 Goals

- **G1.** Produce performance numbers from the existing six slides that are
  methodologically valid and honestly reported.
- **G2.** Establish a dataset registry with complete provenance, licence, and
  label source for every specimen entering the project.
- **G3.** Replace heuristic patch labels with pathologist-derived
  patient/slide-level labels from public data.
- **G4.** Move from patch-level classification to slide-level prediction using
  methods comparable to the current literature.
- **G5.** Leave the codebase in a state where an external reviewer can
  reproduce every reported number from a single command.

### 2.2 Non-goals for this phase

- No deployment, packaging, Docker, or REST API.
- No user interface.
- No Turkish patient cohort. That requires an ethics committee (KBYY) process
  and a hospital agreement, and is scoped as a later project milestone.
- No new model architectures beyond the baselines required for comparison.
- No experiment tracking infrastructure (MLflow, W&B). Plain CSV and JSON
  artifacts are sufficient and inspectable.

### 2.3 Explicit YAGNI decisions

- `label_smoothing`, `mixup_alpha`, `cutmix_alpha`, `tta_enabled`,
  `tta_augmentations`, `bootstrap_samples`, `confidence_level`,
  `calibration_bins`, and `threshold_method: adaptive_per_patient` are removed
  from the configuration rather than implemented. A configuration file that
  declares unimplemented behaviour is worse than no configuration file,
  because it silently misleads the reader.
- Bootstrap confidence intervals are implemented for the headline metrics
  because with six specimens the variance is the entire story, but no general
  calibration framework is built.

---

## 3. Architecture

### 3.1 Module layout

The current design is a set of standalone scripts that each re-implement
shared logic. `count_blobs` exists in two files (`label.py` and
`validate_labels.py`) and hard-coded settings blocks are duplicated across
every script. The rebuild separates computation from orchestration.

```
blobcount/
  __init__.py
  config.py          # configuration loading, validation, seed control
  slides.py          # slide metadata, MPP normalization, patch extraction
  blobs.py           # single implementation of the blob counter
  labeling.py        # patch -> label assignment
  splits.py          # LOSO and grouped split construction
  data.py            # dataset and dataloader construction
  models.py          # model construction
  train.py           # training and cross-validation loop
  evaluate.py        # metrics, aggregation, bootstrap CI
  interpret.py       # Grad-CAM
  predict.py         # inference helper
  registry.py        # dataset manifest loading and validation
  report.py          # report and figure generation
bin/                 # thin CLI entry points, one per stage
tests/               # pytest suite
configs/
  default.yaml
data/
  manifest.yaml      # provenance for every specimen (committed)
```

Each module has one responsibility and can be tested independently. The
`bin/` scripts stay thin so that the pipeline reads as a sequence of stages.

### 3.2 Configuration

A single `configs/default.yaml` is loaded by every entry point through
`blobcount.config`. The loader must:

- Validate that every declared key is consumed by the code. Unknown keys are a
  hard error, not a warning.
- Reject unimplemented keys. If a key exists in the YAML, something reads it.
- Expose a `--set key=value` override on every entry point so that no
  experiment requires editing a tracked file.
- Call `set_seed()` at the start of every entry point, covering Python
  `random`, NumPy, and torch, and configuring `cudnn.deterministic`.

### 3.3 Resolution normalization (the core fix)

All downstream analysis assumes a fixed physical field of view. This must be
enforced at extraction time.

- **Target:** 0.5 µm/px, giving a 128 µm field of view for a 256 x 256 patch.
  All six existing slides are at or above this resolution, so no upsampling
  occurs for any of them.
- **Implementation:** for each slide, read `aperio.MPP` (falling back to
  `openslide.mpp-x` when available). Compute
  `downsample = slide_mpp / target_mpp`. Select the pyramid level with
  `slide.get_best_level_for_downsample(downsample)` and pass the residual
  scale to the reader. Record the achieved MPP per patch in the patch
  metadata so that drift is detectable.
- **Assertion:** a test asserts that the physical field of view of patches
  from all six slides agrees within 2%.
- **Consequence for blob filtering:** area thresholds are converted from
  physical units (µm²) to pixels at extraction time rather than hard-coded.
  The existing 50-2000 px² filter at 0.25 µm/px corresponds to 3.1-125 µm² and
  becomes 12.5-500 px² at 0.5 µm/px. The circularity threshold of 0.6 is
  scale-invariant and is unchanged.
- **Stain normalization:** the grayscale thresholds currently used for tissue
  detection (210) and blob detection (200) are absolute intensity cutoffs and
  are sensitive to per-slide staining variation. They become per-slide
  adaptive, derived from each slide's intensity distribution, and the derived
  value is recorded in the manifest.

### 3.4 Splitting

With six specimens, a fixed train/test split is not defensible: a two-specimen
test set is two data points.

- **Leave-one-specimen-out (LOSO)** is the primary evaluation. Six folds, each
  training on five specimens and validating on the held-out one.
- Results are reported as mean ± standard deviation across the six folds with
  bootstrap confidence intervals computed over specimens, not over patches.
  Patch-level bootstrap would be meaningless because patches within a specimen
  are not independent.
- The README claim of a patient-level split preventing leakage is removed. The
  unit of splitting is the specimen, and this is stated wherever a split is
  described.
- Once a real cohort exists, `StratifiedGroupKFold` replaces LOSO so that
  grade distribution is balanced across folds. LOSO remains available for the
  six-specimen pilot.

### 3.5 Metrics

- **Model selection** during training uses validation F1, not accuracy. The
  weighted random sampler and class-weighted loss both distort the accuracy
  estimate on imbalanced data.
- **Headline metric** for steatosis grading is quadratic weighted Cohen's κ
  against the reference grade, which is the field standard and is the metric
  the 2022 and 2025 studies above report. Accuracy is reported alongside but is
  not the selection or headline criterion.
- **Reported quantities** for each fold come from the selected (best) epoch.
  The training loop records per-epoch history to CSV and the summary is
  computed from the selected epoch. The previous behaviour of mixing best-epoch
  accuracy with final-epoch F1 is removed.

### 3.6 Evaluation integrity

- `evaluate.py` reads an explicit specimen list for the split being evaluated.
  It never globs the full dataset directory.
- A held-out evaluation is performed on specimens that no fold trained on, and
  the specimen IDs are printed and recorded in the output artifact.
- The report generator refuses to emit a performance table when the evaluated
  specimen set intersects the trained specimen set, failing loudly rather than
  producing a misleading number.

### 3.7 Dataset registry and provenance

`data/manifest.yaml` is committed and is the authoritative record of what is in
the project. One entry per specimen:

```yaml
- id: <stable identifier>
  source: <TCGA-LIHC | GTEx | DLiPath | local | unknown>
  source_case_id: <identifier at the source institution>
  scanner: <vendor and model>
  mpp: <microns per pixel at level 0>
  objective: <nominal magnification>
  stain: <H&E | Masson | other>
  acquisition_date: <ISO 8601 or unknown>
  label_source: <pathologist | report-text | heuristic | none>
  label_detail: <free text, e.g. Kleiner steatosis score from report>
  licence: <identifier and URL>
  ethics: <approval reference or public-domain statement>
```

Validation rules enforced by `registry.py`:

- A specimen without `source_case_id` and `licence` cannot be used for
  training. It may be used for pipeline smoke tests only.
- Slides currently recorded as `1.svs` through `6.svs` have unknown
  provenance. They are marked `source: unknown` and are excluded from any
  reported result until their origin is recovered. They remain usable for
  verifying that the extraction and normalization code runs.

This is not bookkeeping. "Which data, from where, under what permission" is the
first question a TÜBİTAK reviewer asks, and the current answer is that the
provenance was destroyed by renaming.

### 3.8 Phase A data sources

Ordered by suitability for the project's actual question, which is steatosis
quantification on H&E.

| Priority | Source | Contents | Label source | Notes |
|----------|--------|----------|--------------|-------|
| 1 | GTEx liver | 109 H&E WSIs | Kleiner steatosis score extracted from the pathology report text | Postmortem donor livers, not NAFLD biopsies. Closest match to the existing H&E data. Requires assembling the slide-ID list and downloading from the GDC image endpoint. |
| 2 | DLiPath | 636 WSIs, 304 donors | Expert annotation: total and macrovesicular steatosis, ballooning, cholestasis, portal inflammation, fibrosis | Donor liver, where steatosis grade is the accept/discard decision. Public MIL baselines available for comparison. Confirm access terms before committing. |
| 3 | TCGA-LIHC | 377 cases, 365 diagnostic slides | None for steatosis | Hepatocellular carcinoma tissue, not a steatosis cohort. Usable only for a stain and scanner robustness check, not as a primary cohort. |
| 4 | OSF 8e7hd (Heinemann et al.) | 467 biopsies from 3 centres, 282 fully annotated | Expert annotation of steatosis, ballooning, inflammation, fibrosis | Masson-Goldner and Masson trichrome staining, not H&E. Defer until stain handling is addressed. |

Turkish data sources identified but not yet actionable:

- `acikveri.saglik.gov.tr` (Ministry of Health open data portal) has published
  histopathology datasets including Mel-DEPTHS (Yildiz Technical University,
  50 melanoma WSIs with expert-verified labels). No liver dataset is
  currently published. The portal is a demonstrated route for a future
  submission.
- `compseg.ankara.edu.tr` (Ankara University COMPSEG group) published NuSeC and
  MiDeSeC breast histopathology datasets under TÜBİTAK project 121E379. A
  Turkish group with demonstrated histopathology dataset publishing capability
  and a potential collaboration contact.
- `patolojiatlasi.com` (Serdar Balci, Memorial Pathology) maintains 265+
  whole-slide cases including liver, bilingual Turkish and English. A teaching
  atlas rather than a research dataset, but a WSI source and a contact.

### 3.9 Modelling direction

- **Unit of prediction** moves from the patch to the slide. Patch predictions
  are aggregated with a learned or attention-based pooling rather than majority
  vote, which is the standard in the literature.
- **Required baselines**, so that any claimed improvement is meaningful:
  1. The existing ResNet18 fine-tuning baseline, corrected per this document.
  2. A self-supervised pathology foundation model (UNI or Prov-GigaPath)
     with a linear probe on slide-level embeddings.
  3. At least one established MIL model (CLAM-SB, ABMIL, or TransMIL).
- Ablations to report: effect of MPP target, effect of blob threshold
  (recomputed after normalization), effect of stain normalization, and the
  backbone comparison.

---

## 4. Error handling and logging

- No bare `except:`. Every caught exception is logged with the slide
  identifier, the patch coordinates, and the exception, and the count of
  failures is reported at the end of extraction. The current
  `patch.py` silently discards unreadable slides and patches.
- `label.py` clears its output directories before writing. The current
  implementation copies into existing directories, so re-running it after a
  threshold change leaves stale labels behind, and a patch can end up present
  in both class folders with contradictory labels. The current dataset was
  verified clean, but the code permits the failure.
- Configuration errors, missing specimens, and manifest violations abort the
  run with a non-zero exit code and a message naming the offending entry.
- Every pipeline stage writes a JSON artifact recording its inputs, resolved
  parameters, and outputs, so that a result can be traced to the exact
  configuration that produced it.

---

## 5. Testing

A `pytest` suite covering the parts where a silent error would corrupt results:

- **Resolution normalization.** Assert that the physical field of view agrees
  across all six slides within 2%, and that no slide is upsampled.
- **Blob counter.** Synthetic images with a known number of discs of known
  radius assert an exact count, including the empty and all-rejected cases.
- **Configuration.** Every key in `configs/default.yaml` is consumed. Unknown
  keys raise. A configuration missing a required key raises.
- **Split disjointness.** No specimen appears in more than one LOSO fold's
  validation set, and no training set contains a validation specimen.
- **Evaluation integrity.** The report generator raises when the evaluated
  specimen set intersects the trained specimen set. This is the regression test
  for the defect in section 1.1.
- **Fold summary consistency.** The reported per-fold metrics equal the
  selected epoch's metrics from the training log, not the final epoch's.

`test_model.py` is renamed to `predict.py` to remove the collision with the
test suite.

---

## 6. Repository hygiene

- `Colab_Proje.zip` (10.1 GB) is moved out of the project directory. It is
  already excluded by the `*.zip` rule, but a 10 GB archive inside the working
  tree will be synchronized by OneDrive and is a standing risk of accidental
  commit. Its contents should be inspected once, as it may hold the larger
  dataset referenced in the project history.
- `data/` and `dataset/` are moved outside the OneDrive-synchronized tree.
  Patient-derived data synchronized to cloud storage is an ethics problem
  independent of the research question, and OneDrive synchronization of tens of
  thousands of files is a practical failure mode.
- `steatosis_model.pth` (44 MB) in the repository root is a stale artifact from
  an earlier Colab run and is not referenced by any code. It is deleted.
- The `dataset/train` directory is verified free of duplicate filenames across
  class folders before any relabeling. This has been checked once and is
  clean; the check becomes a test.

---

## 7. Sequencing

The two phases run concurrently, with Phase A data acquisition in the
background while Phase B is completed.

**These phases are decomposed into two separate implementation plans.** Phase B
is planned and executed first; it is self-contained, depends on nothing outside
the repository, and yields a reportable result. Phase A is not planned until
open questions O1 through O5 are answered, because O3 in particular (binary
detection versus ordinal grading) changes the label representation, the loss
function, and the evaluation metric, and planning it before that decision would
produce a plan that has to be rewritten.

**Phase B, on the existing six slides.** Deliverable: a technical report whose
numbers are valid.

1. Extract patches with MPP normalization; record resolution per patch.
2. Recompute blob counts under normalized resolution; regenerate labels.
3. Re-run the threshold ablation and report the per-specimen class
   distribution, not only the pooled one.
4. Refactor into the `blobcount` package with real configuration wiring.
5. Replace the fold loop with LOSO; select on F1; report the selected epoch.
6. Add the test suite.
7. Add the dataset manifest and mark the six slides `source: unknown`.
8. Correct the README: remove the leakage claim, remove the threshold
   justification, add the cohort size and the label-definition limitation.

**Phase A, on public data.** Deliverable: a defensible baseline and a
comparison point.

9. Assemble the GTEx slide list with Kleiner scores; download; register.
10. Evaluate DLiPath access terms; register if usable.
11. Port the pipeline to slide-level prediction against pathologist grades.
12. Implement the three baselines and report κ against the published numbers.

**Explicitly deferred.** Turkish cohort and KBYY process; these follow once a
working pipeline and a collaborator exist.

---

## 8. Open questions

- **O1.** ~~Can the provenance of the six existing slides be recovered?~~
  **Partially resolved, 2026-09-27.** The Aperio metadata was read directly from
  the six files. Three carry TCGA slide barcodes in `aperio.Filename` and were
  confirmed against the GDC API as TCGA-LIHC (liver hepatocellular carcinoma):

  | Local file | TCGA case | Scanner | Date |
  |------------|-----------|---------|------|
  | `4.svs` | TCGA-2V-A95S | Aperio SS1764CNTLR | 2015-07-01 |
  | `5.svs` | TCGA-DD-AAEH | Aperio SS1763CNTLR | 2014-08-15 |
  | `6.svs` | TCGA-GJ-A6C0 | Aperio SS1436CNTLR | 2013-05-08 |

  The remaining three (`1.svs`, `2.svs`, `3.svs`) have numeric-only
  `aperio.Filename` values (54008, 36394, 32235), were scanned on Aperio SS1352
  and SS1302 between 2012 and 2013, and carry reporting-template identifiers
  `gleason` and `ap1546-dsr` rather than TCGA barcodes. They are from a
  different source and their origin is not yet established. Slide 1's
  `gleason` template is a prostate reporting protocol and is unexplained; the
  tissue morphology is consistent with liver, so it is most likely a scanner
  configuration default rather than a statement about the specimen. Each
  slide's `openslide.quickhash-1` value is recorded in the manifest so that a
  future match against a public archive does not require re-reading the files.

  Consequence: Phase B has three usable specimens with full provenance and
  three without. Leave-one-out over the three usable specimens gives three
  folds, each training on two specimens. That is a pipeline verification, not a
  result, and the README must say so. A meaningful Phase B number requires
  either recovering the provenance of the other three slides or obtaining
  additional TCGA-LIHC diagnostic slides, which are openly available and would
  need only manifest entries.
- **O2.** Does the 10 GB `Colab_Proje.zip` contain a larger cohort, and is that
  cohort the one the project history refers to?
- **O3.** Is the target clinical question steatosis detection (binary) or
  steatosis grading (ordinal, Kleiner 0-3)? The literature is largely graded,
  and grading is the more useful and more defensible target, but it requires
  ordinal labels rather than a binary split. This should be settled before
  Phase A model work begins.
- **O4.** Which TÜBİTAK programme is targeted (2244 industrial doctorate, 1501,
  1505, or LİGE)? The programme determines the required deliverables, the
  industry partner arrangement, and the reporting cadence, which affects how
  much of this scope is committed to in the first year.
- **O5.** Is a GPU available? The current pipeline is CPU-only by design, and
  foundation-model baselines and MIL training are substantially more practical
  with one. This affects the phase A timeline materially.
