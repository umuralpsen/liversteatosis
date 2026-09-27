# Phase B: Methodology Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the pipeline so that every reported number is produced under a fixed physical resolution, with dataset provenance recorded, evaluated on held-out specimens, and reported from the selected epoch.

**Architecture:** Replace the nine standalone scripts with a `blobcount` package of single-responsibility modules plus thin `bin/` entry points. Patches are extracted at a fixed 0.5 µm/px so that all specimens share one field of view, which makes the blob counter's physical-unit thresholds valid across slides. Evaluation moves to leave-one-specimen-out with a guard that refuses to score a specimen that was trained on.

**Tech Stack:** Python 3.11+, PyTorch 2.x, torchvision, OpenSlide (Python binding 1.4.3), OpenCV, NumPy, scikit-learn, pandas, PyYAML, pytest, matplotlib.

**Spec:** `docs/superpowers/specs/2026-09-27-tubitak-rebuild-design.md`

## Global Constraints

- Target resolution is **0.5 µm/px**, patch size **256 x 256**, giving a 128 µm field of view. This value is fixed by the spec and must not be varied per experiment without recording the change.
- Achieved resolution must agree with the target within **2%** for every specimen. Measured deviation for the current six slides at this target is at most 0.30%.
- The blob area filter is expressed in **square microns**, never in pixels. Equivalent pixel bounds are derived at the achieved resolution.
- Circularity threshold is **0.6** and is scale-invariant; it is not recalculated.
- Specimen-level splitting is **leave-one-specimen-out**. No other split is used in Phase B.
- Model selection during training is on **validation F1**. Accuracy is reported but never selects.
- Reported per-fold metrics come from the **selected epoch**. Final-epoch metrics are never reported.
- `openslide.mpp-x` is read first, `aperio.MPP` is the fallback, and if neither is present the slide is **rejected with an error**. The resolution is never assumed or defaulted.
- Every configuration key present in `configs/default.yaml` must be read by code. A key that no code reads is a hard error at load time, not a warning.
- No bare `except:` anywhere. Every caught exception is logged with specimen, coordinates, and the exception object.
- `pytest` must pass with zero warnings promoted to errors.
- Every task ends with a commit. Conventional Commits format, matching the existing history (`docs:`, `feat:`, `fix:`, `test:`, `refactor:`).
- Repository root contains a non-ASCII path component (`Masaüstü`). All file access uses `pathlib.Path`, never string concatenation with hard-coded separators.

## Review Focus

### Expected scale of the Phase B run

Registry validation admits three of the six specimens for training, because
three have no recoverable `source_case_id` or `licence`. Leave-one-out over
three specimens produces three folds, each training on two specimens. The
resulting metrics are a pipeline verification and must be labelled as such in
every artifact and in the README. Task 15 must not present them as a cohort
result. If more TCGA-LIHC slides are added to `data/manifest.yaml` before Task
11 runs, the fold count rises accordingly and no code change is needed.

### Input classes and failure modes

These are input classes and failure modes the spec implies but that no individual module's own tests would catch. Each has a test pinned to the task that owns the code.

1. **Missing resolution metadata.** A slide where neither `openslide.mpp-x` nor `aperio.MPP` is present must be rejected by name, not silently processed at an assumed 1.0 µm/px. Silent assumption is exactly the class of bug that produced the original defect. → Task 3
2. **Non-ASCII and Turkish-locale paths.** The repository lives under `Masaüstü` and the machine locale is Turkish, where `casefold()` and `lower()` differ. Specimen identifiers derived from filenames must round-trip through the manifest, the patch index, and the label CSV unchanged, including any non-ASCII characters. → Task 2
3. **A patch index written at a different target resolution.** `patches.csv` records the achieved `mpp` per patch. Labeling must use each patch's own recorded value, never a single global constant. Reading a global value here would silently reintroduce the original defect while passing a naive test. → Task 6
4. **Fully blank slide.** Adaptive intensity thresholds are derived from a percentile of tissue pixels. A slide with no tissue produces a percentile over an empty array, which yields `NaN` and propagates into every downstream comparison. Must be detected and reported as a skipped slide, not a `NaN` threshold. → Task 4
5. **Stale labels from a previous threshold.** Re-running labeling after changing the threshold must not leave a patch present in both class directories. The current code permits this; the current dataset happens to be clean, which is why the bug was never noticed. → Task 6

---

## File Structure

**Created:**

| Path | Responsibility |
|------|----------------|
| `pyproject.toml` | Package metadata, dependencies, pytest and ruff config |
| `configs/default.yaml` | The single source of runtime parameters |
| `Makefile` | `make all` reproduces every reported number in one command |
| `blobcount/__init__.py` | Package marker, version |
| `blobcount/config.py` | Load, validate, override, seed control, device selection |
| `blobcount/registry.py` | Load and validate `data/manifest.yaml`, gate specimen usability |
| `blobcount/slides.py` | Slide metadata, resolution planning, patch reading, adaptive tissue threshold |
| `blobcount/blobs.py` | Blob counting in physical units, adaptive gray threshold, Reinhard stain normalization |
| `blobcount/extract.py` | Patch extraction orchestration, index CSV emission |
| `blobcount/labeling.py` | Patch to label assignment, output directory lifecycle |
| `blobcount/ablation.py` | Threshold sweep with per-specimen breakdown |
| `blobcount/splits.py` | Leave-one-specimen-out fold construction |
| `blobcount/data.py` | Dataset, transforms, per-fold class weights, sampler |
| `blobcount/models.py` | Model construction from a name |
| `blobcount/train.py` | Cross-validation training loop, epoch selection, summary |
| `blobcount/evaluate.py` | Metrics, integrity guard, bootstrap confidence intervals |
| `blobcount/interpret.py` | Grad-CAM |
| `bin/extract_patches.py` | CLI: extraction stage |
| `bin/label_patches.py` | CLI: labeling stage |
| `bin/ablate_threshold.py` | CLI: threshold ablation |
| `bin/train.py` | CLI: cross-validation training |
| `bin/evaluate.py` | CLI: evaluation |
| `bin/predict.py` | CLI: inference on a slide or directory |
| `bin/gradcam.py` | CLI: Grad-CAM figures |
| `data/manifest.yaml` | Provenance record, one entry per specimen |
| `tests/conftest.py` | Shared fixtures: synthetic patch, temporary output tree |
| `tests/test_config.py` | Config tests |
| `tests/test_registry.py` | Registry tests |
| `tests/test_slides.py` | Resolution planning and metadata tests |
| `tests/test_blobs.py` | Blob counter and scale-invariance tests |
| `tests/test_extract.py` | Extraction tests |
| `tests/test_labeling.py` | Labeling tests |
| `tests/test_ablation.py` | Ablation tests |
| `tests/test_splits.py` | LOSO tests |
| `tests/test_data.py` | Dataset and sampler tests |
| `tests/test_models.py` | Model construction tests |
| `tests/test_train.py` | Training summary tests |
| `tests/test_evaluate.py` | Metric and guard tests |
| `tests/test_interpret.py` | Grad-CAM tests |

**Deleted:** `patch.py`, `label.py`, `train.py`, `evaluate.py`, `gradcam.py`, `validate_labels.py`, `patient_level_prediction.py`, `test_model.py`, `visualize_samples.py`, `steatosis_model.pth`

**Modified:** `README.md`, `.gitignore`

**Unchanged and not read by any new code:** `data/`, `dataset/`, `results/` (existing artifacts are inputs to nothing; Task 14 relocates the data directories and Task 15 regenerates `results/`).

**Deviation from the spec.** Spec section 3.1 lists a `blobcount/report.py` for report and figure generation. This plan does not create it. Figure and summary generation lives in the module that owns the data it renders — `ablation.plot`, and the JSON and CSV writers in `train` and `evaluate`. A module that only forwards calls to other modules adds an indirection without a boundary, which is the opposite of what the spec asked for. The spec's remaining module list is implemented as written.

---

### Task 1: Package scaffolding and configuration

**Files:**
- Create: `pyproject.toml`
- Create: `configs/default.yaml`
- Create: `blobcount/__init__.py`
- Create: `blobcount/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `blobcount.config.Config` — with `REQUIRED_KEYS` as a module-level frozenset
  - `blobcount.config.load(path: Path | None = None, overrides: Sequence[str] = ()) -> Config`
  - `Config.get(key: str, default: Any = None) -> Any` — dot notation
  - `Config.set(key: str, value: Any) -> None`
  - `Config.path(key: str) -> Path` — applies `os.path.expandvars` then `os.path.expanduser` to the raw value, then resolves a still-relative value against the project root
  - `blobcount.config.set_seed(seed: int | None = None) -> None`
  - `blobcount.config.get_device() -> str` — returns `"cuda"` or `"cpu"`
  - `Config.unknown_keys() -> set[str]` — declared keys never read during load
  - Exception `ConfigError(Exception)`

- [ ] **Step 1: Write the failing tests**

`tests/test_config.py`:

```python
import pytest
from blobcount.config import Config, ConfigError, load, set_seed, get_device


def test_loads_declared_value():
    cfg = load()
    assert cfg.get("extraction.target_mpp") == 0.5
    assert cfg.get("extraction.patch_size") == 256


def test_unknown_top_level_section_raises():
    with pytest.raises(ConfigError, match="unknown section"):
        load(overrides=["not_a_section.key=1"])


def test_dot_notation_get_and_set():
    cfg = load()
    cfg.set("training.batch_size", 8)
    assert cfg.get("training.batch_size") == 8


def test_missing_required_key_raises(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("project:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key"):
        load(partial)


def test_config_with_no_unread_keys_on_load():
    cfg = load()
    assert cfg.unknown_keys() == set()


def test_paths_resolve_against_project_root():
    cfg = load()
    resolved = cfg.path("paths.results")
    assert resolved.is_absolute()
    assert resolved.name == "results"


def test_path_expands_environment_variables(monkeypatch):
    monkeypatch.setenv("BLOBCOUNT_TEST_ROOT", "/tmp/blobcount")
    cfg = load(overrides=["paths.results=%BLOBCOUNT_TEST_ROOT%/out"])
    assert str(cfg.path("paths.results")) == "/tmp/blobcount/out"


def test_path_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg = load(overrides=["paths.results=~/out"])
    assert cfg.path("paths.results") == tmp_path / "out"


def test_set_seed_makes_torch_deterministic():
    import torch
    set_seed(42)
    a = torch.randn(4)
    set_seed(42)
    assert torch.equal(a, torch.randn(4))


def test_get_device_returns_known_device():
    assert get_device() in {"cpu", "cuda"}
```

`REQUIRED_KEYS` is a module-level constant in `blobcount/config.py` listing the dotted keys every stage depends on: `project.seed`, `project.device`, `paths.slides`, `paths.patches`, `paths.labels`, `paths.results`, `paths.manifest`, `extraction.target_mpp`, `extraction.patch_size`, `extraction.step_size`, `blobs.area_um2_min`, `blobs.area_um2_max`, `blobs.circularity_min`, `blobs.gray_percentile`, `labeling.blob_threshold`, `training.architecture`, `training.num_classes`. `load` raises `ConfigError` listing every missing key.

`tests/conftest.py` provides `synthetic_patch` (a 256x256x3 `uint8` NumPy array, mid-gray background at value 180 with three white discs of radius 12 at recorded positions) and `tmp_output_tree` (a `tmp_path` with `patches/` and `labels/` subdirectories).

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount'`

- [ ] **Step 3: Create `pyproject.toml`**

Declares the package, requires Python `>=3.11`, pins the runtime dependencies listed in Tech Stack, and configures pytest with `testpaths = ["tests"]` and `filterwarnings = ["error"]`. Add `[tool.pytest.ini_options] addopts = "-q"`. The project is installed in editable mode with `pip install -e ".[dev]"`.

- [ ] **Step 4: Create `configs/default.yaml` with exactly these keys**

```yaml
project:
  name: liver-steatosis-detection
  seed: 42
  device: auto

paths:
  slides: data
  patches: dataset/patches
  labels: dataset/labels
  results: results
  manifest: data/manifest.yaml

extraction:
  target_mpp: 0.5
  patch_size: 256
  step_size: 256
  tissue_percentile: 5
  tissue_gray_max: 235

blobs:
  area_um2_min: 3.1
  area_um2_max: 125.0
  circularity_min: 0.6
  gray_percentile: 95
  morph_kernel_size: 3
  morph_iterations: 2
  stain_normalization: false

labeling:
  blob_threshold: 5

ablation:
  thresholds: [3, 5, 7, 10, 15, 20, 25, 30]

training:
  architecture: resnet18
  pretrained: true
  num_classes: 2
  input_size: 224
  batch_size: 32
  epochs: 10
  learning_rate: 0.0001
  patience_early_stop: 3
  patience_lr_scheduler: 3
  lr_scheduler_factor: 0.5
  use_weighted_sampler: true
  num_workers: 4

augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotation_degrees: 15
  color_jitter_brightness: 0.15
  color_jitter_contrast: 0.15
  color_jitter_saturation: 0.1
  color_jitter_hue: 0.05
  affine_translate: 0.05

evaluation:
  bootstrap_samples: 2000
  confidence_level: 0.95

gradcam:
  alpha: 0.4
  colormap: jet
```

`area_um2_min: 3.1` and `area_um2_max: 125.0` are the physical equivalents of the original 50–2000 px² filter evaluated at 0.25 µm/px, preserved so that the correction changes resolution handling and not the biological criterion. The config deliberately contains no key that no module reads.

- [ ] **Step 5: Implement `blobcount/config.py`**

`load()` reads the YAML, applies `key=value` overrides from `overrides` (parsed with `yaml.safe_load` on the value so types survive), raises `ConfigError` listing every entry of `REQUIRED_KEYS` that the file does not define, resolves every `paths.*` entry through `Config.path`, and records which declared keys were read. `unknown_keys()` returns declared dotted keys that were never requested through `get` or `set` during the load, so a task that stops reading a key makes it visible; a clean load of `configs/default.yaml` returns the empty set. The project root is the directory containing `pyproject.toml`, located by walking up from `__file__`. `set_seed()` seeds `random`, `numpy`, and `torch`, calls `torch.cuda.manual_seed_all` when CUDA is present, sets `torch.backends.cudnn.deterministic = True` and `benchmark = False`, and sets `PYTHONHASHSEED`. `get_device()` returns `config.get("project.device")` unless it is `auto`, in which case it probes `torch.cuda.is_available()`.

- [ ] **Step 6: Create `tests/conftest.py`**

Define the two fixtures described in Step 1. `synthetic_patch` returns a `namedtuple SyntheticPatch(array, disc_centres, disc_radius)` so tests can assert on ground truth.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS, 10 passed

- [ ] **Step 8: Verify the package installs and imports cleanly**

Run: `python -m pip install -e ".[dev]" && python -c "from blobcount.config import load; print(load().get('extraction.target_mpp'))"`
Expected: prints `0.5`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml configs/default.yaml blobcount/__init__.py blobcount/config.py tests/conftest.py tests/test_config.py
git commit -m "feat: add package scaffolding and validated configuration loader"
```

---

### Task 2: Dataset manifest and registry

**Files:**
- Create: `data/manifest.yaml`
- Create: `blobcount/registry.py`
- Create: `tests/test_registry.py`

**Interfaces:**
- Consumes: `blobcount.config.Config` from Task 1.
- Produces:
  - `blobcount.registry.Specimen` — frozen dataclass with fields `id: str`, `source: str`, `source_case_id: str | None`, `scanner: str | None`, `mpp: float | None`, `objective: int | None`, `stain: str | None`, `acquisition_date: str | None`, `label_source: str | None`, `label_detail: str | None`, `licence: str | None`, `ethics: str | None`, `slide_path: Path`
  - `blobcount.registry.load_registry(path: Path | None = None) -> list[Specimen]`
  - `blobcount.registry.usable_for_training(specimens: Sequence[Specimen]) -> list[Specimen]` — drops entries missing `source_case_id` or `licence`
  - `blobcount.registry.excluded(specimens: Sequence[Specimen]) -> list[Specimen]`
  - Exception `RegistryError(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from blobcount.registry import RegistryError, load_registry, usable_for_training, excluded


def test_loads_six_entries():
    reg = load_registry()
    assert len(reg) == 6
    assert {s.id for s in reg} == {"1", "2", "3", "4", "5", "6"}


def test_tcga_entries_carry_recovered_case_ids():
    reg = {s.id: s for s in load_registry()}
    assert reg["4"].source == "TCGA-LIHC"
    assert reg["4"].source_case_id == "TCGA-2V-A95S"
    assert reg["5"].source_case_id == "TCGA-DD-AAEH"
    assert reg["6"].source_case_id == "TCGA-GJ-A6C0"


def test_unknown_source_is_excluded_from_training():
    reg = load_registry()
    excluded_ids = {s.id for s in excluded(reg)}
    assert excluded_ids == {"1", "2", "3"}


def test_usable_set_is_non_empty():
    assert len(usable_for_training(load_registry())) == 3


def test_entry_without_licence_is_rejected(tmp_path):
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        "specimens:\n"
        "- id: X\n  source: local\n  source_case_id: X1\n  slide_path: x.svs\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="licence"):
        load_registry(bad)


def test_duplicate_ids_rejected(tmp_path):
    bad = tmp_path / "manifest.yaml"
    body = (
        "specimens:\n"
        "- id: X\n  source: local\n  source_case_id: A\n  licence: L\n  slide_path: a.svs\n"
        "- id: X\n  source: local\n  source_case_id: B\n  licence: L\n  slide_path: b.svs\n"
    )
    bad.write_text(body, encoding="utf-8")
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(bad)


def test_non_ascii_id_round_trips(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(
        "specimens:\n"
        "- id: \"hasta-ğü-01\"\n  source: local\n  source_case_id: C\n"
        "  licence: L\n  slide_path: c.svs\n",
        encoding="utf-8",
    )
    assert load_registry(p)[0].id == "hasta-ğü-01"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.registry'`

- [ ] **Step 3: Create `data/manifest.yaml`**

Six entries, keys `1` through `6`, each with the values recovered by reading the Aperio metadata of the corresponding slide. These are measured facts, not estimates:

| id | source | source_case_id | scanner | mpp | objective | stain | acquisition_date |
|----|--------|-----------------|---------|-----|-----------|-------|------------------|
| 1 | unknown | null | Aperio SS1352 | 0.4990 | 20 | H&E | 2013-05-23 |
| 2 | unknown | null | Aperio SS1302 | 0.2520 | 40 | H&E | 2013-02-28 |
| 3 | unknown | null | Aperio SS1302 | 0.2520 | 40 | H&E | 2012-01-27 |
| 4 | TCGA-LIHC | TCGA-2V-A95S | Aperio SS1764CNTLR | 0.2527 | 40 | H&E | 2015-07-01 |
| 5 | TCGA-LIHC | TCGA-DD-AAEH | Aperio SS1763CNTLR | 0.2525 | 40 | H&E | 2014-08-15 |
| 6 | TCGA-LIHC | TCGA-GJ-A6C0 | Aperio SS1436CNTLR | 0.2485 | 40 | H&E | 2013-05-08 |

For entries 1 to 3 also record `label_source: null` and a `provenance_note` stating that the slide was renamed to an integer during the graduation project, that the original identifier is not recoverable from the file, and that `aperio.DSR ID` on slide 1 is `gleason` while slides 2 and 3 are `ap1546-dsr`, which is unresolved. For entries 4 to 6 set `licence: "TCGA data use certification (dbGaP phs000178)"` and `ethics: "public, de-identified"`; `label_source: null` because TCGA-LIHC carries no steatosis grade.

Add a header comment recording that `aperio.Filename` and the Aperio metadata were the provenance source, and listing the `openslide.quickhash-1` value of each slide so that a future match against a public archive can be made without re-reading the files.

- [ ] **Step 4: Implement `blobcount/registry.py`**

`load_registry` parses the YAML, constructs `Specimen` for each entry, and raises `RegistryError` on a duplicate `id`, a missing required field, or a `licence` that is absent while `source` is not `unknown`. `usable_for_training` returns entries where `source_case_id` and `licence` are both truthy. `excluded` returns the complement. `slide_path` is resolved against the manifest's parent directory so that callers never build paths by concatenation.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_registry.py -v`
Expected: PASS, 7 passed

- [ ] **Step 6: Verify the recovered provenance against GDC**

Run: `python -m pytest tests/test_registry.py -k tcga -v`
Expected: PASS — the assertion pins `TCGA-LIHC` and the three case IDs, so a manifest edit that loses them fails.

- [ ] **Step 7: Commit**

```bash
git add data/manifest.yaml blobcount/registry.py tests/test_registry.py
git commit -m "feat: add dataset manifest with recovered TCGA-LIHC provenance"
```

---

### Task 3: Slide metadata and resolution planning

**Files:**
- Create: `blobcount/slides.py`
- Create: `tests/test_slides.py`

**Interfaces:**
- Consumes: `blobcount.config.Config` (`extraction.target_mpp`, `extraction.patch_size`, `extraction.step_size`).
- Produces:
  - `blobcount.slides.SlideInfo` — frozen dataclass: `slide_id: str`, `path: Path`, `width: int`, `height: int`, `mpp: float`, `objective: int | None`, `level: int`, `level_downsample: float`, `read_px: int`, `achieved_mpp: float`
  - `blobcount.slides.read_mpp(slide) -> float` — `openslide.mpp-x`, then `aperio.MPP`, else raise
  - `blobcount.slides.plan_resolution(slide_mpp: float, target_mpp: float, patch_size: int, level_downsamples: Sequence[float]) -> tuple[int, float, int, float]` — returns `(level, level_downsample, read_px, achieved_mpp)`
  - `blobcount.slides.open_info(path: Path, slide_id: str, target_mpp: float, patch_size: int) -> SlideInfo`
  - `blobcount.slides.iter_coords(info: SlideInfo, patch_size: int, step_size: int) -> Iterator[tuple[int, int]]`
  - Exception `SlideError(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from blobcount.slides import SlideError, iter_coords, plan_resolution, read_mpp

# level downsamples measured from all six slides: 1.0, ~4.0, ~16.0
PYRAMID = (1.0, 4.0002, 16.0043)


def test_near_target_slide_is_read_at_level_zero():
    level, ds, read_px, achieved = plan_resolution(0.4990, 0.5, 256, PYRAMID)
    assert (level, read_px) == (0, 257)
    assert achieved == pytest.approx(0.4990, rel=0.01)


def test_finer_slide_is_downsampled_to_the_target_field_of_view():
    # 0.252 um/px is finer than the 0.5 um/px target, so more source pixels are
    # read than the patch width and the region is shrunk. Reading only 256 source
    # pixels here would upsample by 2x and silently change the field of view.
    level, ds, read_px, achieved = plan_resolution(0.2520, 0.5, 256, PYRAMID)
    assert level == 0
    assert read_px == 508
    assert achieved == pytest.approx(0.5, rel=0.01)


def test_coarser_slide_is_upsampled_and_achieves_the_target():
    level, ds, read_px, achieved = plan_resolution(4.0, 0.5, 256, PYRAMID)
    assert level == 0
    assert read_px == 32
    assert achieved == pytest.approx(0.5, rel=0.01)


@pytest.mark.parametrize("slide_mpp,expected_read_px,expected_achieved", [
    (0.4990, 257, 0.5010),
    (0.2520, 508, 0.5002),
    (0.2527, 507, 0.5004),
    (0.2525, 507, 0.5000),
    (0.2485, 515, 0.4999),
])
def test_measured_slides_hit_expected_read_size(slide_mpp, expected_read_px, expected_achieved):
    _, _, read_px, achieved = plan_resolution(slide_mpp, 0.5, 256, PYRAMID)
    assert read_px == expected_read_px
    assert achieved == pytest.approx(expected_achieved, abs=5e-4)


def test_all_measured_slides_within_two_percent_of_target():
    for mpp in (0.4990, 0.2520, 0.2520, 0.2527, 0.2525, 0.2485):
        _, _, _, achieved = plan_resolution(mpp, 0.5, 256, PYRAMID)
        assert abs(achieved - 0.5) / 0.5 < 0.02


def test_missing_resolution_metadata_raises():
    class Fake:
        properties = {"aperio.AppMag": "40"}
    with pytest.raises(SlideError, match="resolution"):
        read_mpp(Fake())


def test_iter_coords_skips_incomplete_edge_tiles():
    info = type("I", (), {"width": 600, "height": 300})()
    coords = list(iter_coords(info, 256, 256))
    assert coords == [(0, 0), (256, 0)]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_slides.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.slides'`

- [ ] **Step 3: Implement `blobcount/slides.py` resolution planning**

A patch of `patch_size` output pixels at `target_mpp` covers `patch_size * target_mpp` microns of tissue. A level-`L` source pixel covers `slide_mpp * level_downsample(L)` microns. Therefore the number of source pixels to read is

```
required_downsample = target_mpp / slide_mpp
level               = the pyramid level minimizing |level_downsample - required_downsample|
read_px             = round(patch_size * target_mpp / (slide_mpp * level_downsample))
achieved_mpp        = read_px * slide_mpp * level_downsample / patch_size
```

When `slide_mpp` is finer than `target_mpp` the required downsample is greater than 1, `read_px` exceeds `patch_size`, and the region is shrunk; when `slide_mpp` is coarser, `read_px` is smaller than `patch_size` and the region is enlarged. Both cases land on `target_mpp` up to the integer rounding of `read_px`, and that rounding is the only source of deviation. The direction of this formula is the whole point: reading exactly `patch_size` source pixels from a slide finer than the target would upsample and change the field of view, which is the defect being corrected.

`read_mpp` reads `openslide.mpp-x` as a float, falls back to `aperio.MPP`, and raises `SlideError` naming the slide and the two property names it looked for when neither is present. It never returns a default.

`iter_coords` yields `(x, y)` in row-major order and stops a row early when `x + patch_size > width`, mirroring the bounds check in the original `patch.py` so that no partial tile is ever produced.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_slides.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Verify the plan against the real slides**

Run: `python -c "from pathlib import Path; from blobcount.slides import open_info; [print(open_info(p, p.stem, 0.5, 256)) for p in sorted(Path('data').glob('*.svs'))]"`
Expected: six lines, each with `achieved_mpp` between 0.490 and 0.510

- [ ] **Step 6: Commit**

```bash
git add blobcount/slides.py tests/test_slides.py
git commit -m "feat: add resolution planning that normalizes field of view across slides"
```

---

### Task 4: Patch extraction

**Files:**
- Create: `blobcount/extract.py`
- Create: `bin/extract_patches.py`
- Create: `tests/test_extract.py`

**Interfaces:**
- Consumes: `blobcount.slides.SlideInfo`, `open_info`, `iter_coords` from Task 3; `Config` from Task 1; `Specimen` from Task 2.
- Produces:
  - `blobcount.extract.tissue_gray_threshold(patch: np.ndarray) -> float` — the 5th percentile of the patch's grayscale, the background level
  - `blobcount.extract.is_tissue(patch: np.ndarray, threshold: float, gray_max: float) -> bool`
  - `blobcount.extract.read_patch(slide, info: SlideInfo, x: int, y: int) -> np.ndarray | None` — `x` and `y` are **level-0 source pixels**, exactly as yielded by `iter_coords`; `None` when the read returns fewer than `read_px` pixels on either axis
  - `blobcount.extract.extract_slide(specimen: Specimen, out_dir: Path, cfg: Config) -> ExtractionStats`
  - `blobcount.extract.prepare_output(out_dir: Path) -> None` — deletes and recreates the directory; used by extraction and, later, by labeling
  - `blobcount.extract.ExtractionStats` — frozen dataclass: `slide_id: str`, `patches_written: int`, `tissue_rejected: int`, `read_failures: int`, `skipped_blank: bool`
  - `blobcount.extract.write_index(rows: Sequence[dict], path: Path) -> None` — CSV with header `patch_id,slide_id,x,y,mpp,tissue_gray`
  - `bin/extract_patches.py` — CLI accepting `--slide` (repeatable) and `--force`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from blobcount.extract import is_tissue, tissue_gray_threshold, write_index


def test_background_level_is_low_percentile(synthetic_patch):
    assert 150 <= tissue_gray_threshold(synthetic_patch.array) <= 200


def test_white_disc_patch_is_tissue(synthetic_patch):
    assert is_tissue(synthetic_patch.array, tissue_gray_threshold(synthetic_patch.array), 235)


def test_blank_patch_is_not_tissue():
    blank = np.full((256, 256, 3), 250, dtype=np.uint8)
    assert not is_tissue(blank, tissue_gray_threshold(blank), 235)


def test_blank_slide_threshold_is_not_nan():
    blank = np.full((256, 256, 3), 250, dtype=np.uint8)
    assert not np.isnan(tissue_gray_threshold(blank))


def test_index_round_trips_non_ascii_slide_id(tmp_path):
    rows = [{"patch_id": "p1", "slide_id": "hasta-ğü-01", "x": 0, "y": 0,
             "mpp": 0.5, "tissue_gray": 181.0}]
    out = tmp_path / "patches.csv"
    write_index(rows, out)
    assert "hasta-ğü-01" in out.read_text(encoding="utf-8")


def test_extraction_clears_output_directory(tmp_output_tree):
    from blobcount.extract import prepare_output
    stale = tmp_output_tree / "patches" / "stale.png"
    stale.write_bytes(b"x")
    prepare_output(tmp_output_tree / "patches")
    assert not stale.exists()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.extract'`

- [ ] **Step 3: Implement `blobcount/extract.py`**

`tissue_gray_threshold` returns `float(np.percentile(grayscale, 5))` and is total: for a constant image the percentile equals the constant, so the value is never `NaN`, which is what pins Review Focus item 4. `is_tissue` returns `True` when the patch's mean grayscale is below `gray_max` and the patch is not uniformly background, that is when at least one pixel is darker than `threshold + 20`.

`read_patch` calls `slide.read_region((x, y), info.level, (info.read_px, info.read_px))` with `x` and `y` passed through untransformed, because `iter_coords` yields level-0 source coordinates. It converts to an RGB array and resizes to `extraction.patch_size` with `cv2.INTER_AREA`.

**An out-of-bounds `read_region` does not raise.** Verified on `data/2.svs` at an origin 5000 px past the right edge: it returns a 16x16 RGBA image that is entirely `[0, 0, 0, 0]`, where an in-bounds read of the same size has 77 distinct colours. A fully transparent patch converts to solid black, whose mean grayscale is 0 and therefore **passes** `is_tissue`. A coordinate bug therefore does not announce itself — it writes black patches that are then labelled. `read_patch` must reject a read whose alpha channel is entirely zero, treating it as a `read_failure` and counting it, in addition to the size check. This is the only guard between a grid arithmetic error and a silently corrupted dataset.

`extract_slide` iterates `iter_coords`, and **must** call
`open_info(specimen.slide_path, specimen.id, target_mpp, patch_size,
expected_mpp=specimen.mpp)`, passing the manifest's recorded resolution so that
`open_info` cross-checks it against the slide's own `openslide.mpp-x` and raises
on divergence beyond 1e-3 relative. Omitting `expected_mpp` is a defect, not a
default: the cross-check is the only thing that detects the manifest ceasing to
describe the file feeding the patient-level split, and the parameter is opt-in by
construction, so this call site is what makes it real. It then calls `read_patch`, counts `read_failures` on `None`, calls `is_tissue` and counts `tissue_rejected`, and on acceptance writes a PNG named `{slide_id}_{x}_{y}.png` where `x` and `y` are the source coordinates from `iter_coords`, so the filename records where on the slide the patch came from. Every exception is caught with `except Exception as exc` and logged with slide id, coordinates, and `exc`. `prepare_output` deletes and recreates the directory. `write_index` writes UTF-8 with the pinned header.

- [ ] **Step 4: Create `bin/extract_patches.py`**

A thin entry point: parse `--slide` and `--force`, call `load_config`, `load_registry`, filter to the requested specimens, call `extract_slide` for each, write `patches.csv` under `paths.patches`, print the per-slide `ExtractionStats` as a table, and exit non-zero if any slide was skipped as blank.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_extract.py -v`
Expected: PASS, 6 passed

- [ ] **Step 6: Commit**

```bash
git add blobcount/extract.py bin/extract_patches.py tests/test_extract.py
git commit -m "feat: add patch extraction at normalized resolution with failure accounting"
```

---

### Task 5: Blob counting in physical units

**Files:**
- Create: `blobcount/blobs.py`
- Create: `tests/test_blobs.py`

**Interfaces:**
- Consumes: `Config` (`blobs.*`) from Task 1.
- Produces:
  - `blobcount.blobs.BlobParams` — frozen dataclass: `area_um2_min: float`, `area_um2_max: float`, `circularity_min: float`, `gray_threshold: float`, `morph_kernel_size: int`, `morph_iterations: int`
  - `blobcount.blobs.area_px2(area_um2: float, mpp: float) -> float` — returns `area_um2 / (mpp ** 2)`
  - `blobcount.blobs.gray_threshold_from_patch(patch: np.ndarray, percentile: float) -> float`
  - `blobcount.blobs.count_blobs(patch: np.ndarray, params: BlobParams, mpp: float) -> int`
  - `blobcount.blobs.normalize_stain(patch: np.ndarray) -> np.ndarray` — Reinhard colour normalization in Lab space toward fixed per-channel target means and standard deviations `(128, 128, 128)` and `(60, 60, 60)`

- [ ] **Step 1: Write the failing tests**

```python
import cv2
import numpy as np
import pytest
from blobcount.blobs import BlobParams, area_px2, count_blobs, gray_threshold_from_patch


def _discs(n, radius, size=256):
    img = np.full((size, size, 3), 180, dtype=np.uint8)
    for i in range(n):
        cv2.circle(img, (60 + i * 90, 128), radius, (250, 250, 250), -1)
    return img


def _params(threshold=200.0):
    return BlobParams(area_um2_min=3.1, area_um2_max=125.0, circularity_min=0.6,
                      gray_threshold=threshold, morph_kernel_size=3, morph_iterations=2)


def test_area_conversion_matches_physical_units():
    # 50 px^2 at 0.25 um/px is 3.1 um^2; the same physical area at 0.5 um/px is 12.4 px^2
    assert area_px2(3.1, 0.25) == pytest.approx(49.6, rel=0.01)
    assert area_px2(3.1, 0.5) == pytest.approx(12.4, rel=0.01)


def test_counts_exact_number_of_discs():
    assert count_blobs(_discs(3, 12), _params(), 0.5) == 3


def test_empty_patch_counts_zero():
    assert count_blobs(np.full((256, 256, 3), 180, dtype=np.uint8), _params(), 0.5) == 0


def test_same_physical_disc_counts_same_at_two_resolutions():
    # a disc of 6 um radius is 24 px at 0.25 um/px and 12 px at 0.5 um/px
    coarse = _discs(4, 24, size=512)
    fine = _discs(4, 12, size=256)
    assert count_blobs(coarse, _params(), 0.25) == count_blobs(fine, _params(), 0.5) == 4


def test_mpp_argument_actually_changes_the_area_filter():
    # the same 12 px disc is 2.8 um^2 at 0.25 um/px (rejected, below the 3.1 floor)
    # and 11.3 um^2 at 0.5 um/px (accepted). If the mpp argument were ignored the
    # two counts would be equal, which is the original defect in miniature.
    assert count_blobs(_discs(3, 12), _params(), 0.25) == 0
    assert count_blobs(_discs(3, 12), _params(), 0.5) == 3


def test_disc_outside_area_range_is_rejected():
    # radius 40 px at 0.5 um/px is 1257 um^2, far above the 125 um^2 ceiling
    assert count_blobs(_discs(2, 40), _params(), 0.5) == 0


def test_gray_threshold_from_patch_is_a_high_percentile():
    p = _discs(3, 12)
    t = gray_threshold_from_patch(p, 95.0)
    assert 200 <= t <= 250


def test_gray_threshold_is_constant_for_uniform_patch():
    flat = np.full((256, 256, 3), 180, dtype=np.uint8)
    assert gray_threshold_from_patch(flat, 95.0) == 180.0


def test_stain_normalization_changes_mean_toward_target():
    from blobcount.blobs import normalize_stain
    src = np.full((64, 64, 3), (140, 90, 160), dtype=np.uint8)
    out = normalize_stain(src)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_blobs.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.blobs'`

- [ ] **Step 3: Implement `blobcount/blobs.py`**

`count_blobs` converts the patch to grayscale, applies `cv2.threshold(gray, params.gray_threshold, 255, cv2.THRESH_BINARY)`, runs `cv2.morphologyEx` with `MORPH_OPEN` using a `np.ones((k, k), np.uint8)` kernel for `morph_iterations`, finds external contours with `RETR_EXTERNAL` and `CHAIN_APPROX_SIMPLE`, and accepts a contour when `area_px2_min < cv2.contourArea(cnt) < area_px2_max` and `4 * pi * area / perimeter**2 > circularity_min`, where `area_px2_min = area_px2(params.area_um2_min, mpp)` and `area_px2_max = area_px2(params.area_um2_max, mpp)` are computed from the `mpp` argument. The `mpp` parameter is required, not optional, which is what prevents Review Focus item 3.

`normalize_stain` converts BGR to Lab, shifts each channel so its mean matches the target `(128, 128, 128)`, and scales each channel by `target_std / observed_std`. The scale factor must be computed as `np.where(observed_std > 1e-6, target_std / np.where(observed_std > 1e-6, observed_std, 1.0), 1.0)`, because a uniform input image has zero per-channel standard deviation and a bare division would emit a `RuntimeWarning` that the project's `filterwarnings = ["error"]` setting turns into a test failure. `normalize_stain` then clips, converts back to BGR, and returns `uint8`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_blobs.py -v`
Expected: PASS, 8 passed

- [ ] **Step 5: Verify the counter on real slide material**

Run: `python -c "import cv2, openslide; from blobcount.slides import open_info, read_patch; from blobcount.blobs import BlobParams, gray_threshold_from_patch, count_blobs; i = open_info(__import__('pathlib').Path('data/4.svs'), '4', 0.5, 256); s = openslide.OpenSlide(str(i.path)); p = read_patch(s, i, i.width // 2, i.height // 2); t = gray_threshold_from_patch(p, 95); print('achieved_mpp', round(i.achieved_mpp, 4), 'gray_p95', round(t, 1), 'blobs', count_blobs(p, BlobParams(3.1, 125.0, 0.6, t, 3, 2), i.achieved_mpp))"`
Expected: `achieved_mpp` between 0.495 and 0.505, `gray_p95` between 200 and 255, and an integer blob count

- [ ] **Step 6: Commit**

```bash
git add blobcount/blobs.py tests/test_blobs.py
git commit -m "fix: express blob area thresholds in square microns instead of pixels"
```

---

### Task 6: Labeling

**Files:**
- Create: `blobcount/labeling.py`
- Create: `bin/label_patches.py`
- Create: `tests/test_labeling.py`

**Interfaces:**
- Consumes: `count_blobs`, `BlobParams`, `gray_threshold_from_patch` from Task 5; `Specimen` from Task 2; `Config` (`labeling.blob_threshold`, `blobs.*`, `extraction.patch_size`).
- Produces:
  - `blobcount.labeling.read_index(path: Path) -> list[dict]`
  - `blobcount.labeling.load_patch(path: Path, size: int) -> np.ndarray` — the single seam through which the module reads image files, so tests can substitute it
  - `blobcount.labeling.label_patches(index_path: Path, out_dir: Path, cfg: Config) -> LabelingStats`
  - `blobcount.labeling.LabelingStats` — frozen dataclass: `blob_threshold: int`, `per_slide: dict[str, dict[str, int]]` keyed by slide then class plus an `n` count, `per_slide_mpp: dict[str, float]`, `per_slide_gray_threshold: dict[str, float]`, `duplicates_removed: int`
  - `bin/label_patches.py` — CLI accepting `--index` and `--threshold`

`label_patches` additionally writes `out_dir/labels.csv` with header `patch_id,slide_id,label,blob_count,mpp,path`, one row per patch, where `label` is `0` for normal and `1` for steatosis and `path` is the written file's path. This CSV is the only handoff from labeling to training, ablation, and evaluation; those three tasks read it rather than re-deriving labels from the directory layout.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from blobcount.labeling import read_index


def test_index_reads_utf8_slide_ids(tmp_path):
    p = tmp_path / "patches.csv"
    p.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "1_hasta-ğü-01_0_0,hasta-ğü-01,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    assert read_index(p)[0]["slide_id"] == "hasta-ğü-01"
    assert float(read_index(p)[0]["mpp"]) == pytest.approx(0.5010)


def test_relabeling_leaves_no_patch_in_both_classes(tmp_output_tree, monkeypatch):
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5,181.0\n"
        "B_0_0,B,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    for threshold in (5, 25):
        labeling.label_patches(idx, out, _cfg_with_threshold(threshold))
    names = [f"{f.parent.name}/{f.name}" for f in out.rglob("*.png")]
    assert len(names) == len(set(names))


def test_per_patch_mpp_is_used_not_a_global(tmp_output_tree, monkeypatch):
    # Two patches of the same slide at different achieved mpp must be thresholded
    # with their own value; a global value silently reintroduces the original defect.
    from blobcount import labeling
    seen = []
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs",
                        lambda patch, params, mpp: seen.append(mpp) or 0)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5010,181.0\n"
        "A_256_0,A,256,0,0.5015,181.0\n",
        encoding="utf-8",
    )
    labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    assert seen == [pytest.approx(0.5010), pytest.approx(0.5015)]


def test_stats_json_is_written(tmp_output_tree, monkeypatch):
    import json
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    written = json.loads((tmp_output_tree / "labels" / "labeling_stats.json").read_text("utf-8"))
    assert written["blob_threshold"] == 5
    assert written["per_slide"]["A"]["n"] == 1


def test_labels_csv_carries_path_label_and_mpp(tmp_output_tree, monkeypatch):
    import csv
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs", lambda patch, params, mpp: 7)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    labeling.label_patches(idx, out, _cfg_with_threshold(5))
    with (out / "labels.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["patch_id"] == "A_0_0"
    assert rows[0]["slide_id"] == "A"
    assert int(rows[0]["label"]) == 1
    assert rows[0]["blob_count"] == "7"
    assert float(rows[0]["mpp"]) == pytest.approx(0.5010)
    assert rows[0]["path"].endswith("steatosis/A_0_0.png")
```

`_cfg_with_threshold(t)` is a module-level helper in `tests/test_labeling.py` that calls `blobcount.config.load()` and sets `labeling.blob_threshold` to `t`, returning the `Config`. The tests supply image data by monkeypatching `blobcount.labeling.load_patch`, so no image files are needed on disk.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.labeling'`

- [ ] **Step 3: Implement `blobcount/labeling.py`**

`label_patches` calls `prepare_output` (imported from `blobcount.extract`) on `out_dir` before writing anything, which is what pins Review Focus item 5. It reads the index, groups rows by `slide_id`, derives each slide's adaptive gray threshold from its `tissue_gray` values at `blobs.gray_percentile`, constructs `BlobParams` per slide, and for each row calls `load_patch` then `count_blobs(patch, params, mpp=float(row["mpp"]))` using that row's own recorded value, copying the file into `out_dir/<class>/<patch_id>.png` where class is `steatosis` when the count is at or above `cfg.get("labeling.blob_threshold")` and `normal` otherwise. It accumulates `per_slide` counts keyed by slide then class, `per_slide_mpp`, and `per_slide_gray_threshold`, writes `out_dir/labels.csv` with header `patch_id,slide_id,label,blob_count,mpp,path`, scans the output tree afterwards for any filename present in both class directories, removes duplicates, and counts them in `duplicates_removed`. Finally it writes `out_dir/labeling_stats.json` containing the blob threshold, the per-slide counts, the per-slide achieved resolution, the per-slide gray threshold, and the duplicate count, so that every label is traceable to the parameters that produced it.

- [ ] **Step 4: Create `bin/label_patches.py`**

Parse `--index` and `--threshold`, call `label_patches`, print the per-slide count table and the per-slide achieved resolution as a table, and exit non-zero if `duplicates_removed` is greater than zero.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_labeling.py -v`
Expected: PASS, 5 passed

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add blobcount/labeling.py bin/label_patches.py tests/test_labeling.py
git commit -m "fix: clear label output before writing and threshold each patch at its own resolution"
```

---

### Task 7: Threshold ablation with per-specimen breakdown

**Files:**
- Create: `blobcount/ablation.py`
- Create: `bin/ablate_threshold.py`
- Create: `tests/test_ablation.py`

**Interfaces:**
- Consumes: `read_index` from Task 6; `count_blobs`, `BlobParams` from Task 5; `Config` (`ablation.thresholds`, `blobs.*`).
- Produces:
  - `blobcount.ablation.sweep(counts_by_patch: Mapping[str, int], slide_of: Mapping[str, str], thresholds: Sequence[int]) -> dict[int, dict]`
  - `blobcount.ablation.summarise(per_threshold: dict[int, dict]) -> dict` — pooled and per-slide steatosis ratios, plus the spread of per-slide ratios at each threshold
  - `blobcount.ablation.plot(summarised: dict, out_path: Path) -> None` — two panels: per-slide ratio curves over thresholds, and a per-slide bar chart at the selected threshold
  - `bin/ablate_threshold.py` — CLI accepting `--index` and `--selected`

- [ ] **Step 1: Write the failing test**

```python
def test_pooled_balance_hides_per_slide_spread():
    from blobcount.ablation import sweep, summarise
    # 100 patches on slide A all high, 100 on slide B all low
    counts, slide_of = {}, {}
    for i in range(100):
        counts[f"A{i}"] = 20; slide_of[f"A{i}"] = "A"
        counts[f"B{i}"] = 0;  slide_of[f"B{i}"] = "B"
    res = sweep(counts, slide_of, [5])
    s = summarise(res)
    assert s[5]["pooled_steatosis_ratio"] == pytest.approx(0.5)
    assert s[5]["per_slide_ratio"]["A"] == pytest.approx(1.0)
    assert s[5]["per_slide_ratio"]["B"] == pytest.approx(0.0)
    assert s[5]["per_slide_ratio_spread"] == pytest.approx(1.0)


def test_threshold_appears_in_output():
    from blobcount.ablation import sweep, summarise
    counts = {f"p{i}": i for i in range(40)}
    slide_of = {f"p{i}": "A" for i in range(40)}
    s = summarise(sweep(counts, slide_of, [3, 5, 7]))
    assert set(s) == {3, 5, 7}
    assert s[5]["n_steatosis"] == 35
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_ablation.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.ablation'`

- [ ] **Step 3: Implement `blobcount/ablation.py`**

`sweep` returns, for each threshold, `n_normal`, `n_steatosis`, `pooled_steatosis_ratio`, `per_slide_ratio` mapping slide id to that slide's steatosis fraction, and `per_slide_ratio_spread` as the max minus min of the per-slide ratios. `summarise` assembles the same structure from `sweep` output and adds the selected threshold. `plot` writes a two-panel matplotlib figure.

- [ ] **Step 4: Create `bin/ablate_threshold.py`**

Recompute blob counts from the index (it does not read the labels produced by Task 6, because the sweep must be independent of the chosen threshold), call `sweep` and `summarise`, write `results/label_analysis/threshold_ablation.json` and `threshold_ablation.png`, and print the per-slide ratio table at the selected threshold. The printed table is the deliverable that shows the magnification artifact is gone.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/test_ablation.py -v`
Expected: PASS, 2 passed

- [ ] **Step 6: Commit**

```bash
git add blobcount/ablation.py bin/ablate_threshold.py tests/test_ablation.py
git commit -m "feat: report per-specimen class distribution alongside the pooled ratio"
```

---

### Task 8: Leave-one-specimen-out splits

**Files:**
- Create: `blobcount/splits.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `blobcount.splits.loso_folds(specimen_of_row: Sequence[str]) -> list[Fold]` — takes one specimen id per dataset row, in dataset order, and returns one fold per distinct specimen
  - `blobcount.splits.Fold` — frozen dataclass: `fold: int`, `held_out: str`, `train_idx: np.ndarray`, `val_idx: np.ndarray`
  - Exception `SplitError(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from blobcount.splits import SplitError, loso_folds


def test_six_specimens_give_six_folds():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    assert len(folds) == 6


def test_each_specimen_is_held_out_exactly_once():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    assert sorted(f.held_out for f in folds) == ["1", "2", "3", "4", "5", "6"]


def test_train_and_val_are_disjoint_and_complementary():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    for f in folds:
        assert set(f.train_idx) & set(f.val_idx) == set()
        assert len(f.train_idx) + len(f.val_idx) == 6


def test_fold_order_is_deterministic():
    a = loso_folds(["3", "1", "2"])
    b = loso_folds(["1", "2", "3"])
    assert [f.held_out for f in a] == [f.held_out for f in b] == ["1", "2", "3"]


def test_single_specimen_raises():
    with pytest.raises(SplitError, match="at least two"):
        loso_folds(["only"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_splits.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.splits'`

- [ ] **Step 3: Implement `blobcount/splits.py`**

`loso_folds` takes specimen ids, requires at least two distinct values, sorts them with `sorted()` for determinism, and for each produces `val_idx` as the indices of every row belonging to that specimen and `train_idx` as the complement, both as `np.ndarray` of `int64`. It raises `SplitError` when fewer than two distinct specimens are supplied. Note that the indices are positions in the caller's array, so the caller must pass per-row specimen ids, not a deduplicated list; the signature says `Sequence[str]` and the docstring must state that it expects one id per row.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_splits.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add blobcount/splits.py tests/test_splits.py
git commit -m "feat: add leave-one-specimen-out split construction"
```

---

### Task 9: Dataset, transforms, and per-fold class weights

**Files:**
- Create: `blobcount/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: `Config` (`training.*`, `augmentation.*`).
- Produces:
  - `blobcount.data.PatchDataset(Dataset)` — `__init__(paths: Sequence[Path], labels: Sequence[int], transform=None)`, `__getitem__` returns `(Tensor, int)`
  - `blobcount.data.build_transforms(cfg: Config) -> tuple[Callable, Callable]` — returns `(train_transform, eval_transform)`
  - `blobcount.data.class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor`
  - `blobcount.data.build_sampler(labels: Sequence[int], enabled: bool, seed: int) -> WeightedRandomSampler | None`
  - `blobcount.data.build_loader(dataset, batch_size, sampler, shuffle, num_workers, seed) -> DataLoader`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch


def test_class_weights_use_only_the_labels_passed():
    from blobcount.data import class_weights
    w = class_weights([0, 0, 0, 1])
    assert w.tolist() == pytest.approx([4 / 6, 4 / 2])


def test_class_weights_differ_when_given_a_different_subset():
    from blobcount.data import class_weights
    a = class_weights([0] * 90 + [1] * 10)
    b = class_weights([0] * 50 + [1] * 50)
    assert a[1] > b[1]


def test_sampler_balances_classes():
    from blobcount.data import build_sampler
    labels = [0] * 90 + [1] * 10
    s = build_sampler(labels, enabled=True, seed=42)
    drawn = list(iter(s))
    assert abs(sum(drawn) / len(drawn) - 0.1) < 0.05


def test_sampler_disabled_returns_none():
    from blobcount.data import build_sampler
    assert build_sampler([0, 1], enabled=False, seed=42) is None


def test_eval_transform_has_no_randomness():
    from blobcount.data import build_transforms
    from blobcount.config import load
    _, ev = build_transforms(load())
    import numpy as np
    from PIL import Image
    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    assert torch.equal(ev(img), ev(img))


def test_loader_is_deterministic_given_seed(tmp_path):
    import numpy as np
    from PIL import Image
    from blobcount.data import PatchDataset, build_loader
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(p)
        paths.append(p)
    ds = PatchDataset(paths, [0, 1])
    a = build_loader(ds, 2, None, True, 0, 7)
    b = build_loader(ds, 2, None, True, 0, 7)
    assert [y for _, y in a] == [y for _, y in b]


def test_loader_shuffle_is_disabled_when_a_sampler_is_supplied():
    from blobcount.data import build_loader, PatchDataset, build_sampler
    from pathlib import Path
    ds = PatchDataset([Path("a.png")], [0])
    loader = build_loader(ds, 1, build_sampler([0], True, 1), True, 0, 1)
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler) is False
    assert loader.sampler is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.data'`

- [ ] **Step 3: Implement `blobcount/data.py`**

`class_weights` implements `total / (num_classes * count[c])` over the labels it is given and returns a `float32` tensor — the fix for the original defect, where weights were computed once from the pooled dataset and therefore saw the held-out specimens' label distribution. `build_transforms` composes `Resize((input_size, input_size))` with the augmentation keys for train, and `Resize`, `ToTensor`, and ImageNet `Normalize` for eval. `build_sampler` returns a `WeightedRandomSampler` with `replacement=True`, `num_samples=len(labels)`, and a generator seeded from `seed`, or `None` when disabled. `build_loader` passes `shuffle=False` whenever a sampler is supplied, and seeds a `torch.Generator` for the worker seeds.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_data.py -v`
Expected: PASS, 6 passed

- [ ] **Step 5: Commit**

```bash
git add blobcount/data.py tests/test_data.py
git commit -m "fix: compute class weights per fold from training labels only"
```

---

### Task 10: Model construction

**Files:**
- Create: `blobcount/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `Config` (`training.architecture`, `training.num_classes`, `training.pretrained`).
- Produces:
  - `blobcount.models.build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module`
  - `blobcount.models.SUPPORTED` — tuple of supported architecture names

- [ ] **Step 1: Write the failing tests**

```python
import pytest
import torch


def test_resnet18_head_matches_num_classes():
    from blobcount.models import build_model
    m = build_model("resnet18", 2, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_resnet50_head_matches_num_classes():
    from blobcount.models import build_model
    m = build_model("resnet50", 2, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_efficientnet_b0_head_matches_num_classes():
    from blobcount.models import build_model
    m = build_model("efficientnet_b0", 3, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 3)


def test_unknown_architecture_raises():
    from blobcount.models import build_model
    with pytest.raises(ValueError, match="unsupported"):
        build_model("vgg16", 2, pretrained=False)


def test_supported_lists_the_configured_default():
    from blobcount.models import SUPPORTED
    from blobcount.config import load
    assert load().get("training.architecture") in SUPPORTED
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.models'`

- [ ] **Step 3: Implement `blobcount/models.py`**

`SUPPORTED = ("resnet18", "resnet50", "efficientnet_b0")`. `build_model` dispatches to `torchvision.models.resnet18`, `resnet50`, or `efficientnet_b0` with `weights="DEFAULT"` when `pretrained` is true and `weights=None` otherwise, replaces the final linear layer with `nn.Linear(in_features, num_classes)` for the ResNets and `model.classifier[1]` for EfficientNet, and raises `ValueError` listing `SUPPORTED` otherwise.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS, 5 passed

- [ ] **Step 5: Commit**

```bash
git add blobcount/models.py tests/test_models.py
git commit -m "feat: add configurable model construction with a supported-architecture guard"
```

---

### Task 11: Cross-validation training with epoch selection

**Files:**
- Create: `blobcount/train.py`
- Create: `bin/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Consumes: `loso_folds`, `Fold` from Task 8; `PatchDataset`, `build_transforms`, `class_weights`, `build_sampler`, `build_loader` from Task 9; `build_model` from Task 10; `Config` (`training.*`, `augmentation.*`).
- Produces:
  - `blobcount.train.EpochRecord` — frozen dataclass: `fold: int`, `epoch: int`, `train_loss: float`, `val_loss: float`, `val_f1: float`, `val_accuracy: float`, `precision: float`, `recall: float`, `tp: int`, `fp: int`, `fn: int`, `tn: int`, `lr: float`
  - `blobcount.train.select_epoch(records: Sequence[EpochRecord]) -> EpochRecord` — the record with the highest `val_f1`, earliest epoch on a tie
  - `blobcount.train.summarise_folds(selected: Sequence[EpochRecord]) -> dict` — mean and standard deviation of accuracy, precision, recall, and F1 across folds, all taken from the selected epochs
  - `blobcount.train.train_fold(fold: Fold, paths, labels, cfg: Config, out_dir: Path) -> tuple[EpochRecord, list[EpochRecord]]`
  - `blobcount.train.run_cross_validation(paths, labels, specimen_ids, cfg: Config, out_dir: Path) -> dict`
  - `bin/train.py` — CLI accepting `--labels`, `--out`

- [ ] **Step 1: Write the failing tests**

```python
import pytest
from blobcount.train import EpochRecord, select_epoch, summarise_folds


def _rec(epoch, f1, acc=0.8):
    return EpochRecord(fold=1, epoch=epoch, train_loss=0.5, val_loss=0.4,
                       val_f1=f1, val_accuracy=acc, precision=0.7, recall=0.7,
                       tp=1, fp=1, fn=1, tn=1, lr=1e-4)


def test_select_epoch_uses_f1_not_accuracy():
    recs = [_rec(1, 0.60, acc=0.95), _rec(2, 0.90, acc=0.70)]
    assert select_epoch(recs).epoch == 2


def test_select_epoch_breaks_ties_on_earliest_epoch():
    recs = [_rec(1, 0.90), _rec(2, 0.90), _rec(3, 0.80)]
    assert select_epoch(recs).epoch == 1


def test_summary_uses_selected_epochs_only():
    selected = [_rec(1, 0.90), EpochRecord(2, 2, 0.5, 0.4, 0.50, 0.99, 0.4, 0.4, 1, 1, 1, 1, 1e-4)]
    s = summarise_folds([_rec(1, 0.90), _rec(2, 0.90)])
    assert s["mean_f1"] == pytest.approx(0.90)
    assert s["mean_f1_std"] == pytest.approx(0.0)
    assert "epoch2" not in str(s)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.train'`

- [ ] **Step 3: Implement `blobcount/train.py`**

`select_epoch` maximizes `val_f1` and returns the earliest epoch on a tie. `summarise_folds` computes mean and population standard deviation over the supplied records and additionally returns `selected_epochs` listing `fold` and `epoch` for each, so that the artifact is self-documenting — this is the regression test for the original defect, where the summary mixed best-epoch accuracy with final-epoch F1. `train_fold` builds train and eval transforms, constructs `class_weights` from the fold's training labels only, builds the sampler, constructs the model, and loops for `training.epochs` epochs with `Adam(lr=learning_rate)` and `ReduceLROnPlateau(factor=lr_scheduler_factor, patience=patience_lr_scheduler)`. It computes `val_f1` from the confusion matrix, appends an `EpochRecord` per epoch, saves a checkpoint whenever `val_f1` strictly improves, and stops after `patience_early_stop` epochs without improvement. `run_cross_validation` calls `loso_folds`, runs each fold, writes `results/training_log_fold{n}.csv` containing every epoch, writes `results/fold_summary.csv` from the selected epochs, and returns the summary dict.

- [ ] **Step 4: Create `bin/train.py`**

Read `labels.csv` (the handoff written by Task 6), resolve each row's `path` against `paths.labels`, call `run_cross_validation`, print the per-fold selected-epoch table and the mean table, and write `results/training_summary.json`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: PASS, 3 passed

- [ ] **Step 6: Commit**

```bash
git add blobcount/train.py bin/train.py tests/test_train.py
git commit -m "fix: select checkpoints on validation F1 and report the selected epoch only"
```

---

### Task 12: Evaluation with integrity guard and bootstrap intervals

**Files:**
- Create: `blobcount/evaluate.py`
- Create: `bin/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Config` (`evaluation.*`); `Specimen` from Task 2; `build_model` from Task 10; `build_transforms` from Task 9.
- Produces:
  - `blobcount.evaluate.Metrics` — frozen dataclass: `accuracy: float`, `precision: float`, `recall: float`, `specificity: float`, `f1: float`, `npv: float`, `auc: float | None`, `tp: int`, `fp: int`, `fn: int`, `tn: int`, `n: int`
  - `blobcount.evaluate.bootstrap_ci(values: Sequence[float], n_samples: int, level: float, seed: int) -> tuple[float, float]`
  - `blobcount.evaluate.quadratic_weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int], max_rating: int) -> float`
  - `blobcount.evaluate.evaluate_specimens(model, paths, labels, specimen_ids, train_specimens: Sequence[str], cfg: Config) -> Metrics`
  - Exception `EvaluationIntegrityError(Exception)`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from blobcount.evaluate import (EvaluationIntegrityError, bootstrap_ci,
                                 quadratic_weighted_kappa)


def test_overlap_between_trained_and_evaluated_specimens_raises():
    from blobcount.evaluate import evaluate_specimens
    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        evaluate_specimens(None, ["a.png"], [0], ["A"], ["A"], None)


def test_disjoint_specimens_do_not_raise():
    from blobcount.evaluate import assert_disjoint
    assert_disjoint(["A", "B"], ["C"])


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.8, 0.85, 0.9, 0.82, 0.88], 2000, 0.95, 42)
    assert lo < 0.86 < hi


def test_bootstrap_ci_is_seed_stable():
    a = bootstrap_ci([0.8, 0.85, 0.9], 500, 0.95, 7)
    b = bootstrap_ci([0.8, 0.85, 0.9], 500, 0.95, 7)
    assert a == b


def test_kappa_is_one_for_perfect_agreement():
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 3) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_disagreement():
    y = [0, 0, 1, 1]
    assert quadratic_weighted_kappa(y, [0, 1, 0, 1], 1) == pytest.approx(0.0, abs=1e-9)


def test_kappa_penalises_large_errors_more_than_small():
    small = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 3, 2], 3)
    large = quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0], 3)
    assert small > large
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.evaluate'`

- [ ] **Step 3: Implement `blobcount/evaluate.py`**

`assert_disjoint(train_specimens, eval_specimens)` raises `EvaluationIntegrityError` naming the intersecting specimens. `evaluate_specimens` calls it first, then runs inference, and returns `Metrics` with accuracy, precision, recall, specificity, F1, NPV, AUC via `sklearn.metrics.roc_auc_score` when both classes are present and `None` otherwise, and the four confusion counts. `bootstrap_ci` resamples the supplied per-specimen values `n_samples` times with a seeded `numpy.random.default_rng` and returns the empirical `level` quantiles. `quadratic_weighted_kappa` implements Cohen's kappa with weights `((i - j) ** 2) / ((max_rating - 1) ** 2)` and returns 1.0 when observed and expected agreement are both 1.

- [ ] **Step 4: Create `bin/evaluate.py`**

Accept `--specimens` (repeatable), `--checkpoint`, and `--train-specimens` (repeatable, required). The training specimen list is mandatory precisely so that a caller cannot accidentally score a specimen that was trained on without stating it. Write `results/evaluation/medical_metrics.json` and `per_specimen_metrics.csv`, and print the metric table with bootstrap intervals.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS, 7 passed

- [ ] **Step 6: Run the full suite**

Run: `python -m pytest -v`
Expected: PASS, all tests

- [ ] **Step 7: Commit**

```bash
git add blobcount/evaluate.py bin/evaluate.py tests/test_evaluate.py
git commit -m "fix: refuse to score a specimen that was trained on"
```

---

### Task 13: Grad-CAM and inference entry point

**Files:**
- Create: `blobcount/interpret.py`
- Create: `bin/gradcam.py`
- Create: `bin/predict.py`
- Create: `tests/test_interpret.py`

**Interfaces:**
- Consumes: `Config` (`gradcam.*`); `build_model` from Task 10; `build_transforms` from Task 9.
- Produces:
  - `blobcount.interpret.GradCAM` — `__init__(model, target_layer)`, `generate(input_tensor, target_class=None) -> tuple[np.ndarray, int, torch.Tensor]`
  - `blobcount.interpret.overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float) -> np.ndarray`
  - `bin/predict.py` — CLI accepting `--image` or `--dir` and `--checkpoint`
  - `bin/gradcam.py` — CLI accepting `--checkpoint`, `--slide`, `--n`

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
import torch
import torch.nn as nn


def test_gradcam_uses_output_hook_and_produces_a_map():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    m = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(m, m.layer4[-1])
    heat, cls, out = cam.generate(torch.randn(1, 3, 224, 224))
    assert heat.shape == (224, 224)
    assert 0.0 <= heat.min() and heat.max() <= 1.0
    assert cls in (0, 1)
    assert out.shape == (1, 2)


def test_gradcam_map_is_not_constant():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    torch.manual_seed(0)
    m = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(m, m.layer4[-1])
    heat, _, _ = cam.generate(torch.randn(1, 3, 224, 224))
    assert heat.std() > 1e-6


def test_overlay_shape_and_range():
    from blobcount.interpret import overlay
    img = np.random.rand(224, 224, 3).astype(np.float32)
    heat = np.random.rand(224, 224).astype(np.float32)
    out = overlay(img, heat, 0.4)
    assert out.shape == (224, 224, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_hooks_are_removable():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    m = build_model("resnet18", 2, pretrained=False)
    g = GradCAM(m, m.layer4[-1])
    g.remove()
    assert g.activations is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_interpret.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'blobcount.interpret'`

- [ ] **Step 3: Implement `blobcount/interpret.py`**

`GradCAM.__init__` stores the model and layer and registers a forward hook that captures `output.detach()` and calls `output.register_hook(...)` to capture the gradient with respect to that output tensor, replacing `register_full_backward_hook`, which fires with respect to module inputs and is unreliable on residual blocks. `generate` runs a forward pass, picks the predicted class when `target_class` is None, zeroes gradients, calls `backward` on the target logit, computes `weights = gradients.mean(dim=[2, 3], keepdim=True)`, forms `cam = relu((weights * activations).sum(dim=1, keepdim=True))`, squeezes, resizes to 224 x 224 with `cv2.INTER_LINEAR`, and min-max normalizes with an epsilon in the denominator. `remove()` calls `handle.remove()` on both hooks and clears `activations` and `gradients`. `overlay` applies a JET colour map, converts BGR to RGB, blends with `alpha`, and clips to `[0, 1]`.

- [ ] **Step 4: Create `bin/gradcam.py` and `bin/predict.py`**

`bin/gradcam.py` generates per-class sample figures and a summary grid into `results/gradcam/`. `bin/predict.py` predicts on a single image or a directory, writes `results/predictions.csv`, and creates `results/` if absent, which is the crash the old `test_model.py` had.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_interpret.py -v`
Expected: PASS, 4 passed

- [ ] **Step 6: Commit**

```bash
git add blobcount/interpret.py bin/gradcam.py bin/predict.py tests/test_interpret.py
git commit -m "fix: capture Grad-CAM gradients from the layer output tensor"
```

---

### Task 14: Remove the legacy scripts and relocate data

**Files:**
- Delete: `patch.py`, `label.py`, `train.py`, `evaluate.py`, `gradcam.py`, `validate_labels.py`, `patient_level_prediction.py`, `test_model.py`, `visualize_samples.py`
- Delete: `steatosis_model.pth`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: every module produced by Tasks 1 to 13.
- Produces: a repository in which the `blobcount` package is the only implementation.

- [ ] **Step 1: Confirm the replacement is complete before deleting anything**

Run: `python -m pytest -v`
Expected: PASS, all tests, and `python bin/extract_patches.py --help` exits 0 for every file in `bin/`

- [ ] **Step 2: Confirm no module imports a legacy script**

Run: `grep -rn "import patch\|import label\|from patch\|from label\|validate_labels\|patient_level" blobcount/ bin/ tests/`
Expected: no output

- [ ] **Step 3: Delete the legacy scripts and the stale checkpoint**

Run: `git rm patch.py label.py train.py evaluate.py gradcam.py validate_labels.py patient_level_prediction.py test_model.py visualize_samples.py steatosis_model.pth`
Expected: ten files staged for deletion

- [ ] **Step 4: Move `data/` and `dataset/` out of the OneDrive-synchronized tree**

Run: `New-Item -ItemType Directory -Force -Path "$env:LOCALAPPDATA\UmurOS" | Out-Null; Move-Item -LiteralPath "data" -Destination "$env:LOCALAPPDATA\UmurOS\data"; Move-Item -LiteralPath "dataset" -Destination "$env:LOCALAPPDATA\UmurOS\dataset"`
Expected: both directories moved. If a move fails because OneDrive holds a file open, stop and report it rather than copying, because a partial copy plus a delete loses data. Note that `data/manifest.yaml` is tracked by git and must be copied back into the repository before the move, then restored into the relocated `data/` directory; `git mv data/manifest.yaml` is not applicable because the destination is outside the repository, so copy the file, move the directory, and copy the manifest back.

- [ ] **Step 5: Point the configuration at the relocated directories**

Edit `configs/default.yaml` so `paths.slides` and `paths.manifest` are absolute paths under `%LOCALAPPDATA%\UmurOS\data`, and `paths.patches` and `paths.labels` are absolute paths under `%LOCALAPPDATA%\UmurOS\dataset`. Add a commented block recording the original relative locations and the reason for the move.

- [ ] **Step 6: Update `.gitignore`**

Keep the existing rules. Two changes are required, and the first is not
optional.

Replace the bare `data/` line with `data/*`, then add `!data/manifest.yaml`
immediately after it. Git cannot re-include a file inside an excluded
directory, so a bare `data/` plus a negation leaves the manifest ignored and
invisible; `data/*` excludes the directory's contents while permitting the one
negated path. Verify with `git check-ignore -v data/manifest.yaml`, which must
produce no output, and `git ls-files data/`, which must list only the manifest
and none of the six `.svs` files. This is the state the manifest is already
committed in, reached by `git add -f`; the ignore rule must be brought in line
with it.

Add `Colab_Proje.zip` explicitly with a comment recording that it is 10.1 GB and
is a graduation-project archive whose contents were inspected on 2026-09-27 and
found to contain no larger cohort, so that it cannot be committed by accident
even if the `*.zip` rule is ever removed.

- [ ] **Step 7: Verify the configuration still loads and the registry still validates**

Run: `python -c "from blobcount.registry import load_registry, usable_for_training; r = load_registry(); print(len(r), len(usable_for_training(r)))"`
Expected: prints `6 3`

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor: replace standalone scripts with the blobcount package"
```

---

### Task 15: Correct the README

**Files:**
- Create: `Makefile`
- Modify: `README.md`

**Interfaces:**
- Consumes: `results/fold_summary.csv`, `results/evaluation/medical_metrics.json`, `results/label_analysis/threshold_ablation.json` produced by the pipeline.
- Produces: a README whose every claim is traceable to a committed artifact.

- [ ] **Step 1: Create `Makefile`**

Targets `install`, `test`, `extract`, `label`, `ablate`, `train`, `evaluate`, `predict`, `gradcam`, `clean`, and `all`. `all` runs `extract`, `label`, `ablate`, `train`, `evaluate` in that order and is the single command that reproduces every number in the README. Each target invokes the corresponding `bin/` script with `PYTHONPATH=.` set, so the pipeline runs from a fresh clone without an editable install. `clean` removes `results/` and the generated label directory but never touches the slides.

- [ ] **Step 2: Verify the single command runs end to end**

Run: `make all`
Expected: exit 0, and `results/label_analysis/threshold_ablation.json`, `results/fold_summary.csv`, and `results/evaluation/medical_metrics.json` all exist and are newer than the start of the run

- [ ] **Step 3: Collect the numbers to be reported**

Run: `cat results/fold_summary.csv results/evaluation/medical_metrics.json results/label_analysis/threshold_ablation.json`
Expected: the three artifacts that every README claim must cite

- [ ] **Step 4: Rewrite the Results section**

Replace the current tables with the values from Step 3. Report the per-specimen class distribution at the selected threshold from `threshold_ablation.json`, not only the pooled ratio. Report the leave-one-specimen-out mean and standard deviation from `fold_summary.csv`, and state the number of specimens.

- [ ] **Step 5: Remove the two false claims**

Delete the sentence claiming that patient-level grouping prevents data leakage, and delete the justification for threshold 5 based on a balanced class distribution. Both are contradicted by `docs/superpowers/specs/2026-09-27-tubitak-rebuild-design.md` section 1.1.

- [ ] **Step 6: Add the limitations section**

State that the ground truth is a morphological blob counter rather than a pathologist grade, that the ceiling of the reported performance is that counter, that three of the six specimens have no recoverable provenance and are excluded from reported results, that the cohort is six specimens and that this is a pilot, and that the stain is H&E only.

- [ ] **Step 7: Rewrite the pipeline description to match the code**

Replace the stage table with the `bin/` entry points in the order they run, document `make all` as the reproduction command, and replace the folder structure block with the structure recorded in the File Structure section of this plan.

- [ ] **Step 8: Commit**

```bash
git add README.md Makefile results/
git commit -m "docs: correct reported results and state the label-definition limitation"
```

---

## Verification

After all tasks, the following must hold. Each is a command with an observable outcome, not an assertion about intent.
```bash
python -m pytest -v                          # all tests pass, zero warnings
python -m pip install -e ".[dev]"            # installs cleanly on Python 3.11+
grep -rn "GroupKFold\|shutil.copy\|register_full_backward_hook\|except:" blobcount/ bin/
                                           # no output
python -c "from blobcount.config import load; c=load(); print(len(c.get('extraction.target_mpp')))"
                                           # prints 0.5; load() performs no reads of its own
git status --porcelain                      # clean
```

### Final gate on the config-readability debt

`tests/test_config.py` carries a module-level `_KEYS_WITHOUT_READER` set listing
declared config keys that no module reads yet, because Tasks 2 to 15 do not exist
when Task 1 lands. That set is accepted debt with a named owner: this plan. The
constraint "a configuration key that no code reads is a hard error" is enforced
against the keys outside the set, and the set is what makes the constraint
satisfiable at Task 1 rather than a softened form of it.

The branch must not merge with the set non-empty. Before merging, run:

```bash
python -c "import re,pathlib; s=pathlib.Path('tests/test_config.py').read_text('utf-8'); m=re.search(r'_KEYS_WITHOUT_READER[^=]*=\s*frozenset\(\{(.*?)\}\)', s, re.S); print('exempt keys:', 0 if not m.group(1).strip() else len([x for x in m.group(1).split(',') if x.strip()]))"
```

Expected: `exempt keys: 0`. Each task that adds a reader for a listed key makes
the source-scan test fail by naming that key as a stale entry, so the set shrinks
one key at a time and cannot be forgotten. Note the direction of the guarantee: a
key that *gains* a reader is caught loudly; a listed key that *loses* its reader
is not caught while it remains listed. The set is the control, and the gate above
is what closes it.

The last `grep` is the mechanical check that the three defects named in the spec
— grouped splitting by specimen count, non-idempotent labeling, and the
unreliable backward hook — no longer appear anywhere in the implementation.