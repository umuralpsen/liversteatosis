"""Patch extraction: the same field of view off every slide, or the study is a fiction.

The original `patch.py` tiled each slide at a fixed 256x256 *level-0* window and
kept the tile when its mean grayscale fell under a constant. That constant is the
last inheritance this package still carries, and it is not independently justified:
`extraction.tissue_gray_max` is a configured parameter, and no measurement in this
repository establishes its value. `patch.py`'s own docstring called 210 "found
empirically"; nothing here repeats that, and a change to the key changes the filter.

`is_tissue` is that mean test and nothing else. The build before this one added a
second clause, that a pixel darker than the patch's own 5th percentile plus a margin
must exist in the patch, and that clause could never be false: a percentile is never
above the minimum of the same values, so the filter was one factor while reading as
two. The clause is deleted rather than given a threshold, because a floor on the
spread or on the median would be a constant nobody has measured.

**The filter has no background-rejection behaviour and its sensitivity is unmeasured.**
A uniform field of any value below `gray_max` is accepted, so the rule cannot tell a
blank field from a flat field of tissue, and a real `3.svs` window at (14732, 3048)
with mean 235.0, std 22.1 and min 63.3 is rejected by the mean clause alone, so
neither a floor on spread nor a raised `gray_max` recovers it. Whether one mean
threshold separates tissue from background across the whole grid is a question about
a full-grid measurement rather than about this function, and `summarise_tissue_gray`
makes it answerable from a committed index instead of from a claim.

**The transparency check in `read_patch` is the only thing standing between a grid
arithmetic error and a silently corrupted dataset.** `slide.read_region` does not
raise for an origin outside the slide. It returns a full-size RGBA region whose alpha
is zero everywhere, and dropping that alpha and taking the RGB channels leaves solid
black, whose mean grayscale is 0, which is *below* `gray_max`. So a coordinate bug
does not announce itself: it writes black tiles that the next task counts blobs in
and labels steatotic. The size check the brief specifies does not catch it, because
the region comes back at the width that was asked for rather than clipped to the
edge. Measured on `data/2.svs` at an origin 5000 px past the right edge: a 508x508
region, alpha zero throughout, one distinct RGB value, against 1128 distinct colours
in an in-bounds read of the same size. A read whose alpha is entirely zero is
therefore a `read_failure` and is counted as one, and the alpha channel is read
before the conversion to RGB, because after the conversion the evidence is gone.

**The manifest cross-check is opt-in and this module opts in.** `open_info` takes
`expected_mpp` with a `None` default, and `None` means the manifest records no scale
for the slide. `extract_slide` passes `specimen.mpp` unconditionally, because
`data/manifest.yaml` is the record the patient-level split is computed from and a
slide replaced under a fixed name is exactly what that record would then misdescribe.
The parameter being opt-in by construction is the reason the call site is written
out: a caller that forgets it gets a plausible run against an unverified file.

**Coordinates are level-0 source pixels.** `iter_coords` yields the origin pair
`slide.read_region` takes for its location, and `info.read_px` is the source-pixel
tile size, which is 507 to 515 for the five 40x slides and 257 for the 20x one.
Neither is transformed here, and the read is then resized to `extraction.patch_size`
with `cv2.INTER_AREA`, which is where the field of view becomes
`patch_size * target_mpp` microns for every slide in the set. The tile grid is walked
by `iter_coords`, which drops incomplete edge tiles, so a `read_failure` means
something actually went wrong rather than that the walk reached a border.

Every accepted patch is written as `{slide_id}_{x}_{y}.png` with the *source*
coordinates, so the file name records where on the slide the patch came from, and one
row per patch goes to `patches.csv` carrying the resolution the patch was actually
read at rather than the one that was requested. Every exception raised per patch is
caught and logged with the slide id, the coordinates and the exception object, and
counted; a run that ends with `read_failures` and no written patches is reported as
`skipped_blank` and exits non-zero from the command line, so an empty dataset cannot
pass for a completed one.

Declared interface, in `__all__` order:

    ExtractionStats
        slide_id: str
        patches_written: int
        tissue_rejected: int
        read_failures: int
        skipped_blank: bool
    OutputPathError
    extract_slide(
        specimen: Specimen,
        out_dir: Path,
        cfg: Config,
        *,
        rows: list[dict],
    ) -> ExtractionStats
    is_tissue(patch: np.ndarray, threshold: float, gray_max: float) -> bool
    patches_dir(cfg: Config | None = None) -> Path
    prepare_output(
        out_dir: Path, *, root: Path | None = None, protected: Sequence[Path] | None = None
    ) -> None
    read_patch(
        slide: Any, info: SlideInfo, x: int, y: int, *, patch_size: int | None = None
    ) -> np.ndarray | None
    summarise_tissue_gray(index_path: Path, *, blank_slides: Collection[str] = ()) -> dict
    tissue_gray_threshold(patch: np.ndarray, percentile: float = TISSUE_PERCENTILE) -> float
    write_index(rows: Sequence[dict], path: Path) -> None
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blobcount.config import PROJECT_ROOT, Config, default_config
from blobcount.registry import Specimen
from blobcount.slides import SlideInfo, iter_coords_from_config, open_info_from_config

__all__ = [
    "ExtractionStats",
    "OutputPathError",
    "extract_slide",
    "is_tissue",
    "patches_dir",
    "prepare_output",
    "read_patch",
    "summarise_tissue_gray",
    "tissue_gray_threshold",
    "write_index",
]

LOGGER = logging.getLogger(__name__)

# The column order `write_index` pins, and the keys every index row carries.
INDEX_COLUMNS = ("patch_id", "slide_id", "x", "y", "mpp", "tissue_gray")

# The default `extraction.tissue_percentile`: the background level of a patch is the
# dark end of it, and 5 keeps a handful of the darkest pixels from setting the level
# of a patch that is otherwise background. A test pins this constant to the
# configured value, because the two are the same fact stated twice and a change to
# the key must not leave the constant behind.
TISSUE_PERCENTILE = 5


class OutputPathError(Exception):
    """Raised when a directory `prepare_output` was asked to delete is not one to delete."""


@dataclass(frozen=True)
class ExtractionStats:
    """What one slide's extraction produced, and what it cost.

    `patches_written` counts accepted patches and nothing else. `tissue_rejected`
    counts reads that succeeded and were filtered as background, and `read_failures`
    counts reads that produced no patch: a short read, a region with no opaque pixel
    in it, an exception raised by the reader, or a PNG that could not be written. The
    two failure modes are counted apart because they mean different things, a read
    that is filtered is a measurement and a read that fails is a hole in the record.
    A failed write is counted with the failed reads rather than apart from them
    because the outcome is the same: a window that was read, accepted, and then did
    not reach the disk, and its coordinates appear in no index row.

    `skipped_blank` is True when the slide produced no accepted patch, whether because
    every read was rejected as background or because every read failed. A slide in
    that state is one the rest of the pipeline has nothing to train or evaluate on,
    and reporting it is the difference between an empty run and a completed one.
    """

    slide_id: str
    patches_written: int
    tissue_rejected: int
    read_failures: int
    skipped_blank: bool


def _grayscale(patch: np.ndarray) -> np.ndarray:
    """Return the mean of the colour channels, as a float array.

    The channels are averaged rather than combined through a luma weighting. The
    weighted form is the better estimator of perceived brightness and the wrong one
    here: the threshold it feeds was calibrated against a plain channel mean, and
    changing the definition under a fixed threshold changes what the threshold means
    without any of it being a decision anyone made.
    """
    return np.asarray(patch, dtype=np.float64).mean(axis=2)


def tissue_gray_threshold(patch: np.ndarray, percentile: float = TISSUE_PERCENTILE) -> float:
    """Return the background level of `patch`: the `percentile` of its grayscale.

    A low percentile rather than the minimum, so that one dark speck, a pen mark or a
    dust mote, cannot set the level of an otherwise blank patch. The result is total
    for every array with a channel axis: a constant image has its percentile equal to
    the constant, so a blank patch yields a number and never a `NaN`, and the value
    returned for a blank patch is the same value that rejects it.

    `percentile` defaults to the module constant `TISSUE_PERCENTILE`, which a test
    pins to `extraction.tissue_percentile`, so the default and the configured value
    cannot drift apart. `extract_slide` passes the configured value through as an
    argument, so an override reaches a run without touching the constant.

    **The filter does not consult this number.** `is_tissue` is a mean test alone,
    and the function stays because the value it returns is recorded in the
    `tissue_gray` column of `patches.csv`, which the counting, labeling and training
    tasks read to derive a per-slide background level. It is kept for that consumer
    and not because it decides which patches are kept.
    """
    return float(np.percentile(_grayscale(patch), percentile))


def is_tissue(patch: np.ndarray, threshold: float, gray_max: float) -> bool:
    """Report whether `patch` holds tissue rather than background: one mean test.

    A patch is tissue when its mean grayscale is below `gray_max`. That is the whole
    rule, and it is the test `patch.py` made, which is why the key is named for it.

    `threshold` is the background level `tissue_gray_threshold` measures on the same
    patch, and it is **not consulted**. The signature keeps it because the interface
    is declared, and the number is recorded per patch in the `tissue_gray` index
    column that later tasks read; it is kept for that consumer and not because it
    decides this answer. An earlier build also required a pixel darker than
    `threshold + 20` to exist, which could never be false: `threshold` is a percentile
    of this patch's own grayscale and the minimum is the smallest of the same values,
    so the minimum is never above the threshold and the margin is always satisfied.
    It was deleted rather than given a threshold, because any floor on the spread or
    the median would be a constant nobody has measured.

    **The filter has no background-rejection behaviour and its sensitivity is
    unmeasured.** A uniform field of any value below `gray_max` is accepted, so a
    blank field and a flat field of tissue are indistinguishable here, and a real
    `3.svs` window at (14732, 3048) with mean 235.0, std 22.1 and min 63.3 is
    rejected by the mean clause alone, so neither a floor on spread nor a raised
    `gray_max` recovers it. `tests/test_extract.py` pins the acceptance of a uniform
    faint field under a name that says so, and it is deleted by whichever change
    closes the gap.
    """
    return bool(_grayscale(patch).mean() < gray_max)


def read_patch(
    slide: Any,
    info: SlideInfo,
    x: int,
    y: int,
    *,
    patch_size: int | None = None,
) -> np.ndarray | None:
    """Read the `info.read_px` window at source origin `(x, y)` and return a patch.

    `x` and `y` are level-0 source pixels exactly as `iter_coords` yields them, and
    they are handed to `slide.read_region` untransformed, because that is the unit the
    read takes its location in. The read is `info.read_px` square at `info.level`,
    which is 508 px across for a 40x slide planned for a 256 px patch, and the result
    is resized to `patch_size` with `cv2.INTER_AREA`, an area average rather than a
    nearest-neighbour pick, so the field of view the patch covers is
    `patch_size * target_mpp` microns and no pixel is invented by a subsample.

    `None` is returned, never a substitute image, for a read that yielded no usable
    region: one fewer than `read_px` on either axis, a region with no opaque pixel in
    it at all, a region that is not an image, or a region with a channel count no
    image here can be written from. The all-transparent case is the one that matters
    and the size check cannot catch it, because a read outside the slide comes back at
    the width that was asked for rather than clipped to the edge. It converts to solid
    black, which `is_tissue` accepts, so returning it would write a black tile into a
    labelled dataset. See the module docstring.

    `patch_size` is keyword-defaulted to `extraction.patch_size` read from the default
    configuration, because the interface carries no `Config`. `extract_slide` passes
    its own `cfg` through instead, so the window a run is planned for and the window it
    resizes to cannot be read from two different configurations.

    `cv2` is imported here rather than at module scope: `opencv-python` is a native
    wheel that fails to import where its system libraries are absent, and a
    module-scope import would make that a collection error for every test in the file
    including the ones that touch no image. A missing import here surfaces as a
    `read_failure` on every patch, which leaves `patches_written` at zero and the
    slide reported blank, so the run fails rather than quietly producing nothing.
    """
    import cv2

    size = int(patch_size) if patch_size is not None else int(
        default_config().get("extraction.patch_size")
    )
    region = slide.read_region((x, y), info.level, (info.read_px, info.read_px))
    if region is None:
        return None
    raw = np.asarray(region)
    if raw.ndim != 3 or raw.shape[0] < info.read_px or raw.shape[1] < info.read_px:
        return None
    if raw.shape[2] == 4 and not raw[..., 3].any():
        return None
    if raw.shape[2] not in (3, 4):
        return None
    return cv2.resize(
        np.ascontiguousarray(raw[..., :3]), (size, size), interpolation=cv2.INTER_AREA
    )


def extract_slide(
    specimen: Specimen,
    out_dir: Path,
    cfg: Config,
    *,
    rows: list[dict],
) -> ExtractionStats:
    """Extract every accepted patch of `specimen` into `out_dir` and report the counts.

    The read is planned by `open_info_from_config` with `expected_mpp=specimen.mpp`,
    so the slide's own `openslide.mpp-x` and the scale `data/manifest.yaml` records
    are cross-checked before a single pixel is read, and a slide that no longer
    matches its row raises `SlideError` rather than being extracted against a record
    that no longer describes it. The parameter is optional in `open_info` and this
    call site is what makes the check real. The window is then walked by
    `iter_coords_from_config`, which converts the configured step into source pixels
    and drops incomplete edge tiles, so the coordinates that reach the reader come
    from one place and are not recomputed here.

    An accepted patch is written as `{slide_id}_{x}_{y}.png` with the source
    coordinates, which is what the file name means and what `patches.csv` records.
    `is_tissue` decides acceptance and the background level
    `tissue_gray_threshold` measures is recorded beside it, not consulted by the
    filter. `mpp` in an index row is the resolution the patch was actually read at,
    `info.achieved_mpp`, and not the target that was requested, so a run's resolution
    is auditable from the patch record alone.

    Every exception raised by the reader per patch is caught, logged with the slide
    id, the coordinates and the exception object, and counted as a `read_failure`; the
    walk continues, because a run that stops at the first bad tile reports one failure
    and hides the rest. A write that raises is caught and counted the same way, with
    the specimen id, the coordinates and the file name. Nothing else in the loop is
    guarded: `read_patch` has already returned a uint8 `(H, W, 3)` array, and
    `tissue_gray_threshold` and `is_tissue` are total over one, so a `try` around
    them could only catch a defect in this function.
    `SlideError` from the plan is not caught: it describes the slide rather than a
    patch, and it is raised before `out_dir` is created so a rejected slide leaves the
    output tree alone.

    **`rows` is required.** The return type carries counts only, so the index rows a
    run needs in order to write one `patches.csv` come back through the list the
    caller supplies. Making it optional would give the caller a way to write patches
    with no index row and no way to recover their `tissue_gray` afterwards, and that
    is a loss the counts do not repay. A caller that wants only the counts passes an
    empty list and ignores it.

    `PIL` is imported inside the write for the same reason `cv2` is imported inside
    the read: it is a native wheel, and making it a module-scope import would make an
    absent one a collection error for every test in the suite, not only the ones that
    read a slide.

    `out_dir` is created if it is absent but never cleared: `prepare_output` is the
    function that deletes, so a caller decides whether a previous run's patches may be
    destroyed rather than this function deciding it as a side effect of extracting.
    """
    info = open_info_from_config(specimen.slide_path, specimen.id, cfg, expected_mpp=specimen.mpp)
    patch_size = int(cfg.get("extraction.patch_size"))
    percentile = float(cfg.get("extraction.tissue_percentile"))
    gray_max = float(cfg.get("extraction.tissue_gray_max"))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from PIL import Image

    import openslide

    written = rejected = failures = 0
    with openslide.OpenSlide(str(specimen.slide_path)) as slide:
        for x, y in iter_coords_from_config(info, cfg):
            try:
                patch = read_patch(slide, info, x, y, patch_size=patch_size)
            except Exception as exc:
                LOGGER.error("slide %s: read at (%d, %d) raised %r", specimen.id, x, y, exc)
                failures += 1
                continue
            if patch is None:
                LOGGER.warning(
                    "slide %s: read at (%d, %d) returned no usable region: shorter than "
                    "the planned read, outside the slide, or with no opaque pixel in it, "
                    "so it is counted as a read failure rather than written as a patch",
                    specimen.id, x, y,
                )
                failures += 1
                continue
            threshold = tissue_gray_threshold(patch, percentile)
            if not is_tissue(patch, threshold, gray_max):
                rejected += 1
                continue
            name = f"{specimen.id}_{x}_{y}.png"
            try:
                Image.fromarray(patch).save(out / name)
            except Exception as exc:
                LOGGER.error(
                    "slide %s: writing %s at (%d, %d) raised %r", specimen.id, name, x, y, exc
                )
                failures += 1
                continue
            written += 1
            rows.append({
                "patch_id": name,
                "slide_id": specimen.id,
                "x": x,
                "y": y,
                "mpp": info.achieved_mpp,
                "tissue_gray": threshold,
            })

    return ExtractionStats(
        slide_id=specimen.id,
        patches_written=written,
        tissue_rejected=rejected,
        read_failures=failures,
        skipped_blank=written == 0,
    )


def _is_below(target: Path, root: Path) -> bool:
    """Report whether `target` lies strictly below `root`.

    Both are resolved first, so a `..` is collapsed and a link cannot point out, and
    the test is `root in target.parents` rather than a string prefix: `Proje-backup`
    starts with the characters of `Proje` and is a different directory, and a prefix
    test would place it inside the project. Being `root` itself is not below it, which
    is what makes a call that would delete the project root a refusal rather than a
    delete.
    """
    return Path(root).resolve() in Path(target).resolve().parents


def _protected_paths() -> tuple[tuple[str, Path], ...]:
    """Return the `(name, path)` pairs a run may not delete by clearing a directory.

    The slides and the manifest are read from the configuration rather than pinned
    here, so a run whose configuration moves them is guarded against the paths it
    actually uses. They are read from the default configuration because they are
    project facts rather than per-run settings, and the CLI passes nothing.
    """
    active = default_config()
    return (
        ("paths.slides", active.path("paths.slides")),
        ("paths.manifest", active.path("paths.manifest")),
    )


def prepare_output(
    out_dir: Path,
    *,
    root: Path | None = None,
    protected: Sequence[Path] | None = None,
) -> None:
    """Delete `out_dir` and create it empty.

    Extraction and, later, labeling write into one directory that a previous run may
    have filled. Anything left behind is a patch from a slide, a resolution or a
    configuration that is no longer the one being run, and it is indistinguishable
    from a fresh one once it is on disk, so the directory is emptied rather than
    written into. Deleting is not guarded by `ignore_errors`: a tree that cannot be
    cleared must not be written into, because the stale files would survive into the
    dataset and the run would look complete.

    **`paths.patches` is a hand-edited string in `configs/default.yaml`, and this
    function deletes whatever `Path` it is handed.** `paths.patches: .` makes a
    `--force` run an irreversible deletion of the repository, the manifest and every
    staged slide, with no confirmation and no way back. Every target below is refused
    before anything is removed, and the refusal is an `OutputPathError` rather than a
    partial delete:

    - the project root itself, and
    - `paths.slides` and `paths.manifest` themselves, and
    - any directory that contains `paths.slides` or `paths.manifest`, and
    - anything not strictly below the project root, which covers a target outside the
      project and one reached through a `..` that climbs out of it.

    A protected path is refused on equality as well as on containment. `_is_below` is
    strict, so a target that *is* `paths.slides` is not below it and the ancestor clause
    alone would let it through; in the real configuration `paths.slides` is `data`, so
    a `paths.patches` hand-edit to `data` is one character from the case that is
    covered and reaches the delete. That is the same catastrophe, so it is refused
    before the containment test rather than after it.

    `root` defaults to the project root and `protected` to the configured slides and
    manifest. Both are parameters so a test can state the boundary instead of
    inheriting the checkout's, and so the containment rule has one implementation
    rather than one per call site.
    """
    target = Path(out_dir)
    boundary = Path(root) if root is not None else PROJECT_ROOT
    resolved = target.resolve()
    if resolved == boundary.resolve():
        raise OutputPathError(
            f"refusing to delete the project root {resolved}: it holds the source, the "
            f"configuration and the manifest, and --force is not a licence to remove it"
        )
    guarded = (
        _protected_paths() if protected is None
        else tuple((str(item), Path(item)) for item in protected)
    )
    for name, item in guarded:
        if resolved == item.resolve():
            raise OutputPathError(
                f"refusing to delete {resolved}: it is {name} itself, so clearing it "
                f"would destroy the source data"
            )
        if _is_below(item, resolved):
            raise OutputPathError(
                f"refusing to delete {resolved}: {name} is at {item} and is inside it, so "
                f"clearing this directory would destroy the source data"
            )
    if not _is_below(resolved, boundary):
        raise OutputPathError(
            f"refusing to delete {resolved}: it is not strictly below the project root "
            f"{boundary}, so the target is not one this project owns"
        )
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)


def patches_dir(cfg: Config | None = None) -> Path:
    """Return the configured patch output directory, `paths.patches`.

    The patch root is a configuration key and the resolution of that key is a project
    fact, so it is resolved here and in no other place: a caller that builds a patch
    path by hand can disagree with the configured root without anything noticing.
    """
    active = default_config() if cfg is None else cfg
    return active.path("paths.patches")


def write_index(rows: Sequence[dict], path: Path) -> None:
    """Write `rows` to `path` as UTF-8 CSV under the pinned `INDEX_COLUMNS` header.

    The header is written even for no rows, so an empty run leaves a readable file
    rather than a missing one. The file is UTF-8 and slide ids are written as data,
    not spliced into the text, so an id carrying a comma, a quote or a non-ASCII
    character round-trips. `newline=""` leaves the line terminator to `csv` instead
    of letting the text layer translate it, which is what keeps the `\r\n` the csv
    module writes from becoming `\r\r\n` on Windows.
    """
    target = Path(path)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def summarise_tissue_gray(index_path: Path, *, blank_slides: Collection[str] = ()) -> dict:
    """Return the distribution of the `tissue_gray` column over an index.

    The keys are `count`, `min`, `max`, `mean`, `p05`, `p50`, `p95` and
    `skipped_blank_slides`. The statistics are `None` rather than a number for an index
    with no rows, because a percentile of nothing has no value and reporting `0.0`
    would put a plausible number where a missing one is the truth.

    **This is the characterisation the tissue filter is missing, made answerable
    from an artifact rather than from a claim.** `is_tissue` accepts a patch on its
    mean alone and rejects one on the same number, so whether that one threshold
    separates tissue from background across the whole grid is a question about the
    spread of the recorded `tissue_gray` values: a distribution with a long bright
    tail reaching `tissue_gray_max` is one where a real background window was
    accepted, and a distribution that never reaches it is one where nothing needs
    deciding. Running this over the full grid is deferred, because it needs the full
    extraction; what belongs to this module is the function, so that the measurement
    is one command against a committed index.

    **`skipped_blank_slides` counts the slides the caller reports blank, not the rows
    the file carries for them.** A slide reported `skipped_blank` wrote no patches and
    no rows, so the index of a run that saw one names it nowhere and a count taken from
    the file is 0 for every index this pipeline writes. The flag is passed in rather
    than read, because the index has no such column and its header is pinned for the
    consumers downstream. Called with no `blank_slides` the count is 0, which is then
    the correct answer for an index this pipeline writes and not a default to be
    trusted.
    """
    target = Path(index_path)
    with target.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    values = [float(row["tissue_gray"]) for row in rows]
    summary = {
        "count": len(values),
        "skipped_blank_slides": len(set(blank_slides)),
    }
    if not values:
        return {**summary, **{name: None for name in ("min", "max", "mean", "p05", "p50", "p95")}}
    percentiles = np.percentile(np.asarray(values, dtype=np.float64), [5, 50, 95])
    return {
        **summary,
        "min": min(values),
        "max": max(values),
        "mean": float(np.mean(values)),
        "p05": float(percentiles[0]),
        "p50": float(percentiles[1]),
        "p95": float(percentiles[2]),
    }
