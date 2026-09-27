"""Patch extraction: the same field of view off every slide, or the study is a fiction.

The original `patch.py` tiled each slide at a fixed 256x256 *level-0* window and
kept the tile when its mean grayscale fell under a constant. That constant is the
last inheritance this package still carries, and it is not independently justified:
`extraction.tissue_gray_max` is a configured parameter, and no measurement in this
repository establishes its value. `patch.py`'s own docstring called 210 "found
empirically"; nothing here repeats that, and a change to the key changes the filter.

`is_tissue` keeps that mean test and adds a second clause: the patch must also hold
at least one pixel below `threshold + DARK_MARGIN`, where `threshold` is the 5th
percentile of the patch's own grayscale. That second clause is a tautology, because a
5th percentile is never above the minimum of the same values, so `is_tissue` in this
build reduces to `mean < gray_max` and the filter is the original one with a
different constant. The clause is implemented as specified and kept, and
`is_tissue`'s docstring carries the measurement, because the key beside it is a
configured parameter and changing the rule here would be a decision nobody made.

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
tile size, which is 508 for the five 40x slides and 257 for the 20x one. Neither is
transformed here, and the read is then resized to `extraction.patch_size` with
`cv2.INTER_AREA`, which is where the field of view becomes `patch_size * target_mpp`
microns for every slide in the set. The tile grid is walked by `iter_coords`, which
drops incomplete edge tiles, so a `read_failure` means something actually went wrong
rather than that the walk reached a border.

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
    extract_slide(
        specimen: Specimen,
        out_dir: Path,
        cfg: Config,
        *,
        rows: list[dict] | None = None,
    ) -> ExtractionStats
    is_tissue(patch: np.ndarray, threshold: float, gray_max: float) -> bool
    patches_dir(cfg: Config | None = None) -> Path
    prepare_output(out_dir: Path) -> None
    read_patch(
        slide: Any, info: SlideInfo, x: int, y: int, *, patch_size: int | None = None
    ) -> np.ndarray | None
    tissue_gray_threshold(patch: np.ndarray, percentile: float = TISSUE_PERCENTILE) -> float
    write_index(rows: Sequence[dict], path: Path) -> None
"""

from __future__ import annotations

import csv
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blobcount.config import Config, default_config
from blobcount.registry import Specimen
from blobcount.slides import SlideInfo, iter_coords_from_config, open_info_from_config

__all__ = [
    "ExtractionStats",
    "extract_slide",
    "is_tissue",
    "patches_dir",
    "prepare_output",
    "read_patch",
    "tissue_gray_threshold",
    "write_index",
]

LOGGER = logging.getLogger(__name__)

# The column order `write_index` pins, and the keys every index row carries.
INDEX_COLUMNS = ("patch_id", "slide_id", "x", "y", "mpp", "tissue_gray")

# The default `extraction.tissue_percentile`: the background level of a patch is the
# dark end of it, and 5 keeps a handful of the darkest pixels from setting the level
# of a patch that is otherwise background.
TISSUE_PERCENTILE = 5

# How far below the background level a pixel must fall for the patch to count as more
# than background. See `is_tissue`: because the background level is a low percentile
# of the same grayscale, this margin cannot currently reject anything, and the
# constant is kept because the rule specifies it and the threshold beside it is a
# configured parameter.
DARK_MARGIN = 20


@dataclass(frozen=True)
class ExtractionStats:
    """What one slide's extraction produced, and what it cost.

    `patches_written` counts accepted patches and nothing else. `tissue_rejected`
    counts reads that succeeded and were filtered as background, and `read_failures`
    counts reads that yielded no usable region at all: a short read, a region with no
    opaque pixel in it, or an exception. The two failure modes are counted apart
    because they mean different things, a read that is filtered is a measurement and
    a read that fails is a hole in the record.

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

    `percentile` carries the configured `extraction.tissue_percentile`, which the
    interface this function declares does not otherwise pass; it is keyword-defaulted
    to the configured value so that a caller reading the configuration gets one
    threshold and an extraction run reading it gets the same one.
    """
    return float(np.percentile(_grayscale(patch), percentile))


def is_tissue(patch: np.ndarray, threshold: float, gray_max: float) -> bool:
    """Report whether `patch` holds tissue rather than background.

    Two conditions, both required. The mean grayscale is below `gray_max`, which is
    the test `patch.py` made and the reason the key is named for it. And the patch is
    not uniformly background: at least one pixel must be darker than `threshold` plus
    `DARK_MARGIN`, where `threshold` is the background level `tissue_gray_threshold`
    measured on the same patch.

    **The second condition cannot currently reject anything, and a caller deciding
    whether to keep it needs to know that.** `threshold` is the 5th percentile of this
    patch's own grayscale and `gray.min()` is the smallest of the same values, so
    `min <= threshold` always and `min < threshold + DARK_MARGIN` holds for every
    patch, including a constant one, where the percentile equals the constant. Over
    2000 random patches and every constant value from 0 to 255, the condition was
    never false. The consequence is that `is_tissue` reduces to `mean < gray_max`: the
    mean test of the original `patch.py` with a different threshold, and a uniform
    field of anything below `gray_max` is accepted, which is the shape the second
    condition exists to reject. A sampled sweep of 240 real windows over `1.svs`,
    `2.svs` and `5.svs` found no obviously non-tissue window passing, so this is a
    documented gap in the rule rather than an observed corruption.

    The rule is implemented as specified and left in place. It is a parameter pair in
    `configs/default.yaml`, and a change to either key is a decision to be made
    deliberately, not a correction to be slipped in here.

    To make the second condition able to reject something, the background level has
    to come from outside the patch, or the comparison has to be against the patch's
    own median rather than its 5th percentile. Both change what the number means, and
    both are a different rule from the one specified.
    """
    gray = _grayscale(patch)
    return bool(gray.mean() < gray_max and gray.min() < threshold + DARK_MARGIN)


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
    rows: list[dict] | None = None,
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

    A patch is accepted when `is_tissue` accepts it against the background level
    `tissue_gray_threshold` measures, and is then written as
    `{slide_id}_{x}_{y}.png` with the source coordinates, which is what the file name
    means and what `patches.csv` records. `mpp` in an index row is the resolution the
    patch was actually read at, `info.achieved_mpp`, and not the target that was
    requested, so a run's resolution is auditable from the patch record alone.

    Every exception raised per patch is caught, logged with the slide id, the
    coordinates and the exception object, and counted as a `read_failure`; the walk
    continues, because a run that stops at the first bad tile reports one failure and
    hides the rest. `SlideError` from the plan is not caught: it describes the slide
    rather than a patch, and it is raised before `out_dir` is created so a rejected
    slide leaves the output tree alone.

    The return type carries counts only, so the index rows a run needs in order to
    write one `patches.csv` are appended to `rows` when a caller supplies a list. A
    caller that wants only the counts passes none.

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
            try:
                threshold = tissue_gray_threshold(patch, percentile)
                accepted = is_tissue(patch, threshold, gray_max)
            except Exception as exc:
                LOGGER.error("slide %s: patch at (%d, %d) raised %r", specimen.id, x, y, exc)
                failures += 1
                continue
            if not accepted:
                rejected += 1
                continue
            name = f"{specimen.id}_{x}_{y}.png"
            try:
                Image.fromarray(patch).save(out / name)
            except Exception as exc:
                LOGGER.error("slide %s: writing %s raised %r", specimen.id, name, exc)
                failures += 1
                continue
            written += 1
            if rows is not None:
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


def prepare_output(out_dir: Path) -> None:
    """Delete `out_dir` and create it empty.

    Extraction and, later, labeling write into one directory that a previous run may
    have filled. Anything left behind is a patch from a slide, a resolution or a
    configuration that is no longer the one being run, and it is indistinguishable
    from a fresh one once it is on disk, so the directory is emptied rather than
    written into. Deleting is not guarded by `ignore_errors`: a tree that cannot be
    cleared must not be written into, because the stale files would survive into the
    dataset and the run would look complete.
    """
    target = Path(out_dir)
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
