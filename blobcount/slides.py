"""Slide metadata and the resolution plan that makes every patch the same size on tissue.

Six slides that are not one magnification are the defect this module exists to correct.
The original `patch.py` read a fixed 256x256 window at level 0 from every slide, so a
20x slide at 0.499 um/px produced patches covering twice the tissue area of the 40x
slides at ~0.25 um/px. A blob-area filter expressed in pixels then labelled 90.9% of one
slide steatotic and 24.7% of another normal, and both numbers were arithmetic on an
uncontrolled field of view rather than a measurement.

A patch of `patch_size` output pixels at `target_mpp` covers `patch_size * target_mpp`
microns of tissue, and a level-`L` source pixel covers `slide_mpp * level_downsample(L)`
microns. Setting the two equal and solving for the read width gives

    required_downsample = target_mpp / slide_mpp
    level               = the pyramid level minimising |level_downsample - required|
    read_px             = round(patch_size * target_mpp / (slide_mpp * level_downsample))
    achieved_mpp        = read_px * slide_mpp * level_downsample / patch_size

A slide finer than the target has a required downsample above 1, so `read_px` exceeds
`patch_size` and the region is shrunk; a slide coarser than the target has `read_px`
below `patch_size` and the region is enlarged. Both land on `target_mpp` up to the
integer rounding of `read_px`, and that rounding is the only source of deviation: at
256 px it is under 0.2% for the six slides, against a 2% budget. The direction is the
whole point. Reading exactly `patch_size` source pixels from a 0.25 um/px slide and
stretching it to 256 would upsample by 2x and change the field of view, which is the
defect being corrected, and it would do so silently.

`achieved_mpp` is carried on `SlideInfo` per slide rather than asserted once, so the
resolution a patch was actually extracted at is auditable from the patch record alone.

`read_mpp` reads `openslide.mpp-x`, then `aperio.MPP`, and raises `SlideError` naming
the slide and both properties when neither yields a number. It never returns a
default: an assumed scale is the error being corrected here, and inventing one would
reinstate it undetectably.

`plan_resolution` and `iter_coords` are pure and take no `Config`, so the arithmetic is
testable without a slide. The two `*_from_config` adapters are the boundary that reads
`extraction.target_mpp`, `extraction.patch_size` and `extraction.step_size`, keeping
that key access in one place per value instead of spreading it across the call sites
that assemble a run.

`open_info` takes an optional `expected_mpp`, which is the scale the dataset manifest
records for the slide. The manifest documents that value as transcribed `aperio.MPP`
and this module reads the slide's own metadata, so the two are one claim read twice,
and a divergence means the provenance record no longer describes the file feeding the
patient-level split. It is cross-checked and not trusted: both present and differing by
more than `MPP_TOLERANCE` relative raises, because everywhere else in this package a
disagreement is an error rather than a preference.

**Coordinates are in source pixels.** `iter_coords` yields level-0 read origins, which
is the unit `slide.read_region` takes for its location, so a caller passes the pair
through untransformed. Its `patch_size` and `step_size` arguments are in source pixels
*at `info.level`*, and `info.width` and `info.height` are the level-0 dimensions, so
the fit test compares `patch_size * info.level_downsample` against them. At level 0
the factor is 1.0 and the two units coincide, which is why the mixing that produced
the earlier defect was invisible here and not on the slides: `2.svs` at `read_px = 508`
was walked with a 256 px step over level-0 bounds, giving 22,842 tiles where the output
grid has 5,734, and an origin 122,936 px across a 62,350 px slide. `1.svs` has
`read_px = 257`, a ratio of 1.004, so it read as correct throughout. The conversion from
the config's output-pixel `patch_size` and `step_size` into the source-space tile and
step happens once, in `iter_coords_from_config`, by the factor `read_px / patch_size`.

Two deliberate deviations from the brief's formula, both recorded because a reader
comparing code to brief should not have to guess. `read_px` is `max(1, round(...))`
rather than a bare `round(...)`: a coarse slide and a small patch can otherwise round
the read to zero pixels, and a read of nothing is not a coarser patch. The step in
`iter_coords_from_config` carries the same floor, for the same reason. Neither guard is
reachable from the brief's own test values.

Declared interface, in `__all__` order:

    SlideError
    SlideInfo
        slide_id: str
        path: Path
        width: int
        height: int
        mpp: float
        objective: int | None
        level: int
        level_downsample: float
        read_px: int
        achieved_mpp: float
    iter_coords(info: SlideInfo, patch_size: int, step_size: int) -> Iterator[tuple[int, int]]
    iter_coords_from_config(info: SlideInfo, cfg: Config | None = None) -> Iterator[tuple[int, int]]
    open_info(
        path: Path,
        slide_id: str,
        target_mpp: float,
        patch_size: int,
        *,
        expected_mpp: float | None = None,
    ) -> SlideInfo
    open_info_from_config(
        path: Path, slide_id: str, cfg: Config | None = None, *, expected_mpp: float | None = None
    ) -> SlideInfo
    plan_resolution(
        slide_mpp: float, target_mpp: float, patch_size: int, level_downsamples: Sequence[float]
    ) -> tuple[int, float, int, float]
    read_mpp(slide: Any) -> float
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blobcount.config import Config, default_config

__all__ = [
    "SlideError",
    "SlideInfo",
    "iter_coords",
    "iter_coords_from_config",
    "open_info",
    "open_info_from_config",
    "plan_resolution",
    "read_mpp",
]

MPP_PROPERTY = "openslide.mpp-x"
FALLBACK_MPP_PROPERTY = "aperio.MPP"
OBJECTIVE_PROPERTIES = ("openslide.objective-power", "aperio.AppMag")

# The relative distance the slide's own scale and the manifest's may differ by before
# the two are treated as describing different tissue. Aperio writes `aperio.MPP` to
# four decimals and OpenSlide derives `openslide.mpp-x` from the same value, so the
# two agree exactly on every slide read so far; 1e-3 is loose enough to survive a
# transcription of either, and far tighter than the ~2% a resolution budget allows.
MPP_TOLERANCE = 1e-3


class SlideError(Exception):
    """Raised when a slide cannot be read, or carries no usable resolution."""


@dataclass(frozen=True)
class SlideInfo:
    """What one slide is, and how it must be read to land on the target resolution.

    `mpp` is the scale read from the slide's own metadata, never assumed.
    `level`, `level_downsample` and `read_px` are the pyramid level chosen and the
    source width to read at it, and `achieved_mpp` is the resolution that width
    actually yields on the output patch, so a caller records what happened rather
    than what was requested. `objective` is descriptive and nullable; unlike `mpp`
    it decides nothing.
    """

    slide_id: str
    path: Path
    width: int
    height: int
    mpp: float
    objective: int | None
    level: int
    level_downsample: float
    read_px: int
    achieved_mpp: float


def _slide_name(slide: Any) -> str:
    """Return something a human recognises the slide by."""
    return str(getattr(slide, "name", None) or type(slide).__name__)


def read_mpp(slide: Any) -> float:
    """Return the microns-per-pixel scale declared by an open slide.

    `openslide.mpp-x` is read first and `aperio.MPP` second, because the first is
    the standardised property and the second is the Aperio-specific one. A property
    that is present but not a number is as unusable as a missing one, and the next
    property is tried rather than the read abandoned: a corrupt value in the
    standard key says nothing about the vendor key, and the alternative loses a
    scale that is sitting right there. When no property yields a number
    `SlideError` names the slide and both properties looked for, and says which of
    them were present. No default scale is ever returned.
    """
    properties = slide.properties
    unreadable: list[str] = []
    for name in (MPP_PROPERTY, FALLBACK_MPP_PROPERTY):
        if name not in properties:
            continue
        try:
            return float(properties[name])
        except (TypeError, ValueError):
            unreadable.append(name)
    if unreadable:
        raise SlideError(
            f"slide {_slide_name(slide)} declares resolution propert"
            f"{'y' if len(unreadable) == 1 else 'ies'} "
            f"{', '.join(repr(name) for name in unreadable)} as "
            f"{', '.join(repr(properties[name]) for name in unreadable)}, which "
            f"{'is' if len(unreadable) == 1 else 'are'} not a number, so no scale can be "
            f"read from it; {MPP_PROPERTY!r} and {FALLBACK_MPP_PROPERTY!r} are the only "
            f"resolution properties read, and neither yielded a number"
        )
    raise SlideError(
        f"slide {_slide_name(slide)} carries no resolution metadata: neither "
        f"{MPP_PROPERTY!r} nor {FALLBACK_MPP_PROPERTY!r} is present, so the "
        f"microns-per-pixel scale is unknown and no default is assumed"
    )


def _read_objective(slide: Any) -> int | None:
    """Return the nominal objective power, or None when absent or unreadable.

    Unlike the resolution this decides nothing about how a slide is read, so a
    missing or malformed value is reported as unknown rather than raised.
    """
    properties = slide.properties
    for name in OBJECTIVE_PROPERTIES:
        if name not in properties:
            continue
        try:
            return int(float(properties[name]))
        except (TypeError, ValueError):
            return None
    return None


def plan_resolution(
    slide_mpp: float,
    target_mpp: float,
    patch_size: int,
    level_downsamples: Sequence[float],
) -> tuple[int, float, int, float]:
    """Plan a read for `slide_mpp` that puts every patch at `target_mpp`.

    Returns `(level, level_downsample, read_px, achieved_mpp)`, where `read_px` is
    the number of source pixels to read across for a `patch_size` output patch and
    `achieved_mpp` is the resolution that read actually produces. A slide finer than
    the target yields a `read_px` above `patch_size`, so the region is shrunk; a
    slide coarser than the target yields a `read_px` below it, so the region is
    enlarged. Neither direction is a special case here: it falls out of the ratio.

    The level chosen is the one whose downsample is nearest the required downsample,
    with the lower level winning a tie so the result is deterministic. `read_px` is
    held at 1 or more, because a coarse slide and a small patch can otherwise round
    to a read of nothing.

    `SlideError` is raised for a pyramid with no level 0, a level 0 whose downsample
    is not 1.0, a non-positive scale or a non-positive patch size, since each of those
    makes the plan undefined rather than merely inaccurate. The level-0 check is not
    pedantry: `read_px` and the tile grid are both expressed against the level-0
    dimensions, so a slide whose first level is already downsampled would be planned
    and walked against coordinates that are not its own.
    """
    if not level_downsamples:
        raise SlideError(
            "cannot plan a resolution: the pyramid holds no levels, and level 0 is "
            "the full-resolution image every plan is expressed against"
        )
    if level_downsamples[0] != 1.0:
        raise SlideError(
            f"cannot plan a resolution: level 0 must be the full-resolution image, so "
            f"its downsample must be exactly 1.0, got {level_downsamples[0]!r}; every "
            f"read size and grid coordinate in this module is expressed against the "
            f"level-0 dimensions, which that pyramid does not have"
        )
    for name, value in (("slide_mpp", slide_mpp), ("target_mpp", target_mpp), ("patch_size", patch_size)):
        if value <= 0:
            raise SlideError(f"cannot plan a resolution: {name} must be positive, got {value!r}")

    required_downsample = target_mpp / slide_mpp
    level, level_downsample = min(
        enumerate(level_downsamples),
        key=lambda item: (abs(item[1] - required_downsample), item[0]),
    )
    read_px = max(1, round(patch_size * target_mpp / (slide_mpp * level_downsample)))
    achieved_mpp = read_px * slide_mpp * level_downsample / patch_size
    return level, level_downsample, read_px, achieved_mpp


def _check_declared_mpp(slide_id: str, mpp: float, expected_mpp: float | None) -> None:
    """Fail closed when the manifest and the slide disagree about the scale.

    `data/manifest.yaml` records `mpp` as transcribed `aperio.MPP` and this module
    reads the slide's own metadata, so the two are the same claim read twice, and a
    divergence means the provenance record no longer describes the file that feeds
    the patient-level split. The slide is not silently preferred over the manifest
    either: the manifest is the record the split is computed from, and a slide that
    was replaced under a fixed name is exactly the case a silent preference hides.
    `expected_mpp` of `None` is the manifest carrying no value, which is a fact about
    the record and not a contradiction, so it is not checked.
    """
    if expected_mpp is None:
        return
    if abs(expected_mpp) == 0.0:
        raise SlideError(
            f"slide {slide_id} declares {mpp!r} um/px but the manifest records 0.0 "
            f"um/px, which cannot be compared relatively; fix the manifest row"
        )
    divergence = abs(mpp - expected_mpp) / abs(expected_mpp)
    if divergence > MPP_TOLERANCE:
        raise SlideError(
            f"slide {slide_id} declares {mpp!r} um/px but the manifest records "
            f"{expected_mpp!r} um/px, a relative difference of {divergence:.2%} against a "
            f"{MPP_TOLERANCE:.1e} tolerance, so the manifest no longer describes this "
            f"file; fix the manifest rather than the reader"
        )


def open_info(
    path: Path,
    slide_id: str,
    target_mpp: float,
    patch_size: int,
    *,
    expected_mpp: float | None = None,
) -> SlideInfo:
    """Open a slide, read its metadata, and plan the read that hits `target_mpp`.

    `path` is opened read-only and closed before this returns: reading metadata is
    cheap, and the planned region is read later by the caller, which reopens the
    slide itself. The dimensions are the level-0 dimensions, and `level` is
    expressed against them.

    `expected_mpp` is optional and defaults to `None`, which is the manifest
    recording no scale for the slide; when it is given it is the `Specimen.mpp` the
    dataset manifest carries, and it is cross-checked against the slide's own
    metadata rather than preferred to it. See `_check_declared_mpp`.

    `SlideError` is raised for a slide that will not open, for one that declares no
    resolution, for one whose declared resolution contradicts `expected_mpp`, and
    for a pyramid that cannot be planned against.
    """
    import openslide

    slide_path = Path(path)
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except (OSError, openslide.OpenSlideError) as error:
        raise SlideError(f"slide {slide_id} at {slide_path} cannot be opened: {error}") from error

    with slide:
        mpp = read_mpp(slide)
        _check_declared_mpp(slide_id, mpp, expected_mpp)
        objective = _read_objective(slide)
        width, height = slide.dimensions
        level, level_downsample, read_px, achieved_mpp = plan_resolution(
            mpp, target_mpp, patch_size, tuple(slide.level_downsamples)
        )

    return SlideInfo(
        slide_id=slide_id,
        path=slide_path,
        width=int(width),
        height=int(height),
        mpp=mpp,
        objective=objective,
        level=level,
        level_downsample=level_downsample,
        read_px=read_px,
        achieved_mpp=achieved_mpp,
    )


def _starts(limit: int, extent: float, stride: float) -> Iterator[int]:
    """Yield the tile starts spaced by `stride` whose `extent` still fits in `limit`.

    The starts are level-0 coordinates and `limit` is a level-0 dimension, while
    `extent` and `stride` arrive in source pixels at some level above 0. Walking by
    index and rounding the product rather than by a float `range` step keeps every
    yielded coordinate an `int`, which matters because these become pixel offsets and
    file names.
    """
    if extent > limit:
        return
    for index in range(int((limit - extent) // stride) + 1):
        yield round(index * stride)


def iter_coords(info: SlideInfo, patch_size: int, step_size: int) -> Iterator[tuple[int, int]]:
    """Yield `(x, y)` read origins in row-major order, dropping partial tiles.

    Every number here is a source pixel. `patch_size` and `step_size` are the tile
    width and the stride **at `info.level`**, so the reading task passes `info.read_px`
    and the configured step converted by `read_px / patch_size`; the yielded pair is a
    level-0 coordinate, which is what `slide.read_region` takes for its location, so a
    caller hands it over untransformed.

    `info.width` and `info.height` are the level-0 dimensions and a tile read at
    `info.level` covers `patch_size * info.level_downsample` of them, so the fit test
    carries that factor. At level 0 it is 1.0 and the source and output units coincide;
    above it they do not, and a tile bounded as if they did runs off the edge of the
    slide, where `read_region` returns transparent black rather than raising.

    A tile is yielded only when the whole window fits, so the trailing strip on the
    right and bottom edges is dropped rather than read as a smaller tile, which is the
    rule `patch.py` applied and the reason a partial tile never reaches the disk. A
    `step_size` below `patch_size` overlaps neighbours, which is the point of the
    setting.
    """
    for name, value in (("patch_size", patch_size), ("step_size", step_size)):
        if value <= 0:
            raise SlideError(f"cannot walk a slide grid: {name} must be positive, got {value!r}")

    extent = patch_size * info.level_downsample
    stride = step_size * info.level_downsample
    for y in _starts(info.height, extent, stride):
        for x in _starts(info.width, extent, stride):
            yield x, y


def open_info_from_config(
    path: Path,
    slide_id: str,
    cfg: Config | None = None,
    *,
    expected_mpp: float | None = None,
) -> SlideInfo:
    """Plan a read for `path` using the configured target resolution and patch size.

    `expected_mpp` is passed through to `open_info` unchanged; see that function for
    what it cross-checks.
    """
    active = default_config() if cfg is None else cfg
    return open_info(
        Path(path),
        slide_id,
        target_mpp=float(active.get("extraction.target_mpp")),
        patch_size=int(active.get("extraction.patch_size")),
        expected_mpp=expected_mpp,
    )


def iter_coords_from_config(info: SlideInfo, cfg: Config | None = None) -> Iterator[tuple[int, int]]:
    """Walk `info` using the configured patch size and step, in source coordinates.

    `extraction.patch_size` and `extraction.step_size` are expressed in output pixels
    at `extraction.target_mpp` and `iter_coords` is expressed in source pixels at
    `info.level`, so this is the one place the two grids meet: the tile becomes
    `info.read_px`, which is what the read actually takes, and the step is scaled by
    `info.read_px / patch_size`. Passing either through unscaled is the defect this
    module exists to correct, so the conversion is here and nowhere else.

    The scaled step is held at 1 or more, for the same reason `read_px` is: a fine
    configured step and a fine slide can otherwise round to no movement at all, which
    would read the same patch repeatedly.
    """
    active = default_config() if cfg is None else cfg
    patch_size = int(active.get("extraction.patch_size"))
    step_size = int(active.get("extraction.step_size"))
    return iter_coords(
        info,
        info.read_px,
        max(1, round(step_size * info.read_px / patch_size)),
    )
