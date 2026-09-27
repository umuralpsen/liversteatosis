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
the slide and both properties when neither is present. It never returns a default: an
assumed scale is the error being corrected here, and inventing one would reinstate it
undetectably.

`plan_resolution` and `iter_coords` are pure and take no `Config`, so the arithmetic is
testable without a slide. The two `*_from_config` adapters are the boundary that reads
`extraction.target_mpp`, `extraction.patch_size` and `extraction.step_size`, keeping
that key access in one place per value instead of spreading it across the call sites
that assemble a run.

`iter_coords` walks the output-pixel grid and stops a row early at the right and bottom
edges, mirroring the `x + PATCH_SIZE > w or y + PATCH_SIZE > h` guard in `patch.py`, so
no partial tile is ever produced. A coordinate on that grid is in output pixels while
`read_px` is in source pixels, so a caller hands `read_region` a location scaled by
`read_px / patch_size`; that conversion belongs to the reading task, not here.

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
    open_info(path: Path, slide_id: str, target_mpp: float, patch_size: int) -> SlideInfo
    open_info_from_config(path: Path, slide_id: str, cfg: Config | None = None) -> SlideInfo
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
    the standardised property and the second is the Aperio-specific one. A value
    that is present but not a number is as unusable as a missing one, so it raises
    `SlideError` naming the property that failed rather than falling through to a
    guess. When neither property is present `SlideError` names the slide and both
    properties looked for. No default scale is ever returned.
    """
    properties = slide.properties
    for name in (MPP_PROPERTY, FALLBACK_MPP_PROPERTY):
        if name not in properties:
            continue
        try:
            return float(properties[name])
        except (TypeError, ValueError) as error:
            raise SlideError(
                f"slide {_slide_name(slide)} declares resolution property {name!r} as "
                f"{properties[name]!r}, which is not a number, so no scale can be read "
                f"from it"
            ) from error
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

    `SlideError` is raised for a pyramid with no level 0, a non-positive scale or a
    non-positive patch size, since each of those makes the plan undefined rather
    than merely inaccurate.
    """
    if not level_downsamples:
        raise SlideError(
            "cannot plan a resolution: the pyramid holds no levels, and level 0 is "
            "the full-resolution image every plan is expressed against"
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


def open_info(path: Path, slide_id: str, target_mpp: float, patch_size: int) -> SlideInfo:
    """Open a slide, read its metadata, and plan the read that hits `target_mpp`.

    `path` is opened read-only and closed before this returns: reading metadata is
    cheap, and the planned region is read later by the caller, which reopens the
    slide itself. The dimensions are the level-0 dimensions, and `level` is
    expressed against them.

    `SlideError` is raised for a slide that will not open, for one that declares no
    resolution, and for a pyramid that cannot be planned against.
    """
    import openslide

    slide_path = Path(path)
    try:
        slide = openslide.OpenSlide(str(slide_path))
    except (OSError, openslide.OpenSlideError) as error:
        raise SlideError(f"slide {slide_id} at {slide_path} cannot be opened: {error}") from error

    with slide:
        mpp = read_mpp(slide)
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


def iter_coords(info: SlideInfo, patch_size: int, step_size: int) -> Iterator[tuple[int, int]]:
    """Yield `(x, y)` patch origins in row-major order, dropping partial tiles.

    A tile is yielded only when the whole `patch_size` window fits, so the trailing
    strip on the right and bottom edges of the slide is dropped rather than read as
    a smaller tile, which is the same rule `patch.py` applied and the reason a
    partial tile never reaches the disk. A `step_size` below `patch_size` overlaps
    neighbours, which is the point of the setting. Coordinates are in output pixels
    at the target resolution; the caller scales them by `info.read_px / patch_size`
    before passing them to a source-pixel read.
    """
    for name, value in (("patch_size", patch_size), ("step_size", step_size)):
        if value <= 0:
            raise SlideError(f"cannot walk a slide grid: {name} must be positive, got {value!r}")

    for y in range(0, info.height, step_size):
        if y + patch_size > info.height:
            break
        for x in range(0, info.width, step_size):
            if x + patch_size > info.width:
                break
            yield x, y


def open_info_from_config(path: Path, slide_id: str, cfg: Config | None = None) -> SlideInfo:
    """Plan a read for `path` using the configured target resolution and patch size."""
    active = default_config() if cfg is None else cfg
    return open_info(
        Path(path),
        slide_id,
        target_mpp=float(active.get("extraction.target_mpp")),
        patch_size=int(active.get("extraction.patch_size")),
    )


def iter_coords_from_config(info: SlideInfo, cfg: Config | None = None) -> Iterator[tuple[int, int]]:
    """Walk `info` using the configured patch size and step."""
    active = default_config() if cfg is None else cfg
    return iter_coords(
        info,
        patch_size=int(active.get("extraction.patch_size")),
        step_size=int(active.get("extraction.step_size")),
    )
