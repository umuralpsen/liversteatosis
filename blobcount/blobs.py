"""Blob counting in square microns: the second half of the fix, and the half that makes Task 3 pay.

`patch.py` counted lipid-vacuole-like blobs with an area filter written in pixels,
50 to 2000 px^2, and that filter meant a different area of tissue on every slide in
the set. At 0.252 um/px the floor is 3.2 um^2 and the ceiling 127 um^2; at 0.499 um/px,
which is the one 20x slide, the same numbers are 6.2 um^2 and 498 um^2. The filter was
therefore not a criterion applied to tissue, it was a criterion applied to a
magnification. It is what labelled 90.9% of one slide steatotic and 24.7% of another
normal, and both figures were arithmetic on an uncontrolled field of view.

The criterion itself was not the problem, and it is preserved exactly. `blobs.area_um2_min`
is 3.1 um^2 and `blobs.area_um2_max` is 125.0 um^2, which are the 50 px^2 and 2000 px^2
the original filter used *evaluated at 0.25 um/px*, so the biological question asked is
unchanged and the two can be compared against the original numbers without conversion.
What changed is the unit, and the conversion belongs here:

    a patch covers `mpp` um per pixel, so `A` um^2 is `A / mpp**2` px^2

`count_blobs` takes the `mpp` of the patch in front of it and converts both bounds with
it. That argument is **required, not defaulted**, and the default is the defect: a
callable `mpp` with a fallback resolution is a call that can silently count at the wrong
scale, and the original code is what that looks like. The number to pass is
`SlideInfo.achieved_mpp`, the resolution the patch was actually read at and recorded in
the `mpp` column of `patches.csv`, not `extraction.target_mpp` and not the slide's own
`mpp`. Task 3 made every patch cover the same physical area; this is the filter that
makes that normalisation mean something, and the two are only correct together.

`circularity_min` is `4 * pi * area / perimeter**2`, a ratio of two squared lengths, so
it is already scale-invariant and is applied to the pixel measurements unchanged. Only
the area needs converting.

**The threshold is a percentile of the patch, so it always cuts the patch, and what it
cuts depends on how far the patch's own grayscale spreads.** `gray_threshold_from_patch`
returns the configured percentile of `patch`'s grayscale, which is a statement about the
distribution in front of it and not about the tissue: by construction roughly
`100 - percentile` percent of the patch becomes foreground, whatever the patch contains.
On a synthetic field of 98% mid-gray with a few pale discs in it, the 95th percentile is
the field value 180, the threshold is equal to the background, nothing is foreground and
the patch counts zero blobs.

Four of the five windows of `data/4.svs` sampled at 0.5 um/px are nearly flat: a mean
near 236 over a range of about ten levels, a grayscale standard deviation of 1.3 to 1.7,
and a 95th percentile that selects the top 5% of a two-level spread. **Those four
windows were never checked for tissue and they are slide background, not tissue**, so
their flatness is a fact about glass and not a description of the material the counting
runs over. What they do show is that the count is unstable: over five sampled windows of
the same slide at the same percentile the counts ran 0, 1, 1, 7 and 18, and at the 99th
percentile every one of them was 0. None of those is yet a measurement of steatosis.

The distribution that does describe the material is the one over the accepted tissue
patches. That figure is **one measurement and not a settled property of the accepted
patches**: over 120 of the patches the original extraction accepted, which is known tissue
under a mean of 210, the grayscale standard deviation is 16.8 to 59.5 levels and the 95th
percentile is 84 to 247 with a median of 213. A second, independent measurement of the
same population, drawn across all six slides from a pool of 12,400 accepted patches and
subsampled to 120 at a recorded seed, gives a **narrower** standard deviation of 24.3 to
57.9 and a 95th percentile of 183.3 to 248.3 with a median of 225.3. The two disagree past
the first decimal and neither is a constant; what they agree on is the point of this
paragraph, which is that the accepted patches have a structure to cut rather than a
rounding error to resolve. So the two-level spread of the background windows is not the
number the ablation needs to start from: that accepted-patch distribution is.

The rule is the brief's and the parameter is `blobs.gray_percentile` in
`configs/default.yaml`; this module reads it rather than restating it, and a threshold
smoothed, floored or replaced here would be a constant nobody has measured. What the
rule costs across the six slides is a question for the Task 7 ablation, which is written
to measure it.

**Stain normalization is implemented, tested, and off.** `blobs.stain_normalization` is
`false` in `configs/default.yaml`, `prepare_patch` is the only thing that consults it, and
with the key false it returns the patch untouched without calling `normalize_stain` at
all. `tests/test_blobs.py` asserts that with a stub that raises if it is called, so
"off by default" is a tested statement rather than a claim about a boolean.

Declared interface, in `__all__` order:

    BlobParams
        area_um2_min: float
        area_um2_max: float
        circularity_min: float
        gray_threshold: float
        morph_kernel_size: int
        morph_iterations: int
    area_px2(area_um2: float, mpp: float) -> float
    count_blobs(patch: np.ndarray, params: BlobParams, mpp: float) -> int
    gray_threshold_from_patch(patch: np.ndarray, percentile: float) -> float
    normalize_stain(patch: np.ndarray) -> np.ndarray
    params_from_config(patch: np.ndarray, cfg: Config | None = None) -> BlobParams
    prepare_patch(patch: np.ndarray, cfg: Config | None = None) -> np.ndarray
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from blobcount.config import Config, default_config
from blobcount.extract import _grayscale

__all__ = [
    "BlobParams",
    "area_px2",
    "count_blobs",
    "gray_threshold_from_patch",
    "normalize_stain",
    "params_from_config",
    "prepare_patch",
]

# The Reinhard targets, in the 8-bit Lab encoding `cv2` produces from a uint8 BGR
# image: L* scaled to 0-255, a* and b* offset by 128. (128, 128, 128) is therefore a
# mid-lightness neutral in that encoding and not the sRGB gray 128, which sits at L 137.
_TARGET_LAB_MEAN = np.array([128.0, 128.0, 128.0])
_TARGET_LAB_STD = np.array([60.0, 60.0, 60.0])

# Below this a per-channel standard deviation is treated as zero and left unscaled. A
# uniform patch has one, and a bare `target / std` divides by it, which NumPy reports
# as a RuntimeWarning and `filterwarnings = ["error"]` turns into a test failure.
_STD_FLOOR = 1e-6


@dataclass(frozen=True)
class BlobParams:
    """The counting criterion for one patch, in the units the criterion is stated in.

    The two area bounds are square microns, not pixels, and nothing in this class
    converts them: `count_blobs` needs the patch's own `mpp` to do that, and a
    conversion that happened here would be one reading a resolution from somewhere
    other than the patch record. `gray_threshold` is a grayscale level measured on
    that patch, so it is per patch too and arrives from `gray_threshold_from_patch`
    rather than from the configuration; `params_from_config` is the one function that
    joins the two.
    """

    area_um2_min: float
    area_um2_max: float
    circularity_min: float
    gray_threshold: float
    morph_kernel_size: int
    morph_iterations: int


def area_px2(area_um2: float, mpp: float) -> float:
    """Return `area_um2` square microns as a pixel count at `mpp` microns per pixel.

    A patch covers `mpp**2` um^2 per pixel, so the conversion divides rather than
    multiplies, and the square is what makes it a length that has to be squared: a
    scale factor applied once would convert a length and leave the area wrong by a
    factor of `mpp`.

    A non-positive `mpp` raises `ValueError` rather than returning a number. At zero the
    conversion is undefined, and a negative one is not an inversion but something worse
    for a filter to meet: `mpp**2` is positive, so `-0.5` returns the pixel area for
    `0.5` and the caller gets a plausible count for a resolution no patch was read at.
    `SlideInfo.achieved_mpp` comes from a plan that already refuses a non-positive
    scale, so this is the boundary check for a caller that passes something else, and
    the failure it prevents is a confident count at a scale nothing recorded.
    """
    if mpp <= 0:
        raise ValueError(f"mpp must be positive to convert an area, got {mpp!r}")
    return area_um2 / (mpp**2)


def gray_threshold_from_patch(patch: np.ndarray, percentile: float) -> float:
    """Return the `percentile` of `patch`'s own grayscale, as the threshold to binarise at.

    A high percentile rather than a constant, so the threshold follows the patch
    instead of assuming a staining intensity. Grayscale is the mean of the colour
    channels, and `_grayscale` is imported from `blobcount.extract` rather than
    written again here so that the two modules cannot come to mean different things by
    that word: the threshold this returns is binarised by `count_blobs`, and
    `extraction.tissue_gray_max` is compared against the same definition in
    `is_tissue`.

    **This number says how far the patch spreads, not what the patch contains.** A
    percentile is taken over the whole patch, so a patch that is 98% mid-gray field with
    a few pale discs in it returns the field value at the 95th percentile and the disc
    value only from the 98th, and the threshold at 95% is therefore equal to the
    background it is supposed to rise above: `cv2.threshold` keeps pixels strictly above
    it, and a patch whose 95th percentile *is* its background value counts nothing. A
    patch whose grayscale is nearly flat puts the threshold in the middle of a very small
    spread, where which pixels clear it is decided by a level or two. Both are what the
    rule says; the rule is `blobs.gray_percentile`, and see the module docstring for what
    it measured on real material.

    Total for every array with a channel axis, as `tissue_gray_threshold` is: a
    constant patch has its percentile equal to the constant, so a blank patch yields a
    number and never a `NaN`.
    """
    return float(np.percentile(_grayscale(patch), percentile))


def count_blobs(patch: np.ndarray, params: BlobParams, mpp: float) -> int:
    """Return how many blobs in `patch` match `params`, with the area bounds in um^2.

    The patch is reduced to its channel-mean grayscale, binarised at
    `params.gray_threshold`, opened with a `params.morph_kernel_size` square kernel
    `params.morph_iterations` times to drop the single-pixel and thin responses a
    threshold leaves behind, and the remaining external contours are the candidates.
    A candidate is counted when its area in pixels falls strictly between
    `area_px2(params.area_um2_min, mpp)` and `area_px2(params.area_um2_max, mpp)` and
    its circularity is above `params.circularity_min`.

    **`mpp` is required.** It is the microns per pixel of this patch and nothing else
    can stand in for it: the configuration carries the target resolution, the slide
    carries its own, and the patch carries the one it was actually read at, which is
    the third of those and the only one that describes the array in hand. A default
    here would be a way to call this function without stating the resolution, which is
    how a pixel-valued filter came to be applied to a 20x slide as if it were a 40x
    one. `SlideInfo.achieved_mpp` is what a caller passes. The conversion runs first,
    before the patch is thresholded, opened and contoured, because a non-positive
    `mpp` is a caller error and is worth reporting before the work rather than after it.

    Both area bounds are exclusive. A contour whose area equals a bound exactly is not
    counted, which is stated because the bound is a limit on the criterion and a blob
    sitting on a limit is not inside it; `cv2.contourArea` returns a float derived from
    an integer count, so exact equality is rare in practice.

    The grayscale is **rounded** to `uint8`, not truncated, before the threshold is
    applied. A mean of three channels lands on a third of a level, so truncation moves
    the effective cut up to a whole level above the threshold that was asked for, in
    one direction, always. Rounding leaves an error of at most half a level whose sign
    follows the third, so it has no direction to correct: the argument for it is that a
    bias of a known sign is worth removing, not that the bias is large. The spread of the
    accepted tissue patches is a measured sample and is cited as one. One measurement over
    120 of the patches the original extraction accepted gives a grayscale standard
    deviation of 16.8 to 59.5 levels and a second, independent one over the same population
    gives 24.3 to 57.9, so a half-level bias is about three per cent of that spread at the
    narrow end of the first and under one per cent at its wide end, and is not what decides
    a count. The module docstring carries the 95th percentile of the same two measurements.

    Where the difference is large is the flat case, and there the
    material is glass: on one unverified background window of `data/4.svs` at 0.5 um/px,
    standard deviation 1.4 levels, the 95th percentile counted 8 truncated against 18
    rounded, with 4.9% of pixels above the threshold against 5.8% where the percentile
    asks for 5%. Both figures describe the rule on a near-flat distribution and neither
    describes an accepted patch; the module docstring carries the tissue distribution.

    The circularity denominator is not guarded. A contour reaching that line has an
    area above `area_px2_min` and therefore above zero, and a closed contour enclosing
    a positive area has a positive perimeter, so a zero divisor is not reachable and a
    test for it would be a clause that could never fail. `cv2.arcLength` on a one-pixel
    or two-point contour returns a positive length in any case.

    `cv2` is imported here rather than at module scope, for the reason
    `blobcount.extract.read_patch` imports it inside the read: it is a native wheel
    that fails to import where its system libraries are absent, and a module-scope
    import would make that failure a collection error for every test in the suite,
    including the ones that touch no image.
    """
    import cv2

    area_min_px = area_px2(params.area_um2_min, mpp)
    area_max_px = area_px2(params.area_um2_max, mpp)

    gray = np.rint(_grayscale(patch)).astype(np.uint8)
    binary = cv2.threshold(gray, params.gray_threshold, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((params.morph_kernel_size, params.morph_kernel_size), np.uint8)
    opened = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, kernel, iterations=params.morph_iterations
    )
    contours = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    counted = 0
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if not area_min_px < area < area_max_px:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if 4.0 * math.pi * area / (perimeter**2) > params.circularity_min:
            counted += 1
    return counted


def normalize_stain(patch: np.ndarray) -> np.ndarray:
    """Return `patch` with its Lab channels shifted and scaled onto fixed targets.

    Reinhard normalization, per channel, in the 8-bit Lab space of a uint8 BGR image:
    each channel is moved so its mean is `(128, 128, 128)` and scaled so its standard
    deviation is `(60, 60, 60)`, which is a statement about the distribution of a
    staining and not about any one pixel. The result is clipped, converted back to BGR
    and returned as `uint8`, so it is a drop-in replacement for the patch it was given.

    A uniform patch has a per-channel standard deviation of zero, and `np.where`
    evaluates both of its branches, so the divisor is replaced before it is used and
    no `RuntimeWarning` is emitted for a value the project turns into a test failure.
    A channel left at that zero deviation is already exactly on the target mean, so
    the substitution is not visible in the output: the two forms of the guard, an
    `np.where` on the divisor alone and an `np.where` on the ratio as well, produce
    bit-identical arrays on a uniform patch, on random noise and on a patch one pixel
    off uniform, which is why only the first is here.

    `patch` is a uint8 BGR image, the format `read_patch` returns. A float array or a
    two-dimensional one is rejected by `cv2.cvtColor` rather than by a check here: the
    precondition is stated by the interface and `cv2`'s error names it.
    """
    import cv2

    lab = cv2.cvtColor(np.ascontiguousarray(patch), cv2.COLOR_BGR2LAB)
    channels = lab.reshape(-1, 3).astype(np.float64)
    observed_mean = channels.mean(axis=0)
    observed_std = channels.std(axis=0)
    scale = _TARGET_LAB_STD / np.where(observed_std > _STD_FLOOR, observed_std, 1.0)
    normalized = (channels - observed_mean) * scale + _TARGET_LAB_MEAN
    return cv2.cvtColor(
        np.clip(np.rint(normalized), 0, 255).reshape(lab.shape).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def params_from_config(patch: np.ndarray, cfg: Config | None = None) -> BlobParams:
    """Return the counting criterion for `patch`, from the configuration.

    This is the boundary that reads `blobs.area_um2_min`, `blobs.area_um2_max`,
    `blobs.circularity_min`, `blobs.morph_kernel_size`, `blobs.morph_iterations` and
    `blobs.gray_percentile`, so those six keys are read in one place instead of at
    every call site that assembles a run. `gray_threshold` is the sixth field and is
    the one value a configuration cannot supply, because it is a measurement of this
    patch: `blobs.gray_percentile` states which percentile of it to take, and that is
    what this function does with the key.

    `cfg` defaults to the process-wide default configuration, as the other
    `*_from_config` adapters in this package do.

    **The `gray_threshold` here is superseded by a per-slide derivation and is not a
    second live way to set one.** No task in the plan calls this function:
    `label_patches` reads the `tissue_gray` column of the patch index and derives one
    threshold per slide at `blobs.gray_percentile`, and the Task 7 ablation reads the
    same column. The two boundaries disagree: a percentile of the patch in hand
    against a percentile of the slide's own background levels. The per-slide one is
    the one the study uses, because a patch's own percentile moves with the patch
    and a slide's does not. This function is kept for its other five keys, and a later
    reader should not treat both thresholds as live.
    """
    active = default_config() if cfg is None else cfg
    return BlobParams(
        area_um2_min=float(active.get("blobs.area_um2_min")),
        area_um2_max=float(active.get("blobs.area_um2_max")),
        circularity_min=float(active.get("blobs.circularity_min")),
        gray_threshold=gray_threshold_from_patch(
            patch, float(active.get("blobs.gray_percentile"))
        ),
        morph_kernel_size=int(active.get("blobs.morph_kernel_size")),
        morph_iterations=int(active.get("blobs.morph_iterations")),
    )


def prepare_patch(patch: np.ndarray, cfg: Config | None = None) -> np.ndarray:
    """Return `patch` with the configured preprocessing applied, which by default is none.

    `blobs.stain_normalization` is the only key this reads, and it is `false` in
    `configs/default.yaml`, so the default path returns the array it was handed without
    calling `normalize_stain` at all. The normalization is implemented and tested because
    the key exists and an ablation needs the switch; whether turning it on moves a
    study's numbers is a measurement to be taken and recorded, not a default to be
    picked here.

    `cfg` defaults to the process-wide default configuration, as the other
    `*_from_config` adapters in this package do.
    """
    active = default_config() if cfg is None else cfg
    if bool(active.get("blobs.stain_normalization")):
        return normalize_stain(patch)
    return patch
