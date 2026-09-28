import numpy as np
import pytest

import blobcount.blobs as blobs_module
from blobcount.blobs import (
    BlobParams,
    area_px2,
    count_blobs,
    gray_threshold_from_patch,
    normalize_stain,
    params_from_config,
    prepare_patch,
)
from blobcount.config import load


def _cv2():
    """Return the `cv2` module, skipping the test when it will not import.

    `cv2` is not imported at module scope here, for the reason
    `blobcount.extract.read_patch` imports it inside the read: it is a native wheel
    that fails to import where its system libraries are absent, and a module-scope
    import turns that into a collection *error* for every test in this file,
    including `test_area_conversion_matches_physical_units`, which touches no
    image. A skip from inside this helper costs the tests that draw or count, which
    is the honest price.
    """
    return pytest.importorskip(
        "cv2",
        reason="opencv-python cannot be imported, so no disc can be drawn and no "
        "patch can be thresholded, opened or contoured",
    )


def _discs(n, radius, size=256):
    cv2 = _cv2()
    img = np.full((size, size, 3), 180, dtype=np.uint8)
    for i in range(n):
        cv2.circle(img, (60 + i * 90, 128), radius, (250, 250, 250), -1)
    return img


def _bar(w, h, size=256):
    cv2 = _cv2()
    img = np.full((size, size, 3), 180, dtype=np.uint8)
    cv2.rectangle(img, (60, 128 - h // 2), (60 + w, 128 + h // 2), (250, 250, 250), -1)
    return img


def _params(threshold=200.0):
    return BlobParams(area_um2_min=3.1, area_um2_max=125.0, circularity_min=0.6,
                      gray_threshold=threshold, morph_kernel_size=3, morph_iterations=2)


def _uniform(value=(140, 90, 160), size=16):
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_area_conversion_matches_physical_units():
    # 50 px^2 at 0.25 um/px is 3.1 um^2; the same physical area at 0.5 um/px is 12.4 px^2
    assert area_px2(3.1, 0.25) == pytest.approx(49.6, rel=0.01)
    assert area_px2(3.1, 0.5) == pytest.approx(12.4, rel=0.01)
    # the ceiling the same way: 2000 px^2 at 0.25 um/px is 125 um^2, and 500 px^2 at
    # 0.5 um/px. Both are exact, not approximate, so they are pinned exactly: a
    # conversion that quietly rounded would still satisfy the two `approx` lines above.
    assert area_px2(125.0, 0.25) == 2000.0
    assert area_px2(125.0, 0.5) == 500.0


def test_counts_exact_number_of_discs():
    assert count_blobs(_discs(3, 12), _params(), 0.5) == 3


def test_empty_patch_counts_zero():
    assert count_blobs(np.full((256, 256, 3), 180, dtype=np.uint8), _params(), 0.5) == 0


def test_same_physical_disc_counts_same_at_two_resolutions():
    # a disc of 6 um radius is 24 px at 0.25 um/px and 12 px at 0.5 um/px
    #
    # The fine image is 512 px, not the 256 px the brief used. `_discs` spaces the
    # centres 90 px apart from x=60, so the fourth of four sits at x=330 and a 256 px
    # canvas truncates it: the fine image held three discs and the coarse one four, and
    # the comparison that failed was 4 against 3 rather than one resolution against
    # another. The canvas is the only thing this case is not about; the discs are.
    coarse = _discs(4, 24, size=512)
    fine = _discs(4, 12, size=512)
    assert count_blobs(coarse, _params(), 0.25) == count_blobs(fine, _params(), 0.5) == 4


def test_mpp_argument_actually_changes_the_area_filter():
    # A 4 px disc opens to 34 px^2 (measured), which is 2.1 um^2 at 0.25 um/px and so
    # falls below the 3.1 um^2 floor, and 8.5 um^2 at 0.5 um/px, which is inside the
    # 3.1-125 um^2 range. If the mpp argument were ignored the two counts would be
    # equal, which is the original defect in miniature.
    #
    # The task brief specified a 12 px radius here and claimed 2.8 um^2 for it, which
    # is the area of a 3.8 px disc: a 12 px radius opens to 404 px^2, or 25 um^2 at
    # 0.25 um/px, comfortably inside the range, and the brief's own
    # `test_counts_exact_number_of_discs` counts that same disc at the coarse
    # resolution. No implementation can reject a blob at 0.25 um/px and accept it at
    # 0.5 um/px unless its area falls between 12.4 px^2 and 49.6 px^2. Radius 3 lands
    # there too, at a measured 16 px^2, and radius 4 at 34 px^2 is used because it sits
    # nearer the middle of that window than 16 does. The intent of the case, the name,
    # and the direction of the two counts are unchanged.
    assert count_blobs(_discs(3, 4), _params(), 0.25) == 0
    assert count_blobs(_discs(3, 4), _params(), 0.5) == 3


def test_disc_outside_area_range_is_rejected():
    # radius 40 px at 0.5 um/px is 1257 um^2, far above the 125 um^2 ceiling
    assert count_blobs(_discs(2, 40), _params(), 0.5) == 0


def test_elongated_contour_inside_the_area_range_is_rejected_on_circularity():
    # The area filter and the circularity filter are separate clauses and only the
    # first of them was covered. This bar is 5 x 40 px at 0.25 um/px: a measured
    # 200 px^2 contour, 12.5 um^2, comfortably inside the 3.1-125 um^2 range, so the
    # area clause passes it. Its circularity is 4*pi*200/90**2 = 0.31, under the 0.6
    # floor, so the second clause rejects it. The 5 px width is what lets it survive a
    # 3x3 opening run twice: a 3 px bar is erased outright and would leave nothing to
    # reject.
    #
    # At 0.5 um/px the window is 12.4-49.6 px^2, and the narrowest bar that survives the
    # opening in that window is too stubby to fall under 0.6, so the case is read at
    # 0.25 um/px where the window is 49.6-2000 px^2. The disc is the control for the
    # other half: a measured 170 px^2 at circularity 0.89, over the same floor and the
    # same range, and counted.
    assert count_blobs(_bar(5, 40), _params(), 0.25) == 0
    assert count_blobs(_discs(1, 8), _params(), 0.25) == 1


def test_gray_threshold_from_patch_is_a_high_percentile():
    # The brief pins `200 <= t <= 250` on this fixture, and it cannot hold on it: three
    # 12 px discs cover 2.02% of a 256 px patch, so every percentile below the 98th is
    # the field value 180 and the 98th and above are the disc value 250. Measured. A
    # band that starts at 200 would need a patch that is mostly bright, which is what
    # a real field of glass and tissue looks like and what step 5 measures on a slide;
    # this fixture is a field with three dots in it.
    #
    # The property under test is that the number is a high percentile of *this* patch
    # and not a constant, and it is stated on the fixture that is here.
    patch = _discs(3, 12)
    assert gray_threshold_from_patch(patch, 50.0) == 180.0
    assert gray_threshold_from_patch(patch, 95.0) == 180.0
    assert gray_threshold_from_patch(patch, 99.0) == 250.0


def test_gray_threshold_is_constant_for_uniform_patch():
    flat = np.full((256, 256, 3), 180, dtype=np.uint8)
    assert gray_threshold_from_patch(flat, 95.0) == 180.0


def test_stain_normalization_changes_mean_toward_target():
    cv2 = _cv2()
    src = np.full((64, 64, 3), (140, 90, 160), dtype=np.uint8)
    out = normalize_stain(src)
    assert out.shape == src.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, src)
    # A uniform patch has a zero per-channel standard deviation, so this is the case the
    # guard exists for, and it is the one where the mean is the whole of the change: the
    # source sits at L 122 and comes back at L 128. Asserting shape and dtype alone
    # would pass on an identity function.
    before = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).reshape(-1, 3)[:, 0].mean()
    after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).reshape(-1, 3)[:, 0].mean()
    assert after == pytest.approx(128.0, abs=1.0)
    assert abs(after - 128.0) < abs(before - 128.0)


def test_mpp_cannot_be_omitted():
    with pytest.raises(TypeError):
        count_blobs(np.zeros((8, 8, 3), dtype=np.uint8), _params())


@pytest.mark.parametrize("mpp", [0.0, -0.5])
def test_area_px2_rejects_a_non_positive_scale(mpp):
    with pytest.raises(ValueError, match="mpp"):
        area_px2(3.1, mpp)


def test_count_blobs_rejects_a_non_positive_mpp_before_doing_the_work():
    # The conversion runs first, so the failure is a caller error rather than something
    # found after the patch has been thresholded, opened and contoured.
    with pytest.raises(ValueError, match="mpp"):
        count_blobs(_discs(3, 12), _params(), 0.0)


def test_params_from_config_reads_the_configured_blob_keys():
    cfg = load(overrides=[
        "blobs.area_um2_min=4.0",
        "blobs.area_um2_max=90.0",
        "blobs.circularity_min=0.7",
        "blobs.gray_percentile=50",
        "blobs.morph_kernel_size=5",
        "blobs.morph_iterations=1",
    ])
    patch = _discs(3, 4)
    params = params_from_config(patch, cfg)
    assert params.area_um2_min == 4.0
    assert params.area_um2_max == 90.0
    assert params.circularity_min == 0.7
    assert params.morph_kernel_size == 5
    assert params.morph_iterations == 1
    assert params.gray_threshold == gray_threshold_from_patch(patch, 50.0)


def test_params_from_config_uses_the_default_run_configuration():
    cfg = load()
    params = params_from_config(_discs(3, 4))
    assert params.area_um2_min == cfg.get("blobs.area_um2_min")
    assert params.area_um2_max == cfg.get("blobs.area_um2_max")
    assert params.circularity_min == cfg.get("blobs.circularity_min")
    assert params.morph_kernel_size == cfg.get("blobs.morph_kernel_size")
    assert params.morph_iterations == cfg.get("blobs.morph_iterations")
    # The percentile is read from the configuration rather than written into the
    # assertion, so the case still describes the default run if the key ever moves.
    assert params.gray_threshold == gray_threshold_from_patch(
        _discs(3, 4), float(cfg.get("blobs.gray_percentile"))
    )


def test_params_from_config_threshold_follows_the_configured_percentile():
    patch = _discs(3, 12)
    low = params_from_config(patch, load(overrides=["blobs.gray_percentile=50"]))
    high = params_from_config(patch, load(overrides=["blobs.gray_percentile=99"]))
    assert low.gray_threshold < high.gray_threshold


def test_prepare_patch_leaves_the_patch_alone_by_default():
    assert load().get("blobs.stain_normalization") is False
    patch = _uniform()
    assert np.array_equal(prepare_patch(patch), patch)


def test_prepare_patch_does_not_call_normalize_stain_by_default(monkeypatch):
    def _forbidden(_patch):
        raise AssertionError("normalize_stain ran while blobs.stain_normalization is false")

    monkeypatch.setattr(blobs_module, "normalize_stain", _forbidden)
    patch = _uniform()
    assert np.array_equal(prepare_patch(patch), patch)


def test_prepare_patch_normalizes_when_the_configuration_asks_for_it():
    _cv2()
    cfg = load(overrides=["blobs.stain_normalization=true"])
    patch = _uniform()
    assert not np.array_equal(prepare_patch(patch, cfg), patch)


def test_normalize_stain_moves_lightness_onto_the_target_mean_and_spread():
    cv2 = _cv2()
    ramp = np.tile(np.linspace(100, 160, 64, dtype=np.uint8), (64, 1))
    src = np.repeat(ramp[:, :, None], 3, axis=2)
    out = normalize_stain(src)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)
    assert lab[:, 0].mean() == pytest.approx(128.0, abs=1.0)
    assert lab[:, 0].std() == pytest.approx(60.0, abs=1.0)


def test_normalize_stain_of_a_uniform_patch_is_one_neutral_grey():
    _cv2()
    out = normalize_stain(_uniform(size=64))
    assert len(np.unique(out.reshape(-1, 3), axis=0)) == 1
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]


def test_normalize_stain_leaves_a_degenerate_chroma_channel_at_the_target():
    cv2 = _cv2()
    ramp = np.tile(np.linspace(100, 160, 64, dtype=np.uint8), (64, 1))
    src = np.repeat(ramp[:, :, None], 3, axis=2)
    lab = cv2.cvtColor(normalize_stain(src), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    assert lab[:, 1].mean() == pytest.approx(128.0, abs=1.0)
    assert lab[:, 2].mean() == pytest.approx(128.0, abs=1.0)
    assert lab[:, 1].std() < 2.0
    assert lab[:, 2].std() < 2.0
