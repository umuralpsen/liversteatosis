import pytest

from blobcount.config import default_config
from blobcount.slides import (
    SlideError,
    iter_coords,
    open_info,
    plan_resolution,
    read_mpp,
)

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


class _Properties:
    def __init__(self, values):
        self.values = values

    def __contains__(self, key):
        return key in self.values

    def __getitem__(self, key):
        return self.values[key]


def _slide_with(properties):
    return type("S", (), {"name": "9.svs", "properties": _Properties(properties)})()


def test_read_mpp_prefers_openslide_mpp_x():
    slide = _slide_with({"openslide.mpp-x": "0.2485", "aperio.MPP": "0.9"})
    assert read_mpp(slide) == pytest.approx(0.2485)


def test_read_mpp_falls_back_to_aperio_mpp():
    assert read_mpp(_slide_with({"aperio.MPP": "0.4990"})) == pytest.approx(0.4990)


def test_unreadable_resolution_metadata_raises():
    with pytest.raises(SlideError, match="openslide.mpp-x"):
        read_mpp(_slide_with({"openslide.mpp-x": ""}))


def test_missing_resolution_error_names_the_slide_and_both_properties():
    with pytest.raises(SlideError) as caught:
        read_mpp(_slide_with({"aperio.AppMag": "40"}))
    message = str(caught.value)
    assert "9.svs" in message, message
    assert "openslide.mpp-x" in message, message
    assert "aperio.MPP" in message, message


def test_plan_resolution_rejects_an_empty_pyramid():
    with pytest.raises(SlideError, match="level 0"):
        plan_resolution(0.2520, 0.5, 256, ())


def test_plan_resolution_rejects_a_non_positive_scale():
    with pytest.raises(SlideError, match="slide_mpp"):
        plan_resolution(0.0, 0.5, 256, PYRAMID)
    with pytest.raises(SlideError, match="target_mpp"):
        plan_resolution(0.2520, 0.0, 256, PYRAMID)
    with pytest.raises(SlideError, match="patch_size"):
        plan_resolution(0.2520, 0.5, 0, PYRAMID)


def test_plan_resolution_never_plans_an_empty_read():
    _, _, read_px, _ = plan_resolution(400.0, 0.5, 256, PYRAMID)
    assert read_px == 1


def test_plan_resolution_picks_the_nearest_pyramid_level():
    # 0.5 um/px from a 0.5 um/px slide at level 0 is a 1.0x downsample, and 1.0 is
    # the level the pyramid offers that is nearest to it.
    level, downsample, _, _ = plan_resolution(0.5, 0.5, 256, PYRAMID)
    assert (level, downsample) == (0, 1.0)

    # A target coarser than level 0 by 4x lands on level 1, whose downsample
    # already supplies the 4x, so the read collapses back to the patch width
    # instead of enlarging a 16x region.
    level, downsample, read_px, achieved = plan_resolution(0.125, 0.5, 256, PYRAMID)
    assert (level, downsample) == (1, 4.0002)
    assert read_px == 256
    assert achieved == pytest.approx(0.5, rel=0.01)


def test_plan_resolution_uses_finer_levels_only_when_they_are_closer():
    # Required downsample 8.0 sits nearer the 4.0 level than the 16.0 level.
    level, _, _, _ = plan_resolution(0.0625, 0.5, 256, PYRAMID)
    assert level == 1

    # Required downsample 12.0 sits nearer the 16.0 level.
    level, _, _, _ = plan_resolution(0.5 / 12.0, 0.5, 256, PYRAMID)
    assert level == 2


def test_iter_coords_uses_the_configured_step():
    info = type("I", (), {"width": 700, "height": 700})()
    assert list(iter_coords(info, 256, 128)) == [
        (0, 0), (128, 0), (256, 0), (384, 0),
        (0, 128), (128, 128), (256, 128), (384, 128),
        (0, 256), (128, 256), (256, 256), (384, 256),
        (0, 384), (128, 384), (256, 384), (384, 384),
    ]


def test_iter_coords_yields_nothing_when_no_tile_fits():
    info = type("I", (), {"width": 200, "height": 200})()
    assert list(iter_coords(info, 256, 256)) == []


def test_iter_coords_rejects_a_non_positive_grid():
    info = type("I", (), {"width": 600, "height": 600})()
    with pytest.raises(SlideError, match="patch_size"):
        list(iter_coords(info, 0, 256))
    with pytest.raises(SlideError, match="step_size"):
        list(iter_coords(info, 256, 0))


def _staged_slides():
    """Skip unless a slide can be opened, and say which of the two reasons holds.

    Two independent things can be missing and they call for opposite remedies, so
    they are reported apart. `openslide` is a native binding, not a pure-Python
    package: `pyproject.toml` declares `openslide-python`, and the shared library
    it binds arrives as the `openslide-bin` wheel on Windows and macOS, which has no
    Linux build, so a Linux runner needs a system `libopenslide0` installed by the
    image. The import is attempted inside this helper rather than at module scope,
    because a module-scope import would turn one absent native library into a
    collection *error* for every test in this file, including all the ones that
    never open a slide and would otherwise pass anywhere.

    The slides are the second reason. `.gitignore` holds `data/` and `*.svs`, so
    the images are out of the repository on purpose and a fresh clone has the
    manifest but no slides. A test that opens a slide is meaningful only where the
    slides are staged, and failing on a clone that was never given 30 GB of WSI
    would be a false alarm about the metadata reader.

    Returns `(module, paths)` for the caller.
    """
    openslide = pytest.importorskip(
        "openslide",
        reason="openslide cannot be imported, so no .svs can be opened; install the "
        "native library (the openslide-bin wheel on Windows and macOS, a system "
        "libopenslide0 on Linux, which openslide-bin has no wheel for)",
    )
    slides = sorted(default_config().path("paths.slides").glob("*.svs"))
    if not slides:
        pytest.skip(
            "no .svs under paths.slides: the slides are git-ignored by design, so a "
            "fresh clone carries the manifest and no image files"
        )
    return openslide, slides


def test_every_staged_slide_lands_within_two_percent_of_the_target():
    _, slides = _staged_slides()
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        deviation = abs(info.achieved_mpp - 0.5) / 0.5
        assert deviation < 0.02, f"{info.slide_id}: achieved {info.achieved_mpp} um/px, {deviation:.2%} off"
        assert info.read_px >= 1, info
        assert info.mpp > 0, info
        assert info.level_downsample >= 1.0, info


def test_staged_slides_agree_on_the_field_of_view():
    _, slides = _staged_slides()
    target_field = 256 * 0.5
    widths = []
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        widths.append(info.read_px * info.mpp * info.level_downsample)
    assert abs(max(widths) - min(widths)) < 0.02 * target_field, widths


def test_staged_slide_report_names_the_resolution_actually_read():
    _, slides = _staged_slides()
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        assert info.slide_id == path.stem
        assert info.path == path
        assert info.width > 0 and info.height > 0
        assert info.objective in (20, 40, None), info
        assert info.level == 0, info


def test_open_info_on_a_missing_slide_raises():
    _staged_slides()
    missing = default_config().path("paths.slides") / "not-a-slide.svs"
    with pytest.raises(SlideError, match="not-a-slide.svs"):
        open_info(missing, "not-a-slide", 0.5, 256)
