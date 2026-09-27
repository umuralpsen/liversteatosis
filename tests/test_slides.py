from pathlib import Path

import pytest

from blobcount.config import default_config, load
from blobcount.registry import load_registry
from blobcount.slides import (
    SlideError,
    SlideInfo,
    _check_declared_mpp,
    iter_coords,
    iter_coords_from_config,
    open_info,
    open_info_from_config,
    plan_resolution,
    read_mpp,
)

# level downsamples measured from all six slides: 1.0, ~4.0, ~16.0
PYRAMID = (1.0, 4.0002, 16.0043)


def _info(*, slide_mpp=0.2520, patch_size=256, level_downsamples=PYRAMID, **overrides):
    """A real `SlideInfo`, so the grid tests walk the fields the walk reads.

    The plan is run here rather than hand-written, which keeps `read_px`,
    `level`, `level_downsample` and `achieved_mpp` consistent with each other, and it
    keeps the default `read_px` at 508 against a 256 px patch. That inequality is the
    whole point: with a stub carrying only `width` and `height` the two units of the
    grid coincide and a walk in the wrong one is indistinguishable from a walk in the
    right one, which is how the source/output mix survived a passing suite.
    """
    level, level_downsample, read_px, achieved_mpp = plan_resolution(
        slide_mpp, 0.5, patch_size, level_downsamples
    )
    values = {
        "slide_id": "9",
        "path": Path("data/9.svs"),
        "width": 900,
        "height": 600,
        "mpp": slide_mpp,
        "objective": 40,
        "level": level,
        "level_downsample": level_downsample,
        "read_px": read_px,
        "achieved_mpp": achieved_mpp,
    }
    return SlideInfo(**(values | overrides))


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
    # The brief's fixture carried only `width` and `height`, so its 256 px tile and
    # its 256 px step were the same number and the test could not tell a source-pixel
    # walk from an output-pixel one. `read_px` is 508 here and the tile is `read_px`,
    # on a 900 x 600 level-0 slide: 508 fits, 256 + 508 = 764 fits, 512 + 508 = 1020
    # does not, and the row at y = 256 needs 764 of a 600 px height and is dropped.
    # The two coordinates the brief asserts are therefore the same two, read in the
    # unit the read actually takes.
    info = _info(slide_mpp=0.2520, width=900, height=600)
    assert info.read_px == 508
    coords = list(iter_coords(info, info.read_px, 256))
    assert coords == [(0, 0), (256, 0)]


def _slide_with(properties):
    return type("S", (), {"name": "9.svs", "properties": properties})()


def test_read_mpp_prefers_openslide_mpp_x():
    slide = _slide_with({"openslide.mpp-x": "0.2485", "aperio.MPP": "0.9"})
    assert read_mpp(slide) == pytest.approx(0.2485)


def test_read_mpp_falls_back_to_aperio_mpp():
    assert read_mpp(_slide_with({"aperio.MPP": "0.4990"})) == pytest.approx(0.4990)


def test_read_mpp_falls_through_a_corrupt_standard_property():
    # A value that is present but not a number says nothing about the vendor key, and
    # the alternative is to lose a scale that is sitting right there.
    slide = _slide_with({"openslide.mpp-x": "", "aperio.MPP": "0.4990"})
    assert read_mpp(slide) == pytest.approx(0.4990)


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


def test_plan_resolution_rejects_a_first_level_that_is_not_full_resolution():
    # read_px and the whole tile grid are expressed against the level-0 dimensions,
    # so a pyramid whose first level is already downsampled is planned against
    # coordinates that are not the slide's own.
    with pytest.raises(SlideError, match="full-resolution"):
        plan_resolution(0.2520, 0.5, 256, (4.0, 16.0))
    with pytest.raises(SlideError, match="full-resolution"):
        plan_resolution(0.2520, 0.5, 256, (1.0004, 4.0))


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


def test_iter_coords_walks_source_pixels_at_the_configured_step():
    # 254 is round(128 * 508 / 256), a 128 px output step in source pixels. A 2000 px
    # slide fits (2000 - 508) // 254 + 1 = 6 columns and a 1400 px height 4 rows.
    info = _info(slide_mpp=0.2520, width=2000, height=1400)
    assert list(iter_coords(info, info.read_px, 254)) == [
        (0, 0), (254, 0), (508, 0), (762, 0), (1016, 0), (1270, 0),
        (0, 254), (254, 254), (508, 254), (762, 254), (1016, 254), (1270, 254),
        (0, 508), (254, 508), (508, 508), (762, 508), (1016, 508), (1270, 508),
        (0, 762), (254, 762), (508, 762), (762, 762), (1016, 762), (1270, 762),
    ]


def test_iter_coords_yields_nothing_when_no_tile_fits():
    info = _info(slide_mpp=0.2520, width=200, height=200)
    assert list(iter_coords(info, info.read_px, 254)) == []


def test_iter_coords_rejects_a_non_positive_grid():
    info = _info(slide_mpp=0.2520, width=600, height=600)
    with pytest.raises(SlideError, match="patch_size"):
        list(iter_coords(info, 0, 254))
    with pytest.raises(SlideError, match="step_size"):
        list(iter_coords(info, 508, 0))


def test_iter_coords_bounds_tiles_by_the_level_downsample():
    # A 320 px read at a level whose downsample is 4.0 covers 1280 level-0 px, so a
    # 2000 px level-0 slide fits two tiles across, not eleven, and the walk is bounded
    # by the level-0 dimensions it is given rather than by the read width alone.
    info = _info(slide_mpp=0.1, level_downsamples=(1.0, 4.0, 16.0), width=2000, height=2000)
    assert (info.level, info.level_downsample, info.read_px) == (1, 4.0, 320)
    assert list(iter_coords(info, info.read_px, 160)) == [
        (0, 0), (640, 0), (0, 640), (640, 640),
    ]


def test_iter_coords_from_config_scales_the_output_pixel_step():
    # The configured 256 px step is an output-pixel step, and the walk is in source
    # pixels, so it becomes round(256 * 508 / 256) = 508. Left at 256 this same slide
    # yields six columns and four rows instead of three and two, which is 24 tiles
    # where a 508 px stride fits 6.
    info = _info(slide_mpp=0.2520, width=2000, height=1400)
    coords = list(iter_coords_from_config(info))
    assert sorted({x for x, _ in coords}) == [0, 508, 1016]
    assert sorted({y for _, y in coords}) == [0, 508]
    assert len(coords) == 6


def test_iter_coords_from_config_honours_an_overridden_step():
    info = _info(slide_mpp=0.2520, width=2000, height=1400)
    cfg = load(overrides=["extraction.step_size=128"])
    # 128 px output step * 508 / 256 = 254 source px: 6 columns, 4 rows.
    coords = list(iter_coords_from_config(info, cfg))
    assert sorted({x for x, _ in coords}) == [0, 254, 508, 762, 1016, 1270]
    assert sorted({y for _, y in coords}) == [0, 254, 508, 762]
    assert len(coords) == 24


def test_iter_coords_from_config_tiles_slide_two_exactly_once():
    # 2.svs: read_px 508 against a 256 px patch, so the configured 256 px step becomes
    # 508 source px and so does the tile. Columns (62350 - 508) // 508 + 1 = 122,
    # rows (24247 - 508) // 508 + 1 = 47. Walking the same slide in output pixels over
    # level-0 bounds gives 22,842 tiles and an origin 122,936 px across a 62,350 px
    # slide, which is where this test's fixture inequality comes from.
    info = _info(slide_mpp=0.2520, width=62350, height=24247)
    assert (info.read_px, info.level, info.level_downsample) == (508, 0, 1.0)
    coords = list(iter_coords_from_config(info))
    assert len(coords) == 47 * 122 == 5734
    assert max(x for x, _ in coords) + info.read_px == 122 * 508 <= info.width
    assert max(y for _, y in coords) + info.read_px == 47 * 508 <= info.height


def test_declared_resolution_agreement_is_accepted():
    _check_declared_mpp("2", 0.252, 0.252)
    _check_declared_mpp("2", 0.252, 0.2520)
    # 5e-4 relative is inside the 1e-3 tolerance, so a four-decimal transcription
    # against a three-decimal property is not treated as a contradiction.
    _check_declared_mpp("2", 0.252, 0.252 * (1 + 5e-4))
    # No manifest value is a fact about the record, not a disagreement.
    _check_declared_mpp("2", 0.252, None)


def test_declared_resolution_divergence_raises():
    with pytest.raises(SlideError, match="manifest"):
        _check_declared_mpp("2", 0.252, 0.30)
    with pytest.raises(SlideError, match="0.0"):
        _check_declared_mpp("2", 0.252, 0.0)


def test_declared_resolution_divergence_names_both_values():
    with pytest.raises(SlideError) as caught:
        _check_declared_mpp("2", 0.252, 0.30)
    message = str(caught.value)
    assert "0.252" in message, message
    assert "0.3" in message, message
    assert "16.00%" in message, message


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
    openslide, slides = _staged_slides()
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        deviation = abs(info.achieved_mpp - 0.5) / 0.5
        assert deviation < 0.02, f"{info.slide_id}: achieved {info.achieved_mpp} um/px, {deviation:.2%} off"
        with openslide.OpenSlide(str(path)) as slide:
            assert info.mpp == pytest.approx(float(slide.properties["openslide.mpp-x"])), info


def test_staged_slides_agree_on_the_field_of_view():
    _, slides = _staged_slides()
    target_field = 256 * 0.5
    widths = []
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        widths.append(info.read_px * info.mpp * info.level_downsample)
    assert abs(max(widths) - min(widths)) < 0.02 * target_field, widths


def test_staged_slide_report_names_the_resolution_actually_read():
    openslide, slides = _staged_slides()
    for path in slides:
        info = open_info(path, path.stem, 0.5, 256)
        assert info.slide_id == path.stem
        assert info.path == path
        assert info.objective in (20, 40), info
        assert info.level == 0, info
        with openslide.OpenSlide(str(path)) as slide:
            width, height = slide.dimensions
            declared = float(slide.properties["openslide.mpp-x"])
            planned = plan_resolution(declared, 0.5, 256, tuple(slide.level_downsamples))
        assert (info.width, info.height) == (width, height), info
        assert info.mpp == pytest.approx(declared), info
        assert (info.level, info.level_downsample, info.read_px) == planned[:3], info
        assert info.achieved_mpp == pytest.approx(planned[3]), info


def test_staged_slide_grid_reads_no_pixel_past_the_edge():
    _, slides = _staged_slides()
    cfg = default_config()
    patch_size = int(cfg.get("extraction.patch_size"))
    step_size = int(cfg.get("extraction.step_size"))
    for path in slides:
        info = open_info(path, path.stem, 0.5, patch_size)
        coords = list(iter_coords_from_config(info, cfg))
        assert coords, info
        reach = info.read_px * info.level_downsample
        step = max(1, round(step_size * info.read_px / patch_size))
        columns = (info.width - reach) // step + 1
        rows = (info.height - reach) // step + 1
        assert len(coords) == columns * rows, (info.slide_id, len(coords), columns * rows)
        assert max(x for x, _ in coords) + reach <= info.width
        assert max(y for _, y in coords) + reach <= info.height


def test_staged_slides_agree_with_the_manifest_resolution():
    _, slides = _staged_slides()
    staged = {path.stem: path for path in slides}
    checked = 0
    for specimen in load_registry():
        path = staged.get(specimen.id)
        if path is None:
            continue
        assert specimen.mpp is not None, specimen.id
        info = open_info(path, specimen.id, 0.5, 256, expected_mpp=specimen.mpp)
        assert info.mpp == pytest.approx(specimen.mpp, rel=1e-3), specimen.id
        checked += 1
    assert checked, "no manifest specimen matched a staged slide"


def test_staged_slide_contradicting_the_manifest_raises():
    _, slides = _staged_slides()
    path = slides[0]
    specimen = next(item for item in load_registry() if item.id == path.stem)
    assert specimen.mpp is not None, specimen.id
    with pytest.raises(SlideError, match="manifest"):
        open_info(path, specimen.id, 0.5, 256, expected_mpp=specimen.mpp * 1.01)


def test_open_info_from_config_reads_the_configured_target():
    _, slides = _staged_slides()
    path = slides[0]
    assert open_info_from_config(path, path.stem) == open_info(path, path.stem, 0.5, 256)


def test_open_info_from_config_honours_overridden_extraction_keys():
    _, slides = _staged_slides()
    path = slides[0]
    cfg = load(overrides=["extraction.patch_size=512", "extraction.target_mpp=1.0"])
    assert open_info_from_config(path, path.stem, cfg) == open_info(path, path.stem, 1.0, 512)


def test_open_info_from_config_passes_the_declared_resolution_through():
    _, slides = _staged_slides()
    path = slides[0]
    specimen = next(item for item in load_registry() if item.id == path.stem)
    assert specimen.mpp is not None, specimen.id
    with pytest.raises(SlideError, match="manifest"):
        open_info_from_config(path, path.stem, expected_mpp=specimen.mpp * 1.01)


def test_open_info_on_a_missing_slide_raises():
    _staged_slides()
    missing = default_config().path("paths.slides") / "not-a-slide.svs"
    with pytest.raises(SlideError, match="not-a-slide.svs"):
        open_info(missing, "not-a-slide", 0.5, 256)
