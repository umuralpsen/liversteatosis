import csv
import dataclasses
import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from blobcount.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load
from blobcount.extract import (
    INDEX_COLUMNS,
    TISSUE_PERCENTILE,
    ExtractionStats,
    OutputPathError,
    _protected_paths,
    extract_slide,
    is_tissue,
    patches_dir,
    prepare_output,
    read_patch,
    summarise_tissue_gray,
    tissue_gray_threshold,
    write_index,
)
from blobcount.registry import load_registry
from blobcount.slides import SlideError, iter_coords_from_config, open_info_from_config

PATCH_SIZE = 256
TISSUE_BACKGROUND = 180
TISSUE_DISC = 255
DISC_RADIUS = 12
GRAY_MAX = 235

# The plan the real 40x slides get: 0.252 um/px read for a 0.5 um/px target over a
# 256 px patch, which is 508 source pixels. Every fixture plans its read from this
# pair rather than writing the numbers out, so the source/output inequality that
# makes a walk in the wrong unit observable is present here too.
REAL_SLIDE_MPP = 0.2520
REAL_SLIDE_DIMS = (2000, 2000)
PYRAMID = (1.0, 4.0002, 16.0043)
FAKE_GRID = ((0, 0), (508, 0), (1016, 0), (0, 508), (508, 508), (1016, 508),
             (0, 1016), (508, 1016), (1016, 1016))


def _pil():
    """Return the PIL `Image` module, skipping the test when it will not import.

    Neither `PIL` nor `cv2` is imported at module scope here, for the reason
    `blobcount.extract.read_patch` imports `cv2` inside the read: both are native
    wheels, and a module-scope import turns one absent wheel into a collection error
    for every test in this file, including the ones that touch no image at all. A skip
    from inside a helper costs the tests that need the image, which is the honest
    price.
    """
    return pytest.importorskip(
        "PIL.Image",
        reason="Pillow cannot be imported, so no region can be built or no PNG written",
    )


def _tissue_region(size, background=TISSUE_BACKGROUND, disc=TISSUE_DISC):
    """An RGBA region shaped like a patch of tissue: a gray field, one white disc."""
    Image = _pil()
    rows, columns = np.mgrid[0 : size[1], 0 : size[0]]
    array = np.full((size[1], size[0], 4), background, dtype=np.uint8)
    array[..., 3] = 255
    centre = (size[0] // 2, size[1] // 2)
    array[(columns - centre[0]) ** 2 + (rows - centre[1]) ** 2 <= DISC_RADIUS**2, :3] = disc
    return Image.fromarray(array)


def _flat_region(size, value):
    """An RGBA region of one gray value, fully opaque."""
    Image = _pil()
    array = np.full((size[1], size[0], 4), value, dtype=np.uint8)
    array[..., 3] = 255
    return Image.fromarray(array)


def _gradient_region(size):
    """An RGBA region darkening left to right, so percentiles of it differ.

    A 508 px-wide ramp reaches 250 gray: the 5th percentile lands near 12 and the
    50th near 127, so a run records a different background level for each and the
    configured percentile becomes observable in the index rather than assumed.
    """
    Image = _pil()
    width = size[0]
    row = (np.arange(width) * 250 // (width - 1)).astype(np.uint8)
    array = np.repeat(row[np.newaxis, :, np.newaxis], size[1], axis=0).repeat(3, axis=2)
    return Image.fromarray(np.dstack([array, np.full((size[1], width, 1), 255, np.uint8)]))


def _transparent_region(size):
    """The region OpenSlide hands back for a read that starts outside the slide."""
    Image = _pil()
    return Image.fromarray(np.zeros((size[1], size[0], 4), dtype=np.uint8))


class FakeSlide:
    """An OpenSlide stand-in whose reads come from a table.

    `read_region` records every call, so a test can assert the origin, the level and
    the read width that reached the slide. It returns the full size it was asked for
    and never clips, which is deliberate: a fake that clipped would reproduce the
    reader's benign out-of-bounds case and leave the dangerous one, a full-width read
    with no pixels in it, untestable.
    """

    def __init__(self, dimensions=REAL_SLIDE_DIMS, mpp=REAL_SLIDE_MPP,
                 level_downsamples=PYRAMID, default=_tissue_region):
        self.name = "2.svs"
        self.dimensions = dimensions
        self.level_downsamples = level_downsamples
        self.properties = {
            "openslide.mpp-x": str(mpp),
            "aperio.MPP": str(mpp),
            "openslide.objective-power": "40",
        }
        self.regions: dict[tuple[int, int], object] = {}
        self.reads: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        self._default = default

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read_region(self, location, level, size):
        origin = (int(location[0]), int(location[1]))
        self.reads.append((origin, level, (int(size[0]), int(size[1]))))
        if origin in self.regions:
            return self.regions[origin]
        return self._default(size)


def _staged_slide():
    """Skip unless a slide can be opened, and say which of the two reasons holds.

    `openslide` is a native binding rather than a pure-Python package, so it can be
    installed and still fail to load its shared library. The import is attempted
    inside this helper rather than at module scope, because a module-scope import
    turns one absent native library into a collection *error* for every test in this
    file, including the ones that touch no slide and would otherwise pass anywhere.

    The slides are the second reason, and the two call for opposite remedies.
    `.gitignore` holds `data/` and `*.svs`, so the images are out of the repository
    on purpose and a fresh clone carries the manifest and no images. A test that
    reads real pixels is meaningful only where the slides are staged.
    """
    openslide = pytest.importorskip(
        "openslide",
        reason="openslide cannot be imported, so no .svs can be opened; install the "
        "native library (the openslide-bin wheel on Windows and macOS, a system "
        "libopenslide0 on Linux, which openslide-bin has no wheel for)",
    )
    slides = sorted(load().path("paths.slides").glob("*.svs"))
    if not slides:
        pytest.skip(
            "no .svs under paths.slides: the slides are git-ignored by design, so a "
            "fresh clone carries the manifest and no image files"
        )
    return openslide, slides


def _install_fake_slide(monkeypatch, slide):
    pytest.importorskip("openslide")
    monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
    return slide


def _specimen(slide_id, path, mpp=REAL_SLIDE_MPP):
    """A real manifest row, pointed at `path` and carrying `mpp`."""
    original = next(item for item in load_registry() if item.id == slide_id)
    return dataclasses.replace(original, slide_path=Path(path), mpp=mpp)


def _plan(cfg):
    return open_info_from_config(Path("data/2.svs"), "2", cfg)


def _blank_slide(monkeypatch, dimensions=REAL_SLIDE_DIMS):
    return _install_fake_slide(monkeypatch, FakeSlide(dimensions=dimensions, default=_blank))


def _blank(size):
    return _flat_region(size, 250)


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


def test_prepare_output_creates_a_directory_that_does_not_exist(project_scratch):
    target = project_scratch / "patches"
    prepare_output(target)
    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_prepare_output_refuses_the_project_root():
    # `paths.patches: .` in a hand-edited config, and `--force` on it. The repository,
    # the manifest and the staged slides are all under here and none of them come back.
    with pytest.raises(OutputPathError, match="project root"):
        prepare_output(PROJECT_ROOT)
    assert (PROJECT_ROOT / "blobcount" / "extract.py").is_file()
    assert (PROJECT_ROOT / "configs" / "default.yaml").is_file()


def test_prepare_output_refuses_a_directory_that_holds_the_slides(project_scratch):
    slides = project_scratch / "data" / "slides"
    slides.mkdir(parents=True)
    (slides / "2.svs").write_bytes(b"x")
    with pytest.raises(OutputPathError, match="slides") as refused:
        prepare_output(project_scratch / "data", protected=[slides])
    assert "destroy the source data" in str(refused.value)
    assert (slides / "2.svs").read_bytes() == b"x"


def test_prepare_output_refuses_a_target_that_is_the_slides_directory(project_scratch):
    # The same slide directory as the test above, handed over as the target rather
    # than as a directory containing it. `_is_below` is strict, so a target that *is*
    # the protected path is not below it and the ancestor clause does not fire.
    slides = project_scratch / "data" / "slides"
    slides.mkdir(parents=True)
    (slides / "2.svs").write_bytes(b"x")
    with pytest.raises(OutputPathError, match="slides") as refused:
        prepare_output(slides, protected=[slides])
    assert "destroy the source data" in str(refused.value)
    assert (slides / "2.svs").read_bytes() == b"x"


def test_prepare_output_refuses_the_configured_slides_directory():
    # `paths.patches` hand-edited to the directory `paths.slides` names, with
    # `--force`. The target and the protected path are then the same directory, and
    # clearing it takes the manifest and every staged slide. Asserted against the
    # real configuration because the equality to be refused is between two configured
    # paths, and the target is read here through a separate load from the one the
    # guard reads, so a caller and the guard cannot agree by accident.
    slides = load().path("paths.slides")
    staged = sorted(path.name for path in slides.glob("*.svs"))
    if not staged:
        # Not an assert: `data/` and `*.svs` are git-ignored, so a fresh clone carries
        # the manifest and no images and this test would be red for a reason that has
        # nothing to do with the code. The default protected set is pinned to
        # `paths.slides` by `test_the_default_percentile_constant_is_the_configured_key`,
        # which needs no data, so what this test adds here is the end-to-end pass on a
        # fully staged checkout rather than the wiring itself.
        pytest.skip("no staged slides, so the configured-path equality cannot be exercised here")
    with pytest.raises(OutputPathError, match="paths.slides") as refused:
        prepare_output(slides)
    assert "destroy the source data" in str(refused.value)
    assert slides.is_dir()
    assert sorted(path.name for path in slides.glob("*.svs")) == staged


def test_prepare_output_refuses_a_target_outside_the_project_root(tmp_path):
    outside = tmp_path / "patches"
    outside.mkdir()
    (outside / "stale.png").write_bytes(b"x")
    with pytest.raises(OutputPathError, match="not strictly below"):
        prepare_output(outside)
    assert (outside / "stale.png").read_bytes() == b"x"


def test_prepare_output_refuses_a_sibling_that_shares_the_root_prefix():
    # `...Masaüstü\Proje-backup` starts with `...Masaüstü\Proje` and is a different
    # directory. A string-prefix containment test would call it inside the project.
    sibling = Path(f"{PROJECT_ROOT}-backup")
    assert sibling.name == f"{PROJECT_ROOT.name}-backup"
    # The target does not exist, so the refusal is the only thing that keeps this test
    # from making a directory in the user's OneDrive parent. If the guard ever
    # regresses, `prepare_output` reaches `mkdir` and the test removes what it made
    # rather than leaving a `Proje-backup` behind and reporting only a bare failure.
    created = not sibling.exists()
    # Asserted here rather than left implicit in the `finally`: a user who has a real
    # `Proje-backup` beside the checkout gets this message and stops, instead of a bare
    # "not exists" failure at the bottom and a cleanup that has nothing to do. Nothing
    # is deleted on this path, and the `finally` below still only removes what the test
    # made.
    assert created, (
        f"{sibling} already exists, so this test would be refusing and then deleting a "
        f"directory the user made; move it aside to run the suite"
    )
    try:
        with pytest.raises(OutputPathError, match="not strictly below"):
            prepare_output(sibling)
    finally:
        if created:
            shutil.rmtree(sibling, ignore_errors=True)


def test_prepare_output_refuses_a_target_reached_through_a_dot_dot(project_scratch):
    # `..` is collapsed before the comparison, so climbing from the scratch tree out
    # to the project's own parent is not inside the project. `protected` is empty so
    # that this is the containment rule refusing and not the ancestor rule, which
    # would otherwise fire first for a target at the project's parent.
    (project_scratch / "dataset").mkdir()
    escaped = project_scratch / "dataset" / ".." / ".." / ".." / ".."
    assert escaped.resolve() == PROJECT_ROOT.parent
    with pytest.raises(OutputPathError, match="not strictly below"):
        prepare_output(escaped, protected=[])
    assert PROJECT_ROOT.is_dir()


def test_index_header_is_pinned(tmp_path):
    out = tmp_path / "patches.csv"
    write_index([], out)
    assert out.read_text(encoding="utf-8").splitlines()[0] == "patch_id,slide_id,x,y,mpp,tissue_gray"


def test_index_preserves_a_slide_id_carrying_csv_punctuation(tmp_path):
    rows = [{"patch_id": "p1", "slide_id": 'a,b "c"', "x": 1, "y": 2,
             "mpp": 0.5, "tissue_gray": 181.0}]
    out = tmp_path / "patches.csv"
    write_index(rows, out)
    with out.open(encoding="utf-8", newline="") as handle:
        read_back = list(csv.DictReader(handle))
    assert read_back == [{"patch_id": "p1", "slide_id": 'a,b "c"', "x": "1", "y": "2",
                          "mpp": "0.5", "tissue_gray": "181.0"}]


def test_a_transparent_read_becomes_a_patch_that_is_tissue():
    # Why the transparency check in read_patch is not optional. A read that starts
    # outside the slide comes back as this, and this passes the tissue test, so a
    # coordinate error would write black tiles that are then counted and labelled.
    transparent = np.zeros((256, 256, 4), dtype=np.uint8)
    assert not transparent[..., 3].any()
    black = np.ascontiguousarray(transparent[..., :3])
    assert np.array_equal(black, np.zeros((256, 256, 3), dtype=np.uint8))
    assert is_tissue(black, tissue_gray_threshold(black), GRAY_MAX)


def test_tissue_gray_threshold_honours_a_percentile():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))[:, :, None].repeat(3, axis=2)
    assert tissue_gray_threshold(ramp, 5) == pytest.approx(12.0)
    assert tissue_gray_threshold(ramp, 50) == pytest.approx(127.5)
    assert tissue_gray_threshold(ramp, 95) == pytest.approx(243.0)


def test_tissue_gray_threshold_is_total_on_a_constant_image():
    for value in (0, 37, 180, 250, 255):
        patch = np.full((256, 256, 3), value, dtype=np.uint8)
        assert tissue_gray_threshold(patch) == pytest.approx(float(value))


def test_the_default_percentile_constant_is_the_configured_key():
    # The docstring of `tissue_gray_threshold` says its default is the configured
    # `extraction.tissue_percentile`. That is a claim about two facts in two files, so
    # it is asserted here rather than left to the reader: change the key and this
    # fails until the constant follows it.
    assert TISSUE_PERCENTILE == float(load().get("extraction.tissue_percentile"))

    # The same shape of claim, about the paths `prepare_output` refuses to delete when
    # it is given no `protected` argument: they are read from the configuration rather
    # than pinned as constants, so a run whose configuration moves them is guarded
    # against the paths it actually uses. Asserted from a separate `load()` than the
    # one `_protected_paths` reads, so the guard and the caller cannot agree by
    # accident. The equality refusal itself is covered by
    # `test_prepare_output_refuses_a_target_that_is_the_slides_directory`, which passes
    # `protected` explicitly and so bypasses this default; this assertion is what keeps
    # the default tied to the configuration, and it needs no slide on disk.
    assert dict(_protected_paths())["paths.slides"] == load().path("paths.slides")


def test_is_tissue_rejects_a_field_above_gray_max():
    patch = np.full((256, 256, 3), 200, dtype=np.uint8)
    assert not is_tissue(patch, 200.0, 199.0)


def test_is_tissue_is_the_mean_test_alone():
    # The sample, so the claim is reproducible: 200 patches of 64x64x3 drawn from
    # `numpy.random.default_rng(2026)` in four bands whose means straddle `gray_max`
    # from both sides, plus every constant field from 0 to 255 in steps of 8. Each is
    # compared against `mean < 235` computed here, which is the whole rule. The
    # straddle is asserted rather than assumed, because a sample entirely below the
    # threshold would pass this test without ever reaching the clause under test.
    rng = np.random.default_rng(2026)
    patches = [
        rng.integers(low, high, size=(64, 64, 3), dtype=np.uint8)
        for low, high in ((150, 200), (200, 226), (226, 240), (240, 256))
        for _ in range(50)
    ]
    patches += [np.full((64, 64, 3), value, dtype=np.uint8) for value in range(0, 256, 8)]
    for patch in patches:
        assert is_tissue(patch, tissue_gray_threshold(patch), GRAY_MAX) == (
            patch.mean(axis=2).mean() < GRAY_MAX
        )
    means = [patch.mean(axis=2).mean() for patch in patches]
    assert len(patches) == 232
    assert sum(1 for value in means if value < GRAY_MAX) == 180
    assert sum(1 for value in means if value >= GRAY_MAX) == 52
    assert sum(1 for value in means if 230.0 < value < 240.0) == 51


def test_is_tissue_accepts_a_uniform_faint_field__known_gap():
    # The filter has no background-rejection behaviour and its sensitivity is
    # unmeasured: a field of one value holds no structure at all and is accepted,
    # because the only test is its mean against `gray_max`. Delete this test in the
    # same commit that gives the filter a structure test.
    faint = np.full((256, 256, 3), 200, dtype=np.uint8)
    assert is_tissue(faint, tissue_gray_threshold(faint), GRAY_MAX)


def test_read_patch_passes_the_origin_and_level_through_untransformed(monkeypatch):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    read_patch(slide, info, 1016, 508)
    assert slide.reads == [((1016, 508), 0, (508, 508))]


def test_read_patch_resizes_to_the_configured_patch_size(monkeypatch):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    patch = read_patch(slide, _plan(load()), 0, 0)
    assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3)
    assert patch.dtype == np.uint8


def test_read_patch_honours_an_overridden_patch_size(monkeypatch):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    cfg = load(overrides=["extraction.patch_size=128"])
    assert read_patch(slide, _plan(cfg), 0, 0, patch_size=128).shape == (128, 128, 3)


def test_read_patch_resizes_with_area_interpolation(monkeypatch):
    cv2 = pytest.importorskip("cv2", reason="opencv cannot be imported, so the read "
                                "cannot be resized and no patch can be produced")
    Image = _pil()
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    source = np.full((info.read_px, info.read_px, 3), 200, dtype=np.uint8)
    source[::2] = 100
    slide.regions[(0, 0)] = Image.fromarray(source).convert("RGBA")
    patch = read_patch(slide, info, 0, 0)
    expected = cv2.resize(source, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)
    assert np.array_equal(patch, expected)


def test_read_patch_rejects_a_fully_transparent_read(monkeypatch):
    # The guard. A read that starts past the right edge of the slide comes back at the
    # full 508x508 it asked for, not clipped, so the size check cannot see it and the
    # alpha channel is the only thing left that can.
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    origin = (info.width, 0)
    slide.regions[origin] = _transparent_region((info.read_px, info.read_px))
    raw = np.asarray(slide.regions[origin])
    assert raw.shape == (info.read_px, info.read_px, 4)
    assert not raw[..., 3].any()
    assert read_patch(slide, info, *origin) is None


def test_read_patch_keeps_a_read_holding_one_opaque_pixel(monkeypatch):
    Image = _pil()
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    region = _tissue_region((info.read_px, info.read_px))
    array = np.array(region)
    array[0, 0] = (255, 255, 255, 255)
    origin = (info.width, 0)
    slide.regions[origin] = Image.fromarray(array)
    assert read_patch(slide, info, *origin) is not None


def test_read_patch_rejects_a_short_read(monkeypatch):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    slide.regions[(0, 0)] = _tissue_region((info.read_px - 1, info.read_px))
    assert read_patch(slide, info, 0, 0) is None


def test_read_patch_rejects_a_read_that_is_not_an_image(monkeypatch):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    slide.regions[(0, 0)] = None
    assert read_patch(slide, info, 0, 0) is None


def test_read_patch_rejects_a_read_it_cannot_be_written_from(monkeypatch):
    Image = _pil()
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    info = _plan(load())
    two_channel = np.zeros((info.read_px, info.read_px, 2), dtype=np.uint8)
    slide.regions[(0, 0)] = Image.fromarray(two_channel)
    assert read_patch(slide, info, 0, 0) is None


def test_extraction_cross_checks_the_manifest_resolution(monkeypatch, tmp_path):
    import blobcount.slides as slides_module

    calls = []
    original = slides_module.open_info

    def spy(path, slide_id, target_mpp, patch_size, **kwargs):
        calls.append({"path": Path(path), "slide_id": slide_id, "target_mpp": target_mpp,
                      "patch_size": patch_size, **kwargs})
        return original(path, slide_id, target_mpp, patch_size, **kwargs)

    monkeypatch.setattr(slides_module, "open_info", spy)
    _install_fake_slide(monkeypatch, FakeSlide())
    specimen = _specimen("2", "data/2.svs")
    extract_slide(specimen, tmp_path / "patches", load(), rows=[])
    assert calls, "extract_slide never planned a read"
    assert calls[0]["expected_mpp"] == specimen.mpp
    assert calls[0]["slide_id"] == specimen.id
    assert calls[0]["path"] == Path("data/2.svs")
    assert calls[0]["target_mpp"] == 0.5
    assert calls[0]["patch_size"] == PATCH_SIZE


def test_extraction_uses_the_configured_tissue_gray_max(monkeypatch, tmp_path):
    # 180-gray tissue is below the configured 235 and above this override, so the key
    # actually reaching is_tissue is the difference between nine written patches and
    # a slide reported blank.
    _install_fake_slide(monkeypatch, FakeSlide())
    stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches",
                          load(overrides=["extraction.tissue_gray_max=100"]), rows=[])
    assert (stats.patches_written, stats.tissue_rejected, stats.skipped_blank) == (0, 9, True)


def test_extraction_records_the_configured_tissue_percentile(monkeypatch, tmp_path):
    # A 508 px ramp reaches 250 gray, so its 5th percentile lands near 12 and its 50th
    # near 127. Both runs accept the tile, so only the recorded background level can
    # show that the key was read rather than assumed.
    _install_fake_slide(monkeypatch, FakeSlide(default=_gradient_region))
    rows = []
    extract_slide(_specimen("2", "data/2.svs"), tmp_path / "at5", load(), rows=rows)
    rows_mid = []
    extract_slide(_specimen("2", "data/2.svs"), tmp_path / "at50",
                  load(overrides=["extraction.tissue_percentile=50"]), rows=rows_mid)
    assert rows[0]["tissue_gray"] == pytest.approx(12.0, abs=3.0)
    assert rows_mid[0]["tissue_gray"] == pytest.approx(125.0, abs=4.0)
    assert rows_mid[0]["tissue_gray"] - rows[0]["tissue_gray"] > 100.0


def test_extraction_writes_patches_named_by_source_coordinates(monkeypatch, tmp_path):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    cfg = load()
    out = tmp_path / "patches"
    rows = []
    stats = extract_slide(_specimen("2", "data/2.svs"), out, cfg, rows=rows)
    coords = list(iter_coords_from_config(_plan(cfg), cfg))
    assert coords == list(FAKE_GRID)
    assert sorted(path.name for path in out.glob("*.png")) == sorted(
        f"2_{x}_{y}.png" for x, y in coords
    )
    assert stats == ExtractionStats("2", len(coords), 0, 0, False)
    assert [row["patch_id"] for row in rows] == [f"2_{x}_{y}.png" for x, y in coords]
    assert [(row["x"], row["y"]) for row in rows] == coords
    assert rows[0]["slide_id"] == "2"
    assert rows[0]["mpp"] == pytest.approx(0.5002, abs=5e-4)
    assert rows[0]["tissue_gray"] == pytest.approx(TISSUE_BACKGROUND)


def test_an_index_row_carries_exactly_the_pinned_columns(monkeypatch, tmp_path):
    # `INDEX_COLUMNS` is declared where `write_index` reads it and the row is built
    # where `extract_slide` appends it, and nothing tied the two together: a key added
    # to one and not the other would drop silently. This is what ties them.
    _install_fake_slide(monkeypatch, FakeSlide())
    rows = []
    extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load(), rows=rows)
    assert rows
    assert tuple(rows[0]) == INDEX_COLUMNS
    assert set(rows[0]) == set(INDEX_COLUMNS)


def test_extraction_counts_a_read_with_no_pixels_as_a_failure(monkeypatch, tmp_path):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    out = tmp_path / "patches"
    rows = []
    transparent = (1016, 0)
    slide.regions[transparent] = _transparent_region((508, 508))
    stats = extract_slide(_specimen("2", "data/2.svs"), out, load(), rows=rows)
    assert (stats.read_failures, stats.patches_written) == (1, 8)
    assert len(rows) == 8
    assert not (out / f"2_{transparent[0]}_{transparent[1]}.png").exists()


def test_extraction_counts_a_short_read_as_a_failure(monkeypatch, tmp_path):
    slide = _install_fake_slide(monkeypatch, FakeSlide())
    slide.regions[(0, 0)] = _tissue_region((16, 16))
    stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load(), rows=[])
    assert (stats.read_failures, stats.patches_written) == (1, 8)


def test_extraction_logs_a_read_that_raises_with_its_coordinates(monkeypatch, tmp_path, caplog):
    slide = _install_fake_slide(monkeypatch, FakeSlide())

    def explode(location, level, size):
        raise OSError("tile pyramid is truncated")

    slide.read_region = explode
    with caplog.at_level("ERROR"):
        stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load(), rows=[])
    assert (stats.read_failures, stats.patches_written, stats.skipped_blank) == (9, 0, True)
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "slide 2" in message
    assert "(0, 0)" in message
    assert "OSError" in message
    assert "tile pyramid is truncated" in message


def test_extraction_reports_a_slide_with_no_tissue_as_blank(monkeypatch, tmp_path):
    _blank_slide(monkeypatch)
    rows = []
    stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load(), rows=rows)
    assert stats == ExtractionStats("2", 0, 9, 0, True)
    assert rows == []


def test_extraction_stats_are_frozen():
    stats = ExtractionStats("2", 1, 2, 3, False)
    with pytest.raises(dataclasses.FrozenInstanceError):
        stats.patches_written = 9


def test_patches_dir_follows_the_configured_path(tmp_path):
    cfg = load(overrides=[f"paths.patches={tmp_path}/elsewhere"])
    assert patches_dir(cfg) == tmp_path / "elsewhere"


def test_out_of_bounds_read_on_a_real_slide_is_rejected():
    openslide, slides = _staged_slide()
    path = next(item for item in slides if item.stem == "2")
    cfg = load()
    info = open_info_from_config(path, "2", cfg)
    gray_max = float(cfg.get("extraction.tissue_gray_max"))
    beyond = info.width + 5000
    with openslide.OpenSlide(str(path)) as slide:
        raw = np.asarray(slide.read_region((beyond, 0), info.level, (info.read_px, info.read_px)))
        assert raw.shape == (info.read_px, info.read_px, 4)
        assert not raw[..., 3].any()
        black = np.ascontiguousarray(raw[..., :3])
        assert is_tissue(black, tissue_gray_threshold(black), gray_max)
        assert read_patch(slide, info, beyond, 0) is None
        in_bounds = np.asarray(slide.read_region((0, 0), info.level, (info.read_px, info.read_px)))
        assert in_bounds[..., 3].any()
        assert read_patch(slide, info, 0, 0) is not None


def test_real_patches_from_the_planned_grid_survive_the_guard():
    _, slides = _staged_slide()
    openslide = pytest.importorskip("openslide")
    path = next(item for item in slides if item.stem == "2")
    cfg = load()
    info = open_info_from_config(path, "2", cfg)
    grid = iter_coords_from_config(info, cfg)
    origins = [next(grid), next(grid), next(grid)]
    with openslide.OpenSlide(str(path)) as slide:
        for origin in origins:
            patch = read_patch(slide, info, *origin)
            assert patch is not None, origin
            assert patch.shape == (PATCH_SIZE, PATCH_SIZE, 3), origin


def test_extraction_rejects_a_manifest_that_no_longer_describes_the_file(tmp_path):
    _, slides = _staged_slide()
    path = next(item for item in slides if item.stem == "2")
    declared = next(item for item in load_registry() if item.id == "2")
    assert declared.mpp is not None
    out = tmp_path / "patches"
    with pytest.raises(SlideError, match="manifest"):
        extract_slide(dataclasses.replace(declared, mpp=declared.mpp * 1.01), out, load(), rows=[])
    assert not out.exists()


def test_summarise_tissue_gray_reports_the_distribution_of_an_index(tmp_path):
    # A synthetic index rather than a real one: reading 1.0 GB of slides to
    # characterise a column is the deferred measurement, not this task's. The values
    # are hand-picked so every number in the summary is arithmetic a reader can check:
    # percentiles of five sorted values at index 0.2, 2.0 and 3.8 of the range.
    rows = [
        {"patch_id": f"p{index}", "slide_id": "2", "x": index, "y": 0,
         "mpp": 0.5, "tissue_gray": value}
        for index, value in enumerate((12.0, 40.0, 150.0, 181.0, 250.0))
    ]
    index = tmp_path / "patches.csv"
    write_index(rows, index)
    assert summarise_tissue_gray(index) == {
        "count": 5,
        "min": 12.0,
        "max": 250.0,
        "mean": pytest.approx(126.6),
        "p05": pytest.approx(17.6),
        "p50": pytest.approx(150.0),
        "p95": pytest.approx(236.2),
        "skipped_blank_slides": 0,
    }


def test_summarise_tissue_gray_counts_the_slides_reported_blank(monkeypatch, tmp_path):
    # A slide reported blank writes no patches and no rows, so the index of a run that
    # saw one carries no evidence that it was skipped: the count is slides the caller
    # reports, not rows found in the file. Both the rows and the index here come from a
    # real run over a fake blank slide, and the slide that is reported blank is named in
    # no row of it, which is why `count` is 0 while the blank count is 1.
    _install_fake_slide(monkeypatch, FakeSlide(default=_blank))
    rows = []
    stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load(), rows=rows)
    assert (stats.skipped_blank, rows, stats.patches_written) == (True, [], 0)
    index = tmp_path / "patches.csv"
    write_index(rows, index)
    assert summarise_tissue_gray(index, blank_slides=[stats.slide_id]) == {
        "count": 0,
        "min": None,
        "max": None,
        "mean": None,
        "p05": None,
        "p50": None,
        "p95": None,
        "skipped_blank_slides": 1,
    }
    assert summarise_tissue_gray(index)["skipped_blank_slides"] == 0


def test_summarise_tissue_gray_reports_no_number_for_an_index_with_no_rows(tmp_path):
    index = tmp_path / "patches.csv"
    write_index([], index)
    summary = summarise_tissue_gray(index)
    assert summary["count"] == 0
    assert summary["skipped_blank_slides"] == 0
    assert all(summary[name] is None for name in ("min", "max", "mean", "p05", "p50", "p95"))


def _load_cli(monkeypatch):
    path = PROJECT_ROOT / "bin" / "extract_patches.py"
    spec = importlib.util.spec_from_file_location("blobcount_cli_extract_patches", path)
    module = importlib.util.module_from_spec(spec)
    # Through `monkeypatch` rather than `sys.modules[...] = ...`, so the entry is torn
    # down with the test. Five tests load this module and a plain assignment leaves
    # whichever copy ran last in `sys.modules` for the rest of the session.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _config_pointing_patches_at(target):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["patches"] = str(target)
    path = target.parent / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def cli(project_scratch, monkeypatch):
    module = _load_cli(monkeypatch)
    out = project_scratch / "patches"
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", _config_pointing_patches_at(out))
    monkeypatch.setattr(module, "load_registry", lambda *a, **k: [
        _specimen("2", "data/2.svs"), _specimen("3", "data/3.svs"),
    ])
    return module, out


def test_cli_extracts_only_the_slides_it_is_asked_for(cli, monkeypatch, capsys):
    module, out = cli
    _install_fake_slide(monkeypatch, FakeSlide())
    assert module.main(["--slide", "3", "--slide", "3"]) == 0
    assert sorted({path.name.split("_")[0] for path in out.glob("*.png")}) == ["3"]
    assert (out / "patches.csv").read_text(encoding="utf-8").splitlines()[0] == (
        "patch_id,slide_id,x,y,mpp,tissue_gray"
    )
    assert len((out / "patches.csv").read_text(encoding="utf-8").splitlines()) == 10
    assert "3" in capsys.readouterr().out


def test_cli_refuses_a_slide_the_manifest_does_not_name(cli, monkeypatch, capsys):
    module, out = cli
    assert module.main(["--slide", "9"]) == 2
    assert "9" in capsys.readouterr().err
    assert not out.exists()


def test_cli_refuses_to_overwrite_without_force(cli, monkeypatch, capsys):
    module, out = cli
    _install_fake_slide(monkeypatch, FakeSlide())
    out.mkdir(parents=True, exist_ok=True)
    (out / "2_0_0.png").write_bytes(b"stale")
    assert module.main([]) == 2
    assert (out / "2_0_0.png").read_bytes() == b"stale"
    assert "--force" in capsys.readouterr().err
    assert module.main(["--force"]) == 0
    assert len(list(out.glob("*.png"))) == 18


def test_cli_refuses_to_delete_a_target_outside_the_project(cli, tmp_path, capsys):
    module, _ = cli
    outside = tmp_path / "escape"
    outside.mkdir()
    (outside / "keep.png").write_bytes(b"x")
    module.patches_dir = lambda cfg=None: outside
    assert module.main(["--force"]) == 2
    assert "refusing" in capsys.readouterr().err
    assert (outside / "keep.png").read_bytes() == b"x"


def test_cli_exits_non_zero_when_a_slide_yields_no_tissue(cli, monkeypatch):
    module, out = cli
    _blank_slide(monkeypatch)
    assert module.main([]) == 1
    assert (out / "patches.csv").read_text(encoding="utf-8").splitlines() == [
        "patch_id,slide_id,x,y,mpp,tissue_gray"
    ]


def test_cli_reports_a_manifest_contradiction_as_a_refusal(cli, monkeypatch, capsys):
    # The refusal code and the blank-slide code have to differ, or a slide that
    # stopped describing its own file reports the same thing as one that yielded
    # nothing, and both read as a finished run.
    module, out = cli
    monkeypatch.setattr(module, "load_registry",
                        lambda *a, **k: [_specimen("2", "data/2.svs", mpp=9.0)])
    _install_fake_slide(monkeypatch, FakeSlide())
    assert module.main([]) == 2
    assert "manifest" in capsys.readouterr().err
    assert not out.exists() or not any(out.glob("*.png"))


def test_cli_indexes_the_slides_it_finished_before_a_manifest_contradiction(cli, monkeypatch, capsys):
    # Without the index, the tree holds nine patches that no `patches.csv` describes,
    # and the next run is refused for holding a previous run, so recovering needs
    # `--force`, which deletes what this run just produced.
    module, out = cli
    monkeypatch.setattr(module, "load_registry", lambda *a, **k: [
        _specimen("2", "data/2.svs"), _specimen("3", "data/3.svs", mpp=9.0),
    ])
    _install_fake_slide(monkeypatch, FakeSlide())
    assert module.main([]) == 2
    assert "manifest" in capsys.readouterr().err
    assert len(list(out.glob("*.png"))) == 9
    lines = (out / "patches.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "patch_id,slide_id,x,y,mpp,tissue_gray"
    assert len(lines) == 10
    assert {line.split(",")[1] for line in lines[1:]} == {"2"}
