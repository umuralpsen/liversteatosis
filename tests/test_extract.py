import csv
import dataclasses
import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from PIL import Image

from blobcount.config import DEFAULT_CONFIG_PATH, load
from blobcount.extract import (
    ExtractionStats,
    extract_slide,
    is_tissue,
    patches_dir,
    prepare_output,
    read_patch,
    tissue_gray_threshold,
    write_index,
)
from blobcount.registry import load_registry
from blobcount.slides import SlideError, iter_coords_from_config, open_info_from_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_SIZE = 256
TISSUE_BACKGROUND = 180
TISSUE_DISC = 255
DISC_RADIUS = 12

# The plan the real 40x slides get: 0.252 um/px read for a 0.5 um/px target over a
# 256 px patch, which is 508 source pixels. Every fixture plans its read from this
# pair rather than writing the numbers out, so the source/output inequality that
# makes a walk in the wrong unit observable is present here too.
REAL_SLIDE_MPP = 0.2520
REAL_SLIDE_DIMS = (2000, 2000)
PYRAMID = (1.0, 4.0002, 16.0043)
FAKE_GRID = ((0, 0), (508, 0), (1016, 0), (0, 508), (508, 508), (1016, 508),
             (0, 1016), (508, 1016), (1016, 1016))


def _tissue_region(size, background=TISSUE_BACKGROUND, disc=TISSUE_DISC):
    """An RGBA region shaped like a patch of tissue: a gray field, one white disc."""
    rows, columns = np.mgrid[0 : size[1], 0 : size[0]]
    array = np.full((size[1], size[0], 4), background, dtype=np.uint8)
    array[..., 3] = 255
    centre = (size[0] // 2, size[1] // 2)
    array[(columns - centre[0]) ** 2 + (rows - centre[1]) ** 2 <= DISC_RADIUS**2, :3] = disc
    return Image.fromarray(array)


def _flat_region(size, value):
    """An RGBA region of one gray value, fully opaque."""
    array = np.full((size[1], size[0], 4), value, dtype=np.uint8)
    array[..., 3] = 255
    return Image.fromarray(array)


def _gradient_region(size):
    """An RGBA region darkening left to right, so percentiles of it differ.

    A 508 px-wide ramp reaches 250 gray: the 5th percentile lands near 12 and the
    50th near 127, so a run records a different background level for each and the
    configured percentile becomes observable in the index rather than assumed.
    """
    width = size[0]
    row = (np.arange(width) * 250 // (width - 1)).astype(np.uint8)
    array = np.repeat(row[np.newaxis, :, np.newaxis], size[1], axis=0).repeat(3, axis=2)
    return Image.fromarray(np.dstack([array, np.full((size[1], width, 1), 255, np.uint8)]))


def _transparent_region(size):
    """The region OpenSlide hands back for a read that starts outside the slide."""
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
        self.regions: dict[tuple[int, int], Image.Image] = {}
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
    stale = tmp_output_tree / "patches" / "stale.png"
    stale.write_bytes(b"x")
    prepare_output(tmp_output_tree / "patches")
    assert not stale.exists()


def test_prepare_output_creates_a_directory_that_does_not_exist(tmp_path):
    target = tmp_path / "patches"
    prepare_output(target)
    assert target.is_dir()
    assert list(target.iterdir()) == []


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


def test_a_pure_black_patch_passes_is_tissue():
    # Why the transparency check in read_patch is not optional. A read that starts
    # outside the slide converts to this, and this passes the tissue test, so a
    # coordinate error would write black tiles that are then counted and labelled.
    black = np.zeros((256, 256, 3), dtype=np.uint8)
    assert is_tissue(black, tissue_gray_threshold(black), 235)


def test_a_transparent_read_becomes_a_patch_that_is_tissue():
    transparent = np.zeros((256, 256, 4), dtype=np.uint8)
    assert not transparent[..., 3].any()
    rgb = np.ascontiguousarray(transparent[..., :3])
    assert np.array_equal(rgb, np.zeros((256, 256, 3), dtype=np.uint8))
    assert is_tissue(rgb, tissue_gray_threshold(rgb), 235)


def test_tissue_gray_threshold_is_the_5th_percentile():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))[:, :, None].repeat(3, axis=2)
    assert tissue_gray_threshold(ramp) == pytest.approx(np.percentile(ramp.mean(axis=2), 5))


def test_tissue_gray_threshold_honours_a_percentile():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (256, 1))[:, :, None].repeat(3, axis=2)
    assert tissue_gray_threshold(ramp, 5) == pytest.approx(12.0)
    assert tissue_gray_threshold(ramp, 50) == pytest.approx(127.5)
    assert tissue_gray_threshold(ramp, 95) == pytest.approx(243.0)


def test_tissue_gray_threshold_is_total_on_a_constant_image():
    for value in (0, 37, 180, 250, 255):
        patch = np.full((256, 256, 3), value, dtype=np.uint8)
        assert tissue_gray_threshold(patch) == pytest.approx(float(value))


def test_is_tissue_rejects_a_field_above_gray_max():
    patch = np.full((256, 256, 3), 200, dtype=np.uint8)
    assert not is_tissue(patch, 200.0, 199.0)


def test_is_tissue_reduces_to_the_mean_test_on_a_constant_patch():
    # The second condition cannot reject a uniform patch. `threshold` is a 5th
    # percentile of this patch's own grayscale and the minimum is the smallest value
    # of the same set, so the minimum is never above the threshold and the margin is
    # always satisfied. A uniform field below gray_max is therefore accepted, which is
    # the shape the second condition exists to reject. Pinned here because it is a
    # property of the rule as specified, so a change to it should be visible in the
    # diff rather than discovered on a slide.
    for value in range(0, 256, 8):
        patch = np.full((256, 256, 3), value, dtype=np.uint8)
        assert is_tissue(patch, tissue_gray_threshold(patch), 235) == (value < 235)


def test_is_tissue_reduces_to_the_mean_test_on_textured_patches():
    rng = np.random.default_rng(11)
    for _ in range(50):
        patch = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        mean = patch.mean(axis=2).mean()
        assert is_tissue(patch, tissue_gray_threshold(patch), 235) == (mean < 235)


def test_a_40x_slide_is_planned_as_508_source_pixels_for_a_256_patch():
    info = _plan(load())
    assert (info.read_px, info.level, info.level_downsample) == (508, 0, 1.0)
    assert info.achieved_mpp == pytest.approx(0.5002, abs=5e-4)


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
    extract_slide(specimen, tmp_path / "patches", load())
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
                          load(overrides=["extraction.tissue_gray_max=100"]))
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
    stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load())
    assert (stats.read_failures, stats.patches_written) == (1, 8)


def test_extraction_logs_a_read_that_raises_with_its_coordinates(monkeypatch, tmp_path, caplog):
    slide = _install_fake_slide(monkeypatch, FakeSlide())

    def explode(location, level, size):
        raise OSError("tile pyramid is truncated")

    slide.read_region = explode
    with caplog.at_level("ERROR"):
        stats = extract_slide(_specimen("2", "data/2.svs"), tmp_path / "patches", load())
    assert (stats.read_failures, stats.patches_written, stats.skipped_blank) == (9, 0, True)
    message = "\n".join(record.getMessage() for record in caplog.records)
    assert "2" in message
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
        extract_slide(dataclasses.replace(declared, mpp=declared.mpp * 1.01), out, load())
    assert not out.exists()


def _load_cli():
    path = PROJECT_ROOT / "bin" / "extract_patches.py"
    spec = importlib.util.spec_from_file_location("blobcount_cli_extract_patches", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config_pointing_patches_at(target):
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["patches"] = str(target)
    path = target.parent / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def cli(tmp_path, monkeypatch):
    module = _load_cli()
    out = tmp_path / "patches"
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
    assert not any(out.glob("*.png")) if out.exists() else True
