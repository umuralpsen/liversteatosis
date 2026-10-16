import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from blobcount.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load
from blobcount.extract import write_index
from blobcount.labeling import _CONFIG_KEYS, read_index

# A read of a declared key, in the shape `tests/test_config.py` scans for. Recorded
# because this file has to answer the same question about a single module that file asks
# about the package, and a second scan written as a second pattern would be a second
# definition of "a read".
_KEY_READ = re.compile(r"""\.(?:get|path)\(\s*["']([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)["']""")


def _pil():
    """Return the PIL `Image` module, skipping the test when it will not import.

    Copied from `tests/test_extract.py` for the reason given there: both are native
    wheels, and a module-scope import turns one absent wheel into a collection error for
    every test in this file, including the ones that touch no image at all.
    """
    return pytest.importorskip(
        "PIL.Image",
        reason="Pillow cannot be imported, so no patch can be written or read",
        # A native wheel absent with a plain `ImportError` is a `PytestDeprecationWarning`
        # rather than a skip, and a deprecation warning is a failure under this suite's
        # `filterwarnings = ["error"]`. The guard is the one that skips; naming the
        # exception is what keeps it skipping.
        exc_type=ImportError,
    )


def _cfg_with_threshold(threshold):
    """Return the default configuration carrying `labeling.blob_threshold = threshold`.

    A fresh `load()` per call rather than a mutation of the process-wide default: one
    test below runs the labeling twice with two different thresholds, and
    `default_config()` is a single mutable global that the conftest fixture resets
    between tests and not between the two runs inside one of them.
    """
    cfg = load()
    cfg.set("labeling.blob_threshold", threshold)
    return cfg


def _read_label_rows(out):
    """Return `labels.csv` as a list of its rows, read the way `read_index` reads.

    `newline=""` for the reason `extract.write_index` writes it that way: the line
    terminator belongs to the csv module and not to the text layer.
    """
    with (Path(out) / "labels.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_labels(out):
    """Return `labels.csv` as `{patch_id: path}`."""
    return {row["patch_id"]: row["path"] for row in _read_label_rows(out)}


def _write_patches(directory, names, array):
    """Write `array` to each name in `directory` as a real PNG and return the index path.

    A real file rather than a stub, for the case it serves: `label_patches` resolves its
    source at `index.parent / name` through `load_patch`, and a stubbed `load_patch` never
    exercises that lookup at all. The index is written by `extract.write_index` for the
    same reason — a hand-typed header is a second definition of `INDEX_COLUMNS`.
    """
    Image = _pil()
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        Image.fromarray(array).save(directory / name)
        slide_id, x, y = name.removesuffix(".png").split("_")
        rows.append({
            "patch_id": name,
            "slide_id": slide_id,
            "x": int(x),
            "y": int(y),
            "mpp": 0.5010,
            "tissue_gray": 181.0,
        })
    index = directory / "patches.csv"
    write_index(rows, index)
    return index


def test_index_reads_utf8_slide_ids(tmp_path):
    p = tmp_path / "patches.csv"
    p.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "1_hasta-ğü-01_0_0,hasta-ğü-01,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    assert read_index(p)[0]["slide_id"] == "hasta-ğü-01"
    assert float(read_index(p)[0]["mpp"]) == pytest.approx(0.5010)


def test_relabeling_leaves_no_patch_in_both_classes(tmp_output_tree, monkeypatch):
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    # A count of 5 is at the first threshold and below the second, so the second run
    # really does relabel. A count taken from the uniform patch above is 0 at both
    # thresholds, which would make the second run byte-identical to the first and leave
    # this test asserting nothing about relabeling at all.
    monkeypatch.setattr(labeling, "count_blobs", lambda patch, params, mpp: 5)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5,181.0\n"
        "B_0_0,B,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    first_stats = labeling.label_patches(idx, out, _cfg_with_threshold(5))
    first = _read_labels(out)
    second_stats = labeling.label_patches(idx, out, _cfg_with_threshold(25))
    second = _read_labels(out)

    # The label moved, so the second run really relabelled rather than rewriting the same
    # tree. Without this a count that crossed neither threshold would leave both runs
    # identical and every assertion below would hold for a run that changed nothing.
    assert first == {"A_0_0": "steatosis/A_0_0.png", "B_0_0": "steatosis/B_0_0.png"}
    assert second == {"A_0_0": "normal/A_0_0.png", "B_0_0": "normal/B_0_0.png"}
    # Nothing was contradictory at any point. The clearing is what guarantees it, and a
    # run that inherited the previous threshold's tree would report a non-zero count here
    # even if `_remove_duplicates` then repaired the tree — the count is the record that
    # the second run had a contradiction to repair, which is the defect itself.
    assert (first_stats.duplicates_removed, second_stats.duplicates_removed) == (0, 0)
    # One file per name, keyed on the name alone. A key that includes the class directory
    # cannot go red here: `normal/A_0_0.png` and `steatosis/A_0_0.png` are two distinct
    # strings, so the duplicate state passes that check.
    names = [item.name for item in out.rglob("*.png")]
    assert sorted(names) == ["A_0_0.png", "B_0_0.png"]
    # And the tree agrees with the CSV, which is the property the clearing buys.
    assert sorted(item.relative_to(out).as_posix() for item in out.rglob("*.png")) == sorted(
        second.values()
    )


def test_duplicate_scan_keeps_the_copy_the_csv_names(tmp_output_tree, monkeypatch):
    # The clearing is replaced by a no-op so the second run inherits the first run's
    # tree, which is the state `_remove_duplicates` exists to catch and the only way to
    # reach it. `label_patches` alone cannot produce it, so the scan's key is pinned here
    # rather than left to a run that never fires it.
    from blobcount import labeling
    monkeypatch.setattr(labeling, "prepare_output", lambda target: None)
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs", lambda patch, params, mpp: 5)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    labeling.label_patches(idx, out, _cfg_with_threshold(5))
    assert (out / "steatosis" / "A_0_0.png").is_file()
    stats = labeling.label_patches(idx, out, _cfg_with_threshold(25))

    # `labelled` is keyed by the resolved name `A_0_0.png`, not by the `patch_id` column
    # `A_0_0`. A map keyed by the column looks `A_0_0.png` up in a map holding `A_0_0`,
    # finds nothing, and deletes both copies including the one `labels.csv` names.
    assert stats.duplicates_removed == 1
    named = _read_labels(out)["A_0_0"]
    assert named == "normal/A_0_0.png"
    assert (out / named).is_file()
    assert not (out / "steatosis" / "A_0_0.png").exists()


def test_an_index_naming_one_patch_twice_labels_it_once(tmp_output_tree, monkeypatch):
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs", lambda patch, params, mpp: 7)
    idx = tmp_output_tree / "patches.csv"
    # The same file under both forms of the column: `2_1016_0.png` is what
    # `extract_slide` writes and `2_1016_0` is the same name without its extension, so
    # the two name one patch rather than two.
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "2_1016_0.png,2,1016,0,0.5010,181.0\n"
        "2_1016_0,2,1016,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    labeling.label_patches(idx, out, _cfg_with_threshold(5))
    assert _read_labels(out) == {"2_1016_0.png": "steatosis/2_1016_0.png"}
    assert [item.name for item in out.rglob("*.png")] == ["2_1016_0.png"]


def test_gray_threshold_is_a_percentile_of_the_slide_not_of_the_patch(
    tmp_output_tree, monkeypatch
):
    # The gray threshold is `blobs.gray_percentile` of that slide's `tissue_gray` column
    # and never of a patch. A percentile of a patch moves with the patch, so a
    # regression to that form gives every patch of a slide its own number — and, on the
    # uniform patch below, the same number (180.0) for every patch of every slide, which
    # is why the two derived thresholds are asserted to differ from each other and from
    # 180.0 as well as to equal the two percentiles.
    from blobcount import blobs, labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    seen = []
    monkeypatch.setattr(
        labeling, "count_blobs", lambda patch, params, mpp: seen.append(params) or 0
    )

    def _boom(*args, **kwargs):
        raise AssertionError("the per-patch gray threshold was used")

    monkeypatch.setattr(blobs, "gray_threshold_from_patch", _boom)
    monkeypatch.setattr(blobs, "params_from_config", _boom)
    # Neither name is imported here today, so neither of the two patches above changes what
    # this run does; they close the form a regression could take, and the two below close
    # the other. A regression written `blobs.params_from_config(...)` resolves the name on
    # the module at call time and trips the first pair; one written as a
    # `from blobcount.blobs import params_from_config` added to this module binds its own
    # name here and trips the second. Between them a regression to the per-patch threshold
    # is caught in either form, which is the state this test exists to hold.
    monkeypatch.setattr(labeling, "gray_threshold_from_patch", _boom, raising=False)
    monkeypatch.setattr(labeling, "params_from_config", _boom, raising=False)

    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5010,181.0\n"
        "A_256_0,A,256,0,0.5015,181.0\n"
        "B_0_0,B,0,0,0.5010,200.0\n"
        "B_256_0,B,256,0,0.5015,240.0\n",
        encoding="utf-8",
    )
    stats = labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))

    assert stats.per_slide_gray_threshold == {
        "A": pytest.approx(np.percentile([181.0, 181.0], 95)),
        "B": pytest.approx(np.percentile([200.0, 240.0], 95)),
    }
    assert stats.per_slide_gray_threshold["A"] != stats.per_slide_gray_threshold["B"]
    assert 180.0 not in stats.per_slide_gray_threshold.values()
    # And the `BlobParams` that reached the count carry those numbers, one per slide, in
    # index order — so the counts were binarised at the slide's value and not at a
    # summary of it.
    assert [item.gray_threshold for item in seen] == [
        stats.per_slide_gray_threshold["A"],
        stats.per_slide_gray_threshold["A"],
        stats.per_slide_gray_threshold["B"],
        stats.per_slide_gray_threshold["B"],
    ]


def test_per_patch_mpp_is_used_not_a_global(tmp_output_tree, monkeypatch):
    # Two patches of the same slide at different achieved mpp must be thresholded
    # with their own value; a global value silently reintroduces the original defect.
    from blobcount import labeling
    seen = []
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs",
                        lambda patch, params, mpp: seen.append(mpp) or 0)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5010,181.0\n"
        "A_256_0,A,256,0,0.5015,181.0\n",
        encoding="utf-8",
    )
    labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    assert seen == [pytest.approx(0.5010), pytest.approx(0.5015)]


def test_stats_json_is_written(tmp_output_tree, monkeypatch):
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    written = json.loads((tmp_output_tree / "labels" / "labeling_stats.json").read_text("utf-8"))
    assert written["blob_threshold"] == 5
    assert written["per_slide"]["A"]["n"] == 1


def test_labels_csv_carries_path_label_and_mpp(tmp_output_tree, monkeypatch):
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    monkeypatch.setattr(labeling, "count_blobs", lambda patch, params, mpp: 7)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    labeling.label_patches(idx, out, _cfg_with_threshold(5))
    with (out / "labels.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["patch_id"] == "A_0_0"
    assert rows[0]["slide_id"] == "A"
    assert int(rows[0]["label"]) == 1
    assert rows[0]["blob_count"] == "7"
    assert float(rows[0]["mpp"]) == pytest.approx(0.5010)
    assert rows[0]["path"].endswith("steatosis/A_0_0.png")


def test_an_index_row_carrying_the_extension_is_labeled_from_a_real_file(
    tmp_output_tree, synthetic_patch
):
    # The `patch_id` this pipeline writes is the file name, extension included
    # (`2_1016_0.png`, `extract.py:380`), so `patch_id` and the file name are the same
    # string and `_patch_file_name` is the identity. Nothing here stubs `load_patch`, so
    # the source lookup at `index.parent / name` has to resolve against a real PNG or the
    # test fails; and `count_blobs` is the real one, so the count of 3 discs is measured
    # rather than asserted by a stub.
    from blobcount import labeling
    index = _write_patches(
        tmp_output_tree / "patches", ["2_1016_0.png"], synthetic_patch.array
    )
    out = tmp_output_tree / "labels"
    stats = labeling.label_patches(index, out, _cfg_with_threshold(5))

    rows = _read_label_rows(out)
    assert [row["patch_id"] for row in rows] == ["2_1016_0.png"]
    assert rows[0]["path"] == "normal/2_1016_0.png"
    # The count the real `count_blobs` returned for the three discs the conftest fixture
    # draws, so the class above is a fact about the pixels and not about a stub.
    assert int(rows[0]["blob_count"]) == 3
    assert stats.per_slide["2"] == {"normal": 1, "steatosis": 0, "n": 1}
    # One file, named once: not `2_1016_0.png.png`, and not a second copy left in the
    # other class directory.
    assert sorted(item.name for item in out.rglob("*.png")) == ["2_1016_0.png"]
    with _pil().open(out / rows[0]["path"]) as written:
        assert written.size == (256, 256)


def test_a_non_ascii_slide_id_round_trips_through_the_csv_and_the_json(
    tmp_output_tree, synthetic_patch
):
    # The read side of a non-ASCII id is `test_index_reads_utf8_slide_ids`; this is the
    # write side, which is a different encoding decision in each file. `labels.csv` is
    # written through `open(encoding="utf-8")` and `labeling_stats.json` through
    # `json.dumps(..., ensure_ascii=False)`, so the id is on disk as the characters and
    # not as `\u` escapes — a machine whose code page is Turkish has to be able to read
    # both back.
    from blobcount import labeling
    index = _write_patches(
        tmp_output_tree / "patches", ["hasta-ğü-01_0_0.png"], synthetic_patch.array
    )
    out = tmp_output_tree / "labels"
    labeling.label_patches(index, out, _cfg_with_threshold(5))

    rows = _read_label_rows(out)
    assert [row["slide_id"] for row in rows] == ["hasta-ğü-01"]
    assert (out / rows[0]["path"]).is_file()
    text = (out / "labeling_stats.json").read_text(encoding="utf-8")
    assert "hasta-ğü-01" in text
    assert "\\u" not in text
    assert json.loads(text)["per_slide"]["hasta-ğü-01"]["n"] == 1


def test_a_bad_index_cell_raises_an_input_error_with_the_original_attached(
    tmp_output_tree, monkeypatch
):
    # The wrapper is a re-raise and not a copy: the traceback behind the refusal is the
    # one that says which read refused, and that is only true if the original is attached
    # as `__cause__`. The dedicated type is what lets the command line tell this from a
    # defect, so the two halves are pinned here rather than only through the CLI.
    from blobcount import labeling
    monkeypatch.setattr(labeling, "load_patch", lambda p, s: np.full((s, s, 3), 180, np.uint8))
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,not-a-number,181.0\n",
        encoding="utf-8",
    )
    with pytest.raises(labeling.LabelingInputError) as refused:
        labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    assert isinstance(refused.value.__cause__, ValueError)
    assert "not-a-number" in str(refused.value.__cause__)


def test_a_patch_that_is_not_the_expected_array_is_an_input_error(tmp_output_tree):
    # The fourth thing the type covers: a patch file the index names and `load_patch`
    # refuses. A valid PNG at half the size, so the file is readable and it is the array
    # that is wrong — which is the case the run has to name rather than resize past.
    from blobcount import labeling
    index = _write_patches(
        tmp_output_tree / "patches", ["A_0_0.png"], np.full((256, 256, 3), 180, np.uint8)
    )
    _pil().fromarray(np.full((128, 128, 3), 180, np.uint8)).save(index.parent / "A_0_0.png")
    with pytest.raises(labeling.LabelingInputError) as refused:
        labeling.label_patches(index, tmp_output_tree / "labels", _cfg_with_threshold(5))
    assert isinstance(refused.value.__cause__, ValueError)
    assert "A_0_0.png" in str(refused.value)


def test_a_truncated_row_leaving_no_cell_is_an_input_error(tmp_output_tree):
    # A **pin**, not red evidence: it is green before the per-slide block is narrowed as
    # well as after, because both places wrap a `TypeError` from a cell a short index
    # line leaves `None`. It is here because that cell is the *only* thing the narrow
    # conversion exists for — nothing else in the run converts a cell — so dropping its
    # `TypeError` arm turns this red, and that arm would otherwise be untested.
    from blobcount import labeling
    idx = tmp_output_tree / "patches.csv"
    # Four fields under a six-column header, so `csv.DictReader` fills `mpp` and
    # `tissue_gray` with `None` rather than raising on the short line.
    idx.write_text("patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0\n", encoding="utf-8")
    with pytest.raises(labeling.LabelingInputError) as refused:
        labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    assert isinstance(refused.value.__cause__, TypeError)
    assert "cell that is not the number its column records" in str(refused.value)


def test_a_configured_percentile_numpy_refuses_is_not_an_input_error(tmp_output_tree):
    # The other direction of the boundary, and the one a `try` around the whole
    # per-slide block cannot make: `blobs.gray_percentile: 150` is a value `load`
    # accepts, because it range-checks nothing and only `np.percentile` refuses it. The
    # index below is perfect, so the only thing that can fail here is this project, and
    # reporting it as a bad cell would send the caller off to fix a `patches.csv` that is
    # not broken. It leaves as the `ValueError` it is.
    from blobcount import labeling
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    cfg = load()
    cfg.set("blobs.gray_percentile", 150)
    with pytest.raises(ValueError) as escaped:
        labeling.label_patches(idx, tmp_output_tree / "labels", cfg)
    # Stated explicitly rather than left to `pytest.raises`: `LabelingInputError` derives
    # from `Exception` and not from `ValueError`, so the `raises` alone already excludes
    # it. This line says which exclusion is the point of the test.
    assert not isinstance(escaped.value, labeling.LabelingInputError)
    assert "Percentiles must be in the range" in str(escaped.value)


def test_a_type_error_from_the_patch_reader_is_not_reported_as_a_bad_index(
    tmp_output_tree, monkeypatch
):
    # The narrowed tuple around the read of a patch file. `load_patch` raises a
    # `ValueError` of its own for an array of the wrong shape and lets `OSError` out for
    # a path that is not there; it raises no `TypeError` for any corrupt file, so a
    # `TypeError` can only be a defect in this project. Catching it reported that defect
    # as "your patch file is wrong" and exited 2, which sends the caller to fix a file
    # that is fine. It leaves as itself instead.
    from blobcount import labeling

    def _boom(path, size):
        raise TypeError("injected from the reader")

    monkeypatch.setattr(labeling, "load_patch", _boom)
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0,A,0,0,0.5010,181.0\n",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="injected from the reader") as escaped:
        labeling.label_patches(idx, tmp_output_tree / "labels", _cfg_with_threshold(5))
    # Stated rather than left to the `raises`: `LabelingInputError` derives from
    # `Exception`, so a narrowing back to a `LabelingInputError` would fail the line
    # above already. This one says which exclusion the test is about.
    assert not isinstance(escaped.value, labeling.LabelingInputError)


def _load_cli(monkeypatch):
    path = PROJECT_ROOT / "bin" / "label_patches.py"
    spec = importlib.util.spec_from_file_location("blobcount_cli_label_patches", path)
    module = importlib.util.module_from_spec(spec)
    # Through `monkeypatch` rather than `sys.modules[...] = ...`, so the entry is torn
    # down with the test, for the reason `tests/test_extract.py` gives.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _config_file(directory, name, *, labels, **sections):
    """Return a configuration file with `paths.labels` set to `labels` and `sections` merged.

    `labels` is written as given rather than through `str`, because the values this exists
    to exercise are the ones `load` accepts and `Config.path` refuses — the empty string
    and a non-string — and `str` would turn both into a usable path. The other `sections`
    are merged into the default configuration, so a test can hand a run a value the
    configuration declares and this project nevertheless refuses, such as a
    `blobs.gray_percentile` outside [0, 100].
    """
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["labels"] = labels
    for section, values in sections.items():
        data[section].update(values)
    path = directory / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _config_pointing_labels_at(target):
    return _config_file(target.parent, "config.yaml", labels=str(target))


@pytest.fixture
def cli(project_scratch, synthetic_patch, monkeypatch):
    """The entry point, a real index over real patch files, and the label root.

    `project_scratch` and not pytest's `tmp_path`, because `label_patches` clears
    `out_dir` through `prepare_output`, which refuses a target that is not strictly below
    the project root. The patches are real PNGs and `count_blobs` is the real one, so the
    three discs the conftest fixture draws are counted rather than stubbed: 3 is above
    `--threshold 2` and below the configured 5, which is what lets one test flip a class
    with the flag alone.
    """
    module = _load_cli(monkeypatch)
    index = _write_patches(
        project_scratch / "patches", ["2_1016_0.png", "3_0_0.png"], synthetic_patch.array
    )
    out = project_scratch / "labels"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH", _config_pointing_labels_at(out)
    )
    return module, index, out


def test_cli_labels_every_row_and_exits_zero(cli, capsys):
    module, index, out = cli
    assert module.main(["--index", str(index)]) == 0

    assert _read_labels(out) == {
        "2_1016_0.png": "normal/2_1016_0.png",
        "3_0_0.png": "normal/3_0_0.png",
    }
    written = json.loads((out / "labeling_stats.json").read_text(encoding="utf-8"))
    assert written["per_slide"] == {
        "2": {"normal": 1, "steatosis": 0, "n": 1},
        "3": {"normal": 1, "steatosis": 0, "n": 1},
    }
    printed = capsys.readouterr().out
    # Both tables, in the order the docstring promises: the counts, then the resolution
    # and the threshold the counts were made at.
    assert printed.index("slide_id  normal  steatosis  n") < printed.index(
        "slide_id  achieved_mpp  gray_threshold"
    )
    assert "2 patches labelled" in printed


def test_the_stats_record_the_labeling_rule_that_produced_them(cli):
    # **Spec §4 asks every stage for an artifact recording its inputs, resolved parameters
    # and outputs, and this one recorded the threshold and nothing else.** Every number in
    # the README's ablation section is a function of the eight `blobs.*` keys below, of
    # `labeling.blob_threshold` and of the `extraction.patch_size` every patch array is
    # validated against, so an artifact holding the last and not the first nine cannot be
    # re-examined: a reader who wants to know why specimen 4's counts are 2.63 has the
    # answer's output and none of its inputs. The values here are this configuration's; the
    # rule that keeps the list and the reads in step is the next test.
    #
    # **The keys and not the whole configuration**, for the reason `blobcount.config.
    # resolved` gives, and read through it so each is recorded as a reader the source scan
    # in `tests/test_config.py` can see.
    module, index, out = cli
    assert module.main(["--index", str(index)]) == 0
    written = json.loads((out / "labeling_stats.json").read_text(encoding="utf-8"))
    assert written["config"] == {
        "labeling.blob_threshold": 5,
        "extraction.patch_size": 256,
        "blobs.area_um2_min": 3.1,
        "blobs.area_um2_max": 125.0,
        "blobs.circularity_min": 0.6,
        "blobs.gray_percentile": 95,
        "blobs.morph_kernel_size": 3,
        "blobs.morph_iterations": 2,
        "blobs.stain_normalization": False,
    }
    assert written["provenance"].startswith("written by blobcount.labeling.label_patches")
    # And it says what the labels are, which is the one thing a reader of a labels.csv
    # cannot work out for themselves.
    assert "not a pathologist" in written["provenance"]


def test_the_recorded_config_block_covers_every_key_this_stage_reads(cli):
    # **The rule, not the value.** The assertion above pins eight names and eight values,
    # which is a fact about today's configuration and which cannot notice the ninth key
    # `label_patches` reads: `extraction.patch_size` at the top of the function, passed to
    # `load_patch` for every patch, absent from the block, absent from the test above. A
    # patch array is a `(patch_size, patch_size, 3)` check and that check decides whether a
    # run labels a patch or refuses it, so a block that cannot say what size was required
    # cannot be re-examined by the reader the artifact exists for — and the sibling
    # artifact, written by the same rule at `bin/ablate_threshold.py`, records it.
    #
    # **Two halves, and the second is the one that catches the next omission.** The first
    # says every recorded key has a reader somewhere under `blobcount/`, which is the
    # property `blobcount.config.resolved` documents and which
    # `test_every_declared_config_key_is_read_by_package_source` enforces for declared
    # keys; stated here as well because the two answer different questions and a future
    # record in another stage would not reach the package-wide test. The second says every
    # key *this module* reads is recorded, which is the direction nothing else in the suite
    # checks in either direction for a single stage.
    module, index, out = cli
    assert module.main(["--index", str(index)]) == 0
    recorded = list(_CONFIG_KEYS)
    assert len(set(recorded)) == len(recorded), (
        "the recorded block names a key twice: "
        f"{[key for key in recorded if recorded.count(key) > 1]}"
    )
    reads = set(_KEY_READ.findall((PROJECT_ROOT / "blobcount" / "labeling.py").read_text(encoding="utf-8")))
    package = set(
        _KEY_READ.findall(
            "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted((PROJECT_ROOT / "blobcount").rglob("*.py"))
            )
        )
    )
    # Neither scan is allowed to be empty: a pattern that stopped matching would leave both
    # assertions below comparing an empty set and pass, which is this project's recurring
    # defect and not worth repeating here.
    assert reads, "no cfg.get read was found in blobcount/labeling.py; the scan is broken"
    assert package, "no cfg.get read was found under blobcount/; the scan is broken"

    unreadable = [key for key in recorded if key not in package]
    assert not unreadable, (
        "these keys are recorded in labeling_stats.json but no literal reader under "
        f"blobcount/ can be found for them: {unreadable}"
    )
    unrecorded = [key for key in sorted(reads) if key not in set(recorded)]
    assert not unrecorded, (
        "blobcount/labeling.py reads these configuration keys and labeling_stats.json "
        f"records none of them: {unrecorded}. The block is the inputs of the run, so a "
        "read key missing from it is a reader of the artifact who cannot tell what the "
        "run was given."
    )


def test_the_recorded_threshold_follows_a_hand_edited_configuration(cli):
    # The point of recording resolved values rather than the key names: the number in the
    # block is the number the run used. `--threshold 2` moves `labeling.blob_threshold`,
    # and the block moves with it rather than recording the shipped five.
    module, index, out = cli
    assert module.main(["--index", str(index), "--threshold", "2"]) == 0
    written = json.loads((out / "labeling_stats.json").read_text(encoding="utf-8"))
    assert written["config"]["labeling.blob_threshold"] == 2
    assert written["blob_threshold"] == 2


def test_cli_threshold_reaches_the_labels_and_the_stats(cli):
    module, index, out = cli
    assert module.main(["--index", str(index), "--threshold", "2"]) == 0
    assert _read_labels(out) == {
        "2_1016_0.png": "steatosis/2_1016_0.png",
        "3_0_0.png": "steatosis/3_0_0.png",
    }
    written = json.loads((out / "labeling_stats.json").read_text(encoding="utf-8"))
    assert written["blob_threshold"] == 2
    assert written["per_slide"]["2"]["steatosis"] == 1
    assert written["per_slide"]["2"]["normal"] == 0


def test_cli_refuses_to_delete_a_target_outside_the_project(cli, tmp_path, capsys):
    module, index, _ = cli
    outside = tmp_path / "escape"
    outside.mkdir()
    (outside / "keep.png").write_bytes(b"x")
    assert module.main(["--index", str(index), "--out", str(outside)]) == 2
    assert "refusing" in capsys.readouterr().err
    assert sorted(item.name for item in outside.iterdir()) == ["keep.png"]
    assert (outside / "keep.png").read_bytes() == b"x"


def test_cli_refuses_a_missing_index_with_the_refusal_code(cli, tmp_path, capsys):
    # 1 is `EXIT_CONTRADICTORY`, so a mistyped `--index` exiting 1 would tell any caller
    # reading the code that the labels contradicted themselves. The previous run's labels
    # are still on disk, because the index is read before the tree is cleared.
    module, _, out = cli
    missing = tmp_path / "nope.csv"
    assert module.main(["--index", str(missing)]) == module.EXIT_REFUSED
    assert "nope.csv" in capsys.readouterr().err
    assert not out.exists()


def test_cli_refuses_an_index_without_the_expected_header(cli, capsys):
    module, index, _ = cli
    malformed = index.parent / "malformed.csv"
    malformed.write_text("id,slide\nA_0_0,A\n", encoding="utf-8")
    assert module.main(["--index", str(malformed)]) == module.EXIT_REFUSED
    assert "mpp" in capsys.readouterr().err


def test_cli_refuses_an_index_carrying_a_cell_that_is_not_a_number(cli, capsys):
    # A cell the column says is a number and is not one is the same kind of answer as the
    # missing header above: something the caller pointed at, so it is reported on stderr
    # and not as a traceback. Nothing was labelled, because the conversion fails before
    # the first patch is written.
    module, index, out = cli
    malformed = index.parent / "bad-cell.csv"
    malformed.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "2_1016_0.png,2,1016,0,not-a-number,181.0\n",
        encoding="utf-8",
    )
    assert module.main(["--index", str(malformed)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "not-a-number" in err
    assert "Traceback" not in err
    assert not (out / "labels.csv").exists()


def test_cli_does_not_swallow_a_defect_from_inside_label_patches(cli, monkeypatch):
    # The other direction, and the one a single `except (OSError, ValueError)` around the
    # whole call cannot make: a `ValueError` from the counting is not a statement about
    # the index, and reporting it as one would send the caller off to fix their
    # `--index` for a defect in this project. It leaves `main` as an exception instead.
    module, index, _ = cli
    from blobcount import labeling

    def _boom(patch, params, mpp):
        raise ValueError("injected")

    monkeypatch.setattr(labeling, "count_blobs", _boom)
    with pytest.raises(ValueError, match="injected"):
        module.main(["--index", str(index)])


def test_cli_refuses_an_index_carrying_a_cell_over_the_csv_field_limit(cli, capsys):
    # `csv.DictReader` raises `_csv.Error` on a field longer than `csv.field_size_limit`
    # rather than truncating it, and `csv.Error` is neither a `ValueError` nor an
    # `OSError`, so a catch around `read_index` that omits it lets the exception out of
    # `main`. The process status is then 1, which this program defines as
    # `EXIT_CONTRADICTORY` — "the labels contradicted themselves" — reported for a
    # malformed index, and the answer a caller reading the exit contract gets is the wrong
    # one.
    module, index, _ = cli
    oversized = index.parent / "oversized.csv"
    oversized.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        f"2_1016_0.png,2,1016,0,0.5010,{'1' * 200000}\n",
        encoding="utf-8",
    )
    # Asserted rather than assumed: a limit above the cell would let this pass without
    # ever reaching the guard it exists to exercise.
    assert csv.field_size_limit() < 200000
    assert module.main(["--index", str(oversized)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "field larger than field limit" in err
    assert "Traceback" not in err


def test_cli_does_not_report_a_bad_percentile_as_a_bad_index(cli, project_scratch, monkeypatch):
    # The same defect one level up, and the reason exit 2 has to be reserved for the
    # argument: a configured percentile this project range-checks nowhere is refused by
    # numpy, and 2 would name the caller's `patches.csv` for it. It leaves `main` as the
    # exception it is, so stderr is a traceback naming a numpy call.
    module, index, out = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _config_file(
            project_scratch, "config-bad-percentile.yaml",
            labels=str(out), blobs={"gray_percentile": 150},
        ),
    )
    with pytest.raises(ValueError, match="Percentiles must be in the range"):
        module.main(["--index", str(index)])


@pytest.mark.parametrize("value", ["", 7])
def test_cli_refuses_a_labels_path_config_path_refuses(
    cli, project_scratch, monkeypatch, capsys, value
):
    # `paths.labels` is a hand-edited string and `load` checks that it is declared and
    # nothing else, so an empty string and a non-string both reach `Config.path`, which
    # is where the refusal of them lives. Resolved outside the `try` that catches
    # `ConfigError`, they leave `main` as a traceback and the process exits 1. Neither
    # value is reachable through `--out`, which the parser has already typed as a `Path`;
    # this is the configuration's own refusal surfacing at the command line.
    module, index, _ = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _config_file(
            project_scratch, f"config-labels-{type(value).__name__}.yaml", labels=value
        ),
    )
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "paths.labels" in err
    assert "Traceback" not in err


def test_cli_refuses_before_clearing_anything_when_a_key_it_reads_is_missing(
    cli, project_scratch, monkeypatch, capsys
):
    # A configuration that omits a key `label_patches` reads *after* `prepare_output` has
    # cleared `out`. `load` accepts it while the key is not required, the tree is emptied,
    # and the `cfg.get` then raises `ConfigError` past `main` — so the previous run's
    # labels are gone and the process exits 1, which this program defines as
    # `EXIT_CONTRADICTORY`, on a configuration nothing contradicts. Required keys are the
    # fix and the reason it is the only fix: the key has to fail at load, before the
    # delete, rather than be caught after it.
    module, index, out = cli
    (out / "normal").mkdir(parents=True)
    (out / "normal" / "previous_run.png").write_bytes(b"the previous run")
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["labels"] = str(out)
    del data["blobs"]["morph_kernel_size"]
    config = project_scratch / "config-missing-morph.yaml"
    config.write_text(yaml.safe_dump(data), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", config)

    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "missing required keys" in err
    assert "blobs.morph_kernel_size" in err
    assert "Traceback" not in err
    # The point of the whole finding: the previous run is still on disk.
    assert (out / "normal" / "previous_run.png").read_bytes() == b"the previous run"


def test_cli_refuses_an_out_that_is_an_existing_file(cli, project_scratch, capsys):
    # `--out` naming a regular file. The file is inside the project on purpose: a target
    # outside it is refused by the "not strictly below the project root" rule, which
    # would make this green for a reason that has nothing to do with the file.
    # `shutil.rmtree` on a regular file raises `NotADirectoryError`, and that reached the
    # caller as a traceback exiting 1. The file is not deleted either way, and that is
    # the part worth pinning: this path never held a previous run's labels.
    module, index, _ = cli
    as_file = project_scratch / "labels-as-file"
    as_file.write_text("not a directory\n", encoding="utf-8")
    assert module.main(["--index", str(index), "--out", str(as_file)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "refusing" in err
    assert "not a directory" in err
    assert as_file.read_text(encoding="utf-8") == "not a directory\n"


def test_cli_refuses_a_paths_slides_config_path_refuses(
    cli, project_scratch, monkeypatch, capsys
):
    # R6-1 and R6-2, and the only half of round 5's mechanism this program never got.
    # `prepare_output` resolves the paths it guards the delete against through
    # `_protected_paths`, off the *default* configuration rather than the `cfg` this run
    # loaded, so a `paths.slides` whose value `Config.path` refused reached this program from
    # inside its own guard. `main` caught `OutputPathError` and `LabelingInputError`, so it
    # left as a traceback and the process exited 1 — which this file defines as
    # `EXIT_CONTRADICTORY`, "the labels contradicted themselves", reported to a caller
    # over a hand-edited configuration string.
    #
    # The type of `paths.slides` is no longer part of this: `load` types every `paths.*` key
    # as `str`, so a number under one is refused there, before any `Config` exists and long
    # before this program's own guard runs. What reaches the guard from a *file* is a value
    # of the declared type that `Config.path` still refuses, and the assertion below is
    # about that arm surviving — it is not dead, and the round that made it unreachable from
    # the CLI did not remove it.
    #
    # The stale label is the assertion that matters, and it is the same one the extract
    # side makes: `_protected_paths` runs before `shutil.rmtree`, so a refusal here
    # deletes nothing. A fix that caught the exception after the delete would return the
    # same code and still take the previous run with it.
    module, index, out = cli
    (out / "normal").mkdir(parents=True)
    (out / "normal" / "previous_run.png").write_bytes(b"the previous run")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _config_file(
            project_scratch, "config-slides-not-a-path.yaml",
            labels=str(out), paths={"slides": 7},
        ),
    )
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "paths.slides" in err
    assert "Traceback" not in err
    assert (out / "normal" / "previous_run.png").read_bytes() == b"the previous run"
