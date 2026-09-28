import numpy as np
import pytest

from blobcount.config import load
from blobcount.labeling import read_index


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
    idx = tmp_output_tree / "patches.csv"
    idx.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0,A,0,0,0.5,181.0\n"
        "B_0_0,B,0,0,0.5,181.0\n",
        encoding="utf-8",
    )
    out = tmp_output_tree / "labels"
    for threshold in (5, 25):
        labeling.label_patches(idx, out, _cfg_with_threshold(threshold))
    names = [f"{f.parent.name}/{f.name}" for f in out.rglob("*.png")]
    assert len(names) == len(set(names))


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
    import json
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
    import csv
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
