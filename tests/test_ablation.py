import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from blobcount.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load
from blobcount.extract import write_index

# The two forms of the gray-threshold rule, and what each of them counted on the same 60
# accepted patches of `data/2.svs` at 0.5001 um/px. Task 6 measured them and the
# ablation carries them into its own artifact, so that a reader of the sweep knows what
# distribution the sweep is a sweep of. Written here as well as in the module because a
# constant that only its own module quotes is a constant nothing checks.
_RULE_IN_USE = {
    "rule_in_use": True,
    "rule": "95th percentile of the slide's tissue_gray column",
    "total_blobs": 2166,
    "non_zero_patches": 57,
    "max_per_patch": 86,
}
_SUPERSEDED_RULE = {
    "rule_in_use": False,
    "rule": "95th percentile of the patch's own grayscale",
    "total_blobs": 377,
    "non_zero_patches": 59,
    "max_per_patch": 19,
}


def _pil():
    """Return the PIL `Image` module, skipping the test when it will not import.

    Copied from `tests/test_labeling.py` for the reason given there: a module-scope
    import of a native wheel turns one absent wheel into a collection error for every
    test in this file, including the ones that read no image at all.
    """
    return pytest.importorskip(
        "PIL.Image",
        reason="Pillow cannot be imported, so no patch can be written or read",
    )


def _row(patch_id, slide_id, *, x=0, y=0, mpp=0.5010, tissue_gray=180.0):
    """Return one index row, under the columns `extract.INDEX_COLUMNS` pins."""
    return {
        "patch_id": patch_id,
        "slide_id": slide_id,
        "x": x,
        "y": y,
        "mpp": mpp,
        "tissue_gray": tissue_gray,
    }


def _write_index(directory, rows, array=None):
    """Write `rows` as the patch index of `directory`, and a real PNG per row when given one.

    The header is written by `extract.write_index` rather than typed here, so the index
    a test builds is the one `read_index` accepts and there is no second definition of
    `INDEX_COLUMNS` in this file. `array` is written under each row's `patch_id` as a
    real file, because the count is made through `labeling.load_patch` reading a path
    the caller resolves against `index.parent`; a stubbed reader would leave that lookup
    untested.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if array is not None:
        Image = _pil()
        for row in rows:
            # Only a name that ends in an extension names a file, and the index form
            # `2_1016_0` does not: one test below needs that row to name no patch at all.
            if row["patch_id"].endswith(".png"):
                Image.fromarray(array).save(directory / row["patch_id"])
    index = directory / "patches.csv"
    write_index(list(rows), index)
    return index


def _cfg(directory, name, *, results, **sections):
    """Return a configuration file with `paths.results` set and `sections` merged in.

    Written with `yaml.safe_dump` and `allow_unicode=True` so a non-ASCII value in a
    merged section survives, and a `Path` is written as its string while every other
    value is written as given, because the values this exists to exercise include the
    ones `Config.path` refuses and `str` would turn those into usable paths.
    """
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["results"] = str(results) if isinstance(results, Path) else results
    for section, values in sections.items():
        data[section].update(values)
    path = directory / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _balanced_pair():
    """Return 100 patches at count 20 on slide A and 100 at count 0 on slide B."""
    counts, slide_of = {}, {}
    for i in range(100):
        counts[f"A{i}"] = 20
        slide_of[f"A{i}"] = "A"
        counts[f"B{i}"] = 0
        slide_of[f"B{i}"] = "B"
    return counts, slide_of


# The two tests the plan pins by name, reproduced as written. The first is the one that
# names the defect this rebuild exists to fix; the second is the interface contract a
# caller reads, and on its own it cannot fail against a pass-through, which is why the
# tests below carry the weight it does not.


def test_pooled_balance_hides_per_slide_spread():
    from blobcount.ablation import sweep, summarise
    # 100 patches on slide A all high, 100 on slide B all low
    counts, slide_of = {}, {}
    for i in range(100):
        counts[f"A{i}"] = 20; slide_of[f"A{i}"] = "A"
        counts[f"B{i}"] = 0;  slide_of[f"B{i}"] = "B"
    res = sweep(counts, slide_of, [5])
    s = summarise(res)
    assert s[5]["pooled_steatosis_ratio"] == pytest.approx(0.5)
    assert s[5]["per_slide_ratio"]["A"] == pytest.approx(1.0)
    assert s[5]["per_slide_ratio"]["B"] == pytest.approx(0.0)
    assert s[5]["per_slide_ratio_spread"] == pytest.approx(1.0)


def test_threshold_appears_in_output():
    from blobcount.ablation import sweep, summarise
    counts = {f"p{i}": i for i in range(40)}
    slide_of = {f"p{i}": "A" for i in range(40)}
    s = summarise(sweep(counts, slide_of, [3, 5, 7]))
    assert set(s) == {3, 5, 7}
    assert s[5]["n_steatosis"] == 35


def test_the_spread_is_a_separate_number_at_every_threshold():
    # Four slides, twenty patches each, counts in {0, 4, 6}:
    #   A 15x0  5x4       ratio 0.25 at 1, 0.0 at 5 and above
    #   B  5x0 15x4       ratio 0.75 at 1, 0.0 at 5 and above
    #   C 10x0  5x4 5x6   ratio 0.5 at 1, 0.25 at 5, 0.0 at 7
    #   D 20x6            ratio 1.0 at 1 and 5, 0.0 at 7
    from blobcount.ablation import sweep, summarise

    spec = {
        "A": [0] * 15 + [4] * 5,
        "B": [0] * 5 + [4] * 15,
        "C": [0] * 10 + [4] * 5 + [6] * 5,
        "D": [6] * 20,
    }
    counts, slide_of = {}, {}
    for slide, values in spec.items():
        for index, value in enumerate(values):
            counts[f"{slide}{index}"] = value
            slide_of[f"{slide}{index}"] = slide
    swept = sweep(counts, slide_of, [1, 5, 7, 40])
    s = summarise(swept)

    # 50 of the 80 patches clear 1, 25 clear 5, and none clear 7 or 40.
    assert {t: s[t]["n_steatosis"] for t in (1, 5, 7, 40)} == {1: 50, 5: 25, 7: 0, 40: 0}
    # The spread is the max minus the min of the four per-slide ratios: 0.75 where the
    # pooled ratio is 0.625, 1.0 where it is 0.3125, and 0.0 where it is 0.0. A `sweep`
    # that pooled the counts before grouping reports 0.0 for every threshold here and
    # fails each of these lines, which is the defect the first test names.
    assert {t: s[t]["per_slide_ratio_spread"] for t in (1, 5, 7, 40)} == pytest.approx(
        {1: 0.75, 5: 1.0, 7: 0.0, 40: 0.0}
    )
    # The two numbers are not the same quantity, and that is what stops one from being
    # reported in place of the other. Both of the first two thresholds are within 0.2 of
    # a balanced pooled ratio, and the spread moves by a quarter between them, so no
    # rule over the pooled column alone can tell which of the two the slides disagree
    # about; the narrowest spread is where the pooled ratio is 0.0 and carries nothing.
    balanced = min((1, 5, 7, 40), key=lambda t: abs(s[t]["pooled_steatosis_ratio"] - 0.5))
    assert balanced == 1
    assert s[balanced]["per_slide_ratio_spread"] == pytest.approx(0.75)
    assert abs(s[1]["pooled_steatosis_ratio"] - 0.5) < 0.2
    assert abs(s[5]["pooled_steatosis_ratio"] - 0.5) < 0.2
    assert s[1]["per_slide_ratio_spread"] != pytest.approx(s[5]["per_slide_ratio_spread"])
    narrowest = min((1, 5, 7, 40), key=lambda t: s[t]["per_slide_ratio_spread"])
    assert s[narrowest]["pooled_steatosis_ratio"] == pytest.approx(0.0)
    # The selection is marked on a copy: the sweep's own record is untouched, so a
    # caller holding both sees the sweep as measured and the selection over it.
    assert "selected" not in swept[5]
    assert not any(record["selected"] for record in s.values())


def test_summarise_marks_the_selected_threshold():
    from blobcount.ablation import sweep, summarise
    counts, slide_of = _balanced_pair()
    s = summarise(sweep(counts, slide_of, [3, 5, 7]), 5)
    assert [t for t in (3, 5, 7) if s[t]["selected"]] == [5]
    assert s[3]["pooled_steatosis_ratio"] == s[5]["pooled_steatosis_ratio"]
    assert s[3]["selected"] is False and s[5]["selected"] is True


def test_summarise_refuses_a_selected_threshold_it_was_not_given():
    # The second panel of the figure is drawn at the selected threshold, so a selection
    # naming a threshold the sweep never crossed is a caller error rather than a figure
    # with an empty panel. `bin/ablate_threshold.py` checks the same fact before it
    # reads a single patch and reports it as a refusal.
    from blobcount.ablation import sweep, summarise
    counts, slide_of = _balanced_pair()
    with pytest.raises(ValueError, match="9"):
        summarise(sweep(counts, slide_of, [3, 5, 7]), 9)


def test_sweep_refuses_two_mappings_that_disagree_about_a_patch():
    from blobcount.ablation import sweep
    counts, slide_of = _balanced_pair()
    short = dict(slide_of)
    del short["A7"]
    with pytest.raises(ValueError, match="A7"):
        sweep(counts, short, [5])
    extra = dict(slide_of)
    extra["C9"] = "C"
    with pytest.raises(ValueError, match="C9"):
        sweep(counts, extra, [5])


def test_sweep_refuses_an_empty_count_map():
    # The pooled ratio is `n_steatosis / n` and an empty map has no `n`, so a sweep over
    # an index with no rows would divide by nothing. Refused before the division rather
    # than left to a `ZeroDivisionError` a command line has no answer for.
    from blobcount.ablation import sweep
    with pytest.raises(ValueError, match="no patches"):
        sweep({}, {}, [5])


def test_plot_writes_a_figure(tmp_path):
    from blobcount.ablation import plot, summarise, sweep
    counts, slide_of = _balanced_pair()
    s = summarise(sweep(counts, slide_of, [3, 5, 7]), 5)
    out = tmp_path / "ablation.png"
    plot(s, out)
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_refuses_a_summary_with_no_selected_threshold(tmp_path):
    from blobcount.ablation import plot, summarise, sweep
    counts, slide_of = _balanced_pair()
    unmarked = summarise(sweep(counts, slide_of, [3, 5, 7]))
    with pytest.raises(ValueError, match="selected"):
        plot(unmarked, tmp_path / "ablation.png")
    assert not (tmp_path / "ablation.png").exists()


def test_plot_leaves_no_figure_open(tmp_path, monkeypatch):
    # matplotlib counts the figures a process has open and warns past
    # `figure.max_open_warning` of them, and `filterwarnings = ["error"]` turns that
    # warning into a failure. The bound is lowered to 2 here so that a `plot` which saves
    # and forgets to close fails on the third call instead of on the twenty-first.
    import matplotlib

    monkeypatch.setitem(matplotlib.rcParams, "figure.max_open_warning", 2)
    from blobcount.ablation import plot, summarise, sweep

    counts, slide_of = _balanced_pair()
    s = summarise(sweep(counts, slide_of, [3, 5, 7]), 5)
    for index in range(3):
        plot(s, tmp_path / f"ablation-{index}.png")
    import matplotlib.pyplot as plt

    assert plt.get_fignums() == []


def test_ablate_counts_each_patch_at_its_own_recorded_mpp(
    project_scratch, synthetic_patch, monkeypatch
):
    # Two patches of one slide at two achieved resolutions. `count_blobs` requires the
    # row's own value and a global here is the original defect wearing a different hat,
    # so what each count was handed is recorded rather than inferred from a count.
    from blobcount import ablation

    seen = []
    monkeypatch.setattr(
        ablation, "count_blobs", lambda patch, params, mpp: seen.append(mpp) or 0
    )
    index = _write_index(
        project_scratch / "patches",
        [
            _row("A_0_0.png", "A", x=0, y=0, mpp=0.5010),
            _row("A_256_0.png", "A", x=256, y=0, mpp=0.5015),
        ],
        synthetic_patch.array,
    )
    # `selected` is named because the default is `labeling.blob_threshold` of 5 and this
    # configuration sweeps `[1, 2]`, and a selection outside the sweep is refused.
    ablation.ablate(index, load(overrides=["ablation.thresholds=[1, 2]"]), selected=2)
    assert seen == [0.5010, 0.5015]


def test_ablate_binarises_at_a_per_slide_gray_threshold(
    project_scratch, synthetic_patch, monkeypatch
):
    # The gray threshold is `blobs.gray_percentile` of that slide's `tissue_gray` column
    # and never of a patch. A percentile of the patch in hand moves with the patch, so a
    # regression to that form gives the same 180.0 for every patch of every slide here
    # — the conftest patch is 98% background, so its own 95th percentile is that
    # background value — and both derived thresholds are asserted to differ from it.
    from blobcount import ablation

    seen = []
    monkeypatch.setattr(
        ablation, "count_blobs",
        lambda patch, params, mpp: seen.append(params.gray_threshold) or 0,
    )
    index = _write_index(
        project_scratch / "patches",
        [
            _row("A_0_0.png", "A", tissue_gray=181.0),
            _row("A_256_0.png", "A", x=256, tissue_gray=181.0),
            _row("B_0_0.png", "B", tissue_gray=200.0),
            _row("B_256_0.png", "B", x=256, tissue_gray=240.0),
        ],
        synthetic_patch.array,
    )
    stats = ablation.ablate(index, load())
    assert seen == [
        float(np.percentile([181.0, 181.0], 95)),
        float(np.percentile([181.0, 181.0], 95)),
        float(np.percentile([200.0, 240.0], 95)),
        float(np.percentile([200.0, 240.0], 95)),
    ]
    assert stats.per_slide_gray_threshold["A"] != stats.per_slide_gray_threshold["B"]
    assert 180.0 not in stats.per_slide_gray_threshold.values()
    assert stats.per_slide_mpp == {"A": pytest.approx(0.5010), "B": pytest.approx(0.5010)}


def test_ablate_recomputes_the_counts_and_ignores_the_labels(project_scratch, synthetic_patch):
    # The sweep is independent of the chosen threshold, so it counts the patches again
    # from the index and never reads `labels.csv`. A `labels.csv` carrying counts this
    # run would not produce is written beside the index to make the point falsifiable: an
    # implementation that read it would report 999 blobs per patch, and every ratio below
    # would come out at 1.0.
    from blobcount import ablation

    directory = project_scratch / "patches"
    index = _write_index(
        directory,
        [_row("2_1016_0.png", "2", x=1016), _row("3_0_0.png", "3")],
        synthetic_patch.array,
    )
    (directory / "labels.csv").write_text(
        "patch_id,slide_id,label,blob_count,mpp,path\n"
        "2_1016_0.png,2,1,999,0.5010,steatosis/2_1016_0.png\n"
        "3_0_0.png,3,1,999,0.5010,steatosis/3_0_0.png\n",
        encoding="utf-8",
    )
    stats = ablation.ablate(
        index, load(overrides=["ablation.thresholds=[2, 3, 5]", "labeling.blob_threshold=3"])
    )
    # Three discs, the conftest patch, the real `count_blobs`: three blobs. The count is
    # asserted rather than stubbed because this is the test that says the sweep is made
    # of a real measurement.
    assert stats.counts == {"2_1016_0.png": 3, "3_0_0.png": 3}
    assert stats.slide_of == {"2_1016_0.png": "2", "3_0_0.png": "3"}
    # 3 >= 3 is steatosis and 3 >= 5 is not, so the sweep really moves here.
    assert stats.per_threshold[3]["n_steatosis"] == 2
    assert stats.per_threshold[5]["n_steatosis"] == 0
    assert stats.selected == 3


def test_ablate_reads_the_configured_thresholds(project_scratch, synthetic_patch):
    from blobcount import ablation

    index = _write_index(
        project_scratch / "patches", [_row("A_0_0.png", "A")], synthetic_patch.array
    )
    cfg = load(overrides=["ablation.thresholds=[1, 2, 4, 8]", "labeling.blob_threshold=4"])
    stats = ablation.ablate(index, cfg)
    # `ablation.thresholds` is the one key this task gives a reader to, so the reader it
    # gains is this module's and not the command line's: `tests/test_config.py` scans
    # `blobcount/` and not `bin/`.
    assert set(stats.per_threshold) == {1, 2, 4, 8}
    assert stats.selected == 4


def test_ablate_refuses_an_index_it_cannot_read(project_scratch):
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    directory = project_scratch / "patches"
    directory.mkdir(parents=True, exist_ok=True)
    malformed = directory / "no-header.csv"
    malformed.write_text("id,slide\nA_0_0,A\n", encoding="utf-8")
    with pytest.raises(LabelingInputError) as refused:
        ablation.ablate(malformed, load())
    # The six shapes a labeling run refuses, arriving as the same type, because the same
    # two seams raise them. The type is reused rather than redefined: a second class
    # would have to re-enumerate the shapes or subclass an empty one.
    assert refused.value.__cause__ is not None
    assert "mpp" in str(refused.value)

    bad_cell = directory / "bad-cell.csv"
    bad_cell.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\nA_0_0.png,A,0,0,not-a-number,181.0\n",
        encoding="utf-8",
    )
    with pytest.raises(LabelingInputError, match="not-a-number"):
        ablation.ablate(bad_cell, load())


def test_ablate_logs_the_specimen_the_coordinates_and_the_exception(
    project_scratch, synthetic_patch, monkeypatch, caplog
):
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    def _boom(path, size):
        raise OSError(f"{path} is not an image")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    index = _write_index(
        project_scratch / "patches", [_row("A_512_256.png", "A", x=512, y=256)],
        synthetic_patch.array,
    )
    with caplog.at_level(logging.ERROR, logger="blobcount.ablation"):
        with pytest.raises(LabelingInputError):
            ablation.ablate(index, load())
    logged = "\n".join(record.getMessage() for record in caplog.records)
    # The specimen, the coordinates the index records for that patch, and the exception
    # object itself — the three the plan requires of every caught exception.
    assert "slide A: reading patch A_512_256.png at (512, 256) raised OSError" in logged
    assert "not an image" in logged


def test_ablate_does_not_report_our_own_defect_as_a_bad_patch(
    project_scratch, synthetic_patch, monkeypatch
):
    # A `ValueError` from the counting is a defect in this project and not a statement
    # about the file the caller pointed at. It leaves as itself so the command line can
    # tell a refusal from a traceback, and this states which exclusion is being about.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    def _boom(patch, params, mpp):
        raise ValueError("injected")

    monkeypatch.setattr(ablation, "count_blobs", _boom)
    index = _write_index(
        project_scratch / "patches", [_row("A_0_0.png", "A")], synthetic_patch.array
    )
    with pytest.raises(ValueError, match="injected") as escaped:
        ablation.ablate(index, load())
    assert not isinstance(escaped.value, LabelingInputError)


def test_a_count_summary_reports_the_distribution_the_sweep_is_over(
    project_scratch, synthetic_patch
):
    # The knee question is a question about this distribution, so the run reports its
    # shape. A mean of 36.1 blobs per patch against a swept threshold of 30 is the state
    # the ablation has to show rather than one it may hide behind a ratio, and the
    # summary is over the counts themselves rather than over a ratio of them.
    from blobcount import ablation

    rows = [_row(f"A_{index * 256}_0.png", "A", x=index * 256) for index in range(4)]
    index = _write_index(project_scratch / "patches", rows, synthetic_patch.array)
    summary = ablation.ablate(index, load()).count_summary()
    assert summary == {"count": 4, "min": 3, "max": 3, "mean": 3.0, "p50": 3.0}
    # A dict comparison cannot see `3` against `3.0`, so the types are stated: a count of
    # blobs is a whole number and a mean of counts is not.
    assert isinstance(summary["min"], int) and isinstance(summary["max"], int)
    assert isinstance(summary["mean"], float) and isinstance(summary["p50"], float)


def test_ablate_counts_a_patch_the_index_names_twice_once(project_scratch, synthetic_patch):
    # Two rows for one file. Reading it twice would read every patch in the index twice,
    # and the second row's count would be the one the sweep used while the first row's
    # silently became nothing.
    from blobcount import ablation

    index = _write_index(
        project_scratch / "patches",
        [_row("2_1016_0.png", "2", x=1016), _row("2_1016_0.png", "2", x=1016)],
        synthetic_patch.array,
    )
    stats = ablation.ablate(index, load(overrides=["ablation.thresholds=[3]"]), selected=3)
    assert stats.counts == {"2_1016_0.png": 3}
    assert stats.per_threshold[3]["n_normal"] == 0
    assert stats.per_threshold[3]["n_steatosis"] == 1


def test_ablate_refuses_an_index_naming_a_patch_that_is_not_there(project_scratch):
    # `patch_id` arrives with its `.png` because `extract_slide` writes the file name, and
    # the patch is resolved as `index.parent / patch_id` with no extension appended. An
    # index that names a file which is not there — a `patch_id` carrying no extension at
    # all, or a tree that was extracted under another name — is refused rather than
    # guessed at, because appending an extension would count a file this index did not
    # name and attribute it to a patch id the index never carried.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    index = _write_index(project_scratch / "patches", [_row("2_1016_0", "2", x=1016)])
    with pytest.raises(LabelingInputError, match="2_1016_0") as refused:
        ablation.ablate(index, load())
    assert isinstance(refused.value.__cause__, OSError)


def _load_cli(monkeypatch):
    path = PROJECT_ROOT / "bin" / "ablate_threshold.py"
    spec = importlib.util.spec_from_file_location("blobcount_cli_ablate_threshold", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cli(project_scratch, synthetic_patch, monkeypatch):
    """The entry point, a real index over real patch files, and a results root.

    The results root is inside the checkout because the command line refuses a target
    that is not strictly below the project root and pytest's `tmp_path` is outside it.
    The patches are real PNGs and `count_blobs` is the real one, so the three discs the
    conftest fixture draws are counted: three is below `--selected 2` and at the
    configured 3, which is what lets one test flip a class with the flag alone.
    """
    module = _load_cli(monkeypatch)
    index = _write_index(
        project_scratch / "patches",
        [_row("2_1016_0.png", "2", x=1016), _row("3_0_0.png", "3")],
        synthetic_patch.array,
    )
    results = project_scratch / "results"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            results.parent,
            "config.yaml",
            results=results,
            ablation={"thresholds": [2, 3, 5]},
            labeling={"blob_threshold": 3},
        ),
    )
    return module, index, results


def test_cli_writes_the_json_the_png_and_the_per_slide_table(cli, capsys):
    module, index, results = cli
    assert module.main(["--index", str(index)]) == 0
    out = results / "label_analysis"
    assert (out / "threshold_ablation.json").is_file()
    assert (out / "threshold_ablation.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    written = json.loads((out / "threshold_ablation.json").read_text(encoding="utf-8"))
    assert written["selected_threshold"] == 3
    assert written["n_patches"] == 2
    assert written["n_slides"] == 2
    assert written["blob_counts"] == {"2_1016_0.png": 3, "3_0_0.png": 3}
    assert sorted(written["per_threshold"]) == ["2", "3", "5"]
    assert written["per_threshold"]["3"]["n_steatosis"] == 2
    assert written["per_threshold"]["3"]["per_slide_ratio"] == {"2": 1.0, "3": 1.0}
    assert written["per_threshold"]["3"]["per_slide_ratio_spread"] == 0.0
    assert written["per_threshold"]["3"]["selected"] is True
    assert written["per_threshold"]["2"]["selected"] is False

    printed = capsys.readouterr().out
    # The count distribution first, because the ratios are ratios of it, then the sweep,
    # then the per-slide table at the selected threshold: the deliverable the plan names.
    assert printed.index("blob count") < printed.index("threshold  n_steatosis")
    assert printed.index("threshold  n_steatosis") < printed.index("slide_id")
    assert "2 patches" in printed
    assert "at threshold 3" in printed
    # The deliverable itself, read back: one row per slide, the two integers behind the
    # ratio, and the scale the counts were made at beside it. Taken from everything after
    # the header, because the sweep table above has a row beginning "2" as well.
    table = [line.split() for line in printed.split("slide_id", 1)[1].splitlines()]
    assert [row[:4] for row in table if row and row[0] in {"2", "3"}] == [
        ["2", "1", "1", "1.000"],
        ["3", "1", "1", "1.000"],
    ]


def test_cli_selected_reaches_the_sweep(cli, capsys):
    module, index, results = cli
    assert module.main(["--index", str(index), "--selected", "2"]) == 0
    written = json.loads(
        (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    )
    assert written["selected_threshold"] == 2
    assert written["per_threshold"]["2"]["selected"] is True
    assert written["per_threshold"]["3"]["selected"] is False
    assert "at threshold 2" in capsys.readouterr().out


def test_cli_json_carries_the_two_rule_measurement(cli):
    from blobcount.ablation import RULE_COMPARISON

    module, index, results = cli
    assert module.main(["--index", str(index)]) == 0
    written = json.loads(
        (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    )
    # The artifact has to say what distribution it is a sweep over, or a reader cannot
    # tell a knee in the material from a percentile of a background column.
    comparison = written["counting_rule_comparison"]
    assert comparison["rule_in_use"] == _RULE_IN_USE
    assert comparison["superseded_rule"] == _SUPERSEDED_RULE
    assert comparison == RULE_COMPARISON
    # Stated as the ratio of the two totals rather than as a rounded phrase, because
    # 2166 / 377 is 5.75 and a reader who divides the two totals above must not find a
    # different number in the file.
    assert comparison["total_blob_ratio"] == pytest.approx(2166 / 377)


def test_cli_json_round_trips_a_non_ascii_slide_id(
    project_scratch, synthetic_patch, monkeypatch
):
    module = _load_cli(monkeypatch)
    slide = "hasta-ğü-01"
    index = _write_index(
        project_scratch / "patches", [_row(f"{slide}_0_0.png", slide)], synthetic_patch.array
    )
    results = project_scratch / "results"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            results.parent,
            "config.yaml",
            results=results,
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )
    assert module.main(["--index", str(index)]) == 0
    raw = (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    # Written as data and read back as the same string on a Turkish-locale machine.
    assert slide in raw
    written = json.loads(raw)
    assert written["per_slide_gray_threshold"] == {slide: 180.0}
    assert list(written["per_threshold"]["2"]["per_slide_ratio"]) == [slide]


def test_cli_refuses_a_missing_index_with_the_refusal_code(cli, tmp_path, capsys):
    # 1 is the study-level failure code, so a mistyped `--index` exiting 1 would tell a
    # caller reading the code that the sweep contradicted itself. Nothing was written:
    # the counts are made before the results directory is created.
    module, _, results = cli
    assert module.main(["--index", str(tmp_path / "nope.csv")]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "nope.csv" in err
    assert "Traceback" not in err
    assert not (results / "label_analysis").exists()


def test_cli_refuses_a_selected_threshold_outside_the_sweep(cli, monkeypatch, capsys):
    from blobcount import ablation

    module, index, results = cli

    def _boom(path, size):
        raise AssertionError("a patch was read before --selected was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index), "--selected", "9"]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "9" in err and "2, 3, 5" in err
    assert not (results / "label_analysis").exists()


def test_cli_refuses_a_results_path_outside_the_project(cli, monkeypatch, tmp_path, capsys):
    from blobcount import ablation

    module, index, _ = cli
    outside = tmp_path / "elsewhere"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            tmp_path,
            "outside.yaml",
            results=outside,
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the output target was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    assert str(outside) in capsys.readouterr().err
    assert not outside.exists()


def test_cli_refuses_a_results_path_config_path_refuses(cli, monkeypatch, tmp_path, capsys):
    # `load` validates that the key is declared and nothing about its value, so a
    # `paths.results` that is a number reaches `Config.path` and is refused there. It is
    # caught on the configuration arm rather than left to a traceback, which is the same
    # shape the labeling CLI's `paths.labels` refusal took.
    from blobcount import ablation

    module, index, _ = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            tmp_path,
            "number.yaml",
            results=7,
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the output target was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    assert "path string" in capsys.readouterr().err


def test_cli_refuses_a_results_path_that_is_an_existing_file(cli, monkeypatch, capsys):
    from blobcount import ablation

    module, index, results = cli
    occupied = results.parent / "occupied"
    occupied.write_text("a previous run's figure, or something else", encoding="utf-8")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            occupied.parent,
            "file.yaml",
            results=occupied,
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the output target was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    assert occupied.is_file()
    assert "occupied" in capsys.readouterr().err


def test_cli_does_not_swallow_a_defect_from_inside_ablate(cli, monkeypatch):
    from blobcount import ablation

    module, index, _ = cli

    def _boom(patch, params, mpp):
        raise ValueError("injected")

    monkeypatch.setattr(ablation, "count_blobs", _boom)
    with pytest.raises(ValueError, match="injected"):
        module.main(["--index", str(index)])


def test_cli_refuses_a_configuration_it_cannot_load(cli, tmp_path, monkeypatch, capsys):
    module, index, _ = cli
    broken = tmp_path / "broken.yaml"
    broken.write_text("project:\n  name: x\n", encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    assert "broken.yaml" in capsys.readouterr().err
