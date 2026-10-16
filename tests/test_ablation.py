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

# The criterion `range_coverage` states in its own artifact, as a pin. It names the two
# comparisons and the statistic each end is held against, and it is written so that
# `_criterion_holds` can evaluate it over the numbers in the same block: the test that
# checks the string and the boolean agree is the check that would have caught a predicate
# comparing the low end against the median.
_CRITERION = (
    "brackets_distribution is true iff swept_min <= count_min and swept_max >= count_max"
)


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
    # Counts that straddle 3 and 5, so the two pooled ratios are different numbers rather
    # than the same one twice. `_balanced_pair` is 20 and 0, and a patch at 20 clears every
    # threshold this sweeps, so an equality between the ratios at 3 and 5 was true for a
    # structural reason and asserted nothing at all.
    counts = {
        "A0": 0, "A1": 0, "A2": 4, "A3": 4, "A4": 6,
        "B0": 0, "B1": 0, "B2": 0, "B3": 4, "B4": 6,
    }
    slide_of = {patch: patch[0] for patch in counts}
    s = summarise(sweep(counts, slide_of, [3, 5, 7]), 5)
    assert [t for t in (3, 5, 7) if s[t]["selected"]] == [5]
    # Three clears five of the ten, five clears two of them, seven none.
    assert s[3]["pooled_steatosis_ratio"] == pytest.approx(0.5)
    assert s[5]["pooled_steatosis_ratio"] == pytest.approx(0.2)
    assert s[7]["pooled_steatosis_ratio"] == pytest.approx(0.0)
    assert s[3]["pooled_steatosis_ratio"] != s[5]["pooled_steatosis_ratio"]
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


def test_plot_writes_a_two_panel_figure(tmp_path, monkeypatch):
    # A **pin**, strengthened: it was green before this change, and it is strengthened
    # because a PNG magic number is satisfied by a figure of any shape, while the
    # two-panel layout — the per-slide curves beside the per-slide bars at the selected
    # threshold — is what the plan asked for and nothing else in the file pinned it.
    # `plot` imports `pyplot` inside the function and looks `subplots` up on the module
    # at call time, so replacing the one attribute here records the call without changing
    # what is drawn.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from blobcount.ablation import plot, summarise, sweep

    calls = []
    real = plt.subplots

    def _record(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(plt, "subplots", _record)
    counts, slide_of = _balanced_pair()
    s = summarise(sweep(counts, slide_of, [3, 5, 7]), 5)
    out = tmp_path / "ablation.png"
    plot(s, out)
    assert calls[0][0] == (1, 2)
    assert calls[0][1]["figsize"] == (11.0, 4.5)
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_refuses_a_summary_with_no_selected_threshold(tmp_path):
    from blobcount.ablation import plot, summarise, sweep
    counts, slide_of = _balanced_pair()
    unmarked = summarise(sweep(counts, slide_of, [3, 5, 7]))
    with pytest.raises(ValueError, match="selected"):
        plot(unmarked, tmp_path / "ablation.png")
    assert not (tmp_path / "ablation.png").exists()


def test_plot_refuses_a_threshold_too_large_to_be_a_float(tmp_path):
    # **Red before the fix**, and the exception it raised was not this one: `plot` handed
    # the thresholds straight to matplotlib, which converts a data coordinate to a float in
    # `Axes.add_line` -> `_update_line_limits`, and `float(10 ** 320)` raises
    # `OverflowError: int too large to convert to float` there. CPython's sentence, out of a
    # library, naming no rule — and `_thresholds` above had already recorded that a value
    # this size is a *legitimate* whole number of blobs, so the sweep over it is exact and
    # total and every patch is `normal` at that row. The counting can hold the value; only
    # the drawing cannot, and the function that cannot hold it is the one that refuses.
    #
    # **`PlotError` and not a `ValueError`, and that is the whole reason the escape ends.**
    # `plot`'s two other refusals are `ValueError`s and they are *its* refusals, but a
    # `ValueError` raised anywhere inside matplotlib would be indistinguishable from one at
    # the command line, which catches a refusal around this call. A project-raised type makes
    # the arm as narrow as the defect: this call raises it, nothing else in the figure does.
    # The same argument `bin/ablate_threshold.py` gives for keeping `OverflowError` out of
    # the catch around `_thresholds`, read from the other side — a defect in this project
    # must still print a traceback, and only a defect in this project can raise a `ValueError`
    # out of `plt.subplots`.
    #
    # The `__cause__` is pinned because the refusal has to be *this* limit and not a limit
    # somebody wrote down. The predicate is `float(threshold)` itself, so the boundary is
    # whatever the conversion matplotlib performs accepts, and a `sys.float_info.max`
    # comparison would be a second definition of a number that could drift away from it.
    from blobcount.ablation import PlotError, plot, summarise, sweep

    counts, slide_of = _balanced_pair()
    huge = 10 ** 320
    out = tmp_path / "ablation.png"
    with pytest.raises(PlotError) as refused:
        plot(summarise(sweep(counts, slide_of, [3, huge]), 3), out)
    assert isinstance(refused.value.__cause__, OverflowError)
    message = str(refused.value)
    # The value as the user wrote it, and the rule it broke, because a refusal that does not
    # name either is the CPython sentence this replaces.
    assert str(huge) in message
    assert "float" in message
    # Nothing was drawn, and the check is before the first `plt.subplots`, so the refusal
    # costs no open figure either.
    assert not out.exists()


def test_plot_still_draws_a_threshold_that_is_large_but_is_a_float(tmp_path):
    # The other side of the boundary, and it is here because a predicate that reads as
    # "refuse a big number" would pass the test above without being the rule at all.
    # `2 ** 60` is over 1e18 and converts to a float without a murmur, so a sweep whose
    # counts are anywhere near it is drawn like any other.
    from blobcount.ablation import plot, summarise, sweep

    counts, slide_of = _balanced_pair()
    out = tmp_path / "ablation.png"
    plot(summarise(sweep(counts, slide_of, [3, 2**60]), 3), out)
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


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
    project_scratch, synthetic_patch, monkeypatch
):
    # The knee question is a question about this distribution, so the run reports its
    # shape. A mean of 36.1 blobs per patch against a swept threshold of 30 is the state
    # the ablation has to show rather than one it may hide behind a ratio, and the
    # summary is over the counts themselves rather than over a ratio of them.
    #
    # A **pin**, strengthened: it was green before this change, and it is strengthened
    # because four identical counts make `min`, `max`, `p50` and `mean` the same number,
    # so any one of the four could have been the wrong field and the test would not have
    # known. The counts are stubbed apart for that reason; the real `count_blobs` is
    # pinned by `test_ablate_recomputes_the_counts_and_ignores_the_labels`, which asserts
    # three discs counted as three.
    from blobcount import ablation

    counts = iter([0, 3, 20, 86])
    monkeypatch.setattr(
        ablation, "count_blobs", lambda patch, params, mpp: next(counts)
    )
    rows = [_row(f"A_{index * 256}_0.png", "A", x=index * 256) for index in range(4)]
    index = _write_index(project_scratch / "patches", rows, synthetic_patch.array)
    summary = ablation.ablate(index, load()).count_summary()
    # p50 of 0, 3, 20 and 86 is the mean of the middle two, and the mean of the four is
    # 109 / 4: neither is any of the other three numbers.
    assert summary == {"count": 4, "min": 0, "p50": 11.5, "mean": 27.25, "max": 86}
    # A dict comparison cannot see `0` against `0.0`, so the types are stated: a count of
    # blobs is a whole number and a mean of counts is not.
    assert isinstance(summary["min"], int) and isinstance(summary["max"], int)
    assert isinstance(summary["mean"], float) and isinstance(summary["p50"], float)


def _straddling_index(project_scratch, synthetic_patch):
    """Return an index over four patches, two slides, counts to be stubbed in.

    The distribution is deliberately lopsided once the counts are stubbed — 0, 2, 10 and
    20, so the median is 6.0, the mean is 8.0, the minimum is 0 and the maximum is 20 —
    which lets a swept range stop short of it at either end and tells the two failures
    apart. **The minimum is 0, so a range covering this distribution has to start at or
    below 0**; the two tests below that sweep it at 1 do not, and that is the point of
    them rather than a slip in their fixture.
    """
    return _write_index(
        project_scratch / "patches",
        [
            _row("A_0_0.png", "A"),
            _row("A_256_0.png", "A", x=256),
            _row("B_0_0.png", "B"),
            _row("B_256_0.png", "B", x=256),
        ],
        synthetic_patch.array,
    )


def _stub_counts(monkeypatch, values):
    """Make `count_blobs` hand out `values` in call order, for one `ablate` run.

    A fresh closure per call rather than one iterator for the test, because two `ablate`
    runs over the same index each need all four counts and a shared iterator would be
    empty for the second.
    """
    from blobcount import ablation

    remaining = list(values)
    monkeypatch.setattr(
        ablation, "count_blobs", lambda patch, params, mpp: remaining.pop(0)
    )


def _criterion_holds(coverage):
    """Return what `coverage["criterion"]` says, evaluated over the block's own numbers.

    The criterion the artifact carries is *evaluated* here rather than restated in the
    test. A test that re-typed it would be a second definition of the predicate, and a
    second definition is what `brackets_distribution` was in round 1: the comparison lived
    in a docstring, where a reader of the JSON could not check it and a test that did would
    only have checked itself. `eval` runs over a namespace holding this block's own scalars
    and no builtins, so the expression can only compare what the artifact reports, and the
    string and the boolean either agree or this fails.

    **What this checks is the artifact against itself, and it is not the only check on the
    block — the one that is not this one is the stronger.** The string and the boolean come
    from the same run over the same numbers, so they can agree while both are wrong: a run
    that held the low end against `count_p50` and wrote a criterion string naming
    `count_p50` is perfectly self-consistent and still hands a reader of the JSON the wrong
    answer. **What this function does catch is a block that disagrees with itself**, and
    the example is a `count_min` key holding the median: the string names `count_min` and
    the boolean is computed from the counts themselves, so on a distribution whose median
    clears the swept minimum the two come to different answers and this fails. Only a
    *self-consistent* move needs the caller's independent reading, and the caller of this
    function supplies one — it re-types the comparison over the raw per-patch counts it
    stubbed into `count_blobs`, which the block does not report and cannot be derived
    from. So the rule against re-typing is a rule about the *criterion string*, which this
    function exists to keep honest, and not about the *predicate*, where a second
    independent reading is the only thing that checks anything.
    """
    assert " iff " in coverage["criterion"], (
        f"the criterion has to name its comparison in the form this test evaluates, and "
        f"it reads {coverage['criterion']!r}"
    )
    values = {name: value for name, value in coverage.items() if name != "criterion"}
    return bool(eval(coverage["criterion"].split(" iff ")[-1], {"__builtins__": {}}, values))


def test_range_coverage_compares_the_swept_range_against_the_distribution(
    project_scratch, synthetic_patch, monkeypatch
):
    # **Red before the fix**: `range_coverage` did not exist, so a reader of
    # `blob_count_summary` and `per_threshold` saw the distribution and the sweep and no
    # statement about whether the sweep reached the distribution. The module docstring
    # claimed the run says so and the run did not. **Red again after round 1**, for the
    # second half of the same defect: the range swept here now starts at 0 and not at 1,
    # because the distribution's minimum is 0 and a range starting at 1 leaves that patch
    # below the lowest threshold. Round 1's predicate called that covered, because it
    # compared the low end against the median.
    from blobcount import ablation

    _stub_counts(monkeypatch, [0, 2, 10, 20])
    stats = ablation.ablate(
        _straddling_index(project_scratch, synthetic_patch),
        load(overrides=["ablation.thresholds=[0, 30]"]),
        selected=0,
    )
    assert stats.range_coverage() == {
        "criterion": _CRITERION,
        "swept_min": 0,
        "swept_max": 30,
        "count_min": 0,
        "count_p50": 6.0,
        "count_mean": 8.0,
        "count_max": 20,
        "unswept_below": 0,
        "unswept_above": 0,
        "n_unswept": 0,
        "brackets_distribution": True,
    }
    # The flag is a boolean and not an `int`, because `brackets_distribution` is the whole
    # of what a reader and a test need from it and a `1` reads as a count of something.
    assert isinstance(stats.range_coverage()["brackets_distribution"], bool)


def test_range_coverage_counts_the_patches_the_swept_range_never_cuts(
    project_scratch, synthetic_patch, monkeypatch
):
    # **Red before the fix**, in the direction the reviewer's case takes. Round 1 compared
    # the low end of the swept range against the distribution's *median*, so a range whose
    # bottom sits above the minimum but below the median reported that the sweep covered
    # the distribution. Counts of 1, 50, 51 and 52 swept at [40, 60] gave
    # `brackets_distribution: true` with 1 of the 4 patches below the lowest threshold and
    # therefore never cut at any threshold in the sweep — the exact misreading R7-1 exists
    # to prevent, produced by the fix for R7-1.
    #
    # The second half is the same distribution swept at a range that does cover it, so
    # this test cannot be satisfied by a predicate that answers False to everything. A
    # flag that is always False is not a fix either.
    from blobcount import ablation

    index = _straddling_index(project_scratch, synthetic_patch)
    _stub_counts(monkeypatch, [1, 50, 51, 52])
    short_below = ablation.ablate(
        index, load(overrides=["ablation.thresholds=[40, 60]"]), selected=40
    ).range_coverage()
    assert short_below == {
        "criterion": _CRITERION,
        "swept_min": 40,
        "swept_max": 60,
        "count_min": 1,
        "count_p50": 50.5,
        "count_mean": 38.5,
        "count_max": 52,
        # The one patch at 1, and nothing at the top: the range's top of 60 does clear the
        # maximum of 52, so this failure is at the low end alone.
        "unswept_below": 1,
        "unswept_above": 0,
        "n_unswept": 1,
        "brackets_distribution": False,
    }
    # The count, not the verdict, is what a reader cannot get wrong: a quarter of this
    # cohort was never swept, and the artifact says so in a number.
    assert short_below["n_unswept"] == 1 and short_below["unswept_below"] == 1
    _stub_counts(monkeypatch, [1, 50, 51, 52])
    covering = ablation.ablate(
        index, load(overrides=["ablation.thresholds=[1, 60]"]), selected=1
    ).range_coverage()
    assert covering["count_min"] == 1 and covering["count_max"] == 52
    assert covering["unswept_below"] == 0 and covering["unswept_above"] == 0
    assert covering["n_unswept"] == 0
    assert covering["brackets_distribution"] is True


def test_the_range_coverage_criterion_string_and_its_boolean_agree(
    project_scratch, synthetic_patch, monkeypatch
):
    # **Red before the fix**: the criterion was in a docstring and nowhere else, so a
    # reader of `threshold_range_coverage` saw `brackets_distribution: true` with no key
    # naming what true means, and two readers could derive different answers from the same
    # file. One of them was round 1's, and it was wrong. The agreement is checked by
    # evaluating the artifact's own criterion string, so a predicate that moves without
    # the string moving fails here rather than being taken on trust.
    from blobcount import ablation

    index = _straddling_index(project_scratch, synthetic_patch)
    cases = [
        # The reviewer's case: the bottom is above the minimum and below the median, which
        # is the state round 1 read as covered.
        ([1, 50, 51, 52], [40, 60]),
        # The same distribution with a range that does cover it.
        ([1, 50, 51, 52], [1, 60]),
        # The bottom above the minimum *and* above the median.
        ([1, 50, 51, 52], [52, 60]),
        ([0, 2, 10, 20], [1, 15]),
        ([0, 2, 10, 20], [10, 30]),
        # Covering exactly, at both ends.
        ([0, 2, 10, 20], [0, 20]),
    ]
    for counts, thresholds in cases:
        _stub_counts(monkeypatch, counts)
        coverage = ablation.ablate(
            index,
            load(overrides=[f"ablation.thresholds={thresholds}"]),
            selected=thresholds[0],
        ).range_coverage()
        assert _criterion_holds(coverage) is coverage["brackets_distribution"], (
            f"counts {counts} swept at {thresholds}: the criterion string says "
            f"{_criterion_holds(coverage)} and the flag says "
            f"{coverage['brackets_distribution']}"
        )
        # And the boolean is a statement about the counts rather than about the range
        # alone: it is true exactly when every count lies inside the swept range, which is
        # the definition a reader of the artifact would apply to it.
        #
        # **Re-typing here is the point, and this catches what `_criterion_holds` cannot.**
        # That function checks the criterion *string* against the boolean, so what it
        # catches is a block that disagrees with itself — a `count_min` key holding the
        # median is that, because the string names `count_min` and the boolean is
        # computed from the counts. This checks the *boolean* against `counts`, the
        # values `_stub_counts` handed to `count_blobs` and the block does not carry, so
        # what it catches is a run that is self-consistently wrong: a predicate that read
        # `count_p50` and a string that named it together, which agree with each other
        # and are both wrong. Two checks that fail on different bugs is the reason both
        # are here rather than one. **What this does not read is `count_min`:** it
        # compares the swept range against the raw counts, so a `count_min` holding the
        # wrong statistic is not what it looks for, and the check above is.
        assert coverage["brackets_distribution"] is all(
            coverage["swept_min"] <= value <= coverage["swept_max"] for value in counts
        ), f"counts {counts} swept at {thresholds} are not all inside the swept range"


def test_range_coverage_is_false_at_either_end_of_a_range_that_stops_short(
    project_scratch, synthetic_patch, monkeypatch
):
    # **Red before the fix**, for the same reason as the pair above. Both directions are
    # here because they are different failures: a range whose top stops below the largest
    # count never sees the tail, and a range whose bottom sits above the distribution's
    # smallest count never enters the bulk — the *minimum*, which is what round 1
    # compared against the median instead. The controller's decision is that neither is a
    # reason to widen the configured range from a 60-patch sample, so both are reported
    # and neither is fixed.
    from blobcount import ablation

    index = _straddling_index(project_scratch, synthetic_patch)
    _stub_counts(monkeypatch, [0, 2, 10, 20])
    short_above = ablation.ablate(
        index, load(overrides=["ablation.thresholds=[0, 15]"]), selected=0
    ).range_coverage()
    assert short_above == {
        "criterion": _CRITERION,
        "swept_min": 0,
        "swept_max": 15,
        "count_min": 0,
        "count_p50": 6.0,
        "count_mean": 8.0,
        "count_max": 20,
        # The one patch at 20, and none below: the range's bottom of 0 does reach the
        # distribution's minimum of 0, so this failure is at the top alone.
        "unswept_above": 1,
        "unswept_below": 0,
        "n_unswept": 1,
        "brackets_distribution": False,
    }
    _stub_counts(monkeypatch, [0, 2, 10, 20])
    short_below = ablation.ablate(
        index, load(overrides=["ablation.thresholds=[10, 30]"]), selected=10
    ).range_coverage()
    assert short_below == {
        "criterion": _CRITERION,
        "swept_min": 10,
        "swept_max": 30,
        "count_min": 0,
        "count_p50": 6.0,
        "count_mean": 8.0,
        "count_max": 20,
        # The two patches at 0 and at 2, and none above: 30 clears the maximum of 20.
        "unswept_below": 2,
        "unswept_above": 0,
        "n_unswept": 2,
        "brackets_distribution": False,
    }
    # The reported range is the keys `sweep` was given, not the configured list read a
    # second time: the configured `labeling.blob_threshold` here is 5 and neither range
    # contains it, and both runs still report the range they were actually swept at.
    assert short_below["swept_min"] == 10 and short_above["swept_max"] == 15


def test_ablate_logs_the_index_it_could_not_read(project_scratch, caplog):
    # **Red before the fix**: the index read was caught and re-raised with no `LOGGER`
    # call, and the module docstring claimed every caught exception was logged. An index
    # that cannot be read is the case where a log line matters most, and there are no
    # coordinates to name because no row was read — so the specimen here is the file.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    missing = project_scratch / "patches" / "patches.csv"
    with caplog.at_level(logging.ERROR, logger="blobcount.ablation"):
        with pytest.raises(LabelingInputError):
            ablation.ablate(missing, load())
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert f"reading the patch index {missing} raised" in logged
    assert "patches.csv" in logged


def test_ablate_logs_the_row_whose_cell_is_not_a_number(project_scratch, caplog):
    # **Red before the fix**: the conversion behind `_row_numbers` was caught and re-raised
    # with no `LOGGER` call, for the same reason. This is the one catch that has the row,
    # so all three of the specimen, the coordinates and the exception object are there to
    # be named.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    directory = project_scratch / "patches"
    directory.mkdir(parents=True, exist_ok=True)
    bad_cell = directory / "bad-cell.csv"
    bad_cell.write_text(
        "patch_id,slide_id,x,y,mpp,tissue_gray\n"
        "A_0_0.png,A,512,256,0.5010,not-a-number\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.ERROR, logger="blobcount.labeling"):
        with pytest.raises(LabelingInputError, match="not-a-number"):
            ablation.ablate(bad_cell, load())
    logged = "\n".join(record.getMessage() for record in caplog.records)
    assert "slide A: reading the cell for patch A_0_0.png at (512, 256) raised" in logged
    assert "not-a-number" in logged


def test_ablate_refuses_an_index_with_no_rows(project_scratch):
    # **Red before the fix**: the empty index fell through to `sweep`, which raises a
    # `ValueError` over an empty count map. `ablate` did not wrap it, and
    # `bin/ablate_threshold.py` catches only `LabelingInputError`, so the run printed a
    # traceback and exited 1 — the status this CLI's own contract defines as a fact about
    # this project's code. The message named an empty index, which is a fact about the
    # study's data. An index with a header and no rows is a legitimate state, and it gets
    # the same answer a missing file already gets.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    index = _write_index(project_scratch / "patches", [])
    with pytest.raises(LabelingInputError, match="no rows") as refused:
        ablation.ablate(index, load())
    # Refused before the first cell was converted, so a bad configuration value is
    # reported as the index rather than the other way round.
    assert str(index) in str(refused.value)


def test_ablate_raises_a_value_error_for_a_range_that_is_not_whole_numbers(
    project_scratch, synthetic_patch
):
    # A **pin**, and the other half of the exit-code contract. The command line reports
    # this refusal as `EXIT_REFUSED`; the library boundary keeps the `ValueError`, because
    # it is a statement about the configuration and not about the index, and a
    # `LabelingInputError` would make that type mean two things. It is not a defect either
    # way: `_thresholds` was written to reject this value and it does.
    from blobcount import ablation
    from blobcount.labeling import LabelingInputError

    index = _write_index(
        project_scratch / "patches", [_row("A_0_0.png", "A")], synthetic_patch.array
    )
    with pytest.raises(ValueError, match="5.5") as escaped:
        ablation.ablate(
            index, load(overrides=["ablation.thresholds=[2, 5.5]"]), selected=2
        )
    assert not isinstance(escaped.value, LabelingInputError)


@pytest.mark.parametrize(
    ("literal", "as_repr"),
    [
        # A quoted digit, which is a string and not a number. Round 3's first refusal.
        ("'3'", "'3'"),
        # A YAML flag. `bool` subclasses `int`, so `int(True) == 1` and `1 != True` is
        # False: a `true` in the configured range passed the whole-number test and was
        # swept as the threshold 1, with no message anywhere. Round 3's second refusal.
        ("true", "True"),
        # The three non-finite floats. `isinstance(inf, float)` is True, so the type test
        # admitted all three and `int()` refused them from inside the standard library
        # instead, in two different ways: `int(float("nan"))` raises `ValueError` and
        # `int(float("inf"))` raises `OverflowError`, and only the first is in the
        # `(TypeError, ValueError)` tuple `bin/ablate_threshold.py` catches. So a
        # hand-edited `.inf` left a traceback and exited 1 — the status that CLI documents
        # as a defect in this project's own code. The `.nan` row was already refused, but
        # by that accident: what reached the user was CPython's `cannot convert float NaN
        # to integer`, which names no rule.
        (".inf", "inf"),
        ("-.inf", "-inf"),
        (".nan", "nan"),
        # A fraction, which the type test and the finiteness test both admit and only
        # `int(value) == value` refuses. The row is here because that third condition is
        # one of the three the message has to name, and until this round it named two.
        ("5.5", "5.5"),
        # **Red before the fix**: a negative threshold is a plain number, it is finite and
        # it is a whole number, so nothing above refused it. No blob count is below a
        # negative threshold, so every patch came out `normal` at it, the swept row
        # reported a ratio of 1.0, and the figure drew a curve flat at zero for a
        # threshold no count sits on. A threshold is a cut point on a count of blobs and
        # no count is below zero, so a negative one cuts at nothing and then reports an
        # answer as though it had cut at something.
        ("-5", "-5"),
    ],
)
def test_thresholds_refuses_every_shape_that_is_not_a_threshold(literal, as_repr):
    # **One parametrised test on purpose, so that there is one list of what a threshold
    # is.** These were three tests, each with its own message assertion, and a fourth
    # shape refused today would have been a fourth message to write; one list is a thing
    # a reader can hold. `literal` is the YAML a hand-edit produces and `as_repr` is what
    # this module prints for the same value, so a reader can match the two.
    #
    # **The message is the finding, not the exception.** Every one of these values was
    # already refused, or is refused by this round's predicate, and what a user
    # hand-editing `configs/default.yaml` gets to read is a sentence naming the rule it
    # broke. A refusal that names no rule is a crash the user has to interpret, which is
    # what round 1's `int()` was for `.nan`, and what a CPython sentence about a
    # `cannot convert float NaN to integer` was for the row that got refused by accident.
    from blobcount import ablation

    with pytest.raises(ValueError) as refused:
        ablation._thresholds(load(overrides=[f"ablation.thresholds=[{literal}]"]))
    message = str(refused.value)
    # The offending entry, as this module names it, so the reader knows which one to fix.
    assert as_repr in message
    assert "whole number of blobs" in message
    # **And the whole rule, in one list, checked on every row.** Four conditions are
    # applied to an entry — a plain number, finite, exactly a count, and not below zero —
    # and the message names all four, with the shape that breaks each. Naming two and
    # applying four is the defect this row set exists to end, and a message that named
    # the list only on the rows that need it would be the same defect again.
    for phrase in (
        "a plain number that is finite",
        "exactly a count",
        "not below zero",
        "not '3'",
        "not true",
        ".inf",
        ".nan",
        "5.5",
        "-5",
    ):
        assert phrase in message, f"{phrase!r} is not named in the refusal {message!r}"


def test_thresholds_still_accepts_a_whole_number_written_as_a_float():
    # The other side of the same predicate, and the reason the finiteness test is not a
    # refusal of every float. `3.0` is a whole number of blobs and YAML writes it for a
    # user who computed the range; refusing it would be the same class of mistake as
    # refusing a quoted digit, and `isfinite` rather than `is_integer` is what keeps `3.0`
    # on the accepted side. This one is green before the fix and stays green after it,
    # which is the point: it is here to catch a fix that went one predicate too far.
    from blobcount import ablation

    assert ablation._thresholds(
        load(overrides=["ablation.thresholds=[3.0, 5.0]"])
    ) == (3, 5)


def test_thresholds_accepts_a_count_too_large_to_be_a_float():
    # **The threshold a future reader will want to refuse, and must not.** The finiteness
    # test in `_thresholds` reads `not isinstance(value, float) or math.isfinite(value)`,
    # and the scope to floats is load-bearing in one direction and invisible: `math.isfinite`
    # converts its argument to a float, so the unscoped form raises
    # `OverflowError: int too large to convert to float` on the value below. That is the
    # round 3 escape class back again, one layer down, and nothing in the file could see
    # it: the reviewer rewrote the line unscoped and all 53 tests in this file stayed
    # green while the bug returned. So this test pins the behaviour and not the code, and
    # rewriting the line unscoped now turns this row red and leaves the other 57 in this
    # file green — measured, not asserted.
    #
    # **Why accepting it is right, since it looks like the opposite.** A threshold is a
    # cut point on a count of blobs, and this value is a whole number of blobs however
    # large it is. Python compares two integers exactly at any size, `sweep` only asks
    # `count >= threshold`, and this run labels every patch `normal` at this threshold: a
    # real row in `per_threshold` with a ratio of 0.0, not a failure and not a refusal.
    # What cannot hold a value this size is a *figure*: `plot` puts the thresholds on a
    # float x-axis, and a value too large for a float raises `OverflowError` there, out of
    # matplotlib's transform rather than out of a rule. That is a gap in the drawing
    # path, and it is a different argument from the one this validator answers — refusing
    # a count here to accommodate a plotting library would be this function deciding a
    # question that is not its own, on behalf of a caller that asked for a sweep.
    #
    # So the shape is accepted deliberately, and this test is here so that "deliberately"
    # is a fact the suite carries rather than a comment it has to be trusted about:
    # unscoping the finiteness test turns this row red with that `OverflowError`.
    #
    # **The value is written out in full because that is what YAML has to be given.**
    # `1e320` in a configuration file is the *string* `'1e320'`, so the value falls through
    # to the string tag and is refused above as a quoted digit. **The reason it is a string
    # is the exponent's missing sign, not a missing `.`:** PyYAML 6.0.3's float pattern is
    # `[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?|...`, and the exponent clause
    # requires an explicit `+` or `-`, so `1.0e320` — which does have a `.` — is *also* the
    # string `'1.0e320'`, as is `1.0e3`. Measured, not recalled. The spelling that does
    # resolve is `1.0e+320`, and that is a `float` holding `inf`, which the predicate refuses
    # as a non-finite whole number rather than as a quoted digit — a different refusal with
    # a different message, reached by a value no user writes on purpose. Neither spelling
    # gives an `int` of this magnitude, so the 321-digit integer is the only YAML that
    # reaches the case this test is about, and `10 ** 320` is it without a line of digits in
    # the source.
    from blobcount import ablation

    huge = 10 ** 320
    assert ablation._thresholds(
        load(overrides=[f"ablation.thresholds=[3, {huge}]"])
    ) == (3, huge)
    # And that the sweep is total over it, which is the other half of why a float
    # conversion is not this validator's business: a comparison of two ints cannot
    # overflow, so the row below is a row and not an exception.
    swept = ablation.sweep({"A_0": 4, "A_1": 9}, {"A_0": "A", "A_1": "A"}, (huge,))
    assert swept[huge]["n_normal"] == 2
    assert swept[huge]["n_steatosis"] == 0
    assert swept[huge]["pooled_steatosis_ratio"] == 0.0


class _UntypedThresholds:
    """A stand-in for the one method `_thresholds` reads.

    **This class exists because `Config.__init__` type-checks as of 2026-10-16**, so a
    `Config` can no longer carry a value of the wrong type for `ablation.thresholds` and
    there is nothing left to hand `_thresholds` that `load` refuses first. Before that,
    `Config({"ablation": {"thresholds": b"35"}})` was the way these tests reached the
    function's own check, and they read that route's history in
    `tests/test_config.py::test_a_thresholds_key_that_is_not_a_list_is_refused`.

    **What is given up, stated rather than hidden:** these tests no longer prove that a
    `Config` carrying a mistyped key is caught here, because no such `Config` can exist.
    What they still prove — and what is the reason the check stays in the function rather
    than being deleted along with its route — is that `_thresholds` answers the wrong key
    with a refusal of its own, for a caller that is not a `Config` and so was never inside
    the constructor's reach.
    """

    def __init__(self, value):
        self._value = value

    def get(self, key, default=None):
        if key != "ablation.thresholds":
            raise AssertionError(f"_thresholds read an unexpected key: {key!r}")
        return self._value


@pytest.mark.parametrize(
    ("value", "as_repr"),
    [
        # `!!binary 'MzU='`, which a hand-edit can write and which iterates as the two
        # integers 51 and 53 — a sweep over a range nobody configured, reported as a
        # range like any other.
        (b"35", "b'35'"),
        # `!!set {3, 5}`, which iterates in an order YAML did not choose and which is not
        # a range at all.
        ({3, 5}, "{3, 5}"),
    ],
)
def test_thresholds_refuses_a_key_that_is_not_a_sequence_of_thresholds(value, as_repr):
    # **Red before the fix**, on both rows, and inconsistently with itself: the loop was
    # `for value in cfg.get("ablation.thresholds")`, so anything iterable was swept, and
    # the same bytes value was refused as an *entry* by the type test two lines below.
    # Bytes and a set are not a range, and a range read out of a string or a set is a
    # threshold nobody configured wearing the costume of one.
    #
    # The check is a `Sequence` test and not a YAML type policy: this project asks for no
    # general policy, `load` already refuses a file whose top level is not a mapping, and
    # the shape that matters is the one a reader can write by accident. `str`, `bytes` and
    # `bytearray` are excluded by name because they *are* sequences — of characters, and
    # of each other — so a `Sequence` test alone would sweep `'35'` as 51 and 53 and
    # `b"35"` the same way.
    #
    # **A `TypeError`, and not a `ValueError`,** because this is the key and not an entry,
    # and the two are different answers: the command line's catch is `(TypeError,
    # ValueError)` around this one call and its own docstring already attributes the
    # `TypeError` arm to "a key that is not a sequence" — which was, until this round, a
    # statement about an accident and not about a check.
    #
    # **Delivered through `_UntypedThresholds`, and why that is the only remaining
    # route.** `ablation.thresholds` is declared `list`, so `load` refuses both of these
    # as a `ConfigError` naming the key before `_thresholds` is reached, and since
    # 2026-10-16 `Config.__init__` refuses them too — so a `Config` cannot carry the
    # value at all. A reader wanting to know what a hand-edit does today reads
    # `tests/test_config.py::test_a_thresholds_key_that_is_not_a_list_is_refused`, which
    # is the earlier and better refusal.
    from blobcount import ablation

    with pytest.raises(TypeError) as refused:
        ablation._thresholds(_UntypedThresholds(value))
    message = str(refused.value)
    # The value as it reads, and its type, because the YAML tag the user wrote is not what
    # they see back and `b'35'` names the shape where `!!binary 'MzU='` does not.
    assert as_repr in message
    assert type(value).__name__ in message
    assert "not a sequence of thresholds" in message


def test_thresholds_refuses_a_buffer_view_the_exclusion_tuple_missed():
    # **Red before the fix**, and it is the same failure R10-4 was raised about, one class
    # over: that fix excluded `(str, bytes, bytearray)` by name and left `memoryview` out,
    # which is a `Sequence` by registration and sweeps `memoryview(b"35")` as the two
    # thresholds 51 and 53. So the tuple named three of the four classes it was standing in
    # for, and a reader of it would conclude the shape was covered.
    #
    # **Unreachable from a YAML file, and from a `Config` since 2026-10-16; that is stated
    # rather than used as the reason to skip it.** PyYAML emits `bytes` for `!!binary` and
    # refuses `!!python` tags, and `yaml.safe_dump` cannot represent a memoryview at all.
    # The check stays because the predicate means what it says rather than because the
    # shape is a live hole on one particular route.
    #
    # The fix is the predicate, not a list of every buffer type in the standard library:
    # the test is a `Sequence` test, and the exclusions are the sequence types that carry
    # *elements* rather than *thresholds*. `memoryview`, `bytearray` and `bytes` are all
    # that, and a reader of `blobcount.sequences` can now see the whole set.
    from blobcount.ablation import _thresholds

    with pytest.raises(TypeError) as refused:
        _thresholds(_UntypedThresholds(memoryview(b"35")))
    message = str(refused.value)
    assert "not a sequence of thresholds" in message
    assert "memoryview" in message


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


def _directory_link(target: Path, link: Path) -> bool:
    """Make `link` a directory link to `target`; report whether this platform allowed it.

    The same helper `tests/test_evaluate.py` states for its own junction test, kept
    here rather than imported across test modules: `os.symlink` needs a privilege a
    test suite should not assume, and `mklink /J` is the Windows form that needs none.
    `stdout`/`stderr` are discarded rather than captured as text, for the reason the
    sibling file measured: `mklink` echoes the two paths back, the project root holds
    `Masaüstü`, and `subprocess` decodes that with the machine's cp1254 — which raises
    `UnicodeDecodeError` inside the reader thread and surfaces as a warning
    `filterwarnings = ["error"]` turns into a failed run.
    """
    target.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        import subprocess

        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        return made.returncode == 0
    return False


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
    # The per-slide spread is a column of the printed sweep table, and nothing else in
    # this file would notice its absence. A **pin**: the header carries it today.
    assert "spread" in printed
    # The deliverable itself, read back: one row per slide, the two integers behind the
    # ratio, and the scale the counts were made at beside it. Taken from everything after
    # the header, because the sweep table above has a row beginning "2" as well.
    table = [line.split() for line in printed.split("slide_id", 1)[1].splitlines()]
    assert [row[:4] for row in table if row and row[0] in {"2", "3"}] == [
        ["2", "1", "1", "1.000"],
        ["3", "1", "1", "1.000"],
    ]


def test_cli_writes_and_prints_the_range_coverage(cli, capsys):
    # **Red before the fix**: the JSON carried `blob_count_summary` and `per_threshold`
    # and nothing comparing them, and nothing on stdout did either. A reader of the two
    # side by side concludes the sweep covered the distribution whatever it covers, and
    # that is the misreading this task exists to prevent. **Red again after round 1**: the
    # block carried a verdict and no criterion, so the verdict was a claim about the
    # distribution that only a docstring supported.
    module, index, results = cli
    assert module.main(["--index", str(index)]) == 0
    written = json.loads(
        (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    )
    # Both patches count three blobs and the range is [2, 5], so the distribution is
    # 3 to 3, the range reaches both ends of it, and nothing is left unswept.
    assert written["threshold_range_coverage"] == {
        "criterion": _CRITERION,
        "swept_min": 2,
        "swept_max": 5,
        "count_min": 3,
        "count_p50": 3.0,
        "count_mean": 3.0,
        "count_max": 3,
        "unswept_below": 0,
        "unswept_above": 0,
        "n_unswept": 0,
        "brackets_distribution": True,
    }
    printed = capsys.readouterr().out
    assert "swept range 2 to 5" in printed
    # The word is the flag and not a phrase chosen per run, so it can be read by a person
    # and matched by a test.
    assert "the swept range brackets the distribution" in printed
    # And the number behind the word, on stdout as well as in the JSON, because a flag a
    # reader has to take on trust is the shape this block was corrected away from.
    assert "0 of 2 patches outside the swept range" in printed


def test_cli_says_so_when_the_swept_range_does_not_reach_the_distribution(
    cli, project_scratch, monkeypatch, capsys
):
    # **Red before the fix**, for the same reason as the pair above, and on the branch the
    # controller's decision is about: a range that stops below the distribution is
    # reported, and `ablation.thresholds` is not widened on the strength of a 60-patch
    # sample. The fixture's two patches each count three blobs, so a range of `[2]` has a
    # top of 2 against a maximum of 3 and does not reach the distribution.
    module, index, results = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "narrow.yaml",
            results=results,
            ablation={"thresholds": [2]},
            labeling={"blob_threshold": 2},
        ),
    )
    assert module.main(["--index", str(index)]) == 0
    written = json.loads(
        (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    )
    assert written["threshold_range_coverage"]["brackets_distribution"] is False
    assert written["threshold_range_coverage"]["swept_max"] == 2
    assert written["threshold_range_coverage"]["count_max"] == 3
    # Both patches are above the top of the range, so the artifact says which end failed
    # and how many patches it left out, rather than only that the range was short.
    assert written["threshold_range_coverage"]["unswept_above"] == 2
    assert written["threshold_range_coverage"]["unswept_below"] == 0
    assert written["threshold_range_coverage"]["n_unswept"] == 2
    # The criterion in the artifact, and the flag agreeing with it: what a reader of the
    # JSON has to work with, with nothing outside the file.
    coverage = written["threshold_range_coverage"]
    assert _criterion_holds(coverage) is coverage["brackets_distribution"]
    printed = capsys.readouterr().out
    assert "the swept range does not bracket the distribution" in printed
    assert "2 of 2 patches outside the swept range" in printed
    # The run still wrote everything a run writes: a narrow range is a finding about the
    # configuration, not a reason to refuse.
    assert (results / "label_analysis" / "threshold_ablation.png").is_file()


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
    # **A pin, strengthened**: the line above is green whether or not the constant carries
    # it, and `comparison == RULE_COMPARISON` above compares the artifact to a copy of
    # itself, so neither of them can see a ratio that disagrees with the two totals the
    # same file carries. This one derives the ratio from the artifact's own numbers, so a
    # `total_blob_ratio` that is not 2166 / 377 fails here whatever `RULE_COMPARISON` says.
    assert comparison["total_blob_ratio"] == pytest.approx(
        comparison["rule_in_use"]["total_blobs"] / comparison["superseded_rule"]["total_blobs"]
    )


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


def test_cli_refuses_an_empty_index_with_the_refusal_code(cli, tmp_path, capsys):
    # **Red before the fix**: an index with a header and no rows reached `sweep`, raised a
    # `ValueError`, and left `main` by exception — a traceback and a status of 1, which
    # this CLI defines as a fact about this project's code rather than about the study.
    module, _, results = cli
    empty = _write_index(tmp_path / "empty", [])
    assert module.main(["--index", str(empty)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "no rows" in err
    assert "Traceback" not in err
    assert not (results / "label_analysis").exists()


def test_cli_refuses_a_configured_range_that_is_not_whole_numbers(
    cli, project_scratch, monkeypatch, capsys
):
    # **Red before the fix**: `[2, 5.5]` passed the selection guard, because the guard
    # compared against the configured list read raw, and the refusal came from `_thresholds`
    # inside `ablate` as a `ValueError` that escaped `main` — a traceback and a status of
    # 1. `_thresholds` was written to reject this value; a validator that validates and then
    # reports by traceback looks like a check and is not one.
    from blobcount import ablation

    module, index, results = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "half.yaml",
            results=results,
            ablation={"thresholds": [2, 5.5]},
            labeling={"blob_threshold": 2},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the configured range was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "5.5" in err and "whole number of blobs" in err
    assert "Traceback" not in err
    assert not (results / "label_analysis").exists()


def test_cli_converts_the_configured_range_once_for_its_selection_guard(
    cli, project_scratch, monkeypatch, capsys
):
    # **Red before the fix**, in the direction the guard's own message shows. `["3", 5]`
    # was compared as a raw list, so `3 in ["3", 5]` was False and the guard answered
    # `--selected 3 is not one of the ablation.thresholds this run sweeps (3, 5)` — a
    # message naming a range that contains 3 while refusing 3. The point of a validator is
    # one answer and this file had two; what is left is the validator's, and the question
    # of whether a quoted number is a whole number of blobs is the validator's to answer.
    module, index, results = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "quoted.yaml",
            results=results,
            ablation={"thresholds": ["3", 5]},
            labeling={"blob_threshold": 3},
        ),
    )
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    # The validator's answer, not the guard's, and not a traceback.
    assert "whole number of blobs" in err
    # **Red before the fix**: the message did not say that a quoted digit is the thing
    # being refused, so a user hand-editing the YAML could not tell a deliberate rule from
    # a parse accident. `_thresholds`' own refusal now states it.
    assert "a plain number" in err and "not '3'" in err
    assert "is not one of the ablation.thresholds" not in err
    assert "Traceback" not in err
    assert not (results / "label_analysis").exists()


def test_cli_refuses_a_configured_range_that_is_not_a_sequence(
    cli, project_scratch, monkeypatch, capsys
):
    # **Red before the fix**, on the message and not on the status. `!!binary 'MzU='` is
    # iterable, so the run swept it as the two thresholds 51 and 53 and refused
    # `--selected 2` from the selection guard: `EXIT_REFUSED`, but a refusal reporting a
    # range nobody configured as the range it swept. The `TypeError` arm of the catch in
    # `bin/ablate_threshold.py` had no shape that could reach it — only an accident, an
    # int where a list was configured — and that file's docstring already attributed the
    # arm to "a key that is not a sequence". The key check is that shape, so this pins
    # the claim at the boundary that makes it.
    #
    # **The message this asserts is no longer the one a hand-edit produces**, and the test
    # says so rather than being deleted: `ablation.thresholds` is declared `list`, so
    # `load` refuses `!!binary` as a type before `ablate_threshold` reaches its own check,
    # and the run is refused earlier with the earlier message. Both are `EXIT_REFUSED`, no
    # patch is read, and neither names 51 — the property the test was written to hold still
    # holds, and it now holds earlier. What moved is *which* refusal says it, so the
    # assertion is on the properties rather than on one function's wording.
    from blobcount import ablation

    module, index, results = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "binary.yaml",
            results=results,
            ablation={"thresholds": b"35"},
            labeling={"blob_threshold": 2},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the configured range was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    # The rule, and not the range the bytes spelled: a message naming 51 would be a
    # report of a sweep nobody configured. Either rule names it — `load`'s type refusal
    # today, `_thresholds`' own `not a sequence of thresholds` for a `Config` built in
    # memory — and the run must be refused before a patch is read either way.
    assert "ablation.thresholds" in err, err
    assert ("not a sequence of thresholds" in err) or (
        "type bytes" in err and "as list" in err
    ), err
    assert "51" not in err
    assert "Traceback" not in err
    assert not (results / "label_analysis").exists()


def test_cli_counts_the_slides_it_counted_patches_for(
    cli, project_scratch, synthetic_patch, capsys
):
    # **Red before the fix**: `n_slides` counted every index row's slide, and this index
    # names one `patch_id` under two `slide_id`s. `ablate` counts the patch once and skips
    # the repeat, so slide 3 is in `per_slide_gray_threshold` — a gray threshold derived
    # over a group nothing was counted for — and in no `per_slide_ratio` at all. Reporting
    # it would say the run measured two specimens and the artifact beside it has ratios
    # for one.
    module, _, results = cli
    index = _write_index(
        project_scratch / "patches",
        [_row("2_1016_0.png", "2", x=1016), _row("2_1016_0.png", "3")],
        synthetic_patch.array,
    )
    assert module.main(["--index", str(index)]) == 0
    written = json.loads(
        (results / "label_analysis" / "threshold_ablation.json").read_text(encoding="utf-8")
    )
    assert written["n_patches"] == 1
    assert written["n_slides"] == 1
    assert list(written["per_threshold"]["3"]["per_slide_ratio"]) == ["2"]
    assert "1 patches counted over 1 slides" in capsys.readouterr().out


def test_cli_refuses_a_results_path_that_is_the_project_root(
    cli, project_scratch, monkeypatch, capsys
):
    # **Red before the fix**: the guard tested `results_dir / "label_analysis"`, and a
    # `paths.results: .` makes the results directory itself the project root while leaving
    # the subdirectory below it. The run passed the guard and would have written
    # `threshold_ablation.json` and `threshold_ablation.png` into the repository. The
    # docstring claimed the project root is refused; the test now says which path.
    from blobcount import ablation

    module, index, _ = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "root.yaml",
            results=".",
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the output target was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    assert str(PROJECT_ROOT) in err
    assert "not strictly below the project root" in err
    assert not (PROJECT_ROOT / "label_analysis").exists()


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


def test_cli_refuses_a_non_finite_configured_threshold(cli, monkeypatch, capsys):
    # **Red before the fix**: `_thresholds` raised `OverflowError` on a `.inf`, and that is
    # not in the tuple at `bin/ablate_threshold.py:403`, so the run left a traceback and
    # the interpreter exited 1 — this CLI's documented code for a defect in its own code,
    # produced by a hand-edited YAML value. The validator is where that is fixed; this
    # test is here so the *exit code* is pinned as well as the exception type, because a
    # CLI that raises the right exception and returns the wrong status is still a
    # traceback to whoever ran it. The `.nan` half of the pair is covered by the
    # validator test; this one is about the code, and one config is enough to reach it.
    from blobcount import ablation

    module, index, results = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            results.parent,
            "non_finite.yaml",
            results=results,
            ablation={"thresholds": [2, float("inf"), 5]},
            labeling={"blob_threshold": 2},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the configured range was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    # The project's rule, not the standard library's sentence about infinity: this arm's
    # whole argument is that a user hand-editing `configs/default.yaml` reads a decision.
    assert "whole number of blobs" in err
    assert "inf" in err
    assert not (results / "label_analysis").exists()


def test_cli_refuses_a_results_path_outside_the_project(
    cli, project_scratch, tmp_path, monkeypatch, capsys
):
    # Re-pointed at the invariant that survives K14. The results directory the
    # configuration names sits inside the checkout, so it is owned; the
    # `label_analysis` subdirectory this run would write its figure and its JSON
    # into is a **link** that resolves to a *sibling* of the tree the same
    # configuration says the project owns — `tmp_path/owned-typo`, beside the
    # owned `tmp_path/owned` — which is neither the configured path itself nor
    # below it, so the guard refuses it and nothing is written through the link.
    # The old staging pointed `paths.results` itself at `tmp_path`, which made the
    # target one the configuration had just named — owned by definition — and the
    # refusal asserted the superseded invariant "everything outside the checkout is
    # refused". A link resolving to the near-miss beside the owned tree is the
    # mistyped `--out` the guard exists for; the `_boom` canary below still proves
    # the refusal happens before a single patch is read.
    from blobcount import ablation

    module, index, _ = cli
    results = project_scratch / "results"
    results.mkdir()
    outside = tmp_path / "owned-typo"
    if not _directory_link(outside, results / "label_analysis"):
        pytest.skip("this platform refused to make a directory link")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            project_scratch,
            "sibling.yaml",
            results=results,
            paths={"labels": str(tmp_path / "owned" / "labels")},
            ablation={"thresholds": [2, 3]},
            labeling={"blob_threshold": 3},
        ),
    )

    def _boom(path, size):
        raise AssertionError("a patch was read before the output target was refused")

    monkeypatch.setattr(ablation, "load_patch", _boom)
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    err = capsys.readouterr().err
    # The guard prints the path it was handed — the link, not where it resolves —
    # so the assertion names that link and the message it was refused by.
    assert "label_analysis" in err, err
    assert "project root" in err, err
    assert sorted(path.name for path in outside.iterdir()) == [], (
        "nothing was written through the link"
    )


def test_cli_refuses_a_results_path_that_is_not_a_string(
    cli, monkeypatch, tmp_path, capsys
):
    # **The pin moved, and the name moved with it.** This test used to be named
    # `..._config_path_refuses` and asserted `"path string"`, which is what `Config.path`
    # says when it is handed a number. `load` now type-checks the file path too, so
    # `paths.results: 7` is refused before a `Config` exists and the message names the type
    # instead. That is the better of the two answers and it is the earlier one: the refusal
    # now comes from the same line that refuses an undeclared key, rather than from a second
    # function reached later on a value the first one had already been asked about.
    #
    # `Config.path`'s own check is not dead and is not what this test covers any more: it
    # still refuses a value that is not a string in a `Config` built in memory, which
    # `tests/test_config.py::test_path_rejects_a_non_string_value` covers, and it still
    # refuses a blank string, which is a value question `load` deliberately makes no claim
    # about.
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
    err = capsys.readouterr().err
    assert "paths.results" in err, err
    assert "int" in err and "str" in err, err
    assert "Traceback" not in err


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


def test_cli_refuses_a_threshold_too_large_to_be_a_float(cli, monkeypatch, capsys):
    # **The end-to-end defect, and the one that made it worth closing**: before the fix this
    # run wrote `threshold_ablation.json` to disk and then let `OverflowError: int too large
    # to convert to float` out of `plot`, uncaught, out of `main`, out of the interpreter.
    # Exit status 1, which `bin/ablate_threshold.py`'s own module docstring reserves for "a
    # fact about this project's code rather than about the study" — produced by a caller
    # editing one line of `configs/default.yaml`, on a value the validator had already
    # accepted *by design*. Going through the CLI is the point: the exception itself is not
    # the defect, the uncaught escape is, and a test that called `plot` directly could not
    # see a status at all.
    #
    # **The range is a whole number of blobs and stays accepted.** `10 ** 320` is written out
    # because no short YAML spelling reaches an `int` of this magnitude: `1e320` is the
    # *string* `'1e320'` (PyYAML's float pattern wants an explicit sign on the exponent, so
    # `1.0e320` is a string too) and is refused as a quoted digit, while `1.0e+320` is a
    # `float` holding `inf` and is refused as a non-finite whole number. The count
    # comparison against the long spelling is exact, so the sweep is total: both patches come
    # out `normal` at that row and the row is a row.
    module, index, results = cli
    huge = 10 ** 320
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cfg(
            results.parent,
            "huge.yaml",
            results=results,
            ablation={"thresholds": [3, huge]},
            labeling={"blob_threshold": 3},
        ),
    )
    assert module.main(["--index", str(index)]) == module.EXIT_REFUSED
    out = results / "label_analysis"
    # The JSON is on disk, because the run got all the way through the counting and wrote
    # what it measured before the figure could not be drawn. The figure is not, and that is
    # the state a refusal leaves behind rather than a half-written artifact.
    written = json.loads((out / "threshold_ablation.json").read_text(encoding="utf-8"))
    assert written["per_threshold"][str(huge)] == {
        "n_normal": 2,
        "n_steatosis": 0,
        "pooled_steatosis_ratio": 0.0,
        "per_slide_ratio": {"2": 0.0, "3": 0.0},
        "per_slide_ratio_spread": 0.0,
        "selected": False,
    }
    assert not (out / "threshold_ablation.png").exists()
    err = capsys.readouterr().err
    # The project's rule and not the library's sentence about a float, for the reason the
    # `.inf` arm above gives: a user hand-editing `configs/default.yaml` reads a decision.
    assert "float" in err
    assert str(huge) in err
    assert "OverflowError" not in err


def test_cli_does_not_swallow_a_defect_from_inside_plot(cli, monkeypatch):
    # **A pin on the new arm, and the reason the arm catches `PlotError` and not
    # `ValueError`.** The module docstring claims that a `ValueError` raised anywhere inside
    # matplotlib still reaches the interpreter as a traceback, and that claim is what makes
    # the catch narrow rather than merely small: catch `ValueError` here and every defect in
    # the drawing would be printed on stderr as a decision about the caller's YAML, on a
    # status that file defines as "a fact about this project's code rather than about the
    # study". Mutating the arm to `except ValueError` turns this row red and leaves every
    # other test in the file green — measured, not asserted.
    #
    # The pair with `test_cli_does_not_swallow_a_defect_from_inside_ablate`, and it replaces
    # the name in the CLI module's own namespace, so it is the arm in *this* file that is
    # under test and not some other caller's.
    module, index, _ = cli

    def _boom(summarised, out_path):
        raise ValueError("injected")

    monkeypatch.setattr(module, "plot", _boom)
    with pytest.raises(ValueError, match="injected"):
        module.main(["--index", str(index)])


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
