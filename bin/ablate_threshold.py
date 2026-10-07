"""Sweep the blob-count threshold over a patch index and report it per slide.

The third entry point, after extraction and labeling. It reads the `patches.csv` a
previous `extract_patches` run wrote, counts every patch in it again, sweeps the
thresholds in `ablation.thresholds`, and writes the JSON, the figure and the printed
tables that say where the class distribution sits, per slide, at every threshold.

    python bin/ablate_threshold.py --index dataset/patches/patches.csv
    python bin/ablate_threshold.py --index ... --selected 20

**The counts are made again here and `labels.csv` is not read.** The sweep exists to
inform the choice of `labeling.blob_threshold`, so a count read from a labelling run
made at a threshold already chosen would make the sweep a function of its own answer.
`blobcount.ablation.ablate` re-reads each patch through the same two seams the labeling
stage uses and counts it at its own recorded `mpp` and its own slide's gray threshold.

**`--selected` is the threshold the study labels at, and it defaults to
`labeling.blob_threshold`.** The printed per-slide table and the right-hand panel of the
figure are drawn at it. It is checked against the converted `ablation.thresholds` — the
list `blobcount.ablation._thresholds` returns, and not the configured list read raw —
*before* the first patch is read, because a run that counts every patch in the index and
then finds no record to report a threshold on has wasted the run.
`blobcount.ablation.summarise` refuses the same selection, and that refusal is the
invariant behind this check rather than a second rule.

**Eight refusals, in six groups.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration or the
dataset, never about this program:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two.
  `load` refusing the file: it is not there, its top level is not a mapping, it declares
  a key this project does not, it omits one the project requires, or a value for a declared
  key has a type this project does not read it as. And a `paths.results` whose *value*
  `Config.path` refused because it is empty — the type of that key is `load`'s business now,
  and the empty string is a value question that stays here. Both are refused before a patch
  is read.
- **The output target**, as two checks this program makes for itself, each run against
  both `paths.results` and `paths.results / "label_analysis"`: one that exists and is a
  file rather than a directory, and one that is not strictly below the project root. Both
  paths are tested because a file at the results directory makes the subdirectory
  un-creatable, and because `paths.results: .` puts the results directory itself at the
  project root while leaving the subdirectory below it — so a guard on the subdirectory
  alone writes a figure and a JSON into the repository. `blobcount.extract.prepare_output`
  is the wrong tool for any of them, because it *deletes* and this program writes two files
  into a directory it does not own: it refuses targets it may not clear, and these are
  targets it may not write. Its `_is_below` is the predicate the containment check uses, so
  "is this ours" has one implementation in this project and not one per guard.
  `paths.results` is a hand-edited string in `configs/default.yaml`, and a
  `paths.results: /` or `..` would put a figure and a JSON somewhere this repository is not
  responsible for.
- **The selection**, as a check this program makes for itself: a `--selected`, or a
  configured `labeling.blob_threshold`, that is not one of the `ablation.thresholds`
  this run sweeps. The range it is compared against is the one
  `blobcount.ablation._thresholds` returns, and not the configured list read raw, so that
  the check and the sweep cannot come to different answers about what is being swept.
- **The configured range itself**, as `blobcount.ablation._thresholds`'s own refusal: a
  `ValueError` or `TypeError` out of that one call, caught here. A quoted digit, a
  fraction, a YAML `true`, a negative number and a non-finite float are all refused as
  `ValueError`s and the message names the rule — a plain number that is finite, exactly a
  count and not below zero, `3`, and not `'3'`, not `true`, not `-5` and not `.inf` or
  `.nan` — so a user hand-editing `configs/default.yaml` reads a decision and not an
  accident; a key that is not a sequence at all is the `TypeError`, which is what
  `!!binary 'MzU='` and a `!!set` are.
  All six are the same hand-edit of the same key, and the catch is around that one call
  and not around `ablate`, so it is wide enough to hold a mistyped configuration value
  and too narrow to hold a defect in the counting. **`OverflowError` is not in the tuple
  and adding it would be a mistake rather than caution.** `isinstance(float("inf"),
  float)` is `True`, so a `.inf` reached the validator's `int()` and raised
  `OverflowError`, which this tuple did not hold: a hand-edited `.inf` left a traceback
  and exited 1, the status the `EXIT_CONTRADICTORY` paragraph below reserves for a defect
  in this project's own code. The fix is in the validator, where a non-finite value is
  now refused by the same predicate that refuses a quoted digit, and the predicate is why
  this tuple is still two types: `int()` is reached only for a value already accepted as a
  finite number, so an
  `OverflowError` out of this call would be a bug in the validator, and this file reports
  a bug in this file with a traceback on purpose. Catching it here would print a
  validator defect as a refusal on stderr, which is a defect in this project wearing the
  one costume that reads as a decision. The same reasoning retired the NaN arm's accident:
  `int(float("nan"))` raised `ValueError`, so NaN reached this catch by luck rather than
  by decision, and the message it carried was CPython's `cannot convert float NaN to
  integer`, which names no rule at all.
- **The figure**, as a `PlotError` out of the `plot` call, caught here and nowhere else.
  A swept threshold too large for a `float` — `ablation.thresholds: [3, 10 ** 320]` — is a
  whole number of blobs, so `_thresholds` accepts it by design, `sweep` compares counts
  against it exactly and is total over it, and the JSON this run writes is a real record of
  that row. It is the *figure* that cannot hold the value: every threshold on a matplotlib
  axis is a float coordinate, and the transform raises `OverflowError: int too large to
  convert to float` on one that has no float representation. **The validator does not
  refuse it, and the reason is the point.** Counting and drawing ask different questions —
  "is this a number of blobs?" and "is this a coordinate?" — and the first one has an exact
  answer at any magnitude while the second does not, so the limit is the renderer's and
  `plot` holds it. A validator that refused a count to accommodate a plotting library would
  be deciding a question that is not its own, on behalf of a caller that asked for a sweep
  and may never draw anything. The catch is `PlotError` and not `ValueError` so that a
  `ValueError` out of matplotlib itself still reaches the interpreter as a traceback: this
  arm is wide enough for a hand-edited `ablation.thresholds` and no wider.
  **The JSON is already on disk when this fires, and it stays.** The run finished the
  measurement and stopped at the drawing, so the state is a number a reader can use, no
  figure, and a message on stderr naming the threshold that could not be placed.
- **The index and the patches it names**, as a `LabelingInputError`: the six shapes
  `blobcount.labeling.LabelingInputError` enumerates, an index carrying no rows, and
  those raised by `ablate` around the two reads that can fail that way, around the
  numeric-cell conversion `labeling._row_numbers` wraps, and around nothing else.

All eight are printed on stderr behind this program's name, and none of them is a
traceback.

**There is no `EXIT_CONTRADICTORY` here, and that is a decision rather than an
omission.** The sibling CLIs define 1 for a study-level failure — a slide that yielded
no patches, a labelled tree that contradicted itself — and both of those are states a
run *completes* with. A sweep produces no such state: it reports ratios over counts it
measured, and a ratio that disagrees with a slide is the finding the run exists to
report, not a contradiction to be refused. So this file defines two codes, and a status
of 1 means the interpreter reported a traceback, which is a fact about this project's
code rather than about the study. `main`'s *return value* is the contract, and an
exception never returns one.

**A bad argument is not one of the eight, and not one of this program's outcomes at
all.** `argparse` refuses an option this program does not declare, one given no value, and
a `--selected` that is not an integer, by printing its own message and calling
`sys.exit(2)` inside `parse_args`, before the first statement of `main` runs. Nothing
else reaches it: `--index` is typed `Path` and no string fails that conversion. The
parser's status is the same number as `EXIT_REFUSED` and no code can tell the two
apart, so the separation is on stderr, where the parser names the argument and its
usage and none of the eight names either.

**A defect inside `ablate` or inside `plot` is never a refusal.** A `ValueError` from the
counting, from
`np.percentile` on a `blobs.gray_percentile` outside [0, 100] that this project
range-checks nowhere, from the arithmetic of `sweep` and `summarise`, from a write, or from
anything matplotlib raises while it draws is not caught here and leaves the interpreter to
print a traceback, which is the honest
report of a defect in this project. The status it exits with is 1, which the paragraph
above says is a traceback and not a study-level failure, and a caller that has to tell
the two apart reads stderr: a refusal names what it refused, and a traceback names a file
and a line. The `plot` arm is the one place where a project-raised type is what keeps that
true: `PlotError` is raised by this project and by nothing in a figure library, so catching
it says "this value cannot be drawn" and says nothing else. **A configured value this
project does not check is a limit, and a checked one is not — and the two are separated by
whether a validator exists, not by whether the value is mistyped.** `load` now type-checks
both of its inputs, so a *mistyped* value anywhere in this project's configuration is a
refusal before this program reads a patch, and that covers `extraction.patch_size` as much
as it covers `ablation.thresholds`; the test that says so for this file's own keys is
`tests/test_config.py::test_a_file_value_whose_type_is_not_the_declared_one_is_refused`.
What remains outside `load` is a value of the *declared* type that is nonetheless wrong, and
that splits three ways here. `extraction.patch_size` is read and used, never checked, so a
hand-edit of it to a negative integer reaches the interpreter as a traceback and that is the
recorded limit — a type this project checked, a value it did not. `ablation.thresholds` is
read by a function written to reject a value that is not a whole number of blobs, and a
validator that validates and then reports by traceback looks like a check without being one,
so its refusal is caught above and reported as a refusal. **The same ruling reaches the
drawing path one layer on, and stops at the same line:** a threshold the validator is right
to accept and a renderer is not able to place is a fact about the run's configuration, so it
is caught and reported as a refusal — while a threshold the validator would have accepted and
that a *defect* made impossible to place is still a traceback.

**A summary line, the range coverage, then two tables.** The count distribution comes
first, because the ratios are ratios of it and a reader who has seen a pooled ratio of 0.9
and no mean count has been told the wrong thing by the ratio alone. Under it, the
comparison of the swept range against that distribution: how many of the patches the
range leaves outside itself, and the flag saying whether it brackets the distribution,
because a reader of `blob_count_summary` and `per_threshold` together would otherwise
conclude the sweep covered the distribution whatever it covers. The count is printed
before the flag and not after it, so the line reads as a measurement with a verdict
attached rather than as a verdict with a decoration. Then the sweep, one row per
threshold, and then the per-slide table at the selected threshold: the deliverable the
plan names, and the table that says whether the magnification artifact is gone. It carries
each slide's gray threshold and achieved resolution beside its ratio, because a per-slide
ratio means nothing without the scale its counts were made at.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

# `_thresholds` is private and imported anyway, for the reason
# `blobcount.ablation` imports `blobcount.labeling._slide_gray_threshold` for: the
# conversion of `ablation.thresholds` into whole numbers of blobs has exactly one answer in
# this project, and a second `int()` per entry in this file would be a second answer. The
# leading underscore is about this file not being the module's caller, not about the
# function being single-use. `_is_below` is imported the same way, and for the same reason
# `blobcount.extract.prepare_output` is not: the guard below needs the predicate and not
# the deletion.
from blobcount.ablation import (
    RULE_COMPARISON,
    AblationStats,
    PlotError,
    _thresholds,
    ablate,
    plot,
)
from blobcount.config import PROJECT_ROOT, ConfigError, load
from blobcount.extract import _is_below
from blobcount.labeling import LabelingInputError

EXIT_OK = 0
EXIT_REFUSED = 2

# Where the artifacts go and what they are called, under `paths.results`. Three names in
# one place so the JSON, the figure and the test that reads both name the same file.
_ANALYSIS_DIR = "label_analysis"
_JSON_NAME = "threshold_ablation.json"
_FIGURE_NAME = "threshold_ablation.png"

_SWEEP_COLUMNS = ("threshold", "n_steatosis", "n_normal", "pooled_ratio", "spread", "selected")
_SLIDE_COLUMNS = ("slide_id", "n", "steatosis", "ratio", "gray_threshold", "mpp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ablate_threshold",
        description="Sweep the blob-count threshold over a patch index and report the "
        "class distribution per slide at every threshold.",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        metavar="PATH",
        help="the patches.csv written by bin/extract_patches.py",
    )
    parser.add_argument(
        "--selected",
        type=int,
        metavar="N",
        help="the threshold the study labels at, drawn in the per-slide table and the "
        "right-hand panel; labeling.blob_threshold from the configuration otherwise",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/extract_patches.py` has none: this is a few
    columns of integers, one identifier and a few floats, and `tabulate` is not a
    dependency this project declares. The sibling's `_print_table` is not imported,
    because `bin/` is a directory of scripts loaded by path rather than a package, and
    reaching into one script's private name to share eight lines would couple two entry
    points that are meant to be read one at a time.
    """
    widths = [
        max([len(str(header))] + [len(str(row[index])) for row in rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _sweep_rows(stats: AblationStats) -> list[list[object]]:
    """Return one row per swept threshold, lowest first.

    The rows are the summarised records as they stand, not a re-derivation of them: a
    number printed here and a number written to the JSON are read from one place, and
    the figure beside them reads the same records.
    """
    return [
        [
            threshold,
            record["n_steatosis"],
            record["n_normal"],
            f"{record['pooled_steatosis_ratio']:.3f}",
            f"{record['per_slide_ratio_spread']:.3f}",
            "yes" if record["selected"] else "",
        ]
        for threshold, record in sorted(stats.per_threshold.items())
    ]


def _slide_rows(stats: AblationStats) -> list[list[object]]:
    """Return one row per slide, at the selected threshold, in the sweep's slide order.

    The per-slide counts are counted here from `counts` and `slide_of` rather than
    carried in the record, because the ratio in the record is a fraction and a printed
    table that shows a fraction without the two integers behind it is the table that
    hides the thing this run exists to show. The order is the order of
    `per_slide_ratio`, so the rows and the bars of the figure are the same slides in the
    same order.
    """
    record = stats.per_threshold[stats.selected]
    per_slide: dict[str, list[int]] = {
        slide_id: [0, 0] for slide_id in record["per_slide_ratio"]
    }
    for patch, count in stats.counts.items():
        row = per_slide[stats.slide_of[patch]]
        row[0] += 1
        row[1] += int(count >= stats.selected)
    return [
        [
            slide_id,
            counts[0],
            counts[1],
            f"{record['per_slide_ratio'][slide_id]:.3f}",
            f"{stats.per_slide_gray_threshold[slide_id]:.1f}",
            f"{stats.per_slide_mpp[slide_id]:.4f}",
        ]
        for slide_id, counts in per_slide.items()
    ]


def _n_slides(stats: AblationStats) -> int:
    """Return the number of slides a blob count was actually made for.

    Keyed off `slide_of`, which holds one entry per counted patch, rather than off
    `per_slide_mpp`, which holds one entry per *row* of the index. The two differ on an
    index that names one `patch_id` under two `slide_id`s: `ablate` counts the patch once
    and skips the repeat, so the second slide is in `per_slide_mpp` and
    `per_slide_gray_threshold` — a gray threshold derived over a group nothing was counted
    for — while `per_slide_ratio`, which every per-slide number here is read beside, has
    no key for it. Counting it would report a specimen the artifact has no ratio for.
    """
    return len(set(stats.slide_of.values()))


def _document(stats: AblationStats) -> dict:
    """Return the JSON document this run writes.

    UTF-8 with `ensure_ascii=False` and slide ids as keys and as values, for the reasons
    `extract.write_index` and `labeling._write_labels` give: a slide id in this project
    is a specimen name and has to survive a Turkish-locale machine unchanged. The
    threshold keys are written as strings because JSON object keys are strings, and
    `selected_threshold` is written as a number beside them so that a reader does not
    have to know that.

    `blob_counts` is the measurement itself and is written even though `labels.csv`
    holds a count per patch: this sweep is independent of the threshold that produced
    those labels, and an artifact carrying only ratios could not be re-examined without
    re-reading every patch. `counting_rule_comparison` is `RULE_COMPARISON` verbatim, so
    a reader of a file from any run of this pipeline knows which rule produced the counts
    in it.

    **`threshold_range_coverage` is beside `blob_count_summary` and not below the file,
    because those two are read together.** A reader who has the distribution and the
    sweep and not the comparison between them concludes the sweep covered the
    distribution, whatever it covers. The block carries `n_unswept` — how many patches the
    swept range leaves outside itself — beside `brackets_distribution`, the boolean
    derived from it, because a count of patches is a number a reader cannot misread the
    way a verdict can, and it carries `criterion`, the comparison the boolean is, so the
    two are checked against each other inside the file rather than in a docstring. The
    criterion string is evaluated against this block's own numbers by
    `tests/test_ablation.py`, so the string, the boolean and the predicate cannot come to
    different answers. `n_slides` counts the slides a patch was counted for, for the
    reason `_n_slides` gives.
    """
    return {
        "selected_threshold": stats.selected,
        "n_patches": stats.count_summary()["count"],
        "n_slides": _n_slides(stats),
        "blob_count_summary": stats.count_summary(),
        "threshold_range_coverage": stats.range_coverage(),
        "blob_counts": dict(sorted(stats.counts.items())),
        "per_slide_gray_threshold": stats.per_slide_gray_threshold,
        "per_slide_mpp": stats.per_slide_mpp,
        "per_threshold": {
            str(threshold): record
            for threshold, record in sorted(stats.per_threshold.items())
        },
        "counting_rule_comparison": RULE_COMPARISON,
    }


def _refuse_output_target(results_dir: Path, target: Path) -> str | None:
    """Return why this program will not write its artifacts under `results_dir`, or None.

    The four refusals the module docstring names, as a message rather than an exception
    because this check is this program's own and it is a judgement about an argument:
    no other caller of `ablate` has an output directory to judge, and a type it alone
    raises would be a type every caller has to know. `_is_below` is the same predicate
    `blobcount.extract.prepare_output` uses, imported for the same reason that guard
    resolves the slides and the manifest through the configuration: one implementation
    of "is this ours", which collapses a `..` and refuses the project root itself.

    **Both paths are tested, and the results directory is one of them.** Only
    `results_dir / "label_analysis"` was, and a `paths.results: .` then passed: the
    subdirectory is below the root while the directory itself is the root, so the run
    wrote `threshold_ablation.json` and `threshold_ablation.png` into the repository. The
    sentence above claims the project root is refused, and it is tested on the path whose
    name the refusal prints, so the claim is now true of both.

    **The "is a file" test and the containment test both run against both paths.** A
    `paths.results` that is a regular file makes `results/label_analysis` un-creatable
    — `Path.exists()` on it is False rather than an error, and the `mkdir` below would
    raise `NotADirectoryError` — so checking only the subdirectory would let a typo reach
    a traceback. The subdirectory is checked too, because that is the path a hand-edited
    `paths.results` most often collides with. The file test runs first for each path and
    the containment test second for the same path, so a file outside the project is
    reported as the typo it is rather than as a containment failure.

    The check returns a message and raises nothing, and it runs before the counts, so a
    mistyped `paths.results` costs a message rather than a run over every patch in the
    index.
    """
    for path in (results_dir, target):
        if path.exists() and not path.is_dir():
            return (
                f"refusing to write {path}: it exists and is a file rather than a "
                f"directory, so it is a typo in paths.results and not a previous run"
            )
        if not _is_below(path, PROJECT_ROOT):
            return (
                f"refusing to write {path}: it is not strictly below the project root "
                f"{PROJECT_ROOT}, so it is not a directory this project owns"
            )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ablation over the index the arguments name and return the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        # Resolved inside the arm, for the reason `bin/label_patches.py` gives: `load`
        # types the key as the declared `str`, but an empty one is a value it makes no
        # claim about, so `Config.path` is what refuses it — and outside this arm that
        # refusal would leave a traceback.
        results_dir: Path = cfg.path("paths.results")
        selected = (
            args.selected if args.selected is not None
            else int(cfg.get("labeling.blob_threshold"))
        )
    except ConfigError as error:
        # The two shapes this arm carries: the file `load` refuses, and a `paths.results`
        # `Config.path` refuses. The override the parser produced cannot be the cause:
        # `--selected` has already been typed as the `int` the key is configured as and
        # `--index` as a `Path`.
        print(f"ablate_threshold: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # The configured range, converted once, here, before a patch is read and before the
    # selection is checked against it. `_thresholds` is the conversion `ablate` will make
    # again over the same configuration object, and the point of calling it here is that
    # there is then one answer to "what is this run sweeping" rather than two. The guard
    # below compared `--selected` against the configured list read raw, so `[3, 5.5]`
    # passed the guard and raised past it inside `ablate`, and `["3", 5]` produced
    # `--selected 3 is not one of the ablation.thresholds this run sweeps (3, 5)` — a
    # message that names a range containing 3 and refuses 3.
    #
    # **This arm is above the output-target guard, and neither check depends on the
    # other.** The cost of the order is that a bad `paths.results` is reported after one
    # conversion of a list of a handful of numbers, which is nothing against a patch read;
    # the cost of the other order is that the selection guard below has to read the
    # configured list raw again, which is the two-answers-one-question state the call
    # above exists to end. So the conversion is the one that comes first, and it comes
    # first because it is the only one of the two the next check consumes.
    try:
        swept = list(_thresholds(cfg))
    except (TypeError, ValueError) as error:
        # `ValueError` for a configured entry that is not a plain whole number of blobs —
        # a fraction, a quoted digit, a `true`, a negative number, a `.inf` or a `.nan` — and
        # `TypeError` from the key itself not being a sequence, which `load` used to accept
        # and now refuses: `ablation.thresholds` is declared `list` in `KEY_TYPES`, so a
        # `!!binary` or a `!!set` written into the file is a `ConfigError` on the arm above
        # and never reaches this call. The `TypeError` is still caught because a `Config` is
        # constructible in memory and need not come from a file at all. Both are the same
        # hand-edit of the same key, and the catch is around this one call rather than around
        # `ablate`, so it is too narrow to hold a defect in the counting. This refusal is
        # a statement about the configuration, which is why it is not the `ConfigError` arm
        # above: `load` accepted the file, and `ablation.thresholds` is a key this project
        # wrote a validator for. The message names the rule the entry broke, so a user
        # hand-editing the YAML sees that a quoted digit is refused by choice.
        #
        # `OverflowError` is not in this tuple on purpose, and the module docstring gives
        # the argument. The short form: `_thresholds` refuses a non-finite value before it
        # converts, so nothing a hand-edit can write reaches `int()` and raises one, and an
        # `OverflowError` arriving here would be a bug in that validator rather than a
        # mistyped YAML value — which is a defect in this project, and this file's ruling
        # is that a defect in this project prints a traceback.
        print(f"ablate_threshold: {error}", file=sys.stderr)
        return EXIT_REFUSED

    target = results_dir / _ANALYSIS_DIR
    refused = _refuse_output_target(results_dir, target)
    if refused is not None:
        print(f"ablate_threshold: {refused}", file=sys.stderr)
        return EXIT_REFUSED
    if selected not in swept:
        print(
            f"ablate_threshold: --selected {selected} is not one of the "
            f"ablation.thresholds this run sweeps "
            f"({', '.join(str(threshold) for threshold in swept)}), so there is no "
            f"record to report it on: widen the configured range, or select one of "
            f"those",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        stats = ablate(args.index, cfg, selected=selected)
    except LabelingInputError as error:
        # The six shapes the module docstring groups, at the shape each takes: an index
        # that is not there, cannot be opened, carries a cell longer than
        # `csv.field_size_limit`, or is not the `patches.csv` header; a cell of it that
        # is not the number its column records; and a patch file that is not the `uint8`
        # (patch_size, patch_size, 3) array the count needs. And an index carrying a
        # header and no rows, which `ablate` refuses at the index boundary rather than
        # letting `sweep` divide by a count of nothing. `ablate` raises this around
        # those two reads, `labeling._row_numbers` around the cell conversion, and
        # nothing else, so a `ValueError` from the counting, from the normalization, from
        # the arithmetic of `sweep`, or from a percentile outside [0, 100] is not caught
        # here and reaches the caller as a traceback. Which is the point: a refusal is an
        # answer about the argument and a traceback is the answer about our code, and
        # reporting a defect as the first would send the caller to fix an index that is
        # fine.
        print(f"ablate_threshold: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # The directory is created after the counts and not before, so an index that cannot
    # be read leaves the previous run's artifacts where they are — the reason
    # `bin/label_patches.py` reads its index before it clears its tree.
    target.mkdir(parents=True, exist_ok=True)
    (target / _JSON_NAME).write_text(
        json.dumps(_document(stats), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        plot(stats.per_threshold, target / _FIGURE_NAME)
    except PlotError as error:
        # **The arm is around this one call, and it holds this one type.** A swept
        # threshold too large for a `float` is a whole number of blobs, so it was counted
        # and swept exactly and the JSON above is a real record of it; it is the *figure*
        # that cannot hold the value, because every threshold on a matplotlib axis is a
        # float coordinate. Before this arm existed, `float(10 ** 320)` raised
        # `OverflowError: int too large to convert to float` out of `Axes.add_line` ->
        # `_update_line_limits`, uncaught, out of `main`, on the status the paragraph below
        # reserves for a defect in this project's own code — from a caller who had edited
        # one line of `configs/default.yaml`. That is a caller-reachable traceback in a
        # shipped entry point, which is what rounds 3 to 6 of Task 6 existed to eliminate,
        # and it did not ride to the gate.
        #
        # `PlotError` and not `ValueError` is what makes the arm narrow enough. `plot`
        # raises `ValueError` for two other reasons and those are deliberately not caught
        # here, and neither is a `ValueError` raised anywhere inside matplotlib: a real
        # defect in the drawing has to keep reaching the interpreter as a traceback, because
        # the paragraph below is explicit that a defect in this project is reported with a
        # file and a line and not as a decision about the caller's YAML. The same reasoning
        # that keeps `OverflowError` out of the tuple around `_thresholds` above, read from
        # the other side.
        #
        # **The JSON stays and the figure does not**, because the run finished the
        # measurement and stopped at the drawing. That is the state a refusal leaves here: a
        # number a reader can use, no figure, and a message on stderr naming the threshold
        # that could not be placed. Nothing in this file removes an artifact it did not
        # decide to write, and the directory was created by this run.
        print(f"ablate_threshold: {error}", file=sys.stderr)
        return EXIT_REFUSED

    summary = stats.count_summary()
    print(
        f"{summary['count']} patches counted over {_n_slides(stats)} slides; "
        f"blob counts: min {summary['min']}, p50 {summary['p50']:.2f}, "
        f"mean {summary['mean']:.2f}, max {summary['max']}\n"
    )
    # The range the sweep cut at, against the distribution it cut. Printed under the
    # summary and above the tables because it is the answer to "was this sweep wide
    # enough to place the knee", and a reader of the summary and the sweep table together
    # would otherwise have to do the comparison themselves and would assume it came out
    # favourably. The count comes before the flag: how many patches the range left outside
    # itself is the number, and the flag is that number being zero, so a reader who trusts
    # neither the flag nor this program can still read the answer off the line.
    coverage = stats.range_coverage()
    brackets = (
        "brackets the distribution" if coverage["brackets_distribution"]
        else "does not bracket the distribution"
    )
    print(
        f"swept range {coverage['swept_min']} to {coverage['swept_max']} against a count "
        f"distribution with min {coverage['count_min']}, "
        f"p50 {coverage['count_p50']:.2f}, "
        f"mean {coverage['count_mean']:.2f}, max {coverage['count_max']}: "
        f"{coverage['n_unswept']} of {summary['count']} patches outside the swept range, "
        f"so the swept range {brackets}\n"
    )
    _print_table(_SWEEP_COLUMNS, _sweep_rows(stats))
    print()
    _print_table(_SLIDE_COLUMNS, _slide_rows(stats))
    selected_record = stats.per_threshold[stats.selected]
    print(
        f"\nthe per-slide table is at threshold {stats.selected}, where the pooled ratio "
        f"is {selected_record['pooled_steatosis_ratio']:.3f} and the per-slide spread is "
        f"{selected_record['per_slide_ratio_spread']:.3f}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
