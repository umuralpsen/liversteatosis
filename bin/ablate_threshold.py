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
figure are drawn at it. It is checked against `ablation.thresholds` *before* the first
patch is read, because a run that counts every patch in the index and then finds no
record to report a threshold on has wasted the run. `blobcount.ablation.summarise`
refuses the same selection, and that refusal is the invariant behind this check rather
than a second rule.

**Six refusals, in four groups.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration or the
dataset, never about this program:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two.
  `load` refusing the file: it is not there, its top level is not a mapping, it declares
  a key this project does not, or it omits one the project requires. And a
  `paths.results` that `load` accepted because it validates nothing past the key's
  presence but `Config.path` refused, because it is not a path string or is empty. Both
  are refused before a patch is read.
- **The output target**, as two checks this program makes for itself. A
  `paths.results / "label_analysis"` that is not strictly below the project root, and
  one that exists and is a file rather than a directory — checked on the results
  directory as well, because a file there makes the subdirectory un-creatable.
  `blobcount.extract.prepare_output` is the wrong tool for either, because it *deletes*
  and this program writes two files into a directory it does not own: it refuses targets
  it may not clear, and these are targets it may not write. Its `_is_below` is the
  predicate both checks use, so "is this ours" has one implementation in this project
  and not one per guard. `paths.results` is a hand-edited string in
  `configs/default.yaml`, and a `paths.results: /` or `..` would put a figure and a
  JSON somewhere this repository is not responsible for.
- **The selection**, as a check this program makes for itself: a `--selected`, or a
  configured `labeling.blob_threshold`, that is not one of the `ablation.thresholds`
  this run sweeps.
- **The index and the patches it names**, as a `LabelingInputError`: the six shapes
  `blobcount.labeling.LabelingInputError` enumerates, raised by `ablate` around the two
  reads that can fail that way and around nothing else.

All six are printed on stderr behind this program's name, and none of them is a
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

**A bad argument is not one of the six, and not one of this program's outcomes at all.**
`argparse` refuses an option this program does not declare, one given no value, and a
`--selected` that is not an integer, by printing its own message and calling
`sys.exit(2)` inside `parse_args`, before the first statement of `main` runs. Nothing
else reaches it: `--index` is typed `Path` and no string fails that conversion. The
parser's status is the same number as `EXIT_REFUSED` and no code can tell the two
apart, so the separation is on stderr, where the parser names the argument and its
usage and none of the six names either.

**A defect inside `ablate` is never a refusal.** A `ValueError` from the counting, from
`np.percentile` on a `blobs.gray_percentile` outside [0, 100] that this project
range-checks nowhere, from a configured `ablation.thresholds` holding a value that is
not a whole number of blobs, or from a write is not caught here and leaves the
interpreter to print a traceback, which is the honest report of a defect in this
project. The status it exits with is 1, which the paragraph above says is a traceback
and not a study-level failure, and a caller that has to tell the two apart reads
stderr: a refusal names what it refused, and a traceback names a file and a line. **A
mistyped configured value is a limit rather than a gap**, and the limit is the one
`bin/extract_patches.py` records: `load` checks that a key is declared and nothing at
all about its value, so a conversion this program does not wrap reaches the interpreter
as a traceback. A catch wide enough to hold it would also hold a defect in this
project's own arithmetic, and reporting a defect as a refusal would send the caller to
fix a configuration that is fine.

**A summary line, then two tables.** The count distribution comes first, because the
ratios are ratios of it and a reader who has seen a pooled ratio of 0.9 and no mean count
has been told the wrong thing by the ratio alone. Then the sweep, one row per
threshold, and then the per-slide table at the selected threshold: the deliverable the
plan names, and the table that says whether the magnification artifact is gone. It
carries each slide's gray threshold and achieved resolution beside its ratio, because a
per-slide ratio means nothing without the scale its counts were made at.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from blobcount.ablation import RULE_COMPARISON, AblationStats, ablate, plot
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
    """
    return {
        "selected_threshold": stats.selected,
        "n_patches": stats.count_summary()["count"],
        "n_slides": len(stats.per_slide_mpp),
        "blob_count_summary": stats.count_summary(),
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

    The two refusals the module docstring names, as a message rather than an exception
    because this check is this program's own and it is a judgement about an argument:
    no other caller of `ablate` has an output directory to judge, and a type it alone
    raises would be a type every caller has to know. `_is_below` is the same predicate
    `blobcount.extract.prepare_output` uses, imported for the same reason that guard
    resolves the slides and the manifest through the configuration: one implementation
    of "is this ours", which collapses a `..` and refuses the project root itself.

    **The containment test is on the deepest path and the "is a file" test is on both.**
    A `paths.results` that is a regular file makes `results/label_analysis` un-creatable
    — `Path.exists()` on it is False rather than an error, and the `mkdir` below would
    raise `NotADirectoryError` — so checking only the subdirectory would let a typo reach
    a traceback. The subdirectory is checked too, because that is the path a hand-edited
    `paths.results` most often collides with.

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
    if not _is_below(target, PROJECT_ROOT):
        return (
            f"refusing to write {target}: it is not strictly below the project root "
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
        # validates that a key is declared and nothing about its value, so a
        # `paths.results` that is not a path string is refused by `Config.path`, and
        # outside this arm that refusal would leave a traceback.
        results_dir: Path = cfg.path("paths.results")
        # Read rather than converted here. The value is compared against `--selected`
        # and listed in the message, and the conversion of every entry into a whole
        # number of blobs is `blobcount.ablation._thresholds`'s one job: a second
        # `int()` per entry is a second place for the two to disagree about what the
        # configured range is.
        swept = list(cfg.get("ablation.thresholds"))
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
        # (patch_size, patch_size, 3) array the count needs. `ablate` raises this around
        # those two reads and around nothing else, so a `ValueError` from the counting,
        # from the normalization or from a percentile outside [0, 100] is not caught
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
    plot(stats.per_threshold, target / _FIGURE_NAME)

    summary = stats.count_summary()
    print(
        f"{summary['count']} patches counted over {len(stats.per_slide_mpp)} slides; "
        f"blob counts: min {summary['min']}, p50 {summary['p50']:.2f}, "
        f"mean {summary['mean']:.2f}, max {summary['max']}\n"
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
