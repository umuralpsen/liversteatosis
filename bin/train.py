"""Train the cohort leave-one-specimen-out and report the epoch each fold selected.

The fourth entry point, after extraction, labeling and the threshold sweep. It reads the
`labels.csv` a previous `label_patches` run wrote, splits the cohort into one fold per
specimen, trains each fold, and reports — for the epoch each fold *selected*, and not for
the one it stopped at — the accuracy, precision, recall and F1 the held-out specimen
scored.

    python bin/train.py --labels dataset/labels/labels.csv
    python bin/train.py --labels dataset/labels/labels.csv --out /tmp/run

`--labels` is the handoff file and is required, for the reason `--index` is on the two
sibling entry points: the artifact this run consumes is a file a previous stage wrote and
naming it on the command line says which run's labels are being trained on. **`--out`
overrides `paths.results`** and defaults to it, and is never cleared: a checkpoint and a
log per fold, `fold_summary.csv` and `training_summary.json` are written into it, a file of
the same name is rewritten in place, and a file a run with fewer folds does not produce is
left where the last run put it. **It is refused if it exists as a file or is not strictly
below the project root**, for the reason the output-target group below gives.

**The `path` cell of a row is resolved against `paths.labels`, not against the directory
the `labels.csv` sits in and not against the working directory.** This is the ruling
`blobcount.labeling` writes those cells under — a path relative to the label root,
separated by `/` whatever the platform — so a row resolves to the patch that root names,
and a run started from another directory resolves the same rows to the same files. The read
itself is `blobcount.train.read_labels`, not a copy in this file: `bin/` is a directory of
scripts loaded by path, a read here is outside the source scan in `tests/test_config.py`,
and Task 12 reads the same handoff. A `path` cell that is absolute or that carries a `..`
is refused rather than resolved: a hand-edited `labels.csv` cannot send a run outside the
label root the configuration gave it.

**The handoff must be one a labeling run wrote without `--out`, and nothing here can check
that.** `blobcount/labeling.py:584` writes `target.relative_to(out)` and this project's
`bin/label_patches.py:205` sets `out` to `--out` when it is given one, so a handoff from
`label_patches --out somewhere` carries paths relative to *that* directory: every row here
resolves against the wrong root and the run dies on the first patch with a missing-file
`OSError` naming a path that looks right and is not. **Run `bin/label_patches.py` without
`--out`, or point `paths.labels` at the directory you passed to it.** The format is Task 6's
and is not changed here; this is an inherited precondition, stated in the module docstring,
in `read_labels`'s own docstring and here.

**`training.pretrained` is `true` in `configs/default.yaml`, so a run that finds a cohort
will download the ImageNet weights the backbone was fitted with.** That is the shipped
behaviour and it is not changed here; it is stated because it is the one thing this
command line does that is not local, and because `tests/test_train.py` overrides the key
and points `TORCH_HOME` at an empty directory to show that a run built the other way
downloads nothing.

**Two outcomes, two codes.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration, the
handoff or the output target and never about this program. Fourteen reach it, in four
groups, and nothing this program does does:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two: the
  file `load` refuses — missing, unreadable, not a mapping, a key the code does not
  declare, a required key the file omits, or a value for a declared key whose type this
  project does not read it as — and a `paths.results` whose *value* `Config.path` refuses
  because it is empty. `paths.results` is resolved inside the arm that catches this, for
  the reason `bin/label_patches.py` gives: `load` types it as the declared `str` and an
  empty one is a value question that stays here.
- **The handoff**, as a `ConfigError`, an `OSError`, a `csv.Error` or a `ValueError`: a
  file that is not there or cannot be opened; a cell longer than `csv.field_size_limit`,
  which `csv.DictReader` raises rather than truncating; a header that does not carry every
  column in `blobcount.labeling.LABEL_COLUMNS`; a row with no cell for one of the three
  columns this run reads; a `label` cell that is not a whole number; a `path` cell that is
  absolute or that leaves the label root; and a `paths.labels` whose value `Config.path`
  refuses, which arrives out of `read_labels` because the reader is what resolves every row
  against it. Each of the four row-level refusals names the file, the line and what was
  wrong with the cell, because a hand-edited handoff is otherwise a run that fails on the
  second specimen.
- **The output target**, as this program's own check, and it is the same one
  `bin/ablate_threshold.py` makes: a target that exists and is a file rather than a
  directory, and a target that is not below a root this project owns — the checkout, or
  the parent of a configured `paths.*` value. **A run writes one
  45 MB checkpoint per fold into whatever path it is given**, so a `--out` typo or a
  hand-edited `paths.results` is how a study's weights end up somewhere this repository is
  not responsible for. Nothing is deleted — this run clears no tree, which is a smaller
  blast radius than `bin/label_patches.py`'s and not a reason to be the one entry point
  that does not check. The predicate is `blobcount.extract.is_owned_path`, so "is this
  ours" has one implementation in this project and not one per guard, and it follows the
  configuration rather than the checkout alone, which is what lets a run write to a
  `paths.*` the data move put outside the repository. The check runs before
  the handoff is read so a mistyped target costs a message rather than a training run.
- **The cohort and the model**, as a `SplitError` or a `ValueError`: a specimen column
  with fewer than two distinct ids, or one that is a sequence of characters rather than of
  ids, is `blobcount.splits.loso_folds`'s refusal and reaches here as itself; a `label`
  outside `range(training.num_classes)` is `blobcount.data.class_weights`'s; and a
  `training.architecture` this project does not build, or a `training.num_classes` no head
  can be built for, is `blobcount.models.build_model`'s. All three modules raise stdlib
  types because none of them declares a project exception type, and this file is where
  catching them belongs.

All fourteen are printed on stderr behind this program's name, and none of them is a
traceback.

**There is no `EXIT_CONTRADICTORY` here, and that is a decision rather than an omission.**
The two sibling entry points define `1` for a study-level failure — a slide that yielded no
patches, a labelled tree that contradicted itself — and both are states a run *completes*
with. A cross-validation completes or refuses: every fold either trains and reports or
raises, and there is no third outcome it can arrive at having finished. So this file
defines two codes, and a status of 1 means the interpreter reported a traceback, which is a
fact about this project's code rather than about the study. `main`'s *return value* is the
contract, and an exception never returns one.

**A bad argument is not one of the fourteen, and not one of this program's outcomes at all.**
`argparse` refuses an option this program does not declare, one given no value, and one
given no `--labels`, by printing its own message and calling `sys.exit(2)` inside
`parse_args`, before the first statement of `main` runs. `--labels` and `--out` are both
typed `Path` and no string fails that conversion, so a mistyped path is not one of these:
it reaches `main` and is refused below as a path, which is where its judgement belongs. The
parser's status is the same number as `EXIT_REFUSED` and no code can tell the two apart, so
the separation is on stderr, where the parser names the argument and its usage and none of
the four names either.

**A patch file that is not there is not one of the fourteen, and is deliberate.**
`blobcount.data.PatchDataset` opens one patch and refuses nothing about it, so a row
naming a file that does not exist raises the `OSError` out of the loader, and that error
names the full resolved path — which is the whole fact a caller needs and better than a
sentence about a rule. `blobcount.data` states that ruling and this file is on the other
side of it; the alternative would be a catch wide enough to hold every `OSError` a fold
can raise, which would report a full disk as a missing patch.

**Two tables are printed — the selected epoch of each fold, then the mean of those selected
epochs — and a closing sentence that says what the numbers are.** The per-fold table comes
first because the mean table is a mean *of it* and a reader who has seen a mean of 0.9 and
no fold behind it has been told the wrong thing by the mean alone. Every row of both tables
names the specimen it is about, because a fold number means nothing to a reader who does
not know which patient it was, and every metric of the mean table carries its population
standard deviation beside it. **The first table is rendered out of `fold_summary.csv` and the
second out of the summary dict**, which is what makes `tests/test_train.py` assert the
printed numbers against the file and the dict they came from rather than against the
characters this file formats: a swapped column is a misreport of the study that leaves
every artifact on disk correct, and a test that asserts a column *name* cannot see it. The
run then writes `training_summary.json` into `out_dir`, holding the same summary the mean
table prints and the selected record of every fold.

**The numbers here are from synthetic patches.** The extraction that would produce a
`labels.csv` has never been run on this cohort, so nothing in `results/` has been
regenerated by this command line and nothing this file reports about the study is a
measurement of it. `blobcount.train`'s docstring carries the same statement.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from blobcount.config import PROJECT_ROOT, ConfigError, load
from blobcount.extract import is_owned_path
from blobcount.splits import SplitError
from blobcount.train import read_labels, run_cross_validation

EXIT_OK = 0
EXIT_REFUSED = 2

# Where the artifacts go and what they are called, under `out_dir`. One place, so the
# writer and the tests that read them back name the same file.
_SUMMARY_NAME = "training_summary.json"

_FOLD_COLUMNS = ("fold", "specimen", "epoch", "accuracy", "precision", "recall", "f1", "lr")
_SUMMARY_COLUMNS = ("metric", "mean", "population std")

# The four counters the JSON retypes as `int` and the six numbers it retypes as `float`,
# written here rather than as a rule over the column names: the two sets are disjoint and
# cover the record, and a column added to `EpochRecord` that is in neither set reaches the
# document as the string the CSV holds.
_INT_FIELDS = ("fold", "epoch", "tp", "fp", "fn", "tn")
_FLOAT_FIELDS = ("train_loss", "val_loss", "val_f1", "val_accuracy", "precision", "recall", "lr")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train the cohort leave-one-specimen-out and report the epoch each "
        "fold selected on validation F1.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        metavar="PATH",
        help="the labels.csv written by bin/label_patches.py",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="output directory for the per-fold logs, the checkpoints and the summary; "
        "paths.results from the configuration otherwise. The directory is not cleared, so "
        "a previous run's artifacts stay beside this one's.",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/ablate_threshold.py` gives: this is eight
    columns of two integers, one specimen name and five floats, and `tabulate` is not a
    dependency this project declares. The siblings' `_print_table` is not imported, for
    the reason `bin/ablate_threshold.py` states for its own: `bin/` is a directory of
    scripts loaded by path rather than a package, and reaching into one script's private
    name would couple two entry points that are meant to be read one at a time.
    """
    widths = [
        max([len(str(header))] + [len(str(row[index])) for row in rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _refuse_output_target(target: Path) -> str | None:
    """Return why this program will not write its artifacts into `target`, or None.

    **The same two refusals `bin/ablate_threshold.py` makes, for the same reason, on one
    path rather than two.** That program writes two files into a directory it does not own
    and tests both the results directory and the subdirectory it puts them in; this one
    writes into a single directory and deletes nothing in it, so the blast radius is
    smaller — but a run writes one 45 MB checkpoint per fold into whatever path it is given,
    and a `--out` typo or a hand-edited `paths.results` is how 135 MB lands somewhere this
    repository is not responsible for. "Smaller" is not a reason for this entry point to be
    the one that does not check.

    **The file test runs before the containment test, so a file is reported as the typo it
    is.** A `paths.results` that is a regular file makes `out_dir` un-creatable — the
    `mkdir` would raise `NotADirectoryError` — and the project root itself is not
    strictly below itself, which `_is_below` answers rather than a string comparison
    here: "is this ours" has one implementation in this project and not one per guard,
    and the two entry points that differ in what they *do* to a target do not differ in
    what they decide is a target.

    **The containment test asks `is_owned_path` rather than `_is_below` against the
    checkout alone.** The roots a target must lie under are the ones the project owns —
    the checkout, plus the parent of every configured `paths.*` value, the set
    `blobcount.extract.allowed_roots` derives from the configuration at call time —
    because a path the configuration names is by definition one this project owns, and
    a rule written against the checkout alone would refuse the destinations the data
    move put outside the repository. The predicate stays strict, so the project root
    itself and a directory *beside* an owned one are both refusals, which is the
    mistyped `--out` this check exists to catch.

    The check returns a message and raises nothing, and it runs before the handoff is read,
    so a mistyped `--out` costs a message rather than a training run.
    """
    if target.exists() and not target.is_dir():
        return (
            f"refusing to write {target}: it exists and is a file rather than a directory, "
            f"so it is a typo in --out or in paths.results and not a previous run"
        )
    if not is_owned_path(target):
        return (
            f"refusing to write {target}: it is not strictly below the project root "
            f"{PROJECT_ROOT} or the parent of a configured paths.* value, so it is "
            f"not a directory this project owns"
        )
    return None


def _document(summary: dict, per_fold: Sequence[dict], labels: Path) -> dict:
    """Return the JSON document this run writes.

    UTF-8 with `ensure_ascii=False` and specimen names as values, for the reasons
    `extract.write_index` and `labeling._write_labels` give: a specimen name has to survive
    a Turkish-locale machine unchanged.

    `per_fold` is the selected record of each fold and `summary` the `summarise_folds`
    output over exactly those records, so the file is the mean table and the rows behind it
    in one artifact and cannot be read as though the two came from different epochs. The
    handoff the run trained from is named beside them: a summary whose mean is a number
    without the cohort it is a mean over cannot be compared with anything.
    """
    return {
        "labels": str(labels),
        "summary": summary,
        "per_fold": list(per_fold),
    }


def _read_csv(path: Path) -> list[dict]:
    """Return the rows of a file this run wrote, as the strings it wrote them as.

    The per-fold table is printed from `fold_summary.csv` rather than from the summary
    dict, so that the table and the file a reader will open afterwards cannot disagree — one
    is rendered out of the other. A file this run wrote is UTF-8 with no line ending of its
    own, which is what `blobcount.train._write_rows` wrote it as.
    """
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _records(rows: Sequence[dict]) -> list[dict]:
    """Return the selected rows as the numbers they hold rather than as strings.

    The JSON is a document a downstream reader parses, and `{"val_f1": "0.90"}` is a string
    where a number belongs. The types are the ones `EpochRecord` declares and
    `blobcount.train` wrote them as, which is the only place that answer is known.
    """
    return [
        {
            **{name: int(row[name]) for name in _INT_FIELDS},
            **{name: float(row[name]) for name in _FLOAT_FIELDS},
            "held_out": row["held_out"],
        }
        for row in rows
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run cross-validation over the handoff the arguments name and return the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        # Resolved inside the arm, for the reason `bin/label_patches.py` gives: `load`
        # types this key as the declared `str`, and an empty one is a value it makes no
        # claim about, so `Config.path` is what refuses it and outside this arm that refusal
        # would leave a traceback. `paths.labels` is **not** resolved here: `read_labels`
        # reads it, which is what puts that read under `blobcount/` where the ledger scan
        # in `tests/test_config.py` can see it.
        out_dir: Path = args.out if args.out is not None else cfg.path("paths.results")
    except ConfigError as error:
        # The two shapes this arm carries: the file `load` refuses, and a `paths.results`
        # `Config.path` refuses. The override the parser produced cannot be the cause —
        # `--labels` and `--out` have already been typed as `Path`.
        print(f"train: {error}", file=sys.stderr)
        return EXIT_REFUSED

    refused = _refuse_output_target(out_dir)
    if refused is not None:
        # Before the handoff is read and before a model is built, so a mistyped `--out`
        # costs a message rather than a training run. `EXIT_REFUSED` and not
        # `EXIT_CONTRADICTORY`: nothing here finished, so there is no run whose outcome is
        # being reported, and the two codes mean different things to whoever reads them.
        print(f"train: {refused}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        paths, labels, specimen_ids = read_labels(args.labels, cfg)
    except (ConfigError, OSError, csv.Error, ValueError) as error:
        # The handoff, as the module docstring groups it: a file that is not there or
        # cannot be opened, a cell longer than `csv.field_size_limit`, a header that is not
        # the handoff's, a row with no cell for a column this run reads, a `label` that is
        # not a whole number, and a `path` that leaves the label root. `ConfigError` is in
        # this tuple for one shape, not because the arm is wide: `read_labels` resolves
        # every row against `paths.labels`, so `Config.path`'s refusal of an empty value
        # arrives out of it, and it is an answer about the configuration in a place where
        # the configuration is what was being read. The catch is around this one call, so
        # it is too narrow to hold a defect in the loop below.
        print(f"train: {error}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        summary = run_cross_validation(paths, labels, specimen_ids, cfg, out_dir)
    except (SplitError, ValueError) as error:
        # The cohort and the model, as the module docstring groups them: fewer than two
        # distinct specimens, or a specimen column that is a sequence of characters rather
        # than of ids, is `blobcount.splits.loso_folds`'s own refusal; a label outside
        # `range(training.num_classes)` is `blobcount.data.class_weights`'s; and a
        # `training.architecture` this project does not build, or a `training.num_classes`
        # no head can be built for, is `blobcount.models.build_model`'s. All three are
        # stdlib types because none of those modules declares a project exception type, and
        # this file is where catching them belongs. The catch is around this one call, so a
        # `ValueError` from anything inside the loop itself — an arithmetic error in the
        # metrics, a write that failed — is not caught here and reaches the interpreter as a
        # traceback, which is the honest report of a defect in this project.
        print(f"train: {error}", file=sys.stderr)
        return EXIT_REFUSED

    rows = _read_csv(out_dir / "fold_summary.csv")
    # The table is printed out of `fold_summary.csv` rather than out of the summary dict,
    # and its headers say `accuracy` and `f1` where the file's columns say `val_accuracy`
    # and `val_f1`: a table names the metric and a record names the epoch's field, and the
    # two spellings are not reconciled because reconciling them would put a word on the
    # file's header that `EpochRecord` does not have.
    _print_table(_FOLD_COLUMNS, [
        [
            row["fold"], row["held_out"], row["epoch"],
            f"{float(row['val_accuracy']):.3f}", f"{float(row['precision']):.3f}",
            f"{float(row['recall']):.3f}", f"{float(row['val_f1']):.3f}",
            f"{float(row['lr']):.6f}",
        ]
        for row in rows
    ])
    print()
    _print_table(_SUMMARY_COLUMNS, [
        [name, f"{summary['mean_' + name]:.3f}", f"{summary['mean_' + name + '_std']:.3f}"]
        for name in ("accuracy", "precision", "recall", "f1")
    ])
    (out_dir / _SUMMARY_NAME).write_text(
        json.dumps(
            _document(summary, _records(rows), args.labels),
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    selected = ", ".join(f"{row['held_out']}@{row['epoch']}" for row in rows)
    print(
        f"\n{summary['n_folds']} fold(s) trained over "
        f"{len(specimen_ids)} patches of {len(set(specimen_ids))} specimen(s); the table "
        f"above is the epoch each fold selected on validation F1 ({selected}), and the "
        f"mean is of those epochs only — the epochs a fold did not select are in "
        f"{out_dir / 'training_log_fold<n>.csv'}"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())