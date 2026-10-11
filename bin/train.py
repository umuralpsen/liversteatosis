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
left where the last run put it.

**The `path` cell of a row is resolved against `paths.labels`, not against the directory
the `labels.csv` sits in and not against the working directory.** This is the ruling
`blobcount.labeling` writes those cells under — a path relative to the label root,
separated by `/` whatever the platform — so a row resolves to the patch that root names,
and a run started from another directory resolves the same rows to the same files. A
`path` cell that is absolute or that carries a `..` is refused rather than resolved: a
hand-edited `labels.csv` cannot send a run outside the label root the configuration gave
it.

**`training.pretrained` is `true` in `configs/default.yaml`, so a run that finds a cohort
will download the ImageNet weights the backbone was fitted with.** That is the shipped
behaviour and it is not changed here; it is stated because it is the one thing this
command line does that is not local, and because `tests/test_train.py` overrides the key
and points `TORCH_HOME` at an empty directory to show that a run built the other way
downloads nothing.

**Two outcomes, two codes.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration or the
handoff and never about this program. Eleven reach it, in three groups, and nothing this
program does does:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two: the
  file `load` refuses — missing, unreadable, not a mapping, a key the code does not
  declare, a required key the file omits, or a value for a declared key whose type this
  project does not read it as — and a `paths.labels` or a `paths.results` whose *value*
  `Config.path` refuses because it is empty. The path keys are resolved inside the arm
  that catches this, for the reason `bin/label_patches.py` gives: `load` types them as the
  declared `str` and an empty one is a value question that stays here.
- **The handoff**, as an `OSError`, a `csv.Error` or a `ValueError`: a file that is not
  there or cannot be opened; a cell longer than `csv.field_size_limit`, which
  `csv.DictReader` raises rather than truncating; a header that does not carry every
  column in `blobcount.labeling.LABEL_COLUMNS`; a row with no cell for one of the three
  columns this run reads; a `label` cell that is not a whole number; and a `path` cell
  that is absolute or that leaves the label root. Each of the last four names the file,
  the line and what was wrong with the cell, because a hand-edited handoff is otherwise a
  run that fails on the second specimen.
- **The cohort and the model**, as a `SplitError` or a `ValueError`: a specimen column
  with fewer than two distinct ids, or one that is a sequence of characters rather than of
  ids, is `blobcount.splits.loso_folds`'s refusal and reaches here as itself; a `label`
  outside `range(training.num_classes)` is `blobcount.data.class_weights`'s; and a
  `training.architecture` this project does not build, or a `training.num_classes` no head
  can be built for, is `blobcount.models.build_model`'s. All three modules raise stdlib
  types because none of them declares a project exception type, and this file is where
  catching them belongs.

All eleven are printed on stderr behind this program's name, and none of them is a
traceback.

**There is no `EXIT_CONTRADICTORY` here, and that is a decision rather than an omission.**
The two sibling entry points define `1` for a study-level failure — a slide that yielded no
patches, a labelled tree that contradicted itself — and both are states a run *completes*
with. A cross-validation completes or refuses: every fold either trains and reports or
raises, and there is no third outcome it can arrive at having finished. So this file
defines two codes, and a status of 1 means the interpreter reported a traceback, which is a
fact about this project's code rather than about the study. `main`'s *return value* is the
contract, and an exception never returns one.

**A bad argument is not one of the eleven, and not one of this program's outcomes at all.**
`argparse` refuses an option this program does not declare, one given no value, and one
given no `--labels`, by printing its own message and calling `sys.exit(2)` inside
`parse_args`, before the first statement of `main` runs. `--labels` and `--out` are both
typed `Path` and no string fails that conversion, so a mistyped path is not one of these:
it reaches `main` and is refused below as a path, which is where its judgement belongs. The
parser's status is the same number as `EXIT_REFUSED` and no code can tell the two apart, so
the separation is on stderr, where the parser names the argument and its usage and none of
the four names either.

**A patch file that is not there is not one of the four, and is deliberate.**
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
standard deviation beside it. The run then writes `training_summary.json` into `out_dir`,
holding the same summary the mean table prints and the selected record of every fold.

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
from pathlib import Path, PurePosixPath

from blobcount.config import ConfigError, load
from blobcount.labeling import LABEL_COLUMNS
from blobcount.splits import SplitError
from blobcount.train import run_cross_validation

EXIT_OK = 0
EXIT_REFUSED = 2

# The three columns of `labels.csv` this command line reads: the specimen a patch came
# from, the class it was labelled, and where the patch is. The other three — `patch_id`,
# `blob_count` and `mpp` — belong to the stages that wrote the file and to the artifact's
# auditability rather than to this run; the header is still required to carry all six, for
# the reason `blobcount.labeling.read_index` requires all of the index's: a truncated or
# reordered handoff is a defect in the labeling run that produced it, and it belongs on the
# file that has it rather than on the training that read it.
_READ_COLUMNS = ("slide_id", "label", "path")

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


def _row(index: Path, line: int, row: dict, label_root: Path) -> tuple[str, Path, int]:
    """Return `(specimen, patch path, label)` for one `labels.csv` row, or refuse it.

    **Three refusals, each naming the file, the line and the cell**, because a handoff is a
    file a user can edit and a run that stops on the second specimen with a message naming
    neither is the failure this command line exists not to have. A missing cell is a short
    line — `csv.DictReader` fills the absent columns with `None` — a `label` that is not a
    whole number is a quoted digit or a class name where a class index belongs, and a `path`
    that leaves the label root is a hand-edited cell that would otherwise send the run
    outside the directory the configuration named.

    **The class range is not checked here.** A `label` of `5` under a two-class run is
    refused by `blobcount.data.class_weights` in the fold that holds that row, with the one
    message this project has for a label against a class count; a second rule written here
    would be a second answer to a question `blobcount.data` already answers.

    The row is read from `label_root` rather than from the directory the file is in, and a
    `path` cell is a POSIX-separated relative path because that is what
    `blobcount.labeling._write_labels` writes — `PurePosixPath` is what turns those
    separators into this platform's on a machine whose filesystem separator is not `/`.
    """
    for name in _READ_COLUMNS:
        if not row.get(name):
            # No field count in the message: `csv.DictReader` has already filled the absent
            # columns with `None`, so the row is indistinguishable here from one whose cell
            # is present and empty, and a count would have to be the header's rather than
            # the row's.
            raise ValueError(
                f"{index} line {line} carries no {name!r} cell: the cell is empty or the "
                f"row has fewer fields than the header, which carries "
                f"{', '.join(LABEL_COLUMNS)}. Fix the row, or the labeling run that wrote it"
            )
    try:
        label = int(row["label"])
    except ValueError as error:
        raise ValueError(
            f"{index} line {line} has label {row['label']!r}, which is not a whole number: "
            f"the label column holds the class a labelling run wrote for a patch — 0 for "
            f"normal and 1 for steatosis — and a quoted digit, a class name or a fraction "
            f"reaches CrossEntropyLoss as a target it cannot score, which reports the "
            f"mistake against the loss rather than against the cell"
        ) from error
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(
            f"{index} line {line} has path {row['path']!r}, which is not a path inside "
            f"the label root: blobcount.labeling writes each row's path relative to "
            f"paths.labels, so an absolute path or one holding '..' would send this run "
            f"outside the directory the configuration gave it. Fix the cell, or re-run the "
            f"labeling that wrote it"
        )
    return row["slide_id"], label_root / relative, label


def _read_labels(index: Path, label_root: Path) -> tuple[list[Path], list[int], list[str]]:
    """Return `(paths, labels, specimen_ids)` for the handoff at `index`, in row order.

    The three come back in dataset order and one entry per row, which is the order
    `blobcount.splits.loso_folds` indexes into and the order `PatchDataset` pairs each path
    with its own label. Read as UTF-8 with `newline=""` for the reasons
    `blobcount.labeling.read_index` gives: a slide id in this project is a specimen name and
    has to survive a Turkish-locale machine, and `csv` writes its own line endings.

    The header must carry every name in `blobcount.labeling.LABEL_COLUMNS`, checked here
    rather than left to a `KeyError` on the first row, so that a truncated or reordered
    handoff is reported as the wrong file it is.
    """
    target = Path(index)
    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        missing = [name for name in LABEL_COLUMNS if name not in header]
        if missing:
            raise ValueError(
                f"{target} is not a labels.csv: its header carries "
                f"{', '.join(header) or 'nothing'}, and a labelling run writes "
                f"{', '.join(LABEL_COLUMNS)}. Missing: {', '.join(missing)}. Fix the file, "
                f"or pass the labels.csv another labelling run wrote"
            )
        paths: list[Path] = []
        labels: list[int] = []
        specimens: list[str] = []
        for line, row in enumerate(reader, start=2):
            specimen, path, label = _row(target, line, row, label_root)
            specimens.append(specimen)
            paths.append(path)
            labels.append(label)
    return paths, labels, specimens


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
        # types these two keys as the declared `str`, and an empty one is a value it makes
        # no claim about, so `Config.path` is what refuses it and outside this arm that
        # refusal would leave a traceback.
        label_root: Path = cfg.path("paths.labels")
        out_dir: Path = args.out if args.out is not None else cfg.path("paths.results")
    except ConfigError as error:
        # The two shapes this arm carries: the file `load` refuses, and a `paths.labels` or
        # a `paths.results` `Config.path` refuses. The override the parser produced cannot
        # be the cause — `--labels` and `--out` have already been typed as `Path`.
        print(f"train: {error}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        paths, labels, specimen_ids = _read_labels(args.labels, label_root)
    except (OSError, csv.Error, ValueError) as error:
        # The handoff, as the module docstring groups it: a file that is not there or
        # cannot be opened, a cell longer than `csv.field_size_limit`, a header that is not
        # the handoff's, a row with no cell for a column this run reads, a `label` that is
        # not a whole number, and a `path` that leaves the label root. The catch is around
        # this one read, so it is too narrow to hold a defect in the loop below.
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