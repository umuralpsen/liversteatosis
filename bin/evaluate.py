"""Score a checkpoint over held-out specimens and report the metrics with their intervals.

The fifth entry point, after extraction, labeling, the threshold sweep and training. It
reads the `labels.csv` a previous `label_patches` run wrote, builds the backbone the
configuration names, loads a checkpoint into it, scores the specimens the caller names, and
writes the two artifacts and the printed table that say what that model did on patients it
had never seen.

    python bin/evaluate.py --specimens A --specimens B --checkpoint results/best_model_fold0.pth \
                           --train-specimens C --train-specimens D

`--labels` names the handoff and `--checkpoint` the weights, and both are required.
**`--train-specimens` is required and repeatable, and that is the point of this program
rather than a formality of its parser**: a number computed over a specimen the model was
trained on is not a measurement of anything, and the one place a reader is least likely to
notice is the moment the run reports it. A study has three specimens, a fold trains on two of
them and validates on the third, and a run that scores all three is the easiest mistake to
make with a command that takes one list and not two. So the training cohort is not inferred
from the checkpoint's file name nor from the specimens left over in the handoff — a caller
states it, and `blobcount.evaluate.assert_disjoint` refuses the run when the two lists share a
name. **`--selection-cohort` is required for the same class of reason and is a separate
question from both**: it names the specimen whose validation F1 chose the epoch this
checkpoint holds, it is written into `medical_metrics.json` as `checkpoint_selection_cohort`
beside `train_specimens`, and **it is never inferred**. This program loads a `state_dict`; a
checkpoint carries no specimen id, and the value is not recoverable from anything else it is
given. The tempting inference — the checkpoint is `best_model_fold0.pth`, so fold 0 held out
the first specimen — is a filename convention, and it is exactly the case (fold 0's own
held-out specimen, which is also the cohort this program scores) where a reader would not
notice it being wrong. **`--specimens` is not required, and its default is every specimen the
handoff holds**, which is the cohort a study run means; the guard applies to the default
exactly as it does to a named list. **A specimen the caller names and the handoff holds no row
of is dropped from the run and named in the report**: `medical_metrics.json`'s
`evaluated_specimens` is the cohort that was actually scored, its `requested_specimens` is what
was asked for, and its `not_evaluated` names the difference, and the closing sentence says it
too. The provenance record of a leave-one-specimen-out study has to be the cohort the numbers
are about and not the one a command line hoped for.

**The handoff is read by `blobcount.train.read_labels`, not here.** `bin/` is a directory of
scripts loaded by path rather than a package, so a copy of the CSV contract in this file would
be a second set of answers to it, and a read of `paths.labels` here would be outside the
source scan in `tests/test_config.py`, which would then keep reporting a declared key nothing
under `blobcount/` reads. **Its precondition is inherited and not restated as a rule here: a
handoff from `bin/label_patches.py --out somewhere` carries paths relative to *that*
directory**, so every row resolves against the wrong root and the run dies on the first patch
with a missing-file `OSError` naming a path that looks right and is not. Run
`bin/label_patches.py` without `--out`, or point `paths.labels` at the directory you passed it.

**The backbone is built with `pretrained=false` whatever the configuration says**, because a
checkpoint replaces every parameter of the model a moment later: `training.pretrained: true`
is the shipped value, and honouring it here would download the ImageNet weights of a backbone
whose weights are about to be overwritten, over the network, on every run.
`tests/test_evaluate.py` holds the argument this factory is handed and checks that an empty
`TORCH_HOME` stays empty.

**The artifacts go to `paths.results / "evaluation"`, and nothing here is cleared.**
`medical_metrics.json` holds the pooled metrics, the per-specimen records, the interval every
rate was drawn at with the conditions it was drawn under, the two lists that were scored and
trained, and a sentence saying where its numbers came from; `per_specimen_metrics.csv` holds
one row per specimen with its own counts, so the pooled numbers can be recomputed from the rows
behind them rather than taken on trust. A file of the same name is rewritten in place, and a
file a run with fewer specimens does not produce stays where the last run put it. **The output
directory is guarded before anything is read**, for the reason the group below gives.

**Two outcomes, two codes.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration, the handoff,
the cohort, the checkpoint or the output target, and never about this program. **Six sites in
`main` produce one**, and the bullets below name what each one carries:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two: the file
  `load` refuses — missing, unreadable, not a mapping, a key the code does not declare, a
  required key the file omits, or a value for a declared key whose type this project does not
  read it as — and a `paths.results` whose *value* `Config.path` refuses because it is empty.
  `paths.results` is resolved inside the arm that catches this, for the reason
  `bin/label_patches.py` gives: `load` types it as the declared `str` and an empty one is a
  value question that stays here.
- **The output target**, as this program's own check, and it is the same pair of refusals
  `bin/ablate_threshold.py` makes against both `paths.results` and the subdirectory it puts
  them in: a target that exists and is a file rather than a directory, and a target that is
  not below a root this project owns — the checkout, or the parent of a configured
  `paths.*` value. **Both paths are tested because both can be the one
  that is wrong** — a `paths.results` that is a regular file makes `results/evaluation`
  un-creatable, and a `paths.results: .` leaves `evaluation/` below the root while putting the
  results directory *at* it, so a guard on the subdirectory alone would write a JSON and a CSV
  into the repository. The predicate is `blobcount.extract.is_owned_path`, so "is this ours" has
  one implementation in this project and not one per guard, and it follows the configuration
  rather than the checkout alone, which is what lets a run write to a `paths.*` the data move
  put outside the repository. The check runs before the
  handoff is read and before a checkpoint is loaded, so a mistyped `paths.results` costs a
  message rather than a run.
- **The handoff**, as a `ConfigError`, an `OSError`, a `csv.Error` or a `ValueError`: a file
  that is not there or cannot be opened; a cell longer than `csv.field_size_limit`, which
  `csv.DictReader` raises rather than truncating; a header that does not carry every column in
  `blobcount.labeling.LABEL_COLUMNS`; a row with no cell for one of the three columns this run
  reads; a `label` cell that is not a whole number; a `path` cell that is absolute or that
  leaves the label root; and a `paths.labels` whose value `Config.path` refuses, which arrives
  out of `read_labels` because the reader is what resolves every row against it.
- **The cohort**, as this program's own check: `--specimens` naming a specimen the handoff
  holds no row of, or every specimen it names being one of those. The run would otherwise
  score nothing, report six `0.0` rates beside an `auc` of `None` and call it a result — which
  is the `n == 0` case `blobcount.evaluate` deliberately does not refuse, because a rule about
  a cohort belongs to the caller that has one and this is the caller that can say the caller's
  name was wrong.
- **The checkpoint and the backbone**, as an `OSError`, a `pickle.UnpicklingError`, a
  `RuntimeError` or a `ValueError`: a checkpoint that is not there; one that is not a torch
  archive, which `torch.load` refuses with `UnpicklingError` — measured, its bases are
  `PickleError, Exception, BaseException, object`, so it is neither an `OSError` nor a
  `ValueError` and an arm holding those two would let a `--checkpoint` naming the wrong file
  reach the caller as a traceback out of `main`; one whose weights do not fit the head the
  configuration describes, which `load_state_dict` refuses with `RuntimeError: Error(s) in
  loading state_dict ... size mismatch`; and a `training.architecture` this project does not
  build, or a `training.num_classes` no head can be built for, which is
  `blobcount.models.build_model`'s own `ValueError`.
- **The integrity guard**, as this program's own `EvaluationIntegrityError`, caught in its own
  arm. That type is declared by `blobcount.evaluate` and by nothing else, so catching it says
  "these two cohorts overlap" and says nothing else — the same reason
  `bin/ablate_threshold.py` catches `PlotError` rather than `ValueError`, which every figure
  library also raises. **Nothing is written before it runs**, so a refused run leaves no
  artifact a reader could mistake for a result.

All six are printed on stderr behind this program's name, and none of them is a traceback.

**There is no `EXIT_CONTRADICTORY` here, and that is a decision rather than an omission.**
The two sibling entry points that define `1` do so for a study-level failure — a slide that
yielded no patches, a labelled tree that contradicted itself — and both are states a run
*completes* with. An evaluation has no such state: it reports the metrics of a cohort it was
asked about, and a cohort whose metrics are bad is the finding the run exists to report, not a
contradiction to be refused. So this file defines two codes, and a status of 1 means the
interpreter reported a traceback, which is a fact about this project's code rather than about
the study. `main`'s *return value* is the contract, and an exception never returns one.

**A bad argument is not one of the six, and not one of this program's outcomes at all.**
`argparse` refuses an option this program does not declare, one given no value, a
`--checkpoint` given no path and a run given no `--train-specimens`, by printing its own
message and calling `sys.exit(2)` inside `parse_args`, before the first statement of `main`
runs. `--labels` and `--checkpoint` are both typed `Path` and no string fails that conversion,
so a mistyped path is not one of these: it reaches `main` and is refused below as a path,
which is where its judgement belongs. The parser's status is the same number as `EXIT_REFUSED`
and no code can tell the two apart, so the separation is on stderr, where the parser names the
argument and its usage and none of the six names either.

**A patch file that is not there is not one of the six, and is deliberate.**
`blobcount.data.PatchDataset` opens one patch and refuses nothing about it, so a row naming a
file that does not exist raises the `OSError` out of the loader, and that error names the full
resolved path — which is the whole fact a caller needs and better than a sentence about a
rule. `blobcount.data` states that ruling and this file is on the other side of it.

**One table is printed — the six rates and the AUC, each with its interval — and a closing
sentence that says what the numbers are.** The table is rendered out of the same
`medical_metrics.json` this run writes, so the printed report and the artifact cannot disagree;
`tests/test_evaluate.py` asserts every printed cell against the document. A metric with no
interval prints `n/a` in both bound columns rather than a short row, because a row of a
different width is a table a reader has to re-align. **The closing sentence names the file that
says where the numbers came from rather than saying it here**, because this program cannot
check what it was pointed at: an artifact that asserts `synthetic` about its own input would be
wrong the day the extraction runs. `blobcount.evaluate`'s docstring carries the same statement
about this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from blobcount.config import PROJECT_ROOT, Config, ConfigError, load
from blobcount.evaluate import (
    EvaluationIntegrityError,
    Intervals,
    Metrics,
    bootstrap_metric_intervals,
    evaluate_specimens,
)
from blobcount.extract import is_owned_path
from blobcount.models import build_model
from blobcount.train import read_labels

EXIT_OK = 0
EXIT_REFUSED = 2

# Where the artifacts go and what they are called, under `paths.results`. One place, so the
# writers and the tests that read them back name the same files.
_EVALUATION_DIR = "evaluation"
_METRICS_NAME = "medical_metrics.json"
_PER_SPECIMEN_NAME = "per_specimen_metrics.csv"

# The rows of the printed table, in order. The AUC is last because it is the one that may be
# `None`, and a reader who has read six rates before meeting a missing one is not surprised by
# it. The interval conditions are not in this list: they are in `medical_metrics.json`, and
# this program does not read either `evaluation.*` key — `blobcount.evaluate` does, once, and
# hands them back with the intervals.
_TABLE_ROWS = ("accuracy", "precision", "recall", "specificity", "f1", "npv", "auc")

# The columns `per_specimen_metrics.csv` carries: the specimen name beside the record's own
# fields, so a row is about a patient and not about four integers.
_PER_SPECIMEN_COLUMNS = ("specimen", *(name for name in Metrics.__dataclass_fields__))

# What a cell that is not a number says, in the table. `None` is written nowhere: a JSON
# `null` is a fact a reader can act on, the string `"None"` in a CSV is a `ValueError` for
# every parser downstream, and the table's answer is this.
_ABSENT = "n/a"

# The sentence the artifact carries about where its numbers came from, in the artifact rather
# than in this file's docstring, where a reader of the file alone would not see it.
# **It states what this run read and what those labels are, and nothing about the state of
# the repository**: an earlier draft asserted that no handoff had ever been written and no
# checkpoint ever trained, which was true until the first real run and false the moment it
# happened. A run that asserts "synthetic" about its own input would be making a claim it
# cannot check, and would be wrong the day the extraction runs — the same reasoning, turned
# on the sentence that coined it.
_PROVENANCE = (
    "computed from the patches named by `labels` and the weights named by `checkpoint`, so "
    "every number in this document is a function of those two inputs and of the cohorts "
    "`evaluated_specimens` and `train_specimens` record. Where `labels` is the handoff "
    "bin/label_patches.py writes, its labels are morphological blob counts made at the "
    "configured threshold and not a pathologist's grade, so the ceiling of these numbers is "
    "agreement with that counter: they are a pipeline verification, not a clinical cohort "
    "result"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate",
        description="Score a checkpoint over held-out specimens and report the metrics with "
        "their bootstrap intervals.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        metavar="PATH",
        help="the labels.csv written by bin/label_patches.py",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="PATH",
        help="the weights written by bin/train.py, one results/best_model_fold<n>.pth",
    )
    parser.add_argument(
        "--specimens",
        action="append",
        default=None,
        metavar="ID",
        help="a specimen to score; repeatable. Every specimen the handoff holds when this is "
        "given no value.",
    )
    parser.add_argument(
        "--train-specimens",
        action="append",
        required=True,
        metavar="ID",
        help="a specimen the model was trained on; repeatable and required, because a number "
        "computed over a trained specimen is not a measurement and the caller states the "
        "training cohort rather than have it inferred.",
    )
    parser.add_argument(
        "--selection-cohort",
        required=True,
        metavar="ID",
        help="the specimen whose validation F1 chose the epoch this checkpoint holds; "
        "required and not inferred, because the program loads a state dict and cannot "
        "know which specimen was held out when it was written. Recorded in "
        "medical_metrics.json beside train_specimens, because the two answer different "
        "questions and a reader needs both.",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/ablate_threshold.py` gives: this is four columns of
    a name and three floats, and `tabulate` is not a dependency this project declares. The
    siblings' `_print_table` is not imported, for the reason `bin/train.py` states for its own:
    `bin/` is a directory of scripts loaded by path rather than a package, and reaching into
    one script's private name would couple two entry points that are meant to be read one at a
    time.
    """
    widths = [
        max([len(str(header))] + [len(str(row[index])) for row in rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _refuse_output_target(results_dir: Path, target: Path) -> str | None:
    """Return why this program will not write its artifacts under `results_dir`, or None.

    **The same two refusals `bin/ablate_threshold.py` makes, against both paths, for the same
    reason.** That program writes two files into a directory it does not own and tests both
    the results directory and the subdirectory it puts them in; this one writes two as well
    and deletes nothing, which is a smaller blast radius but not a reason to be the one entry
    point that does not check. A `paths.results` that is a hand-edited string is how a study's
    results end up somewhere this repository is not responsible for.

    **The file test runs before the containment test for each path, so a file is reported as
    the typo it is.** The containment predicate is `blobcount.extract.is_owned_path` rather
    than a string comparison here: "is this ours" has one implementation in this project
    and not one per guard, and the entry points that differ in what they *do* to a target
    do not differ in what they decide is a target. The roots it enforces are the ones the
    project owns — the checkout, plus the parent of every configured `paths.*` value, the
    set `blobcount.extract.allowed_roots` derives from the configuration at call time —
    because a path the configuration names is by definition one this project owns, and a
    rule written against the checkout alone would refuse the destinations the data move
    put outside the repository. A link in the results tree is still resolved before it is
    compared, so one that points under no owned root is a refusal whichever side of the
    widen it lands on.

    The check returns a message and raises nothing, and it runs before the handoff is read and
    before a checkpoint is loaded, so a mistyped `paths.results` costs a message rather than a
    run.
    """
    for path in (results_dir, target):
        if path.exists() and not path.is_dir():
            return (
                f"refusing to write {path}: it exists and is a file rather than a directory, "
                f"so it is a typo in paths.results and not a previous run"
            )
        if not is_owned_path(path):
            return (
                f"refusing to write {path}: it is not strictly below the project root "
                f"{PROJECT_ROOT} or the parent of a configured paths.* value, so it is "
                f"not a directory this project owns"
            )
    return None


def _per_specimen(
    model,
    cohort: Sequence[tuple[Path, int, str]],
    train_specimens: Sequence[str],
    cfg: Config,
) -> list[tuple[str, Metrics]]:
    """Return one `(specimen, Metrics)` per specimen in the cohort, in `sorted()` order.

    **`sorted()` and not the order the command line named them**, for the reason
    `blobcount.splits.loso_folds` gives: the row order of an artifact has to mean the same
    thing on the next run of the pipeline, and a caller who wrote `--specimens B --specimens
    A` has said nothing about which of the two is first.

    One call to `evaluate_specimens` per specimen rather than one over the pooled cohort plus a
    per-specimen tally, because a specimen's own record is what this program reports and one
    function that produces it is one function that can be wrong about it. Every patch is read
    once across the run whichever way it is arranged: the specimens are disjoint, so the sum
    of the per-specimen inferences is the pooled one.
    """
    by_specimen: dict[str, list[tuple[Path, int, str]]] = {}
    for row in cohort:
        by_specimen.setdefault(row[2], []).append(row)
    return [
        (
            specimen,
            evaluate_specimens(
                model,
                [row[0] for row in by_specimen[specimen]],
                [row[1] for row in by_specimen[specimen]],
                [row[2] for row in by_specimen[specimen]],
                train_specimens,
                cfg,
            ),
        )
        for specimen in sorted(by_specimen)
    ]


def _document(
    metrics: Metrics,
    per_specimen: Sequence[tuple[str, Metrics]],
    intervals: Intervals,
    requested: Sequence[str],
    evaluated: Sequence[str],
    absent: Sequence[str],
    trained: Sequence[str],
    selection_cohort: str,
    checkpoint: Path,
    labels: Path,
) -> dict:
    """Return the JSON document this run writes.

    UTF-8 with `ensure_ascii=False` and specimen names as values, for the reasons
    `extract.write_index` and `labeling._write_labels` give: a specimen name has to survive a
    Turkish-locale machine unchanged.

    **The cohorts are three fields and not two.** `evaluated_specimens` is the cohort the run
    scored and `metrics` is the measurement of it, so the two cannot disagree; `requested` is
    what the command line asked for, so a caller who named a specimen the handoff holds no row
    of can see in the artifact that they asked and it was not there; and `not_evaluated`
    names those specimens outright, which is what makes the first field's silence about them
    legible rather than merely quiet. **`train_specimens` is beside them** so an artifact cannot
    be read as though the set that was scored and the set the model was trained on were the same
    by accident. `provenance` is the module's own sentence, so a reader of the file alone learns
    where its numbers came from without having read this module.

    **`checkpoint_selection_cohort` is a fourth field and not an entry in any of the three**,
    because it answers a different question: not what this run scored and not what the
    model learned from, but *which specimen's validation F1 chose the epoch the checkpoint
    holds*. In leave-one-specimen-out that is one specimen, and for fold 0 it is also the
    cohort this program scores — so an artifact recording only `evaluated_specimens` reads
    as though the epoch had never been chosen at all, which is the more optimistic of the
    two readings. **It is written exactly as the caller stated it and is never derived**:
    the program loads a `state_dict`, a checkpoint carries no specimen id, and the value
    would have to be guessed. The guess would be right whenever the caller scored fold 0's
    own held-out specimen, which is precisely the case where a reader would not notice.

    **The interval conditions come out of the `Intervals` record rather than off the
    configuration**, and that is the whole reason that record exists: reading either
    `evaluation.*` key here would be a second reader of it in a place the ledger scan cannot
    see, and the number in this block would then be a value nobody checked against the
    intervals printed beside it.

    **`unit`, `n_units` and `degenerate` are here for the same reason the cohorts are.** The
    block already said `n_samples: 2000` and `confidence_level: 0.95`, which read as a
    95% interval and is not one whenever the cohort holds a single specimen: resampling one
    unit returns that unit, so every bound equals the point estimate and the two numbers
    beside it are decoration. The unit is `specimen` and stays `specimen` — patches from
    one slide are not independent observations, and a patch-level interval on this cohort
    would be a narrower and a wrong one. What was missing was not the unit but the
    arithmetic of how many of them there are, and `degenerate` is that arithmetic written
    down so a reader of the file learns it without the README.
    """
    n_units = len(per_specimen)
    return {
        "provenance": _PROVENANCE,
        "labels": str(labels),
        "checkpoint": str(checkpoint),
        "requested_specimens": list(requested),
        "evaluated_specimens": list(evaluated),
        "not_evaluated": list(absent),
        "train_specimens": list(trained),
        "checkpoint_selection_cohort": selection_cohort,
        "bootstrap": {
            "unit": "specimen",
            "n_units": n_units,
            "degenerate": n_units < 2,
            "n_samples": intervals.n_samples,
            "confidence_level": intervals.level,
            "intervals": {
                name: list(bounds) for name, bounds in intervals.by_metric.items()
            },
        },
        "metrics": asdict(metrics),
        "per_specimen": [
            {"specimen": specimen, **asdict(record)} for specimen, record in per_specimen
        ],
    }


def _write_per_specimen(target: Path, per_specimen: Sequence[tuple[str, Metrics]]) -> None:
    """Write one row per specimen, with an `auc` that is not a number written blank.

    UTF-8 and `newline=""` for the reasons `blobcount.train._write_rows` gives: a specimen name
    has to survive a Turkish-locale machine and `csv` writes its own line endings. `None`
    becomes the empty string and not the text `"None"`, because a downstream reader parsing
    this with `float()` would raise on the text and read the empty cell as a missing value,
    which is what it is.
    """
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_PER_SPECIMEN_COLUMNS)
        writer.writeheader()
        for specimen, record in per_specimen:
            row = {"specimen": specimen, **asdict(record)}
            writer.writerow({name: "" if value is None else value for name, value in row.items()})


def _load_weights(model, checkpoint: Path) -> None:
    """Load `checkpoint` into `model`, naming the file in a refusal that does not.

    **The same `pickle.UnpicklingError` goes out as the one torch raised**, because that type
    is what `main`'s arm catches and a second type would be a second answer to what a
    checkpoint is; only the message is added. The reason it is added at all is measured: a
    `--checkpoint` naming a file that is not a torch archive produces
    `UnpicklingError: Weights only load failed ...` naming torch's policy and not the path,
    and a caller who typed the wrong `--checkpoint` learns nothing from it. `blobcount.train`
    wraps nothing for the opposite reason and the ruling is the same one: its `OSError`
    already carries the full resolved path, which is the fact a caller needs.
    """
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise pickle.UnpicklingError(
            f"{checkpoint} is not a torch archive torch can read into a model: {error}"
        ) from error
    model.load_state_dict(state)


def _cell(value: float | None) -> str:
    """Return `value` at three places, or `_ABSENT` where there is no number."""
    return _ABSENT if value is None else f"{value:.3f}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation the arguments name and return the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        # Resolved inside the arm, for the reason `bin/train.py` gives: `load` types this key
        # as the declared `str`, and an empty one is a value it makes no claim about, so
        # `Config.path` is what refuses it and outside this arm that refusal would leave a
        # traceback. `paths.labels` is **not** resolved here: `read_labels` reads it, which is
        # what puts that read under `blobcount/` where the ledger scan can see it.
        results_dir: Path = cfg.path("paths.results")
    except ConfigError as error:
        # The two shapes this arm carries: the file `load` refuses, and a `paths.results`
        # `Config.path` refuses. The arguments the parser produced cannot be the cause —
        # `--labels` and `--checkpoint` have already been typed as `Path`.
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_REFUSED

    target = results_dir / _EVALUATION_DIR
    refused = _refuse_output_target(results_dir, target)
    if refused is not None:
        # Before the handoff is read and before a checkpoint is loaded, so a mistyped
        # `paths.results` costs a message rather than an inference pass. `EXIT_REFUSED` and
        # not a metrics failure: nothing here finished, so there is no run whose outcome is
        # being reported, and the two codes mean different things to whoever reads them.
        print(f"evaluate: {refused}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        paths, labels, specimen_ids = read_labels(args.labels, cfg)
    except (ConfigError, OSError, csv.Error, ValueError) as error:
        # The handoff, as the module docstring groups it. `ConfigError` is in this tuple for
        # one shape, not because the arm is wide: `read_labels` resolves every row against
        # `paths.labels`, so `Config.path`'s refusal of an empty value arrives out of it, and
        # it is an answer about the configuration in a place where the configuration is what
        # was being read. The catch is around this one call, so it is too narrow to hold a
        # defect in the scoring below.
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_REFUSED

    wanted = set(args.specimens) if args.specimens else set(specimen_ids)
    cohort = [
        (path, label, specimen)
        for path, label, specimen in zip(paths, labels, specimen_ids, strict=True)
        if specimen in wanted
    ]
    # **The cohort the run scored is read off `cohort` and not off `wanted`**, because a
    # specimen the caller named and the handoff holds no row of is silently dropped by the
    # filter above, and a document whose `evaluated_specimens` still names it is a provenance
    # record of a cohort that was never scored. In a leave-one-specimen-out study that field is
    # the one a reader checks first — it is what says which patients these numbers are about —
    # and this project moved `read_labels` into the package for exactly this reason. Measured
    # before the fix: `--specimens A Z` with `Z` absent from the handoff wrote
    # `"evaluated_specimens": ["A", "Z"]` and printed two specimens beside `metrics.n == 2`.
    evaluated = sorted({specimen for _path, _label, specimen in cohort})
    requested = sorted(wanted)
    absent = [name for name in requested if name not in set(evaluated)]

    if not cohort:
        # The caller named a specimen this handoff holds no row of, or every specimen it
        # named is one of those. Refused rather than scored, because six `0.0` rates beside an
        # `auc` of `None` written into `medical_metrics.json` is a report of nothing that a
        # later reader cannot tell apart from a report of a model that scored nothing.
        print(
            f"evaluate: none of the specimens named has a row in {args.labels}"
            + (f": {', '.join(requested)}" if args.specimens else "")
            + f". The handoff holds {sorted(set(specimen_ids))} — score a specimen it holds "
            f"a row of, or pass --labels the handoff that holds one",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    trained = sorted(set(args.train_specimens))
    try:
        model = build_model(
            cfg.get("training.architecture"),
            int(cfg.get("training.num_classes")),
            # `False` whatever the configuration says, for the reason the module docstring
            # gives: the checkpoint loaded below replaces every parameter of this model, and
            # `training.pretrained: true` is the shipped value.
            False,
        )
        _load_weights(model, args.checkpoint)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        # The checkpoint and the backbone, as the module docstring groups them: a file that is
        # not there, one that is not a torch archive, one whose weights do not fit the head the
        # configuration describes, and an architecture this project does not build. The first
        # two are the user's argument and the last two are the configuration; all four reach
        # here as themselves. `pickle.UnpicklingError` is in the tuple for the measured reason
        # the docstring gives — it is neither an `OSError` nor a `ValueError` — and the arm is
        # around these two calls, so a `ValueError` from inside the inference below is not
        # caught here and reaches the interpreter as a traceback, which is the honest report of
        # a defect in this project.
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # Everything below here is the run itself. A patch that is not there, a `ValueError` out of
    # the arithmetic and a failed write are deliberately not caught: a refusal is an answer
    # about the argument and a traceback is the answer about our code.
    try:
        pooled = evaluate_specimens(
            model,
            [row[0] for row in cohort],
            [row[1] for row in cohort],
            [row[2] for row in cohort],
            args.train_specimens,
            cfg,
        )
        per_specimen = _per_specimen(model, cohort, args.train_specimens, cfg)
    except EvaluationIntegrityError as error:
        # This program's own refusal and the one `bin/evaluate.py` exists to make. Nothing has
        # been written — the artifact directory is not created before this point for exactly
        # that reason — so a refused run leaves nothing a reader could mistake for a result.
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # The interval conditions come out of the record rather than off the configuration: this
    # program reads neither `evaluation.*` key, and `blobcount.evaluate` reads each exactly
    # once.
    intervals = bootstrap_metric_intervals(
        [record for _specimen, record in per_specimen], cfg, int(cfg.get("project.seed"))
    )

    target.mkdir(parents=True, exist_ok=True)
    document = _document(
        pooled, per_specimen, intervals, requested, evaluated, absent,
        trained, args.selection_cohort, args.checkpoint, args.labels,
    )
    (target / _METRICS_NAME).write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write_per_specimen(target / _PER_SPECIMEN_NAME, per_specimen)

    metrics = document["metrics"]
    _print_table(
        ("metric", "value", "ci_low", "ci_high"),
        [
            [
                name,
                _cell(metrics[name]),
                *(
                    [_cell(bound) for bound in intervals.by_metric[name]]
                    if name in intervals.by_metric
                    else [_ABSENT, _ABSENT]
                ),
            ]
            for name in _TABLE_ROWS
        ],
    )
    # **A specimen the caller named and the handoff holds no row of is said so here**, because
    # the document records it and a sentence is the only place a person reads. Without this
    # sentence the run is silent about `Z`: the table shows one specimen, `metrics["n"]` is
    # that specimen's rows, and the only record that `Z` was asked for is a JSON field.
    dropped = (
        f" {', '.join(absent)} {'were' if len(absent) > 1 else 'was'} requested and is not in "
        f"the handoff, so {'they were' if len(absent) > 1 else 'it was'} not scored."
        if absent else ""
    )
    print(
        f"\n{pooled.n} patch(es) of {len(evaluated)} specimen(s) "
        f"({', '.join(evaluated)}) scored with {args.checkpoint}.{dropped} Each interval is "
        f"{intervals.n_samples} bootstrap resamples of the per-specimen values at the "
        f"configured level of {intervals.level:.0%}, so it is about the specimens rather "
        f"than about the patches. The specimens this model was trained on were "
        f"{', '.join(trained)} and none of them is among the ones above, and the epoch "
        f"in this checkpoint was chosen on the validation F1 of specimen "
        f"{args.selection_cohort}. Where these numbers came from is the first key of "
        f"{target / _METRICS_NAME}; per-specimen counts are beside it"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())