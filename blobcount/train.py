"""Leave-one-specimen-out cross-validation, and the epoch each fold selects.

**Nothing in this module has been run against the cohort.** `dataset/labels/` does not
exist and the extraction that would fill it has never been run, so the `labels.csv` this
loop trains from has never been written and no number this module produces has been
measured on a real patch. Every figure this module reports is computed from whatever
patches a caller hands it, and the only runs that have ever been made are over synthetic
ones written by `tests/test_train.py`. A reader must not take any property stated here to
mean that the loop has produced a study result.

**The split is by specimen and the model is selected on validation F1.** Both are
decisions and not defaults, and both were not decisions in the code this rebuild replaces.
`train.py:78` grouped patches by the id parsed out of a *file name* and `train.py:297`
split the groups with `GroupKFold`, and it selected on validation accuracy: `train.py:257`
kept the checkpoint whose `val_acc` was highest, while `train.py:281` built each fold's
record with that number beside the **final** epoch's precision, recall and F1, and
`train.py:328` averaged the two as they stood. One number in that summary came from one
epoch and the three beside it from another, so the study reported the accuracy of the
epoch it liked and the F1 of the epoch it stopped at. Here the fold is
`blobcount.splits.loso_folds` — one fold per distinct specimen, the whole specimen held
out — and `select_epoch` is the only thing that chooses, on `val_f1`, with the earliest
epoch winning a tie. `summarise_folds` then averages the **selected** records and returns
`selected_epochs` beside the means, so an artifact that carries the means also carries the
epoch each mean came from and cannot be read as if the means came from anywhere else.

**Accuracy is computed, written, printed and never selects.** `EpochRecord` carries
`val_accuracy` next to `val_f1` because the study reports both, and
`results/fold_summary.csv` has a column for each. The selection is `max` over `val_f1` and
nothing else decides anything from it; the one other reader is `summarise_folds`, which
averages it into `mean_accuracy`, and `tests/test_train.py` pins the selection itself with
two records whose accuracies and F1s disagree.

**The confusion matrix counts every validation row exactly once, and the metrics come out
of it.** `tp`, `fp`, `fn` and `tn` are counted against `labeling.LABEL_BY_CLASS`'s steatosis
class — the class the whole pipeline counts as a positive — and precision, recall, F1 and
accuracy are derived from those four integers rather than accumulated per batch, so
`tp + fp + fn + tn` is the size of the held-out specimen's row set whatever that
specimen's class distribution is. Where a rate has no denominator, `0.0` is the answer and
not a division: a validation set with no predicted positive has a precision of zero, and
one with no actual positive has a recall of zero.

**The class weights come from the fold's training labels, and so do the weights of its
loss.** `train.py:87` counted the classes of the pooled cohort and handed that one tensor
to `nn.CrossEntropyLoss` in every fold, so the weights a fold trained under were derived
from the specimens it was validating on — the leak the split exists to close, arriving
through the loss instead of through the split. `train_fold` calls `class_weights` with the
fold's training labels, and no weight vector it uses can mention a held-out specimen
because no label of one was passed in.

**That same call is what checks the validation labels, and its vector is discarded.**
`class_weights` is the one predicate in this project for "is this a class index of a run
with `num_classes` classes", and a fold's validation rows are labels that no fold's weight
call has looked at: with three specimens admitted for training, a fold validates on a
specimen the other folds train on, so an out-of-range label in a validation column would
reach `CrossEntropyLoss` as a target it cannot score and the report would name the loss
rather than the column. So the validation labels go through the same call and the vector
that comes back is thrown away: what is wanted is the refusal, and a second rule written
here would be a second answer to a question this project already answers once.

**Every `PatchDataset` is given a transform, and the two are not the same one.**
`PatchDataset(transform=None)` applies `ToTensor` and nothing else, and it applies it
happily — a `DataLoader` cannot collate a `PIL.Image`, so a dataset with no transform
still yields tensors, just unresized and unnormalized ones, and a fold built that way
would report a number about a backbone fed pixels in a range it was never fitted on.
Nothing in this module constructs one without an argument for it.

**One checkpoint per fold, and it is written on a strict improvement in validation F1.**
The name is `best_model_fold{fold.fold}.pth`, the number is the one `loso_folds` gave the
fold and it starts at zero — the same number that goes into the fold's log file name and
into the `fold` column of the summary, so an artifact can be matched to a fold without
knowing anything else about it. The comparison is `>` and not `>=`: an epoch that only
*equals* the best F1 is not a new best, and writing it again would replace a checkpoint
with a model trained no further. The first epoch always writes, because the best F1 the
loop starts from is `-inf` rather than `0.0` — a fold whose every epoch scored zero would
otherwise save nothing at all.

**A fold stops after `training.patience_early_stop` epochs in a row without an
improvement**, counted on that same F1. The learning rate is reduced by
`ReduceLROnPlateau` on the validation *loss* with `training.lr_scheduler_factor` and
`training.patience_lr_scheduler`, which is a different question from the one the selection
asks and is left where it is; measured on this stack, a record carries the rate that was in
force while its own epoch trained, read before the scheduler steps, so the reduction a
scheduler makes is the one the *next* record carries.

**A fold is seeded from `project.seed` and its own number**, so two runs of this pipeline
give two identical runs and two folds of one run do not share a head initialisation or a
sampler stream. `set_seed` is the project's one answer to "what seeds a run" and it is
called with the derived value rather than with the configured one, so
`tests/test_train.py` can pin reproducibility by running one fold twice.

**The output directory is an argument and nothing here reads `paths.results`.** Every
artifact this module writes goes to `out_dir`, which is why no test in the suite can touch
the repository's `results/` directory and why the legacy artifacts sitting there are
untouched rather than overwritten by a test run. The `results/` a study run produces are
regenerated by the task that consolidates them; what is in there today is the methodology
this rebuild removes, and its file names are one-based where this module's are zero-based.

**Nothing here is caught, logged, or repaired.** A patch file that is not there raises from
`PatchDataset.__getitem__` inside a loader, and the `OSError` that survives names the path
it could not open, which is the fact a caller needs and better than a sentence about a
rule — this project's `blobcount.data` module states that ruling and this module is on the
other side of it. A `ValueError` out of `class_weights` or `build_model` reaches the caller
as itself for the same reason, and `bin/train.py` is where catching belongs.

Declared interface, in `__all__` order:

    EpochRecord
        fold: int
        epoch: int
        train_loss: float
        val_loss: float
        val_f1: float
        val_accuracy: float
        precision: float
        recall: float
        tp: int
        fp: int
        fn: int
        tn: int
        lr: float
    run_cross_validation(
        paths: Sequence[Path],
        labels: Sequence[int],
        specimen_ids: Sequence[str],
        cfg: Config,
        out_dir: Path,
    ) -> dict
    select_epoch(records: Sequence[EpochRecord]) -> EpochRecord
    summarise_folds(selected: Sequence[EpochRecord]) -> dict
    train_fold(
        fold: Fold,
        paths: Sequence[Path],
        labels: Sequence[int],
        cfg: Config,
        out_dir: Path,
    ) -> tuple[EpochRecord, list[EpochRecord]]
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from blobcount.config import Config, get_device, set_seed
from blobcount.data import (
    PatchDataset,
    build_loader,
    build_sampler,
    build_transforms,
    class_weights,
)
from blobcount.labeling import CLASS_STEATOSIS, LABEL_BY_CLASS
from blobcount.models import build_model
from blobcount.splits import Fold, loso_folds

__all__ = [
    "EpochRecord",
    "run_cross_validation",
    "select_epoch",
    "summarise_folds",
    "train_fold",
]

# The class the whole pipeline counts as a positive, read from the labelling run's own
# table rather than written here: `labeling._write_labels` put the class each directory
# holds into `labels.csv` from that table, and a second place for "which class is
# steatosis" is a second answer to a question the labelling run already answers.
_POSITIVE_CLASS = LABEL_BY_CLASS[CLASS_STEATOSIS]

# The four metrics a summary averages, in the order the mean table prints them, each paired
# with the `EpochRecord` field it is read from. Two of the four are spelled differently on
# the two sides — the record says `val_accuracy` and `val_f1` because it also carries a
# training loss, the summary says `accuracy` and `f1` because a reader of `mean_f1` is
# looking for the metric rather than for an epoch — and the pairing is written down once
# here so that neither spelling is assembled inside a loop.
_METRICS = (
    ("accuracy", "val_accuracy"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("f1", "val_f1"),
)


@dataclass(frozen=True)
class EpochRecord:
    """One epoch of one fold: what it cost, what the held-out specimen scored, and at what rate.

    **The field order is fixed and is the order every artifact writes.**
    `results/training_log_fold{n}.csv`, `results/fold_summary.csv` and the `per_fold` block
    of `results/training_summary.json` all carry these thirteen fields in this order, and a
    `csv.DictWriter` writes its header in the order it is handed the columns rather than the
    order a record was built in — so the order is written down once, here, and read from
    `__dataclass_fields__` by both writers.

    `fold` is `Fold.fold`: the 0-based position of the held-out specimen in the `sorted()`
    order of the distinct specimen ids. The specimen's name is **not** on the record and is
    not lost either — `run_cross_validation` writes it beside these fields in both files,
    and it is what makes a fold number mean something to a reader.

    `epoch` counts from 1, so a fold's records are numbered the way the loop that wrote them
    counted, and `training.epochs` epochs produce epochs 1 to that number.

    `val_loss` is the mean over the fold's validation batches of an **unweighted**
    cross-entropy, and `lr` is the rate that was in force while *this* epoch trained — read
    before the scheduler steps, so a reduction the scheduler makes is carried by the next
    record rather than by the epoch that caused it. `train_fold` says why the validation
    loss carries no class weights where the training loss does.
    """

    fold: int
    epoch: int
    train_loss: float
    val_loss: float
    val_f1: float
    val_accuracy: float
    precision: float
    recall: float
    tp: int
    fp: int
    fn: int
    tn: int
    lr: float


def select_epoch(records: Sequence[EpochRecord]) -> EpochRecord:
    """Return the record with the highest `val_f1`, the earliest epoch winning a tie.

    **F1 is the selector and accuracy is not.** This is the regression the rebuild exists
    for: `train.py:257` kept the checkpoint of the epoch with the highest validation
    accuracy, so a fold whose accuracy was flat while its F1 moved kept the wrong epoch.

    **A tie goes to the earliest epoch**, and the tie-break is on the epoch number and not
    on the position in the sequence, so the answer does not depend on the order the records
    were appended in. One `max` over a `(val_f1, -epoch)` key carries both rules and is the
    whole of the function; measured, a later epoch with the same F1 and a *better* accuracy
    loses to the earlier one, which is what the tie-break is for.

    A sequence with no record in it is refused, naming what a caller has to hand over: the
    selection is undefined over nothing, and CPython's `max() arg is an empty sequence`
    names neither a rule nor an argument.
    """
    if not records:
        raise ValueError(
            "select_epoch was given a sequence with no epoch in it: the epoch a fold "
            "selected is the record with the highest validation F1, and a fold that ran no "
            "epoch has no such record. This happens when training.epochs is 0 — the fold "
            "wrote no checkpoint and there is nothing to select — so check that key before "
            "reading a run's summary as though it held a result"
        )
    return max(records, key=lambda record: (record.val_f1, -record.epoch))


def summarise_folds(selected: Sequence[EpochRecord]) -> dict:
    """Return the mean and population standard deviation of four metrics over `selected`.

    **Every number here is a function of the records this call is given, and of nothing
    else.** The caller passes the selected epoch of each fold and the summary describes
    those; a summary that mixed one fold's best accuracy with its final F1 is not a summary
    of anything — `train.py:281` is where `best_acc` and `final_f1` were put in one record
    and `train.py:328` is where they were averaged as they stood — and the
    `selected_epochs` entry is here so that an artifact carrying these means also carries
    the record each of them came from.

    **The standard deviation is the population one, over the folds as the whole cohort.**
    The folds are not a sample drawn from a larger population of splits — measured on the
    shipped manifest, `usable_for_training` admits three of its six specimens and those
    three are every specimen the study has, so the folds are all of it — so the divisor is
    the count and not the count less one. Measured on the pair `(0.90, 0.70)`, the two
    answers are 0.1 and 0.1414, which is the difference `tests/test_train.py` pins.

    A sequence with no record in it is refused, for the reason `select_epoch` gives: a mean
    and a standard deviation over no fold are `nan`, and a `nan` in a serialized artifact
    loads back as a fact about the run.
    """
    if not selected:
        raise ValueError(
            "summarise_folds was given a sequence with no fold in it: a mean and a "
            "standard deviation over no fold are both nan, and a nan written into a run's "
            "summary reads back as a measurement. A cohort needs at least two distinct "
            "specimens for a fold to hold out, which blobcount.splits.loso_folds refuses"
        )
    summary: dict = {"n_folds": len(selected)}
    for name, field in _METRICS:
        values = [float(getattr(record, field)) for record in selected]
        summary[f"mean_{name}"] = float(np.mean(values))
        summary[f"mean_{name}_std"] = float(np.std(values))
    summary["selected_epochs"] = [
        {"fold": record.fold, "epoch": record.epoch} for record in selected
    ]
    return summary


def _rates(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Return `(precision, recall, f1)` for the positive class, from the confusion matrix.

    **A rate with no denominator is `0.0` and not a division.** A fold that predicted no
    positive has a precision of zero rather than an undefined one, and a fold whose held-out
    specimen carries no positive at all has a recall of zero — and both are reachable: a
    specimen can be entirely normal, which is why `blobcount.data.class_weights` gives an
    absent class `0.0` rather than `inf`. An epoch that scored nothing is still an epoch,
    and the checkpoint rule above is the one that stops such an epoch from being reported
    as a result.
    """
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one pass over `loader` and return the mean loss over its batches.

    The mean is over batches and not over rows, which is what a loss averaged the way
    `train.py:195` averaged it means too: the last batch of a fold is the short one, and
    averaging over batches gives it the same weight as a full one. The value here is a
    training curve and not a measurement of anything — it is carried in the record so that a
    reader can see a fold's loss fall, and no selection reads it.
    """
    model.train()
    total = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / len(loader)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, int, int, int, int]:
    """Return `(mean loss, tp, fp, fn, tn)` over `loader`, with no gradient and no dropout.

    `criterion` is the **unweighted** loss, and that is the caller's argument rather than a
    decision made here: a validation set whose rows all carry the `0.0` weight
    `class_weights` gives an absent class makes a weighted loss `nan`, and `train_fold`
    carries the measurement. The four counts are tallied over the whole loader rather than
    over a batch and merged afterwards, which is the same answer and one place where the
    tally can be wrong instead of four. Every validation row lands in exactly one of them,
    so `tp + fp + fn + tn` is the size of the held-out specimen's row set whatever the
    fold's class distribution is; `tests/test_train.py` pins that identity.
    """
    model.eval()
    total = 0.0
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total += float(criterion(logits, targets).item())
            predicted_positive = logits.argmax(dim=1) == _POSITIVE_CLASS
            actual_positive = targets == _POSITIVE_CLASS
            tp += int((predicted_positive & actual_positive).sum())
            fp += int((predicted_positive & ~actual_positive).sum())
            fn += int((~predicted_positive & actual_positive).sum())
            tn += int((~predicted_positive & ~actual_positive).sum())
    return total / len(loader), tp, fp, fn, tn


def _csv_row(fold: Fold, record: EpochRecord) -> dict:
    """Return `record` as one CSV row, with the held-out specimen named beside it."""
    values = asdict(record)
    return {"fold": values.pop("fold"), "held_out": fold.held_out, **values}


# The columns both files carry, in the record's own order after `fold` and the specimen.
# One list, so `training_log_fold{n}.csv` and `fold_summary.csv` differ only in which rows
# they hold — every epoch against the selected one — and a reader can compare them row for
# row rather than reconciling two headers. `fold` is taken from the record and written once,
# beside the specimen name the record does not carry.
_CSV_COLUMNS = (
    "fold",
    "held_out",
    *(name for name in EpochRecord.__dataclass_fields__ if name != "fold"),
)


def _write_rows(path: Path, rows: Sequence[dict]) -> None:
    """Write `rows` to `path` under `_CSV_COLUMNS`, as UTF-8 and with the spec's newlines.

    UTF-8 because a slide id in this project is a specimen name and has to survive a
    Turkish-locale machine unchanged, and `newline=""` for the reason
    `extract.write_index` and `labeling._write_labels` both give: `csv` writes its own line
    endings and a file opened for translation doubles them.
    """
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def train_fold(
    fold: Fold,
    paths: Sequence[Path],
    labels: Sequence[int],
    cfg: Config,
    out_dir: Path,
) -> tuple[EpochRecord, list[EpochRecord]]:
    """Train one fold and return its selected record and every record the fold wrote.

    The caller must supply **`paths` and `labels`, one entry each per dataset row and in
    dataset order**, which is the contract `blobcount.splits.loso_folds` states and the one
    this function indexes into: `paths[i]` is the patch of row `i` and `labels[i]` is its
    class, and `fold.train_idx` and `fold.val_idx` are positions in that same order. The two
    sequences are checked against each other nowhere, for the reason `PatchDataset` gives:
    with one sequence of paths there is no expected row count to check against. The fold
    itself is handed in whole, so the specimen a fold holds out is not an argument here.

    **Every label the fold uses is checked by `class_weights`, the training ones for the
    weights and the validation ones for the refusal**, which is the whole of what the
    per-fold correction is and the reason a held-out specimen's distribution cannot reach
    the loss. The module docstring carries the argument for the second call.

    **`out_dir` is created when it is not there, a file of the same name is rewritten in
    place, and nothing is deleted.** No `rmtree` runs here: a run whose cohort has one fold
    fewer than the last one leaves the last run's `training_log_fold{n}.csv` for the fold
    that no longer exists, so `paths.results` after two runs is a directory holding both
    rather than the second alone. Clearing is `blobcount.extract.prepare_output`'s business
    and its refusal is about trees of patches, not about five files a fold writes.

    **A `Fold` is never compared here.** `Fold` keeps identity equality (`eq=False`) and
    its generated `__eq__` over `np.ndarray` fields cannot answer a question about two
    folds, so anything that needed to know whether two folds were the same has to compare
    the index arrays with `np.array_equal`. Nothing in this function needs to: the fold it
    was handed is the fold it trains, and the only number it takes from it is `fold.fold`.
    """
    seed = int(cfg.get("project.seed")) + fold.fold
    set_seed(seed)
    device = torch.device(get_device(cfg))
    num_classes = int(cfg.get("training.num_classes"))
    epochs = int(cfg.get("training.epochs"))
    early_stop = int(cfg.get("training.patience_early_stop"))
    batch_size = int(cfg.get("training.batch_size"))
    num_workers = int(cfg.get("training.num_workers"))
    train_transform, eval_transform = build_transforms(cfg)

    train_paths = [Path(paths[index]) for index in fold.train_idx]
    val_paths = [Path(paths[index]) for index in fold.val_idx]
    train_labels = [labels[index] for index in fold.train_idx]
    val_labels = [labels[index] for index in fold.val_idx]
    weights = class_weights(train_labels, num_classes).to(device)
    # Called for what it refuses and not for what it returns: the module docstring gives
    # why a fold's validation labels are held to the same rule its training labels are, and
    # why the vector that comes back is thrown away rather than carried into the loss.
    class_weights(val_labels, num_classes)

    sampler = build_sampler(train_labels, cfg.get("training.use_weighted_sampler"), seed)
    train_loader = build_loader(
        PatchDataset(train_paths, train_labels, train_transform),
        batch_size, sampler, True, num_workers, seed,
    )
    val_loader = build_loader(
        PatchDataset(val_paths, val_labels, eval_transform),
        batch_size, None, False, num_workers, seed,
    )

    model = build_model(
        cfg.get("training.architecture"), num_classes, cfg.get("training.pretrained")
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    # **The validation loss is unweighted, and that is not an oversight.** The weights are
    # built from this fold's *training* labels, and `class_weights` gives a class the fold
    # holds none of a weight of `0.0` rather than `inf`; a weighted loss whose targets all
    # carry weight `0.0` is a division of zero by zero, and measured on this stack
    # `nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.0]))(torch.zeros(2, 2),
    # torch.tensor([1, 1]))` is `nan` where the unweighted loss is `0.693`. That state is
    # reachable and not hypothetical — three specimens are admitted for training, a fold
    # trains on two, and those two can both be class 0 — and the number it poisons is the
    # one `ReduceLROnPlateau` steps on, so a `nan` there shrinks the rate every epoch
    # instead of the configured factor. No metric is affected either way: precision,
    # recall, F1 and accuracy are counted from the confusion matrix, never out of a loss.
    val_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.get("training.learning_rate")))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(cfg.get("training.lr_scheduler_factor")),
        patience=int(cfg.get("training.patience_lr_scheduler")),
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"best_model_fold{fold.fold}.pth"
    best_f1 = float("-inf")
    stale = 0
    records: list[EpochRecord] = []
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, tp, fp, fn, tn = _evaluate(model, val_loader, val_criterion, device)
        precision, recall, f1 = _rates(tp, fp, fn)
        # Read before the scheduler steps: this is the rate this epoch trained under, and
        # the reduction the scheduler may make below is the next record's `lr`.
        lr = float(optimizer.param_groups[0]["lr"])
        records.append(EpochRecord(
            fold=fold.fold,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_f1=f1,
            val_accuracy=(tp + tn) / (tp + fp + fn + tn),
            precision=precision,
            recall=recall,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            lr=lr,
        ))
        if f1 > best_f1:
            best_f1 = f1
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
        scheduler.step(val_loss)
        if stale >= early_stop:
            break
    return select_epoch(records), records


def run_cross_validation(
    paths: Sequence[Path],
    labels: Sequence[int],
    specimen_ids: Sequence[str],
    cfg: Config,
    out_dir: Path,
) -> dict:
    """Train one fold per specimen and return `summarise_folds` over the selected epochs.

    The caller must supply **one specimen id per dataset row, in dataset order**, and that
    is the only shape `loso_folds` accepts: a deduplicated list of specimens is
    indistinguishable from one row per specimen from inside it, and it produces folds of
    one validation row each. The refusal for the two shapes it cannot take — a sequence of
    characters or of bytes, and fewer than two distinct ids — is `blobcount.splits`'s and
    propagates as a `SplitError`, because this function adds no rule about what a specimen
    is.

    **One log and one checkpoint are written per fold and one summary per run, and every
    one of them goes to `out_dir`: `training_log_fold{n}.csv` with every epoch the fold
    ran, `best_model_fold{n}.pth` from inside the fold, and `fold_summary.csv` with one row
    per fold at its selected epoch.** The two files carry the same columns and differ only
    in their rows, which is what makes the summary auditable against the log it summarizes:
    a reader can find the row the summary reported and see the epochs it was chosen over.
    `n` is `Fold.fold` and starts at zero.

    **Each fold's log is written as that fold finishes and the summary last**, so a run
    that raises half way leaves the log of every fold it completed rather than none of them
    — the same order `bin/ablate_threshold.py` creates its directory after the counts it
    may not have if the measurement fails. **Nothing is cleared first**, for the reason
    `train_fold` gives.
    """
    folds = loso_folds(specimen_ids)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[EpochRecord] = []
    for fold in folds:
        best, records = train_fold(fold, paths, labels, cfg, out_dir)
        selected.append(best)
        _write_rows(
            out_dir / f"training_log_fold{fold.fold}.csv",
            [_csv_row(fold, record) for record in records],
        )
    _write_rows(
        out_dir / "fold_summary.csv",
        [_csv_row(fold, record) for fold, record in zip(folds, selected, strict=True)],
    )
    return summarise_folds(selected)