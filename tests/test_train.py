"""Tests for the cross-validation loop, the epoch it selects, and its entry point.

**Every test here trains a fold, and none of them reaches the network or writes into
`results/`.** `configs/default.yaml` ships `training.pretrained: true`, so every
configuration a test builds overrides it — `training.pretrained=false` is in `_cfg` and in
the command line's own configuration file — and
`test_a_run_writes_nothing_under_the_torch_home` points `TORCH_HOME` at an empty directory
and checks that not one byte lands in it. The outcomes `results/` holds are the
methodology this rebuild exists to remove and Task 15 regenerates them, so nothing here
reads or writes them: every run below is handed a directory under `tmp_path`.

The cohort is four rows over two specimens and the patches are real PNGs written by this
file, because the real handoff lives outside the checkout: Task 14 moved the generated
data under `%LOCALAPPDATA%` (the `paths.patches` and `paths.labels` of
`configs/default.yaml`), and a suite that trained on it would depend on gigabytes no clone
carries. Every number this module reports is therefore measured over synthetic patches.
"""

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from blobcount import train as train_module
from blobcount.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load
from blobcount.labeling import LABEL_COLUMNS
from blobcount.splits import SplitError, loso_folds
from blobcount.train import (
    EpochRecord,
    read_labels,
    run_cross_validation,
    select_epoch,
    summarise_folds,
    train_fold,
)

# The shipped `training.*` values, every one of them replaced for a test. `pretrained` is
# the one that matters most: `configs/default.yaml` ships it true, and a fold that reached
# `build_model` with it would ask torchvision for ImageNet weights over the network.
# `input_size` and `batch_size` are cut because a resnet18 forward pass over 224x224 is
# forty-nine times the area of one over 32x32 and this suite trains a real backbone; the
# loss and the metrics are computed from the tensors and not from their size.
_TEST_TRAINING = [
    "training.pretrained=false",
    "training.input_size=32",
    "training.batch_size=2",
    "training.num_workers=0",
    "training.epochs=2",
]


def _cfg(*overrides: str):
    """Return a Config with the test-safe training keys, plus `overrides` on top."""
    return load(overrides=[*_TEST_TRAINING, *overrides])


def _write_patch(path: Path, size: int = 32, seed: int = 0) -> Path:
    """Write one RGB patch of deterministic noise and return the path."""
    random = np.random.default_rng(seed)
    Image.fromarray((random.random((size, size, 3)) * 255).astype("uint8")).save(path)
    return path


def _cohort(root: Path, rows, size: int = 32):
    """Write one patch per `(specimen, label)` row; return paths, labels and specimen ids.

    The three come back in row order, which is the order `blobcount.splits.loso_folds`
    indexes into and the order `PatchDataset` pairs each path with its own label.
    """
    root.mkdir(parents=True, exist_ok=True)
    paths, labels, specimens = [], [], []
    for index, (specimen, label) in enumerate(rows):
        paths.append(_write_patch(root / f"{specimen}_{index}.png", size=size, seed=index))
        labels.append(label)
        specimens.append(specimen)
    return paths, labels, specimens


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stand_in(monkeypatch, model):
    """Make `train_fold` build `model` instead of a backbone, and count its checkpoints.

    `train_fold` owns the loop and not the backbone, so its rules — which epoch is
    selected, when a checkpoint is written, when it stops — can only be pinned against a
    model whose predictions are known. With a real resnet18 they depend on the pixels and
    on the seed, and an assertion of the shape "one checkpoint when nothing improves"
    could not be written at all.
    """
    saves: list[Path] = []
    original = torch.save

    def counting_save(obj, path, *args, **kwargs):
        saves.append(Path(path))
        return original(obj, path, *args, **kwargs)

    monkeypatch.setattr(train_module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(torch, "save", counting_save)
    return saves


class _StandIn(torch.nn.Module):
    """A backbone whose validation prediction this test chooses, one pass at a time.

    Every patch gets the same logits, so the metrics a fold reports are a function of the
    loop and of nothing else. In `training` mode it always predicts class 1, which keeps
    the training step out of the way of what these tests are about.

    - `"constant"` predicts `positive` in every validation pass, so validation F1 never
      changes and no epoch after the first improves on it.
    - `"improving"` predicts class 0 in the first validation pass and class 1 in every one
      after it, so F1 is 0.0 on the first epoch and 1.0 on every one after that.

    It holds one parameter so that `torch.optim.Adam` has something to step, and that
    parameter is added to both logits, so the gradient with respect to it is zero and the
    predictions cannot drift while a fold trains.
    """

    def __init__(self, positive: bool = True, mode: str = "constant"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.positive = positive
        self.mode = mode
        self.validation_passes = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        count = inputs.shape[0]
        if self.training:
            return torch.stack([torch.zeros(count), torch.ones(count)], dim=1) + self.weight
        self.validation_passes += 1
        positive = self.positive if self.mode == "constant" else self.validation_passes > 1
        logits = [torch.zeros(count), torch.zeros(count)]
        logits[1 if positive else 0] = torch.ones(count)
        return torch.stack(logits, dim=1) + self.weight


# ---------------------------------------------------------------------------
# The brief's tests, in the brief's order.
# ---------------------------------------------------------------------------


def _rec(epoch, f1, acc=0.8):
    return EpochRecord(fold=1, epoch=epoch, train_loss=0.5, val_loss=0.4,
                       val_f1=f1, val_accuracy=acc, precision=0.7, recall=0.7,
                       tp=1, fp=1, fn=1, tn=1, lr=1e-4)


def test_select_epoch_uses_f1_not_accuracy():
    recs = [_rec(1, 0.60, acc=0.95), _rec(2, 0.90, acc=0.70)]
    assert select_epoch(recs).epoch == 2


def test_select_epoch_breaks_ties_on_earliest_epoch():
    recs = [_rec(1, 0.90), _rec(2, 0.90), _rec(3, 0.80)]
    assert select_epoch(recs).epoch == 1


def test_summary_uses_selected_epochs_only():
    selected = [_rec(1, 0.90), EpochRecord(2, 2, 0.5, 0.4, 0.50, 0.99, 0.4, 0.4, 1, 1, 1, 1, 1e-4)]
    s = summarise_folds([_rec(1, 0.90), _rec(2, 0.90)])
    assert s["mean_f1"] == pytest.approx(0.90)
    assert s["mean_f1_std"] == pytest.approx(0.0)
    assert "epoch2" not in str(s)


# ---------------------------------------------------------------------------
# What the record is, and what the two functions above do with it.
# ---------------------------------------------------------------------------


def test_a_later_epoch_with_the_same_f1_and_a_better_accuracy_still_loses():
    # The selection reads `val_f1` and the tie-break reads `epoch`; accuracy appears in
    # neither, and a third record carrying the best accuracy of the three and the same F1
    # as the first still loses to it.
    recs = [_rec(1, 0.90, acc=0.70), _rec(2, 0.10, acc=0.10), _rec(3, 0.90, acc=0.99)]
    assert select_epoch(recs).epoch == 1


def test_the_record_is_positional_in_the_field_order_the_artifacts_write():
    # The order the brief fixes, and the order `results/training_log_fold*.csv` carries:
    # fold, epoch, train_loss, val_loss, val_f1, val_accuracy, precision, recall, tp, fp,
    # fn, tn, lr. Positional construction is the only construction that pins it.
    record = EpochRecord(2, 7, 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 7, 8, 9, 10, 0.0005)
    assert (record.fold, record.epoch) == (2, 7)
    assert (record.train_loss, record.val_loss) == (0.11, 0.22)
    assert (record.val_f1, record.val_accuracy) == (0.33, 0.44)
    assert (record.precision, record.recall) == (0.55, 0.66)
    assert (record.tp, record.fp, record.fn, record.tn) == (7, 8, 9, 10)
    assert record.lr == 0.0005
    assert list(EpochRecord.__dataclass_fields__) == [
        "fold", "epoch", "train_loss", "val_loss", "val_f1", "val_accuracy",
        "precision", "recall", "tp", "fp", "fn", "tn", "lr",
    ]


def test_the_summary_reports_the_population_standard_deviation():
    # Population, not sample: the folds are the whole cohort rather than a draw from it.
    # Measured on this pair, the two differ: 0.1 against 0.1414, so `ddof=0` is pinned.
    s = summarise_folds([_rec(1, 0.90), _rec(2, 0.70)])
    assert s["mean_f1"] == pytest.approx(0.80)
    assert s["mean_f1_std"] == pytest.approx(0.10)
    assert s["mean_f1_std"] != pytest.approx(np.std([0.90, 0.70], ddof=1))


def test_the_summary_carries_every_metric_and_not_only_f1():
    # `_rec` holds accuracy at 0.8, F1 at whatever it is given, and precision and recall at
    # 0.7, so the four means are three different numbers rather than one repeated.
    s = summarise_folds([_rec(1, 0.90), _rec(2, 0.70)])
    assert s["mean_accuracy"] == pytest.approx(0.8)
    assert s["mean_precision"] == pytest.approx(0.7)
    assert s["mean_recall"] == pytest.approx(0.7)
    assert s["mean_f1"] == pytest.approx(0.8)
    # The three that never vary between these two records have no spread; F1 does.
    for name in ("accuracy", "precision", "recall"):
        assert s[f"mean_{name}_std"] == pytest.approx(0.0)
    assert s["mean_f1_std"] == pytest.approx(0.1)
    assert s["n_folds"] == 2


def test_the_summary_names_the_fold_and_the_epoch_it_selected():
    s = summarise_folds([_rec(1, 0.90), _rec(2, 0.70)])
    assert s["selected_epochs"] == [{"fold": 1, "epoch": 1}, {"fold": 1, "epoch": 2}]


def test_the_summary_takes_the_records_it_was_given_and_not_the_best_of_each_fold():
    # The original defect: `best_acc` from one epoch beside the final epoch's F1. The
    # best epoch and the last epoch of one fold, and the summary over the best only.
    best = _rec(1, 0.90)
    last = _rec(2, 0.10)
    s = summarise_folds([best, best])
    assert s["mean_f1"] == pytest.approx(0.90)
    assert s["mean_accuracy"] == pytest.approx(0.8)
    assert s["mean_f1"] != pytest.approx(last.val_f1)


@pytest.mark.parametrize("empty", [[], ()])
def test_neither_function_accepts_a_sequence_with_nothing_in_it(empty):
    # Both are otherwise divisions over nothing, and `max()` over nothing raises
    # CPython's `max() arg is an empty sequence`, which names no rule and no argument.
    with pytest.raises(ValueError, match="no epoch"):
        select_epoch(empty)
    with pytest.raises(ValueError, match="no fold"):
        summarise_folds(empty)


# ---------------------------------------------------------------------------
# One fold: what it trains on, what it validates on, and what it writes.
# ---------------------------------------------------------------------------


def test_a_fold_validates_on_every_row_of_the_specimen_it_holds_out(tmp_path):
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    fold = loso_folds(specimens)[0]
    _selected, records = train_fold(fold, paths, labels, cfg, tmp_path / "out")

    assert len(records) == 2, "one record per epoch, for training.epochs epochs"
    assert [record.epoch for record in records] == [1, 2]
    assert {record.fold for record in records} == {fold.fold}
    for record in records:
        # Every validation row is counted exactly once, and no other row is.
        assert record.tp + record.fp + record.fn + record.tn == len(fold.val_idx) == 2
        assert 0.0 <= record.val_f1 <= 1.0
        assert 0.0 <= record.val_accuracy <= 1.0
        assert record.val_accuracy == pytest.approx(
            (record.tp + record.tn) / (record.tp + record.fp + record.fn + record.tn)
        )


def test_a_fold_returns_the_selected_epoch_among_the_records_it_wrote(tmp_path):
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    selected, records = train_fold(loso_folds(specimens)[0], paths, labels, cfg, tmp_path / "out")
    assert selected in records
    assert selected is select_epoch(records)


def test_the_class_weights_are_built_from_the_folds_training_labels_only(tmp_path, monkeypatch):
    # The leak the per-fold correction exists to close: `train.py:87` counted the classes
    # of the pooled cohort and handed that one tensor to every fold. Specimen A's rows are
    # class 1 and specimen B's are class 0, so the fold holding out A can only ever see the
    # two zeros, and a pooled count would see four labels instead of two.
    handed: list[list[int]] = []
    real = train_module.class_weights

    def spy(labels, num_classes=2):
        handed.append([int(label) for label in labels])
        return real(labels, num_classes)

    _stand_in(monkeypatch, _StandIn(positive=True))
    monkeypatch.setattr(train_module, "class_weights", spy)
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    run_cross_validation(paths, labels, specimens, cfg, tmp_path / "out")
    # Two calls per fold, and both are named in `train_fold`: the fold's training labels,
    # whose vector the loss is built from, and the fold's validation labels, whose vector
    # is discarded and whose refusal is the point. Fold 0 holds out specimen A and trains
    # on B's two class-0 rows; fold 1 holds out B and trains on A's two class-1 rows.
    assert handed == [[0, 0], [1, 1], [1, 1], [0, 0]]


def test_every_patch_is_rendered_through_the_transform_build_transforms_returned(
    tmp_path, monkeypatch
):
    # `PatchDataset(transform=None)` applies `ToTensor` and nothing refuses a caller who
    # forgets, so a fold built that way would train on unresized, unnormalized pixels and
    # report a number. The train fold must render through the train transform and the
    # validation fold through the eval one, and the only way to see that is to count.
    seen = {"train": 0, "eval": 0}
    real = train_module.build_transforms

    def spy(cfg):
        train_transform, eval_transform = real(cfg)

        def counting(which, inner):
            def wrapper(image):
                seen[which] += 1
                return inner(image)
            return wrapper

        return counting("train", train_transform), counting("eval", eval_transform)

    _stand_in(monkeypatch, _StandIn(positive=True))
    monkeypatch.setattr(train_module, "build_transforms", spy)
    cfg = _cfg("training.epochs=1")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    train_fold(loso_folds(specimens)[0], paths, labels, cfg, tmp_path / "out")
    # Two training rows and two validation rows, one epoch.
    assert seen == {"train": 2, "eval": 2}


def test_a_checkpoint_is_written_once_when_validation_f1_never_improves(tmp_path, monkeypatch):
    saves = _stand_in(monkeypatch, _StandIn(positive=True))
    cfg = _cfg("training.epochs=4", "training.patience_early_stop=9")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 1), ("B", 1)])
    fold = loso_folds(specimens)[0]
    selected, records = train_fold(fold, paths, labels, cfg, tmp_path / "out")

    assert len(records) == 4
    assert {record.val_f1 for record in records} == {1.0}
    assert selected.epoch == 1, "a tie on F1 selects the earliest epoch"
    # One write, for the first epoch. `>` and not `>=`, which would write four.
    assert saves == [tmp_path / "out" / f"best_model_fold{fold.fold}.pth"]


def test_a_checkpoint_is_written_on_every_strict_improvement_and_on_no_equal_one(
    tmp_path, monkeypatch
):
    saves = _stand_in(monkeypatch, _StandIn(positive=True, mode="improving"))
    cfg = _cfg("training.epochs=4", "training.patience_early_stop=9")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 1), ("B", 1)])
    fold = loso_folds(specimens)[0]
    selected, records = train_fold(fold, paths, labels, cfg, tmp_path / "out")

    assert [record.val_f1 for record in records] == [0.0, 1.0, 1.0, 1.0]
    assert selected.epoch == 2
    # Epoch 1 improves on nothing and epoch 2 improves on epoch 1. Epochs 3 and 4 only
    # equal epoch 2, and an equal F1 is not an improvement.
    assert len(saves) == 2
    assert saves[0] == saves[1]


def test_a_fold_stops_after_the_configured_epochs_without_improvement(tmp_path, monkeypatch):
    _stand_in(monkeypatch, _StandIn(positive=False))
    cfg = _cfg("training.epochs=6", "training.patience_early_stop=2")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 1), ("B", 1)])
    _selected, records = train_fold(loso_folds(specimens)[0], paths, labels, cfg, tmp_path / "out")
    # Epoch 1 improves, epochs 2 and 3 do not, and the third is the patience.
    assert [record.epoch for record in records] == [1, 2, 3]


def test_the_learning_rate_a_record_carries_is_the_one_its_own_epoch_ran_under(
    tmp_path, monkeypatch
):
    # Measured on this stack, with `patience_lr_scheduler=1`: the rate halves on the
    # scheduler step of the third epoch, because `ReduceLROnPlateau` waits for
    # `patience + 1` epochs in a row that fail to improve the validation loss. A record
    # carries the rate that was in force while its epoch trained, read before the step, so
    # the reduction lands on the *next* record and not on the epoch that caused it.
    _stand_in(monkeypatch, _StandIn(positive=False))
    cfg = _cfg("training.epochs=8", "training.learning_rate=0.5",
               "training.patience_early_stop=99", "training.patience_lr_scheduler=1")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 1), ("B", 1)])
    _selected, records = train_fold(loso_folds(specimens)[0], paths, labels, cfg, tmp_path / "out")

    assert [record.lr for record in records] == pytest.approx(
        [0.5, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125, 0.0625]
    )


def test_the_validation_loss_is_a_number_on_a_fold_that_trains_on_one_class_only(tmp_path):
    # `class_weights` gives a class the fold holds none of a weight of `0.0` rather than
    # `inf`, and this fold trains on two class-0 rows and validates on two class-1 rows. A
    # weighted validation loss divides by the sum of the weights of its targets, which is
    # zero here: measured, `nn.CrossEntropyLoss(weight=torch.tensor([0.5, 0.0]))` over two
    # class-1 targets is `nan` where the unweighted loss is 0.693. And `nan` is the number
    # `ReduceLROnPlateau` steps on.
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    fold = loso_folds(specimens)[0]
    assert sorted(set(labels[index] for index in fold.train_idx)) == [0]
    _selected, records = train_fold(fold, paths, labels, cfg, tmp_path / "out")
    for record in records:
        assert record.val_loss == record.val_loss, "a number and not a nan"
        assert record.train_loss == record.train_loss
        assert 0.0 <= record.val_loss <= 100.0


def test_a_label_outside_the_class_range_is_refused_before_it_reaches_the_loss(tmp_path):
    # `class_weights` refuses a label outside `range(num_classes)`, and the fold's own
    # validation rows are labels that no other fold's weight call has looked at: a cohort
    # whose held-out specimen carries a `5` would otherwise reach `CrossEntropyLoss` as a
    # target it cannot score, and the report of that arrives attributed to the loss.
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 5), ("A", 1), ("B", 0), ("B", 0)])
    fold = loso_folds(specimens)[0]
    with pytest.raises(ValueError, match="outside range"):
        train_fold(fold, paths, labels, cfg, tmp_path / "out")


def test_a_run_writes_nothing_under_the_torch_home(tmp_path, monkeypatch):
    home = tmp_path / "torch_home"
    home.mkdir()
    monkeypatch.setenv("TORCH_HOME", str(home))
    cfg = _cfg()
    assert cfg.get("training.pretrained") is False
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    run_cross_validation(paths, labels, specimens, cfg, tmp_path / "out")
    assert sorted(home.rglob("*")) == []


# ---------------------------------------------------------------------------
# The run over every fold.
# ---------------------------------------------------------------------------


def test_a_run_writes_every_epoch_to_the_log_and_only_the_selected_to_the_summary(tmp_path):
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    out = tmp_path / "out"
    summary = run_cross_validation(paths, labels, specimens, cfg, out)

    assert summary["n_folds"] == 2
    assert summary["selected_epochs"] == sorted(
        summary["selected_epochs"], key=lambda entry: entry["fold"]
    )
    rows = _read_csv(out / "fold_summary.csv")
    assert len(rows) == 2
    for row in rows:
        fold_number = int(row["fold"])
        log = _read_csv(out / f"training_log_fold{fold_number}.csv")
        assert len(log) == 2, "the log holds every epoch the fold ran"
        assert [entry["epoch"] for entry in log] == ["1", "2"]
        # The summary row is that fold's selected row, verbatim, and names the specimen.
        assert row in log
        assert row["epoch"] == str(
            next(entry["epoch"] for entry in summary["selected_epochs"]
                 if entry["fold"] == fold_number)
        )
        assert row["held_out"] in specimens
    assert summary["mean_f1"] == pytest.approx(
        np.mean([float(row["val_f1"]) for row in rows])
    )
    assert summary["mean_f1_std"] == pytest.approx(
        np.std([float(row["val_f1"]) for row in rows])
    )
    assert (out / "best_model_fold0.pth").is_file()
    assert (out / "best_model_fold1.pth").is_file()


def test_the_summary_and_the_logs_carry_the_specimen_each_fold_held_out(tmp_path, monkeypatch):
    _stand_in(monkeypatch, _StandIn(positive=True))
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    out = tmp_path / "out"
    run_cross_validation(paths, labels, specimens, cfg, out)
    # `loso_folds` numbers the folds from zero in `sorted()` order of the distinct ids.
    assert [row["held_out"] for row in _read_csv(out / "fold_summary.csv")] == ["A", "B"]
    for fold_number, specimen in enumerate(("A", "B")):
        log = _read_csv(out / f"training_log_fold{fold_number}.csv")
        assert {row["held_out"] for row in log} == {specimen}
        assert {row["fold"] for row in log} == {str(fold_number)}


def test_a_cohort_of_one_specimen_is_refused_before_anything_is_written(tmp_path):
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 0)])
    with pytest.raises(SplitError, match="at least two distinct specimen ids"):
        run_cross_validation(paths, labels, specimens, cfg, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_handoff_reader_is_in_the_package_and_reads_the_configured_label_root(tmp_path):
    # The reader is here and not in `bin/train.py` because `bin/` is a directory of scripts
    # loaded by path: a read of `paths.labels` in there is outside the ledger scan in
    # `tests/test_config.py`, which then reports a declared key nothing under `blobcount/`
    # reads. Task 12 reads the same handoff, so one reader rather than two copies.
    #
    # **The configuration is built by an override and not by `cfg.set`, and that is the
    # difference between an assertion and a tautology.** `Config.set` marks the key it
    # writes as read, so a test that wrote `paths.labels` with `set` and then asserted
    # `"paths.labels" not in cfg.unknown_keys()` was asserting that `set` had run — which
    # it had, on the line above. The assertion passed whether or not `read_labels` read
    # anything at all, which is what `tests/test_config.py::
    # test_set_marks_the_key_it_writes_which_is_why_no_unknown_keys_assertion_is_set_up_with_one`
    # records. `load(overrides=...)` marks nothing, so the ledger below is what the call
    # under test actually did.
    rows = list(_ROWS)
    index = _write_labels_csv(tmp_path / "handoff", rows)
    cfg = _cfg(f"paths.labels={tmp_path / 'labels'}")

    paths, labels, specimens = read_labels(index, cfg)

    assert specimens == ["A", "A", "B", "B"]
    assert labels == [0, 1, 0, 0]
    assert paths == [tmp_path / "labels" / row[5] for row in rows]
    # The key is read and not merely named: `Config` records every value it serves, so a
    # `paths.labels` this call did not use is the one still missing from the ledger.
    assert "paths.labels" not in cfg.unknown_keys()
    # **And the other half of the same claim**: `paths.results` is a declared key this call
    # does not read, and it is still on the ledger afterwards. Without this arm the
    # assertion above is satisfied by any `Config` whose keys have all been read by
    # something, including this test's own setup.
    assert "paths.results" in cfg.unknown_keys()


def test_a_run_given_no_output_directory_writes_into_the_configured_one(tmp_path, monkeypatch):
    # `out_dir` is an argument so a caller can put a run somewhere else, and its absence
    # is the shipped case: the directory is the configured `paths.results`.
    #
    # **An override and not `cfg.set`, for the reason the handoff-reader test above
    # gives**: `set` marks the key it writes, so setting `paths.results` that way and then
    # asserting it is no longer unread asserts that the `set` ran. Here the assertion
    # below is about `run_cross_validation` having read the key, and the arm after it is
    # about a key it did not, so the two together are a claim with a direction.
    _stand_in(monkeypatch, _StandIn(positive=True))
    cfg = _cfg(f"paths.results={tmp_path / 'results'}")
    paths, labels, specimens = _cohort(
        tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)]
    )
    summary = run_cross_validation(paths, labels, specimens, cfg)

    out = tmp_path / "results"
    assert summary["n_folds"] == 2
    assert (out / "fold_summary.csv").is_file()
    assert (out / "training_log_fold0.csv").is_file()
    assert (out / "training_log_fold1.csv").is_file()
    assert (out / "best_model_fold0.pth").is_file()
    assert "paths.results" not in cfg.unknown_keys()
    # `paths.labels` is declared and is not read by a cross-validation run that is handed
    # its own cohort, so it is still on the ledger after this one. Without this the
    # assertion above would also hold for a run that read nothing but its own setup.
    assert "paths.labels" in cfg.unknown_keys()


def test_the_folds_are_numbered_from_zero_in_sorted_order_of_the_specimen_ids():
    # The numbers a run writes into a file name and a column, read here from `loso_folds`
    # itself rather than restated.
    folds = loso_folds(["B", "A", "C", "A"])
    assert [(fold.fold, fold.held_out) for fold in folds] == [(0, "A"), (1, "B"), (2, "C")]


# ---------------------------------------------------------------------------
# The entry point.
# ---------------------------------------------------------------------------


def _load_cli(monkeypatch):
    path = PROJECT_ROOT / "bin" / "train.py"
    spec = importlib.util.spec_from_file_location("blobcount_cli_train", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_labels_csv(directory: Path, rows) -> Path:
    """Write a `labels.csv` under `directory` and return its path.

    The header is `blobcount.labeling.LABEL_COLUMNS`, and `path` is written relative to the
    label root and separated by `/`, which is what `blobcount.labeling._write_labels`
    writes. Every test that calls this puts the file in a directory that is **not** the
    label root — a sibling of it, or a child of it — so no test here can pass by resolving a
    row against the directory the `labels.csv` happens to sit in, and one of them plants a
    decoy there to make sure it does not.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "labels.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)
    return target


def _cli_config(directory: Path, labels_root: Path, results: Path, **training) -> Path:
    """Write a configuration file for the command line, and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["labels"] = str(labels_root)
    data["paths"]["results"] = str(results)
    data["training"].update({
        "pretrained": False, "input_size": 32, "batch_size": 2, "num_workers": 0, "epochs": 2,
    })
    data["training"].update(training)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _label_row(specimen, label, relative):
    return [f"{specimen}_{relative.replace('/', '_').removesuffix('.png')}",
            specimen, label, 7, "0.5", relative]


_ROWS = [
    _label_row("A", 0, "normal/A_0.png"),
    _label_row("A", 1, "steatosis/A_1.png"),
    _label_row("B", 0, "normal/B_2.png"),
    _label_row("B", 0, "normal/B_3.png"),
]


@pytest.fixture
def cli(project_scratch, tmp_path, monkeypatch):
    """The entry point, a `labels.csv` beside no patches at all, and two output roots.

    The patches are written under `paths.labels` and the handoff sits in a directory
    holding nothing else, so every run below resolves a row's path against the configured
    label root or fails to find the patch at all. **`results` is inside the project**,
    because this command line refuses an output target that is not strictly below the
    project root and pytest's `tmp_path` is outside it — the reason
    `tests/conftest.py`'s `project_scratch` exists, and the reason the handoff and the
    label root stay in `tmp_path`: nothing this program writes goes there.
    """
    module = _load_cli(monkeypatch)
    label_root = tmp_path / "labels"
    for number, row in enumerate(_ROWS):
        patch = label_root / row[5]
        patch.parent.mkdir(parents=True, exist_ok=True)
        _write_patch(patch, size=32, seed=number)
    index = _write_labels_csv(tmp_path / "handoff", _ROWS)
    results = project_scratch / "results"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(tmp_path, label_root, results),
    )
    return module, index, label_root, results


def _printed_tables(printed: str) -> list[tuple[list[str], list[list[str]]]]:
    """Return `(header, rows)` for each table the command line printed.

    The tables are the only thing this program puts on stdout, and their shape is the one
    `bin/ablate_threshold.py` writes: two or more spaces between columns, a rule of dashes
    under the header, and a blank line between tables. Cells are split on two or more
    spaces rather than one so a specimen name carrying a space stays in one cell, and a
    block of one cell is not a table — which is how the closing sentence is skipped.
    """
    tables: list[tuple[list[str], list[list[str]]]] = []
    block: list[list[str]] = []
    for line in [*printed.splitlines(), ""]:
        if line.strip():
            block.append([cell for cell in re.split(r"\s{2,}", line.strip()) if cell])
            continue
        rows = [row for row in block if not all(set(cell) <= {"-"} for cell in row)]
        if rows and len(rows[0]) > 1:
            tables.append((rows[0], rows[1:]))
        block = []
    return tables


def test_the_printed_fold_table_carries_the_numbers_its_csv_row_holds(cli, capsys, monkeypatch):
    # **The table is the human-facing report, and a swapped column misreports the study
    # while every artifact on disk stays correct.** So the numbers are asserted as values,
    # twice: once against what this cohort is known to produce and once against the
    # `fold_summary.csv` row the same run wrote.
    #
    # A stand-in backbone predicting class 1 for every patch is what makes them known.
    # Fold 0 holds out specimen A — one class-0 row and one class-1 row — so precision is
    # 1/2 and recall is 1/1, which are different numbers; fold 1 holds out two class-0 rows
    # and so reports zero for all four. Every cell therefore differs from at least one
    # other cell in the column it is in, and no swap can land on an equal value.
    _stand_in(monkeypatch, _StandIn(positive=True))
    module, index, _label_root, results = cli
    assert module.main(["--labels", str(index), "--out", str(results)]) == 0

    tables = _printed_tables(capsys.readouterr().out)
    assert len(tables) == 2, "the per-fold table and the mean table"
    header, printed = tables[0]
    assert tuple(header) == ("fold", "specimen", "epoch", "accuracy", "precision",
                             "recall", "f1", "lr")

    assert [cells[1] for cells in printed] == ["A", "B"]
    assert [cells[2] for cells in printed] == ["1", "1"]
    assert [cells[3] for cells in printed] == ["0.500", "0.000"]
    assert [cells[4] for cells in printed] == ["0.500", "0.000"]
    assert [cells[5] for cells in printed] == ["1.000", "0.000"]
    assert [cells[6] for cells in printed] == ["0.667", "0.000"]

    written = _read_csv(results / "fold_summary.csv")
    assert len(printed) == len(written) == 2
    for cells, row in zip(printed, written):
        assert cells[0] == row["fold"]
        assert cells[1] == row["held_out"]
        assert cells[2] == row["epoch"]
        for position, field in ((3, "val_accuracy"), (4, "precision"),
                                (5, "recall"), (6, "val_f1")):
            assert cells[position] == f"{float(row[field]):.3f}", field
        assert cells[7] == f"{float(row['lr']):.6f}"


def test_the_printed_mean_table_carries_the_summary_numbers(cli, capsys, monkeypatch):
    # The mean table is rendered out of the summary dict rather than out of the file, so
    # what it prints is pinned twice: against the dict `training_summary.json` carries and
    # against the values this cohort is known to produce. **The row order is asserted too**
    # — a reordering of the metric literal still prints each number under its own name, so
    # only the order tells a reader which metric a row is.
    _stand_in(monkeypatch, _StandIn(positive=True))
    module, index, _label_root, results = cli
    assert module.main(["--labels", str(index), "--out", str(results)]) == 0

    tables = _printed_tables(capsys.readouterr().out)
    header, printed = tables[1]
    assert tuple(header) == ("metric", "mean", "population std")
    assert [cells[0] for cells in printed] == ["accuracy", "precision", "recall", "f1"]

    summary = json.loads((results / "training_summary.json").read_text(encoding="utf-8"))
    for cells in printed:
        assert cells[1] == f"{summary['summary']['mean_' + cells[0]]:.3f}"
        assert cells[2] == f"{summary['summary']['mean_' + cells[0] + '_std']:.3f}"

    # Two folds at 0.5 and 0.0 for accuracy and precision, at 1.0 and 0.0 for recall, and at
    # 0.6667 and 0.0 for F1; the population deviation of each pair is its half-range.
    assert [cells[1] for cells in printed] == ["0.250", "0.250", "0.500", "0.333"]
    assert [cells[2] for cells in printed] == ["0.250", "0.250", "0.500", "0.333"]


def test_the_command_line_writes_the_logs_the_summary_and_the_json(cli, capsys):
    module, index, _label_root, results = cli
    assert module.main(["--labels", str(index), "--out", str(results)]) == 0

    assert (results / "training_summary.json").is_file()
    assert (results / "fold_summary.csv").is_file()
    assert (results / "training_log_fold0.csv").is_file()
    assert (results / "training_log_fold1.csv").is_file()

    written = json.loads((results / "training_summary.json").read_text(encoding="utf-8"))
    assert written["summary"]["n_folds"] == 2
    assert [fold["held_out"] for fold in written["per_fold"]] == ["A", "B"]

    printed = capsys.readouterr().out
    # The per-fold table of selected epochs first, and the mean table under it.
    assert printed.index("specimen") < printed.index("std")
    assert "fold" in printed


def test_the_command_line_writes_where_the_configuration_points_when_out_is_absent(cli):
    module, index, _label_root, results = cli
    assert module.main(["--labels", str(index)]) == 0
    written = json.loads((results / "training_summary.json").read_text(encoding="utf-8"))
    assert written["summary"]["n_folds"] == 2


def test_every_row_path_is_resolved_against_the_configured_label_root(cli, tmp_path):
    # A decoy at every row's relative path, inside the directory `labels.csv` sits in and
    # carrying the right row structure but not a patch: a resolution against the file's own
    # directory, or against the working directory, opens one of these and `PIL` refuses it
    # out of `PatchDataset.__getitem__`, which this project deliberately does not wrap.
    module, index, _label_root, results = cli
    for row in _ROWS:
        decoy = index.parent / row[5]
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_text("not an image\n", encoding="utf-8")
    assert module.main(["--labels", str(index), "--out", str(results)]) == 0


def test_a_row_path_that_leaves_the_label_root_is_refused(cli, capsys):
    module, index, label_root, results = cli
    index = _write_labels_csv(label_root / "handoff", [
        _label_row("A", 1, "../outside.png"),
        _label_row("A", 1, "../outside.png"),
        _label_row("B", 0, "../outside.png"),
    ])
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "label root" in capsys.readouterr().err
    assert not results.exists()


def test_an_output_target_that_is_a_file_is_refused(cli, capsys):
    # `paths.results` is a hand-edited string and `--out` is typed by hand too, and a run
    # writes one 45 MB checkpoint per fold into whatever it is given. A target that exists
    # as a file is reported as the typo it is, before a patch is read.
    module, index, _label_root, results = cli
    occupied = results / "occupied.txt"
    occupied.parent.mkdir(parents=True, exist_ok=True)
    occupied.write_text("x", encoding="utf-8")
    assert module.main(["--labels", str(index), "--out", str(occupied)]) == 2
    assert "refusing to write" in capsys.readouterr().err
    assert occupied.read_text(encoding="utf-8") == "x"
    assert not (results / "fold_summary.csv").exists()


def test_an_output_target_outside_the_project_is_refused(cli, tmp_path, monkeypatch, capsys):
    # Re-pointed at the invariant that survives K14: a directory that is a *sibling*
    # of the tree the configuration says this project owns — not the configured path
    # itself, and not below it — is refused. The old staging refused an arbitrary
    # directory under pytest's `tmp_path`, which the widened guard now owns, because
    # the same configuration named a path there; that expectation encoded "everything
    # outside the checkout is refused", the invariant K14 replaced. The mistyped
    # `--out` this guard exists for is the near-miss — one directory off the owned
    # tree — and naming it is a stronger assertion than naming an arbitrary location,
    # because it pins the edge the widened guard still has to hold.
    module, index, _label_root, results = cli
    owned = tmp_path / "owned" / "labels"
    owned.mkdir(parents=True)
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(tmp_path, owned, results),
    )
    sibling = tmp_path / "owned-typo"
    sibling.mkdir()
    assert module.main(["--labels", str(index), "--out", str(sibling)]) == 2
    assert "project root" in capsys.readouterr().err
    assert sorted(path.name for path in sibling.iterdir()) == []


def test_the_project_root_is_not_an_output_target(cli, capsys):
    module, index, _label_root, _results = cli
    assert module.main(["--labels", str(index), "--out", str(PROJECT_ROOT)]) == 2
    assert "project root" in capsys.readouterr().err


def test_an_empty_label_root_is_refused_by_the_command_line(cli, tmp_path, monkeypatch, capsys):
    # `read_labels` is what resolves every row against `paths.labels`, so `Config.path`'s
    # refusal of an empty value arrives out of the read rather than out of the arm around
    # `load`. The refusal is the same either way — an answer about the configuration, on
    # `EXIT_REFUSED`, with no traceback — and this pins which arm it arrives from.
    module, index, _label_root, results = cli
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["labels"] = ""
    empty = tmp_path / "empty_label_root.yaml"
    empty.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", empty)
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "paths.labels" in capsys.readouterr().err


def test_a_labels_file_that_is_not_there_is_refused(cli, capsys):
    module, index, _label_root, results = cli
    assert module.main(["--labels", str(index.parent / "absent.csv"),
                        "--out", str(results)]) == 2
    assert "absent.csv" in capsys.readouterr().err


def test_a_labels_csv_that_is_not_the_handoff_is_refused(cli, capsys):
    module, _index, label_root, results = cli
    index = label_root / "other.csv"
    index.write_text("patch_id,label\nA_0,1\n", encoding="utf-8")
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "is not a labels.csv" in capsys.readouterr().err


def test_a_label_cell_that_is_not_a_whole_number_is_refused(cli, capsys):
    module, _index, label_root, results = cli
    index = _write_labels_csv(label_root / "handoff", [
        _label_row("A", "steatosis", "steatosis/A_0.png"),
        _label_row("A", 1, "steatosis/A_1.png"),
        _label_row("B", 0, "normal/B_2.png"),
    ])
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "line 2" in capsys.readouterr().err


def test_a_cell_wider_than_the_csv_field_size_limit_is_refused(cli, capsys):
    # `csv.DictReader` raises `csv.Error` on a cell over `csv.field_size_limit` rather than
    # truncating it, and that is neither an `OSError` nor a `ValueError`: an arm that left
    # it out would let a malformed handoff reach a caller as a traceback out of `main`.
    module, _index, label_root, results = cli
    index = label_root / "handoff" / "wide.csv"
    index.parent.mkdir(parents=True, exist_ok=True)
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerow(["A_0", "A", 1, 7, "0.5", "x" * (csv.field_size_limit() + 1)])
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "field larger than field limit" in capsys.readouterr().err


def test_a_label_outside_the_class_range_is_refused_by_the_command_line(cli, capsys):
    module, _index, label_root, results = cli
    index = _write_labels_csv(label_root / "handoff", [
        _label_row("A", 5, "steatosis/A_0.png"),
        _label_row("A", 1, "steatosis/A_1.png"),
        _label_row("B", 0, "normal/B_2.png"),
    ])
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "5" in capsys.readouterr().err


def test_a_class_count_the_head_cannot_be_built_for_is_refused(cli, tmp_path, monkeypatch, capsys):
    module, index, label_root, results = cli
    # The output target is the fixture's, which is inside the project: this test is about
    # the class count, and a `--out` outside the project would be refused earlier and for
    # an unrelated reason.
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(tmp_path / "other", label_root, results, num_classes=0),
    )
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "num_classes" in capsys.readouterr().err


def test_a_configuration_the_project_will_not_load_is_refused(cli, tmp_path, monkeypatch, capsys):
    module, index, _label_root, results = cli
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    del data["training"]["architecture"]
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "training.architecture" in capsys.readouterr().err


def test_a_cohort_of_one_specimen_is_refused_by_the_command_line(cli, capsys):
    module, _index, label_root, results = cli
    index = _write_labels_csv(label_root / "handoff", [
        _label_row("A", 1, "steatosis/A_0.png"),
        _label_row("A", 0, "steatosis/A_1.png"),
    ])
    assert module.main(["--labels", str(index), "--out", str(results)]) == 2
    assert "specimen" in capsys.readouterr().err


def test_the_command_line_defines_no_code_for_a_contradictory_run(cli):
    module, _index, _label_root, _results = cli
    assert module.EXIT_OK == 0
    assert module.EXIT_REFUSED == 2
    assert not hasattr(module, "EXIT_CONTRADICTORY")


def test_the_shipped_configuration_asks_for_pretrained_weights_and_every_test_overrides_it():
    # The reason `_TEST_TRAINING` carries `training.pretrained=false`: the file this
    # project ships would otherwise download ImageNet weights on the first fold of a real
    # run. `test_a_run_writes_nothing_under_the_torch_home` is the measurement that a run
    # built from the overridden value downloads nothing.
    shipped = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert shipped["training"]["pretrained"] is True
    assert _cfg().get("training.pretrained") is False
    assert load().get("training.pretrained") is True


def test_every_validation_row_lands_in_exactly_one_bucket_of_the_confusion_matrix(
    tmp_path, monkeypatch
):
    # A constant backbone predicting class 0 for every patch, and a held-out specimen with
    # one class-0 row and one class-1 row, so the four counts are known before the fold
    # runs: nothing is predicted positive, so tp and fp are zero, and each row falls to tn
    # or fn according to its own label. Specimen B is class 0 throughout, so a fold that
    # validated on the training rows instead would report `(0, 0, 0, 2)` rather than
    # `(0, 0, 1, 1)` — the counts are what pins which rows were validated on.
    _stand_in(monkeypatch, _StandIn(positive=False))
    cfg = _cfg("training.epochs=1")
    paths, labels, specimens = _cohort(
        tmp_path / "patches", [("A", 0), ("A", 1), ("B", 0), ("B", 0)]
    )
    fold = loso_folds(specimens)[0]
    _selected, records = train_fold(fold, paths, labels, cfg, tmp_path / "out")
    record = records[0]
    assert (record.tp, record.fp, record.fn, record.tn) == (0, 0, 1, 1)
    assert record.val_accuracy == pytest.approx(0.5)
    # No predicted positive: precision has no denominator and recall is zero of one, so
    # both are `0.0` rather than a division by zero.
    assert record.precision == 0.0
    assert record.recall == 0.0
    assert record.val_f1 == 0.0


def test_the_same_configuration_trains_the_same_fold_twice(tmp_path):
    # `train_fold` seeds from `project.seed` and the fold's own number, so a run repeated is
    # a run reproduced: the same head initialisation, the same sampler draw and the same
    # augmentation stream. The real backbone is what makes this a measurement — a stand-in
    # model carries the same weights into both runs whatever the global RNG does, and
    # removing `set_seed` would leave the two runs identical.
    cfg = _cfg("training.epochs=2")
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    fold = loso_folds(specimens)[0]
    _first, first_records = train_fold(fold, paths, labels, cfg, tmp_path / "out")
    _second, second_records = train_fold(fold, paths, labels, cfg, tmp_path / "out")
    assert [record.train_loss for record in first_records] == [
        record.train_loss for record in second_records
    ]
    assert [record.val_f1 for record in first_records] == [
        record.val_f1 for record in second_records
    ]
    assert [record.tp for record in first_records] == [
        record.tp for record in second_records
    ]


def test_a_shorter_second_run_leaves_the_first_runs_extra_fold_log(tmp_path, monkeypatch):
    # Nothing is cleared: a file of the same name is rewritten in place and a file the
    # second run does not produce stays where the first one put it. Three specimens make
    # three folds, two make two, and fold 2's log is the one that survives.
    _stand_in(monkeypatch, _StandIn(positive=True))
    cfg = _cfg("training.epochs=1")
    out = tmp_path / "out"
    three = _cohort(tmp_path / "patches", [
        ("A", 1), ("A", 1), ("B", 0), ("B", 0), ("C", 1), ("C", 0),
    ])
    run_cross_validation(*three, cfg, out)
    assert sorted(path.name for path in out.glob("*.csv")) == [
        "fold_summary.csv", "training_log_fold0.csv",
        "training_log_fold1.csv", "training_log_fold2.csv",
    ]
    two = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    run_cross_validation(*two, cfg, out)
    assert sorted(path.name for path in out.glob("*.csv")) == [
        "fold_summary.csv", "training_log_fold0.csv",
        "training_log_fold1.csv", "training_log_fold2.csv",
    ]
    assert len(_read_csv(out / "fold_summary.csv")) == 2


def test_a_fold_that_fails_leaves_the_folds_before_it_on_disk(tmp_path, monkeypatch):
    # The log of a fold is written as that fold finishes, so a run refused at the second
    # fold leaves the first fold's log rather than none of them — the order
    # `bin/ablate_threshold.py` creates its directory after the counts it may not have.
    real = train_module.train_fold

    def spy(fold, *args, **kwargs):
        if fold.fold == 1:
            raise ValueError("injected from the second fold")
        return real(fold, *args, **kwargs)

    _stand_in(monkeypatch, _StandIn(positive=True))
    monkeypatch.setattr(train_module, "train_fold", spy)
    cfg = _cfg()
    paths, labels, specimens = _cohort(tmp_path / "patches", [("A", 1), ("A", 1), ("B", 0), ("B", 0)])
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="injected from the second fold"):
        run_cross_validation(paths, labels, specimens, cfg, out)
    assert (out / "training_log_fold0.csv").is_file()
    assert not (out / "training_log_fold1.csv").exists()
    assert not (out / "fold_summary.csv").exists()