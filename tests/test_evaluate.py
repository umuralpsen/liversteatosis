"""Tests for the metric record, the integrity guard, the intervals and the entry point.

**Every test here scores a model this file wrote, and none of them reaches the network.**
`dataset/labels/` does not exist, the extraction that would fill it has never been run and
no checkpoint has ever been trained, so there is no cohort to evaluate and no weights to
load: the cohort below is a handful of uniform PNGs written into `tmp_path`, and the backbone
is a stand-in whose every prediction this file chooses. Every number asserted below is
measured over those synthetic patches, and a reader of this file must not take any of it for
a measurement of a specimen.
"""

import csv
import dataclasses
import importlib.util
import json
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from blobcount.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load
from blobcount.evaluate import (EvaluationIntegrityError, Metrics, assert_disjoint,
                                 bootstrap_ci, bootstrap_metric_intervals,
                                 evaluate_specimens, quadratic_weighted_kappa)
from blobcount.labeling import LABEL_COLUMNS

# The shipped `training.*` values, every one of them replaced for a test. `pretrained` is
# the one that matters: `configs/default.yaml` ships it true, and a call that reached
# `build_model` with it would ask torchvision for ImageNet weights over the network.
_TEST_TRAINING = [
    "training.pretrained=false",
    "training.input_size=32",
    "training.batch_size=2",
    "training.num_workers=0",
]


def _cfg(*overrides: str):
    """Return a Config with the test-safe training keys, plus `overrides` on top."""
    return load(overrides=[*_TEST_TRAINING, *overrides])


@contextmanager
def _no_warning():
    """Fail the test on any warning raised inside the block.

    `pyproject.toml` already promotes warnings to errors for the suite; this makes the claim
    explicit inside the test that is about it, so the assertion does not rest on a
    configuration file this task did not write.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


def _record(**overrides) -> Metrics:
    """Return a `Metrics` holding a distinguishable value in every field."""
    values = {
        "accuracy": 0.9, "precision": 0.8, "recall": 0.7, "specificity": 0.6,
        "f1": 0.75, "npv": 0.65, "auc": 0.8, "tp": 7, "fp": 1, "fn": 2, "tn": 6, "n": 16,
    }
    values.update(overrides)
    return Metrics(**values)


# ---------------------------------------------------------------------------
# The brief's tests, in the brief's order.
# ---------------------------------------------------------------------------


def test_overlap_between_trained_and_evaluated_specimens_raises():
    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        evaluate_specimens(None, ["a.png"], [0], ["A"], ["A"], None)


def test_disjoint_specimens_do_not_raise():
    assert_disjoint(["A", "B"], ["C"])


def test_bootstrap_ci_brackets_the_mean():
    lo, hi = bootstrap_ci([0.8, 0.85, 0.9, 0.82, 0.88], 2000, 0.95, 42)
    assert lo < 0.86 < hi


def test_bootstrap_ci_is_seed_stable():
    a = bootstrap_ci([0.8, 0.85, 0.9], 500, 0.95, 7)
    b = bootstrap_ci([0.8, 0.85, 0.9], 500, 0.95, 7)
    assert a == b


def test_kappa_is_one_for_perfect_agreement():
    assert quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 3) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_level_disagreement():
    y = [0, 0, 1, 1]
    assert quadratic_weighted_kappa(y, [0, 1, 0, 1], 1) == pytest.approx(0.0, abs=1e-9)


def test_kappa_penalises_large_errors_more_than_small():
    small = quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 3, 2], 3)
    large = quadratic_weighted_kappa([0, 1, 2, 3], [3, 2, 1, 0], 3)
    assert small > large


# ---------------------------------------------------------------------------
# The guard: what it refuses, what it names, and when it runs.
# ---------------------------------------------------------------------------


def test_the_guard_names_the_intersection_first_and_then_both_lists():
    with pytest.raises(EvaluationIntegrityError) as caught:
        assert_disjoint(["A", "B", "C"], ["B", "D"])
    message = str(caught.value)
    # The word the refusal is about, then the intersection and nothing else in it: a caller
    # who removed `C` or `D` would have removed nothing, and a message listing the whole of
    # both lists as the overlap would send them to do that. The exact clause is asserted so
    # a message that merely *contained* the letter somewhere could not satisfy it.
    assert message.startswith(
        "overlap between the specimens trained on and the specimens being scored: B."
    ), message
    # Then both lists, separated, so the caller can see which of them to edit.
    assert "train_specimens = A, B, C; eval_specimens = B, D" in message, message
    assert "--train-specimens" in message, message


def test_the_guard_lists_every_overlapping_specimen_rather_than_the_first():
    with pytest.raises(EvaluationIntegrityError) as caught:
        assert_disjoint(["B", "A"], ["A", "B"])
    assert str(caught.value).startswith(
        "overlap between the specimens trained on and the specimens being scored: A, B."
    ), str(caught.value)


def test_the_guard_returns_nothing_and_is_total_on_a_disjoint_call():
    assert assert_disjoint(["A"], ["B"]) is None
    # Empty on either side is disjoint and is not a refusal: a caller naming no training
    # specimen has said what it has, and the evaluation cohort is still its own.
    assert assert_disjoint([], ["A"]) is None
    assert assert_disjoint(["A"], []) is None


def test_the_guard_compares_names_and_ignores_how_many_times_each_appears():
    # A specimen trained on is a specimen trained on whether the training list names it once
    # or nine times, and an evaluation cohort holds every patch of a specimen and therefore
    # repeats its name once per patch.
    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        assert_disjoint(["A", "A", "A", "B"], ["B", "B"])


def test_the_guard_compares_specimens_and_not_their_counts():
    # Two lists of three specimens, disjoint by name and equal in length: a predicate written
    # on lengths alone would pass the overlapping case above, and this one is what catches
    # it.
    assert_disjoint(["A", "B", "C"], ["D", "E", "F"])
    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        assert_disjoint(["A", "B", "C"], ["D", "B", "E"])


def test_the_guard_runs_before_the_model_the_paths_or_the_configuration_are_touched():
    # The brief's own test passes `None` for all three, and that raises rather than tracing
    # back only if `evaluate_specimens` refuses before it dereferences any of them. So the
    # ordering is the property under test, not an accident of the fixture.
    class _Explodes:
        def __getattr__(self, name):
            raise AssertionError(f"an argument was touched: {name}")

        def __call__(self, *args, **kwargs):
            raise AssertionError("an argument was called")

    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        evaluate_specimens(_Explodes(), ["a.png"], [0], ["A", "B"], ["B"], _Explodes())


def test_one_overlapping_specimen_among_several_is_still_an_overlap():
    with pytest.raises(EvaluationIntegrityError, match="overlap"):
        evaluate_specimens(
            None, ["a.png", "b.png", "c.png"], [0, 1, 0], ["A", "B", "C"], ["X", "C"], None
        )


# ---------------------------------------------------------------------------
# The record.
# ---------------------------------------------------------------------------


def test_the_record_is_positional_in_the_field_order_the_artifacts_write():
    record = Metrics(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 8, 9, 10, 11, 12)
    assert record.accuracy == 0.1
    assert record.precision == 0.2
    assert record.recall == 0.3
    assert record.specificity == 0.4
    assert record.f1 == 0.5
    assert record.npv == 0.6
    assert record.auc == 0.7
    assert (record.tp, record.fp, record.fn, record.tn, record.n) == (8, 9, 10, 11, 12)
    assert list(Metrics.__dataclass_fields__) == [
        "accuracy", "precision", "recall", "specificity", "f1", "npv", "auc",
        "tp", "fp", "fn", "tn", "n",
    ]


def test_the_record_is_frozen_and_its_auc_may_be_none():
    record = _record(auc=None)
    assert record.auc is None
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.accuracy = 0.9


# ---------------------------------------------------------------------------
# The metrics, out of a confusion matrix this file knows before the run.
# ---------------------------------------------------------------------------


class _StandIn(torch.nn.Module):
    """A backbone whose every prediction this test chooses, before the run starts.

    `positive=True` predicts class 1 for every patch, so the confusion matrix is decided by
    the labels alone and a one-class cohort is one keypress away. `scored=True` instead puts
    the mean normalised pixel of each patch in the steatosis logit: a black patch lands at
    -1.986 and is predicted class 0, a white one at +2.439 and is predicted class 1, and
    everything in between carries a probability of its own, so the AUC is a real ranking
    rather than `0.5` by construction. One parameter is held so that a checkpoint has a
    state dict to carry; it is added to both logits, so it moves neither the argmax nor the
    softmax ordering.
    """

    def __init__(self, positive: bool = True, scored: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.positive = positive
        self.scored = scored

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        count = inputs.shape[0]
        negative = torch.zeros(count)
        if self.scored:
            positive = inputs.mean(dim=(1, 2, 3))
        else:
            positive = torch.ones(count) if self.positive else torch.zeros(count)
        return torch.stack([negative, positive], dim=1) + self.weight


# The grey level at which the normalised mean crosses zero, measured on this stack: 114
# normalises to -0.0077 and is predicted class 0, 115 to +0.0096 and is predicted class 1.
_BLACK = 0
_WHITE = 255


def _write_patch(path: Path, value: int = _BLACK, size: int = 32) -> Path:
    """Write one uniform RGB patch of grey level `value` and return its path."""
    Image.fromarray(np.full((size, size, 3), value, dtype="uint8")).save(path)
    return path


def _cohort(root: Path, rows, size: int = 32):
    """Write one patch per `(specimen, label, grey)` row; return the three columns.

    The grey level is what the `scored` backbone reads, so the rows of this cohort decide the
    confusion matrix before the run starts — which is what lets the rates below be asserted
    against arithmetic rather than against whatever came out.
    """
    root.mkdir(parents=True, exist_ok=True)
    paths, labels, specimens = [], [], []
    for index, (specimen, label, value) in enumerate(rows):
        paths.append(_write_patch(root / f"{specimen}_{index}.png", value, size))
        labels.append(label)
        specimens.append(specimen)
    return paths, labels, specimens


# Seven rows over one specimen, with their greys chosen so that every one of the four
# confusion buckets is non-zero — a bucket that was zero would let a dropped term leave the sum
# right — and so that the six rates are six different strings at three places. Measured on
# these greys the counts are `(tp, fp, fn, tn) = (1, 1, 2, 3)` over `n = 7`: three greys under
# the threshold and labelled 0, two more under it and labelled 1, and two over it, one of
# each label.
_MIXED = [
    ("A", 0, _BLACK), ("A", 0, 60), ("A", 0, 114),
    ("A", 1, 30), ("A", 1, 90),
    ("A", 0, 250), ("A", 1, 200),
]


def test_the_four_counts_come_out_of_the_predictions_the_backbone_made(tmp_path):
    paths, labels, specimens = _cohort(tmp_path / "p", _MIXED)
    metrics = evaluate_specimens(
        _StandIn(scored=True), paths, labels, specimens, ["B"], _cfg()
    )
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (1, 1, 2, 3)
    assert metrics.n == 7


def test_every_rate_is_the_formula_over_the_four_counts_it_reports(tmp_path):
    # **Asserted as arithmetic over the record's own integers rather than as remembered
    # numbers**, so a rate read off the wrong count cannot satisfy it and a rate that is not
    # derived from the counts at all cannot either.
    paths, labels, specimens = _cohort(tmp_path / "p", _MIXED)
    metrics = evaluate_specimens(
        _StandIn(scored=True), paths, labels, specimens, ["B"], _cfg()
    )
    tp, fp, fn, tn, n = metrics.tp, metrics.fp, metrics.fn, metrics.tn, metrics.n
    assert n == tp + fp + fn + tn
    assert metrics.accuracy == pytest.approx((tp + tn) / n)
    assert metrics.precision == pytest.approx(tp / (tp + fp))
    assert metrics.recall == pytest.approx(tp / (tp + fn))
    assert metrics.specificity == pytest.approx(tn / (tn + fp))
    assert metrics.npv == pytest.approx(tn / (tn + fn))
    assert metrics.f1 == pytest.approx(
        2 * metrics.precision * metrics.recall / (metrics.precision + metrics.recall)
    )
    # Six different numbers on this cohort, so no two of them can be interchanged by a reader
    # or by a column.
    rendered = {f"{value:.3f}" for value in (
        metrics.accuracy, metrics.precision, metrics.recall,
        metrics.specificity, metrics.f1, metrics.npv,
    )}
    assert rendered == {"0.571", "0.500", "0.333", "0.750", "0.600", "0.400"}, rendered


def test_the_counts_are_the_positive_class_the_labelling_run_wrote(tmp_path):
    # `blobcount.train` counts against the class `blobcount.labeling.LABEL_BY_CLASS` names as
    # steatosis and this module reads that same table. A backbone predicting class 0 for
    # every patch over two class-1 rows therefore has everything in `fn`, and a module that
    # counted the *other* class would report everything in `tp` instead.
    paths, labels, specimens = _cohort(tmp_path / "p", [("A", 1, _WHITE), ("A", 1, _WHITE)])
    metrics = evaluate_specimens(
        _StandIn(positive=False), paths, labels, specimens, ["B"], _cfg()
    )
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (0, 0, 2, 0)
    assert metrics.recall == 0.0
    assert metrics.npv == 0.0


def test_every_row_lands_in_exactly_one_bucket_of_the_confusion_matrix(tmp_path):
    # `tp + fp + fn + tn == n` is what says the inference pass saw every row exactly once.
    # Both a positive and a negative backbone are checked, so a dropped term cannot hide
    # behind a bucket that happened to be zero for the other one.
    paths, labels, specimens = _cohort(
        tmp_path / "p", [("A", 0, _BLACK), ("A", 1, _WHITE), ("A", 1, _WHITE), ("B", 0, _BLACK)]
    )
    for positive in (True, False):
        metrics = evaluate_specimens(
            _StandIn(positive=positive), paths, labels, specimens, ["C"], _cfg()
        )
        assert metrics.tp + metrics.fp + metrics.fn + metrics.tn == metrics.n == 4


def test_a_rate_with_no_denominator_is_zero_and_not_a_division(tmp_path):
    # The rule, on all six rates and in both directions. Over an entirely class-0 cohort a
    # backbone predicting class 0 gives `specificity` and `npv` real denominators and
    # `precision` and `recall` none at all; a backbone predicting class 1 gives `precision` a
    # real denominator and `specificity` and `npv` none. Over an entirely class-1 cohort the
    # third case appears: `tp + fn` and `tp + fp` are real and `tn + fp` and `tn + fn` are
    # not. Both states are reachable in this cohort — a specimen can be entirely normal,
    # which is the reason `blobcount.data.class_weights` gives an absent class `0.0` rather
    # than `inf`, and why Task 11 made the validation loss unweighted.
    normal_and_negative = evaluate_specimens(
        _StandIn(positive=False),
        *_cohort(tmp_path / "p", [("A", 0, _BLACK), ("A", 0, _BLACK)]), ["B"], _cfg()
    )
    assert (normal_and_negative.tp, normal_and_negative.fp,
            normal_and_negative.fn, normal_and_negative.tn) == (0, 0, 0, 2)
    assert normal_and_negative.accuracy == 1.0
    assert normal_and_negative.specificity == 1.0
    assert normal_and_negative.npv == 1.0
    assert normal_and_negative.precision == 0.0     # tp + fp == 0
    assert normal_and_negative.recall == 0.0       # tp + fn == 0

    normal_and_positive = evaluate_specimens(
        _StandIn(positive=True),
        *_cohort(tmp_path / "q", [("A", 0, _BLACK), ("A", 0, _BLACK)]), ["B"], _cfg()
    )
    assert (normal_and_positive.tp, normal_and_positive.fp,
            normal_and_positive.fn, normal_and_positive.tn) == (0, 2, 0, 0)
    assert normal_and_positive.accuracy == 0.0
    assert normal_and_positive.precision == 0.0
    assert normal_and_positive.recall == 0.0         # tp + fn == 0
    assert normal_and_positive.specificity == 0.0   # tn + fp == 0
    assert normal_and_positive.npv == 0.0           # tn + fn == 0

    steatotic_and_positive = evaluate_specimens(
        _StandIn(positive=True),
        *_cohort(tmp_path / "r", [("A", 1, _WHITE), ("A", 1, _WHITE)]), ["B"], _cfg()
    )
    assert (steatotic_and_positive.tp, steatotic_and_positive.fp,
            steatotic_and_positive.fn, steatotic_and_positive.tn) == (2, 0, 0, 0)
    assert steatotic_and_positive.accuracy == 1.0
    assert steatotic_and_positive.precision == 1.0
    assert steatotic_and_positive.recall == 1.0
    assert steatotic_and_positive.specificity == 0.0  # tn + fp == 0
    assert steatotic_and_positive.npv == 0.0          # tn + fn == 0

    # None of the three produced a `nan` or an `inf`, and none warned: `filterwarnings =
    # ["error"]` would have failed any of them had a rate divided by zero.
    with _no_warning():
        for record in (normal_and_negative, normal_and_positive, steatotic_and_positive):
            for name in ("accuracy", "precision", "recall", "specificity", "f1", "npv"):
                value = getattr(record, name)
                assert value == value and abs(value) < float("inf"), f"{name}={value}"


def test_the_auc_is_none_wherever_only_one_class_is_present(tmp_path):
    # `sklearn.metrics.roc_auc_score` does not refuse that case — measured on sklearn 1.6.1
    # it returns `nan` and warns `UndefinedMetricWarning: Only one class is present in
    # y_true. ROC AUC score is not defined in that case.` — and `filterwarnings = ["error"]`
    # would turn that warning into a failed run. So the `None` is what keeps a one-class
    # specimen evaluable at all, and both halves are held here.
    with _no_warning():
        paths, labels, specimens = _cohort(tmp_path / "p", [("A", 1, _WHITE), ("A", 1, 200)])
        one_class = evaluate_specimens(
            _StandIn(scored=True), paths, labels, specimens, ["B"], _cfg()
        )
        assert one_class.auc is None
        assert one_class.recall == 1.0

        paths, labels, specimens = _cohort(
            tmp_path / "q", [("A", 0, _BLACK), ("A", 1, _WHITE)]
        )
        both = evaluate_specimens(
            _StandIn(scored=True), paths, labels, specimens, ["B"], _cfg()
        )

    assert isinstance(both.auc, float)
    assert 0.0 <= both.auc <= 1.0


def test_the_auc_is_computed_from_the_probability_of_the_positive_class(tmp_path):
    # The greys are distinct, so the rows carry four different probabilities and the AUC is a
    # real ranking. The expected value is recomputed here from the same patches through the
    # same transform with the same model, so an AUC built out of the argmax instead of the
    # probability — which would be 0.5 for anything — cannot satisfy it.
    from sklearn.metrics import roc_auc_score

    from blobcount.data import PatchDataset, build_loader, build_transforms

    cfg = _cfg()
    paths, labels, specimens = _cohort(
        tmp_path / "p", [("A", 0, 30), ("A", 1, 250), ("B", 0, 150), ("B", 1, 90)]
    )
    model = _StandIn(scored=True)
    metrics = evaluate_specimens(model, paths, labels, specimens, ["C"], cfg)

    _train_transform, eval_transform = build_transforms(cfg)
    loader = build_loader(
        PatchDataset(paths, labels, eval_transform),
        int(cfg.get("training.batch_size")), None, False,
        int(cfg.get("training.num_workers")), int(cfg.get("project.seed")),
    )
    model.eval()
    scores = []
    with torch.no_grad():
        for inputs, _targets in loader:
            scores.append(torch.softmax(model(inputs), dim=1)[:, 1].numpy())
    probabilities = np.concatenate(scores)

    assert metrics.auc == pytest.approx(roc_auc_score(labels, probabilities))
    assert metrics.auc != pytest.approx(0.5), "the probabilities would carry no ranking"


def test_the_cohort_scored_is_the_cohort_handed_over(tmp_path):
    # Three rows over two specimens and a training list that excludes neither, so the guard
    # passes. `n == 3` is what says the loop scored what it was given and not a wider file.
    paths, labels, specimens = _cohort(
        tmp_path / "p", [("A", 1, _WHITE), ("A", 0, _BLACK), ("B", 1, 200)]
    )
    metrics = evaluate_specimens(
        _StandIn(positive=True), paths, labels, specimens, ["C"], _cfg()
    )
    assert metrics.n == 3
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (2, 1, 0, 0)


def test_a_cohort_of_no_rows_is_scored_and_reported_rather_than_refused(tmp_path):
    # The zero-denominator rule covers accuracy as well as the other five, so an empty cohort
    # is a record of `n == 0` and not an exception. The alternative — a refusal here — would
    # be a rule about a cohort, and `bin/evaluate.py` is where the caller that has one can be
    # told its own cohort was empty; this module reports what it was handed.
    metrics = evaluate_specimens(_StandIn(positive=True), [], [], [], ["B"], _cfg())
    assert metrics.n == 0
    assert (metrics.tp, metrics.fp, metrics.fn, metrics.tn) == (0, 0, 0, 0)
    assert metrics.auc is None
    for name in ("accuracy", "precision", "recall", "specificity", "f1", "npv"):
        assert getattr(metrics, name) == 0.0, name


def test_a_patch_that_is_not_there_names_its_own_path_and_is_not_wrapped(tmp_path):
    # `blobcount.data` rules that `PatchDataset` does not wrap its read, because the
    # `OSError` that survives names the resolved path — the fact a caller needs. The test is
    # here so the ruling survives into this module rather than being re-litigated at the
    # first missing patch.
    paths, labels, specimens = _cohort(tmp_path / "p", [("A", 1, _WHITE)])
    paths[0] = paths[0].with_name("absent.png")
    with pytest.raises(OSError) as caught:
        evaluate_specimens(_StandIn(positive=True), paths, labels, specimens, ["B"], _cfg())
    assert "absent.png" in str(caught.value)


# ---------------------------------------------------------------------------
# The bootstrap interval.
# ---------------------------------------------------------------------------


# Ten values rather than the brief's three: with three values both tails of 500 resampled
# means land on the minimum and the maximum whatever was drawn, so every seed agrees and the
# draw is not observable at all. Measured at 200 resamples on these ten values: seed 7 gives
# (0.545, 0.720375) and seed 8 gives (0.535, 0.73525).
_SPREAD = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.55, 0.65, 0.75, 0.45]


def test_the_interval_brackets_every_value_it_was_given():
    # A resampled mean is a mean of the values, so no resample leaves their range and neither
    # can the two tail quantiles of those means.
    lo, hi = bootstrap_ci(_SPREAD, 400, 0.95, 3)
    assert min(_SPREAD) <= lo <= np.mean(_SPREAD) <= hi <= max(_SPREAD)


def test_a_different_seed_draws_a_different_interval():
    assert bootstrap_ci(_SPREAD, 200, 0.95, 7) != bootstrap_ci(_SPREAD, 200, 0.95, 8)


def test_the_level_is_the_two_tails_it_was_given():
    # A level of 1.0 is the whole spread of the resampled means and a level of 0.5 is the
    # central half of it. Measured at 200 resamples on these ten values: 1.0 gives
    # (0.505, 0.78) and 0.5 gives (0.595, 0.655), the second strictly inside the first. The
    # numbers are asserted at three places so the property under test is what the `level`
    # argument states and not the exact float this numpy happened to produce.
    everything = bootstrap_ci(_SPREAD, 200, 1.0, 3)
    half = bootstrap_ci(_SPREAD, 200, 0.5, 3)
    assert everything == pytest.approx((0.505, 0.78))
    assert half == pytest.approx((0.595, 0.655))
    assert everything[0] < half[0] < half[1] < everything[1]


def test_the_interval_is_two_plain_floats():
    interval = bootstrap_ci([0.1, 0.9], 50, 0.95, 1)
    assert isinstance(interval, tuple) and len(interval) == 2
    assert all(type(bound) is float for bound in interval)


def test_the_values_may_be_any_sequence_of_numbers():
    # The argument is a sequence of numbers and not a list, so a caller holding a numpy
    # column of per-specimen scores is not refused and gets the same interval.
    values = [0.4, 0.55, 0.7, 0.85]
    expected = bootstrap_ci(values, 300, 0.95, 11)
    assert bootstrap_ci(tuple(values), 300, 0.95, 11) == expected
    assert bootstrap_ci(np.array(values), 300, 0.95, 11) == expected


def test_the_intervals_are_read_from_the_configuration_rather_than_written_into_the_script():
    # **These two keys have no other reader under `blobcount/`,** so a run whose
    # configuration narrows the interval and whose printed table did not move would report
    # 2000 resamples of a 0.95 level as though they were the configured ones. `Config`
    # records every value it serves, so `unknown_keys()` is what says the read happened.
    shipped = _cfg()
    narrowed = _cfg("evaluation.bootstrap_samples=300", "evaluation.confidence_level=0.5")
    # Six specimens with six different accuracies, so the interval is a spread and not a
    # point: six equal records give (0.9, 0.9) at every level and every sample count.
    per_specimen = [
        _record(accuracy=value) for value in (0.1, 0.35, 0.5, 0.65, 0.8, 0.95)
    ]

    assert shipped.get("evaluation.bootstrap_samples") == 2000
    assert shipped.get("evaluation.confidence_level") == 0.95

    wide = bootstrap_metric_intervals(per_specimen, shipped, 5)
    tight = bootstrap_metric_intervals(per_specimen, narrowed, 5)

    assert wide.by_metric["accuracy"] != tight.by_metric["accuracy"]
    assert wide.by_metric["accuracy"][0] <= tight.by_metric["accuracy"][0]
    assert wide.by_metric["accuracy"][1] >= tight.by_metric["accuracy"][1]
    # **The settings travel out with the intervals**, so a caller that writes them into an
    # artifact never reads either key itself and the number in the artifact is the one the
    # bounds beside it were drawn at.
    assert (wide.n_samples, wide.level) == (2000, 0.95)
    assert (tight.n_samples, tight.level) == (300, 0.5)
    assert "evaluation.bootstrap_samples" not in narrowed.unknown_keys()
    assert "evaluation.confidence_level" not in narrowed.unknown_keys()


def test_the_metric_intervals_cover_the_six_rates_and_not_the_auc():
    # `auc` is `None` wherever a specimen carries one class, which is most specimens of a
    # cohort this size, and a bootstrap over a `None` is not a number. The intervals are over
    # the six rates; the AUC is still in the table, without one.
    records = [
        _record(auc=None),
        _record(accuracy=0.4, precision=0.3, recall=0.2, specificity=0.1,
                f1=0.25, npv=0.15, auc=0.25, tp=2, fp=1, fn=6, tn=1),
    ]
    intervals = bootstrap_metric_intervals(records, _cfg(), 5)
    assert set(intervals.by_metric) == {
        "accuracy", "precision", "recall", "specificity", "f1", "npv"
    }
    for name, (lo, hi) in intervals.by_metric.items():
        values = [getattr(record, name) for record in records]
        assert min(values) <= lo <= max(values) <= hi, name


def test_the_unit_of_resampling_is_the_specimen_and_not_the_patch():
    # Two patches of one slide are not two observations, so the interval is over the values
    # the specimens have rather than over the patches under them. A run of one specimen has an
    # interval of zero width rather than a confident number about a patch.
    one = bootstrap_metric_intervals([_record()], _cfg(), 5)
    assert one.by_metric["accuracy"] == (0.9, 0.9)
    assert bootstrap_ci([0.42], 100, 0.95, 1) == (0.42, 0.42)


@pytest.mark.parametrize("empty", [[], (), np.array([])])
def test_neither_interval_accepts_a_sequence_with_nothing_in_it(empty):
    # Both are otherwise a mean over nothing, and `numpy.quantile` over an empty array raises
    # its own message about an empty array rather than about the rule.
    with pytest.raises(ValueError, match="no value"):
        bootstrap_ci(empty, 100, 0.95, 1)
    with pytest.raises(ValueError, match="no specimen"):
        bootstrap_metric_intervals(empty, _cfg(), 5)


# ---------------------------------------------------------------------------
# The weighted kappa.
# ---------------------------------------------------------------------------


def test_the_weights_are_the_quadratic_ones_an_independent_implementation_agrees_with():
    # scikit-learn computes the same statistic from the same exponent, and the agreement is
    # exact on every case below. **This module does not call sklearn for it**: the weights are
    # the brief's and a second implementation of them inside this project would be a second
    # answer to what a rating is. The comparison is the evidence that the two answers are
    # one. Measured on 1.6.1 with `labels=` given explicitly, so scikit-learn infers nothing:
    # 0.8, -1.0, 0.75 and 0.6363636363636364 respectively.
    from sklearn.metrics import cohen_kappa_score

    for true, predicted, scale in (
        ([0, 1, 2, 3], [0, 1, 3, 2], 3),
        ([0, 1, 2, 3], [3, 2, 1, 0], 3),
        ([0, 1, 2, 3], [0, 1, 1, 2], 3),
        ([0, 1, 2, 2], [0, 1, 1, 2], 1),
    ):
        labels = sorted(set(true) | set(predicted))
        assert quadratic_weighted_kappa(true, predicted, scale) == pytest.approx(
            float(cohen_kappa_score(
                true, predicted, labels=labels,
                weights="quadratic" if scale > 1 else None,
            ))
        )


def test_a_rating_scale_of_one_is_cohens_kappa_and_not_a_division_by_zero():
    # `((i - j) ** 2) / ((max_rating - 1) ** 2)` divides by zero at `max_rating == 1`: the
    # denominator is `(1 - 1) ** 2 == 0` and every disagreement would be `1 / 0`. A
    # two-category rating therefore takes the unweighted rule — weight 1 on agreement and 0 on
    # disagreement, which is Cohen's kappa — and returns what scikit-learn returns with no
    # weights at all.
    from sklearn.metrics import cohen_kappa_score

    assert (1 - 1) ** 2 == 0
    assert quadratic_weighted_kappa([0, 0, 1, 1], [0, 1, 0, 1], 1) == pytest.approx(
        float(cohen_kappa_score([0, 0, 1, 1], [0, 1, 0, 1], labels=[0, 1]))
    )
    # `max_rating == 0` is the same rule and not a different one: a scale cannot be wider
    # than its own ratings, so anything at or under one rating is unweighted. Three
    # categories are still weighted plainly rather than squared — measured above as
    # 0.6363636363636364, which is Cohen's kappa and not the quadratic answer to it.
    assert quadratic_weighted_kappa([0, 1, 2, 2], [0, 1, 1, 2], 0) == pytest.approx(
        0.6363636363636364
    )


def test_a_degenerate_rating_returns_one_rather_than_a_division_by_zero():
    # Every rating the same and predicted perfectly: observed agreement and expected agreement
    # are both 1, so the quotient is `0 / 0`. **scikit-learn does not solve this one** —
    # measured on 1.6.1, `cohen_kappa_score` over two constant sequences returns `nan` and
    # warns `UserWarning: A single label was found in 'y_true' and 'y_pred'` followed by
    # `RuntimeWarning: invalid value encountered in scalar divide` — and
    # `filterwarnings = ["error"]` would fail the run on either.
    with _no_warning():
        assert quadratic_weighted_kappa([2, 2, 2], [2, 2, 2], 3) == 1.0
        assert quadratic_weighted_kappa([1, 1, 1], [1, 1, 1], 1) == 1.0


def test_a_rating_beyond_the_scale_it_is_given_is_held_and_not_folded():
    # `max_rating` is the caller's statement about the scale, and a rating above it is an index
    # the weight matrix has no row for. The matrix widens to hold the rating rather than
    # raising `IndexError`, so the answer is still the kappa of the data: measured on the pair
    # below, a `max_rating` of 4 and one of 9 both return `0.0`.
    ratings = ([0, 4, 0, 4], [0, 4, 4, 0])
    assert quadratic_weighted_kappa(*ratings, 4) == 0.0
    assert quadratic_weighted_kappa(*ratings, 9) == 0.0
    # And a scale *narrower* than the ratings is the same case: a `max_rating` of 1 is the
    # unweighted rule and a rating of 4 is an index four rows down. It returns `0.0` rather
    # than raising, which is the half of the widening that a matrix sized off `max_rating`
    # alone cannot survive.
    assert quadratic_weighted_kappa(*ratings, 1) == 0.0
    assert quadratic_weighted_kappa(*ratings, 0) == 0.0


def test_the_kappa_is_a_float_and_not_a_numpy_scalar():
    assert type(quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 3, 2], 3)) is float


@pytest.mark.parametrize("empty", [[], ()])
def test_the_kappa_refuses_a_sequence_with_no_rating_in_it(empty):
    # Agreement is computed over pairs, and there is no pair here to agree or disagree. The
    # maximum of an empty sequence would be what named the mistake, and it names no rule.
    with pytest.raises(ValueError, match="no rating"):
        quadratic_weighted_kappa(empty, empty, 3)


def test_the_kappa_refuses_two_sequences_of_different_lengths():
    # Pairing the shorter one against the other up to its own length would score a cohort this
    # function was handed half of, and say nothing about the rest.
    with pytest.raises(ValueError, match="different lengths"):
        quadratic_weighted_kappa([0, 1, 2, 3], [0, 1], 3)


# ---------------------------------------------------------------------------
# The entry point.
# ---------------------------------------------------------------------------


def _load_cli(monkeypatch, name: str = "evaluate"):
    path = PROJECT_ROOT / "bin" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"blobcount_cli_{name}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _printed_table(printed: str):
    """Return `(header, rows)` for the one table the command line prints.

    Cells are split on two or more spaces rather than one so a number formatted with a space
    inside it stays one cell, and a block of one cell is not a table — which is how the
    closing sentence is skipped.
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
    assert len(tables) == 1, tables
    return tables[0]


# The cohort the entry point is run against, and the confusion matrix it produces. Specimen A
# is one false positive, one false negative and one true positive; specimen B is three true
# negatives and one more false negative. Pooled: `(tp, fp, fn, tn) = (1, 1, 2, 3)` over seven
# rows, which renders the six rates as 0.571, 0.500, 0.333, 0.750, 0.600 and 0.400 — six
# different strings at three places, so no two columns of the printed table can be exchanged
# without one of the two value assertions going red.
_COHORT = [
    ("A", 0, "normal/A_0.png", 250),
    ("A", 1, "steatosis/A_1.png", _BLACK),
    ("A", 1, "steatosis/A_2.png", 200),
    ("B", 0, "normal/B_3.png", 30),
    ("B", 0, "normal/B_4.png", 90),
    ("B", 0, "normal/B_5.png", 114),
    ("B", 1, "steatosis/B_6.png", _BLACK),
]


def _label_row(specimen, label, relative):
    return [f"{specimen}_{relative.replace('/', '_').removesuffix('.png')}",
            specimen, label, 7, "0.5", relative]


def _write_config(directory: Path, label_root: Path, results: Path, **evaluation) -> Path:
    """Write a configuration file for the command line, and return its path.

    Built from `configs/default.yaml` through the name this module imported at load time, and
    not through the one the fixture has monkeypatched: this helper's job is to produce a file
    the caller names, and the fixture's own file has already had its paths rewritten.
    """
    directory.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["labels"] = str(label_root)
    data["paths"]["results"] = str(results)
    data["training"].update(
        {"pretrained": False, "input_size": 32, "batch_size": 2, "num_workers": 0}
    )
    data["evaluation"].update(evaluation)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _active_config() -> Path:
    """Return the configuration file in force, which the `cli` fixture has replaced.

    `DEFAULT_CONFIG_PATH` is imported by value at the top of this file, so it is still
    `configs/default.yaml` after the fixture has monkeypatched the attribute on
    `blobcount.config`. A test that rewrites the configuration has to read the file the run
    will actually load, or it writes a second one whose `paths.labels` points at a directory
    that holds no patch.
    """
    from blobcount import config as config_module

    return config_module.DEFAULT_CONFIG_PATH


@pytest.fixture
def cli(project_scratch, tmp_path, monkeypatch):
    """The entry point, seven synthetic patches, a handoff, a checkpoint and two outputs.

    **Everything this program writes goes under `project_scratch`**, because the output guard
    refuses a target that is not strictly below the project root and pytest's `tmp_path` is
    outside it — the reason that fixture exists. The patches, the handoff and the checkpoint
    stay in `tmp_path`: they are what the program reads, not what it writes.

    The checkpoint is a real state dict over the stand-in backbone, so `load_state_dict` runs
    for real, and `build_model` is replaced by a factory returning a *fresh* stand-in, so the
    weights really do come out of the file rather than out of the object. The interval is
    narrowed from the shipped 2000 to 300 resamples so the suite does not pay for a resampling
    table it only reads the shape of.
    """
    module = _load_cli(monkeypatch)
    label_root = tmp_path / "labels"
    rows = []
    for specimen, label, relative, value in _COHORT:
        patch = label_root / relative
        patch.parent.mkdir(parents=True, exist_ok=True)
        _write_patch(patch, value)
        rows.append(_label_row(specimen, label, relative))
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    index = handoff / "labels.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)

    checkpoint = tmp_path / "model.pth"
    torch.save(_StandIn(scored=True).state_dict(), checkpoint)
    monkeypatch.setattr(module, "build_model", lambda *a, **k: _StandIn(scored=True))

    results = project_scratch / "results"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(tmp_path, label_root, results, bootstrap_samples=300),
    )
    return module, index, results, checkpoint


def _argv(index, checkpoint, specimens=("A", "B"), train=("Z",)):
    argv = ["--labels", str(index), "--checkpoint", str(checkpoint)]
    for specimen in specimens:
        argv += ["--specimens", specimen]
    for specimen in train:
        argv += ["--train-specimens", specimen]
    return argv


def _document(results: Path) -> dict:
    return json.loads(
        (results / "evaluation" / "medical_metrics.json").read_text(encoding="utf-8")
    )


def test_the_command_line_writes_both_artifacts_under_the_configured_results(cli):
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint)) == 0
    assert (results / "evaluation" / "medical_metrics.json").is_file()
    assert (results / "evaluation" / "per_specimen_metrics.csv").is_file()


def test_the_printed_table_carries_the_numbers_the_json_holds(cli, capsys):
    # **The table is the human-facing report, and a swapped column misreports the study while
    # every artifact on disk stays correct.** So every cell is asserted as a value: against the
    # `medical_metrics.json` the same run wrote, and against the arithmetic of the confusion
    # matrix that JSON also carries.
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint)) == 0

    header, printed = _printed_table(capsys.readouterr().out)
    assert tuple(header) == ("metric", "value", "ci_low", "ci_high")
    assert [cells[0] for cells in printed] == [
        "accuracy", "precision", "recall", "specificity", "f1", "npv", "auc"
    ]

    document = _document(results)
    metrics = document["metrics"]
    intervals = document["bootstrap"]["intervals"]
    for cells in printed:
        name = cells[0]
        assert cells[1] == f"{metrics[name]:.3f}", name
        if name in intervals:
            assert cells[2] == f"{intervals[name][0]:.3f}", name
            assert cells[3] == f"{intervals[name][1]:.3f}", name
        else:
            # The AUC is the one metric with no interval, and it says so in both bound
            # columns rather than printing a short row a reader has to re-align.
            assert cells[2:] == ["n/a", "n/a"], name
    assert "auc" not in intervals, "a bootstrap over a specimen's None is not a number"

    # The pooled cohort is seven rows over two specimens, and every confusion bucket is
    # non-zero, so a dropped term cannot leave the sum right.
    assert metrics["n"] == 7
    assert (metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]) == (1, 1, 2, 3)
    assert metrics["accuracy"] == pytest.approx(4 / 7)
    assert metrics["precision"] == pytest.approx(1 / 2)
    assert metrics["recall"] == pytest.approx(1 / 3)
    assert metrics["specificity"] == pytest.approx(3 / 4)
    assert metrics["npv"] == pytest.approx(3 / 5)
    assert metrics["f1"] == pytest.approx(0.4)
    # Seven different strings at three places, so the two columns a reader would confuse
    # cannot hold equal values and hide a swap.
    assert len({cells[1] for cells in printed}) == 7, printed


def test_the_per_specimen_table_names_every_scored_specimen_with_its_own_counts(cli):
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint)) == 0

    rows = _read_csv(results / "evaluation" / "per_specimen_metrics.csv")
    assert [row["specimen"] for row in rows] == ["A", "B"]
    assert [int(row["n"]) for row in rows] == [3, 4]
    for row in rows:
        assert int(row["tp"]) + int(row["fp"]) + int(row["fn"]) + int(row["tn"]) == int(row["n"])
        # A cell that is not a number is blank and never the text of a missing value: a CSV a
        # downstream reader parses with `float()` would raise `ValueError` on `"None"`.
        assert row["auc"] not in {"None", "nan", "inf"}, row

    document = _document(results)
    assert [entry["specimen"] for entry in document["per_specimen"]] == ["A", "B"]
    # The pooled `n` is the sum of the specimens', which is what says the two tables are about
    # the same rows rather than about two different cohorts.
    assert document["metrics"]["n"] == sum(int(row["n"]) for row in rows) == 7
    # And each specimen's own counts, which are the rows the pooled counts are the sum of.
    assert [(entry["tp"], entry["fp"], entry["fn"], entry["tn"])
            for entry in document["per_specimen"]] == [(1, 1, 1, 0), (0, 0, 1, 3)]


def test_the_interval_is_over_the_specimens_and_cannot_leave_their_range(cli):
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint)) == 0

    document = _document(results)
    # **The interval is the configured one and the artifact says which**, so a run whose
    # printed bounds did not move with its configuration cannot be read as if they had. The
    # fixture narrows `evaluation.bootstrap_samples` from the shipped 2000 to 300, so the
    # number asserted here is one this file chose and not the one `configs/default.yaml` says.
    assert document["bootstrap"]["n_samples"] == 300
    assert document["bootstrap"]["confidence_level"] == 0.95

    per_specimen = [entry["accuracy"] for entry in document["per_specimen"]]
    lo, hi = document["bootstrap"]["intervals"]["accuracy"]
    assert min(per_specimen) <= lo <= hi <= max(per_specimen)
    # Two specimens, one at 1/3 and one at 3/4, so the interval cannot be narrower than the
    # gap between them however many times they are resampled.
    assert per_specimen == pytest.approx([1 / 3, 3 / 4])


def test_the_specimens_named_on_the_command_line_are_the_ones_scored(cli):
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint, specimens=("B",))) == 0
    assert [row["specimen"] for row in
            _read_csv(results / "evaluation" / "per_specimen_metrics.csv")] == ["B"]
    document = _document(results)
    assert document["metrics"]["n"] == 4
    assert document["evaluated_specimens"] == ["B"]


def test_a_run_that_names_no_specimen_scores_every_specimen_in_the_handoff(cli):
    # `--specimens` is the cohort to score and is not required: the default is every specimen
    # the handoff holds, which is the cohort a study run means. The integrity guard applies to
    # it either way, so the default cannot score a trained specimen.
    module, index, results, checkpoint = cli
    argv = ["--labels", str(index), "--checkpoint", str(checkpoint), "--train-specimens", "Z"]
    assert module.main(argv) == 0
    document = _document(results)
    assert document["evaluated_specimens"] == ["A", "B"]
    assert document["metrics"]["n"] == 7


def test_a_one_class_specimen_writes_a_blank_auccell_and_not_the_text_none(cli, tmp_path, monkeypatch):
    # A specimen whose rows are all one class has no AUC — `blobcount.evaluate` returns `None`
    # because `sklearn.metrics.roc_auc_score` warns rather than refuses there — and the CSV
    # writes the empty string for it. **The JSON is the contrast**: the same specimen's record
    # there carries a real `null`, which a JSON reader can test for, while the text `"None"` in
    # the CSV would raise `ValueError` in every reader downstream.
    module, _index, results, checkpoint = cli
    label_root = tmp_path / "one-class"
    rows = []
    for number, (label, value) in enumerate([(0, _BLACK), (0, 60), (0, 114)]):
        relative = f"normal/C_{number}.png"
        patch = label_root / relative
        patch.parent.mkdir(parents=True, exist_ok=True)
        _write_patch(patch, value)
        rows.append(_label_row("C", label, relative))
    handoff = tmp_path / "one-class-handoff"
    handoff.mkdir()
    index = handoff / "labels.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(tmp_path / "one-class-config", label_root, results, bootstrap_samples=300),
    )
    assert module.main(_argv(index, checkpoint, specimens=("C",))) == 0

    written = _read_csv(results / "evaluation" / "per_specimen_metrics.csv")
    assert len(written) == 1
    assert written[0]["auc"] == ""
    assert written[0]["npv"] == "1.0", "the other rates are still numbers"
    document = _document(results)
    assert document["per_specimen"][0]["auc"] is None
    assert document["metrics"]["auc"] is None
    assert "auc" not in document["bootstrap"]["intervals"]


def test_a_specimen_the_handoff_holds_no_row_of_is_refused(cli, capsys):
    # Six `0.0` rates beside an `auc` of `None` written into `medical_metrics.json` is a
    # report of nothing that a later reader cannot tell apart from a report of a model that
    # scored nothing. The caller named a cohort; this says the cohort is not in the handoff.
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint, specimens=("Z",))) == 2
    captured = capsys.readouterr().err
    assert "has a row in" in captured, captured
    # The specimens the handoff does hold, as a Python list, so the reader is told what to
    # choose from rather than only what they chose.
    assert "The handoff holds ['A', 'B']" in captured, captured
    assert not (results / "evaluation").exists()


def test_scoring_a_specimen_that_was_trained_on_is_refused(cli, capsys):
    # The refusal this command line exists to make, and the one its required
    # `--train-specimens` makes a caller state rather than have inferred. The artifact directory
    # is not created, so a refused run leaves nothing behind.
    module, index, results, checkpoint = cli
    assert module.main(_argv(index, checkpoint, train=("B",))) == 2
    captured = capsys.readouterr().err
    assert "overlap" in captured, captured
    assert "B" in captured, captured
    assert not (results / "evaluation").exists()


def test_the_training_specimen_list_may_not_be_omitted(cli):
    # `argparse` refuses it before the first statement of `main` runs, so the refusal is
    # argparse's own and names this program nowhere in it — which is why the status is
    # `EXIT_REFUSED` and the two refusals cannot be told apart by status alone.
    module, index, _results, checkpoint = cli
    with pytest.raises(SystemExit) as caught:
        module.main(["--labels", str(index), "--checkpoint", str(checkpoint),
                     "--specimens", "A"])
    assert caught.value.code == 2


def test_a_checkpoint_that_is_not_there_is_refused(cli, tmp_path, capsys):
    module, index, _results, _checkpoint = cli
    assert module.main(_argv(index, tmp_path / "absent.pth")) == 2
    assert "absent.pth" in capsys.readouterr().err


def test_a_checkpoint_that_is_not_a_torch_file_is_refused(cli, tmp_path, capsys):
    # Measured: `torch.load` of a file that is not a torch archive raises
    # `pickle.UnpicklingError`, whose bases are `PickleError, Exception, BaseException,
    # object` — neither an `OSError` nor a `ValueError`, so an arm holding those two would
    # let a `--checkpoint` pointing at the wrong file reach the caller as a traceback.
    module, index, _results, _checkpoint = cli
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a torch file at all")
    assert module.main(_argv(index, junk)) == 2
    assert "junk.pth" in capsys.readouterr().err


def test_a_checkpoint_whose_weights_do_not_fit_the_model_is_refused(cli, capsys):
    # Measured: `load_state_dict` raises `RuntimeError: Error(s) in loading state_dict ...
    # size mismatch for weight`, which is a disagreement between a file and a configuration
    # rather than a defect in either.
    module, index, _results, checkpoint = cli
    module.build_model = lambda *a, **k: torch.nn.Linear(4, 2)
    assert module.main(_argv(index, checkpoint)) == 2
    assert "size mismatch" in capsys.readouterr().err


def test_an_architecture_this_project_does_not_build_is_refused(cli, tmp_path, monkeypatch, capsys):
    # The one thing the backbone factory refuses for itself, reached through this command line
    # rather than through a traceback out of `main`. The fixture's stand-in factory is replaced
    # by the real one, which is assigned rather than monkeypatched because the module is loaded
    # fresh for every test and leaks nothing.
    from blobcount import models as models_module

    module, index, _results, checkpoint = cli
    data = yaml.safe_load(_active_config().read_text(encoding="utf-8"))
    data["training"]["architecture"] = "ResNet18"
    broken = tmp_path / "mixed-case.yaml"
    broken.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    module.build_model = models_module.build_model
    assert module.main(_argv(index, checkpoint)) == 2
    assert "ResNet18" in capsys.readouterr().err


def test_the_results_directory_itself_is_refused_when_it_is_a_file(cli, project_scratch, monkeypatch, capsys):
    # `paths.results` is a hand-edited string, and a file there makes `results/evaluation`
    # un-creatable — the `mkdir` would raise `NotADirectoryError`, so the guard turns a
    # hand-edit into a message instead of a traceback. **The file is inside the project**, so
    # the containment test passes and this message can only have come from the file test: a
    # file outside the root is reported as a containment failure instead, and a test that
    # accepted either message would not be able to tell the two refusals apart.
    module, index, _results, checkpoint = cli
    occupied = project_scratch / "occupied"
    occupied.write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(project_scratch / "cfg", project_scratch / "labels", occupied),
    )
    assert module.main(_argv(index, checkpoint)) == 2
    captured = capsys.readouterr().err
    assert "refusing to write" in captured, captured
    assert "is a file rather than a directory" in captured, captured
    assert "project root" not in captured, captured
    assert occupied.read_text(encoding="utf-8") == "x"


def test_the_evaluation_subdirectory_is_refused_when_it_is_a_file(cli, project_scratch, monkeypatch, capsys):
    # The second path the guard tests, and the one a `results` that is a valid directory and a
    # `results/evaluation` that is a file is reachable only through. **Without it the `mkdir`
    # below raises `FileExistsError`** out of `main`, which is the traceback the guard exists
    # to turn into a message.
    module, index, _results, checkpoint = cli
    results = project_scratch / "results"
    results.mkdir()
    (results / "evaluation").write_text("x", encoding="utf-8")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(project_scratch / "cfg", project_scratch / "labels", results),
    )
    assert module.main(_argv(index, checkpoint)) == 2
    captured = capsys.readouterr().err
    assert "refusing to write" in captured, captured
    assert "is a file rather than a directory" in captured, captured
    assert (results / "evaluation").read_text(encoding="utf-8") == "x"


def test_an_output_target_outside_the_project_is_refused(cli, tmp_path, monkeypatch, capsys):
    module, index, _results, checkpoint = cli
    outside = tmp_path / "escape"
    outside.mkdir()
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(tmp_path, tmp_path / "labels", outside),
    )
    assert module.main(_argv(index, checkpoint)) == 2
    captured = capsys.readouterr().err
    assert "project root" in captured, captured
    assert sorted(path.name for path in outside.iterdir()) == []


def test_the_project_root_is_not_an_output_target(cli, tmp_path, monkeypatch, capsys):
    # `paths.results: .` resolves to the project root itself while leaving `evaluation/`
    # strictly below it, so a guard on the subdirectory alone would write a JSON and a CSV into
    # the repository. Both paths are tested, which is why this one is here and not only the
    # directory above it.
    module, index, _results, checkpoint = cli
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _write_config(tmp_path, tmp_path / "labels", Path(".")),
    )
    assert module.main(_argv(index, checkpoint)) == 2
    assert "project root" in capsys.readouterr().err


def test_a_handoff_that_is_not_there_is_refused(cli, tmp_path, capsys):
    module, _index, _results, checkpoint = cli
    assert module.main(_argv(tmp_path / "absent.csv", checkpoint)) == 2
    assert "absent.csv" in capsys.readouterr().err


def test_a_configuration_the_project_will_not_load_is_refused(cli, tmp_path, monkeypatch, capsys):
    module, index, _results, checkpoint = cli
    data = yaml.safe_load(_active_config().read_text(encoding="utf-8"))
    del data["training"]["architecture"]
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    assert module.main(_argv(index, checkpoint)) == 2
    assert "training.architecture" in capsys.readouterr().err


def test_the_command_line_defines_no_code_for_a_contradictory_run(cli):
    module, _index, _results, _checkpoint = cli
    assert module.EXIT_OK == 0
    assert module.EXIT_REFUSED == 2
    assert not hasattr(module, "EXIT_CONTRADICTORY")


def test_a_run_asks_for_no_imagenet_weights_because_a_checkpoint_supplies_them(cli, tmp_path, monkeypatch):
    # `configs/default.yaml` ships `training.pretrained: true`, and an evaluation loads a
    # checkpoint that replaces every parameter of the model anyway — so the backbone is built
    # with no weights whatever the configuration says, and a hub cache that was empty stays
    # empty. **The configuration in this test is written with `pretrained: true`** rather than
    # with the fixture's `false`, because a factory handed `false` records `false` either way
    # and the assertion would hold against a program that simply passed the key through.
    module, index, _results, checkpoint = cli
    data = yaml.safe_load(_active_config().read_text(encoding="utf-8"))
    assert data["training"]["pretrained"] is False, "the fixture's file is the base"
    data["training"]["pretrained"] = True
    shipped = tmp_path / "pretrained-true.yaml"
    shipped.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", shipped)

    handed: list[bool] = []

    def factory(architecture, num_classes, pretrained):
        handed.append(pretrained)
        return _StandIn(scored=True)

    module.build_model = factory
    home = PROJECT_ROOT / ".pytest-tmp" / "torch-home"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TORCH_HOME", str(home))
    assert module.main(_argv(index, checkpoint)) == 0
    assert handed == [False]
    assert sorted(home.rglob("*")) == []
