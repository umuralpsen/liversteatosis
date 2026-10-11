"""Metrics for a held-out cohort, the guard that keeps the guard honest, and intervals.

**Nothing in this module has been run against the cohort.** `dataset/labels/` does not
exist, the extraction that would fill it has never been run, and no checkpoint has ever been
trained — `bin/train.py` has never completed a fold. So no number in this file has been
measured on a real patch, and the only runs that have ever been made are over synthetic
patches written by `tests/test_evaluate.py`. A reader must not take any property stated here
to mean that this study has a result.

**The integrity guard is the first thing `evaluate_specimens` does, before it touches the
model, the paths or the configuration.** A number computed over a specimen the model was
trained on is not a measurement of anything, and the one place a reader is least likely to
notice is the point where it happens: the cohort was assembled by somebody who meant to
exclude it, the guard is a check they expect to pass, and the run reports a flattering
accuracy with no signal at all that the check did what they wanted. So the guard is called
on its own, before a single weight is read or a single file opened, and
`EvaluationIntegrityError` names the specimens that are in both lists rather than only
saying that something is. **The lists are the caller's to state and this function's to
compare, and neither is inferred here**: `blobcount.splits` folds a cohort by specimen
because patches of one slide are not independent observations, and the specimens a model
trained on are a fact about the run that produced the checkpoint, which no reader of a
directory can reconstruct. That is why `bin/evaluate.py` makes `--train-specimens` a
required argument rather than inferring it.

**Every rate comes out of one confusion matrix, counted against the class the labelling run
wrote.** `tp`, `fp`, `fn` and `tn` are counted against `blobcount.labeling.LABEL_BY_CLASS`'s
steatosis class — read from that table here, exactly as `blobcount.train` reads it — and
accuracy, precision, recall, specificity, F1 and NPV are derived from those four integers
rather than accumulated per batch. So `tp + fp + fn + tn` is the size of the scored cohort
whatever its class distribution is, and a reader holding the four integers can recompute the
six rates and check this file's arithmetic rather than take it.

**A rate with no denominator is `0.0`, and not a division, and not `inf` and not `None`.**
This covers precision, recall, specificity, NPV, F1 and accuracy: a cohort with no predicted
positive has a precision of zero, one with no true positive has a recall of zero, and a
cohort of no rows at all has an accuracy of zero. Every one of those states is reachable
here rather than hypothetical — a specimen can be entirely normal, which is the reason
`blobcount.data.class_weights` gives an absent class `0.0` rather than `inf`, and the reason
Task 11 made the fold's validation loss unweighted so a one-class fold did not divide by
zero. **`Metrics.auc` is the one declared exception and it is `None` rather than a number,
because `sklearn.metrics.roc_auc_score` does not produce one**: measured on scikit-learn
1.6.1, handed a `y_true` holding one class it returns `nan` and warns
`UndefinedMetricWarning: Only one class is present in y_true. ROC AUC score is not defined
in that case.` — and `filterwarnings = ["error"]` in `pyproject.toml` would turn that warning
into a failed run. So `None` is what keeps a one-class specimen evaluable at all, and the one
place it costs a reader something is a per-specimen CSV cell, which is written blank.

**The interval is over specimens and not over patches.** `bootstrap_ci` resamples the values
it is handed `n_samples` times with a seeded `numpy.random.default_rng` and returns the two
empirical `level` quantiles of the resampled means. The caller decides what a value is, and
the caller this project has is `bin/evaluate.py` feeding it one value per specimen: a
specimen is the unit this study generalises over, and resampling patches of one slide would
report a confidence interval about pixels.
`evaluation.bootstrap_samples` and `evaluation.confidence_level` are read here and not in the
command line, for the reason `blobcount.train.read_labels` was moved into the package:
`bin/` is a directory of scripts loaded by path, a read there is outside the source scan in
`tests/test_config.py`, and the key goes on being reported as one nothing under `blobcount/`
reads.

**`bootstrap_ci` takes its `n_samples` and `level` as arguments, and `bootstrap_metric_
intervals` is what reads the two keys and calls it.** The second is not a wrapper for its own
sake: it is the reason those keys have a reader, because a run whose configuration narrows the
interval and whose printed table did not move would report 2000 resamples of a 0.95 level as
though they were the configured ones. `bootstrap_ci` stays the testable primitive, the six
rates are the ones that get an interval because `auc` is `None` wherever a specimen carries
one class, and **the settings come back out with the intervals in one `Intervals` record**, so
`bin/evaluate.py` writes what level it drew at without reading either key a second time.

**The kappa is Cohen's kappa with quadratic weights, and the weights are the disagreement
weights `((i - j) ** 2) / ((max_rating - 1) ** 2)` written as the agreement they are
complementary to.** The weighted agreement is then `(observed - expected) / (1 - expected)`
over that matrix. **The divisor cancels out of the quotient**, which is worth stating because
it looks load-bearing and is not: measured over 2000 random rating tables drawn on this
stack, the answer computed with `(max_rating - 1) ** 2` in the denominator and the answer
computed with `max_rating ** 2` differ by at most `1.0e-13`. The exponent is what decides
whether a disagreement of one step is punished differently from one of three, and that is
the whole of what the weights are for.

**`max_rating <= 1` is unweighted agreement, as a rule and not as a patch for a division.**
The divisor `(max_rating - 1) ** 2` is `0` at `max_rating == 1` — measured, `(1 - 1) ** 2` is
`0` — and every disagreement would be `1 / 0`. So a scale of one rating takes weight 1 on
agreement and 0 on disagreement, which is Cohen's kappa for a two-category rating, and a
caller with a two-class problem gets exactly what scikit-learn returns with no weights at
all. A three-category rating passed with `max_rating=1` is still unweighted — measured as
`0.6363636363636364` against the same value from `cohen_kappa_score` — so the rule is about
the scale the caller stated and not about how many ratings happen to be present.

**Observed and expected agreement both equal to 1 is `1.0`, because the quotient is `0 / 0`
there.** That is the case where every rating the same class is rated and predicted
perfectly, and **scikit-learn does not solve it**: measured on 1.6.1,
`cohen_kappa_score` over two constant sequences returns `nan` and warns
`UserWarning: A single label was found in 'y_true' and 'y_pred'` followed by
`RuntimeWarning: invalid value encountered in scalar divide`, and `filterwarnings = ["error"]`
would fail the run on either. Enumerating every rating table of three ratings over two,
three and four categories — 1001 tables in total — the expected agreement is `1.0` in 9 of
them and the observed agreement is `1.0` in all 9, so at that size the guard is not leaving
a second degenerate case unhandled.

**The handoff is read by `blobcount.train.read_labels` and not by this module.** `bin/` is a
directory of scripts loaded by path, a copy of the CSV contract here would be a second set
of answers to it, and `read_labels` is the reader `blobcount/train.py` was moved into the
package for. **Its precondition is inherited and not restated as a rule here: a handoff from
`bin/label_patches.py --out somewhere` carries paths relative to *that* directory**, so every
row resolves against the wrong root and the failure is a missing-file `OSError` naming a path
that looks right and is not. Run `bin/label_patches.py` without `--out`, or point
`paths.labels` at the directory you passed it.

**Nothing here is caught, logged, or written.** A patch file that is not there raises from
`PatchDataset.__getitem__` inside a loader, and the `OSError` that survives names the path it
could not open — the fact a caller needs, and better than a sentence about a rule.
`blobcount.data` states that ruling and this module is on the other side of it. The refusal
above is the one exception because it is this module's own type: no other module raises it,
so nothing else catches it, and `bin/evaluate.py` is the only caller that can turn it into a
message on stderr and an exit code.

Declared interface, in `__all__` order:

    Metrics
        accuracy: float
        precision: float
        recall: float
        specificity: float
        f1: float
        npv: float
        auc: float | None
        tp: int
        fp: int
        fn: int
        tn: int
        n: int
    Intervals
        n_samples: int
        level: float
        by_metric: dict[str, tuple[float, float]]
    EvaluationIntegrityError
    assert_disjoint(train_specimens: Sequence[str], eval_specimens: Sequence[str]) -> None
    bootstrap_ci(
        values: Sequence[float], n_samples: int, level: float, seed: int
    ) -> tuple[float, float]
    bootstrap_metric_intervals(
        per_specimen: Sequence[Metrics], cfg: Config, seed: int
    ) -> Intervals
    quadratic_weighted_kappa(
        y_true: Sequence[int], y_pred: Sequence[int], max_rating: int
    ) -> float
    evaluate_specimens(
        model: nn.Module,
        paths: Sequence[Path],
        labels: Sequence[int],
        specimen_ids: Sequence[str],
        train_specimens: Sequence[str],
        cfg: Config,
    ) -> Metrics
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from blobcount.config import Config, get_device
from blobcount.data import PatchDataset, build_loader, build_transforms
from blobcount.labeling import CLASS_STEATOSIS, LABEL_BY_CLASS

__all__ = [
    "Metrics",
    "Intervals",
    "EvaluationIntegrityError",
    "assert_disjoint",
    "bootstrap_ci",
    "bootstrap_metric_intervals",
    "quadratic_weighted_kappa",
    "evaluate_specimens",
]

# The class the whole pipeline counts as a positive, read from the labelling run's own table
# rather than written here: `blobcount.labeling` put the class each directory holds into
# `labels.csv` from that table, and a second place for "which class is steatosis" is a second
# answer to a question the labelling run already answers. `blobcount.train` reads the same
# constant the same way.
_POSITIVE_CLASS = LABEL_BY_CLASS[CLASS_STEATOSIS]

# The classes a `labels.csv` holds, as the labelling run's table states them. Read for one
# purpose: the AUC is a number over two classes and is `None` wherever the scored cohort does
# not hold both of them, so this is the set the check is made against rather than a length.
_BOTH_CLASSES = frozenset(LABEL_BY_CLASS.values())

# The six rates `bootstrap_metric_intervals` puts an interval around, in the order the printed
# table carries them. `auc` is not among them and the module docstring says why: it is `None`
# wherever a specimen carries one class, and a bootstrap over a `None` is not a number.
_INTERVAL_METRICS = ("accuracy", "precision", "recall", "specificity", "f1", "npv")


class EvaluationIntegrityError(Exception):
    """Raised when the specimens about to be scored include one that was trained on.

    One shape reaches it: an overlap between the specimen ids a model was trained on and the
    specimen ids a run is about to score it on. Nothing else in this project declares an
    exception type for it — `blobcount.models` and `blobcount.config` raise `ValueError` and
    `ConfigError` by ruling — so it is this module's own type and this module's to catch, and
    `bin/evaluate.py` is the only caller that turns it into a message and an exit code.
    """


@dataclass(frozen=True)
class Metrics:
    """One scored cohort: six rates, an AUC that may not exist, and the counts behind them.

    **The field order is fixed, and it is the order both artifacts write.**
    `medical_metrics.json` serializes it with `asdict` and `per_specimen_metrics.csv` takes
    its columns from `__dataclass_fields__`, so an artifact carrying the rates also carries
    the integers they were derived from and a reader can recompute the six rather than trust
    them.

    `n` is the number of rows scored and `tp + fp + fn + tn == n` is the identity that says
    the inference pass saw each of them exactly once.

    **A rate with no denominator is `0.0` and `auc` is `None` where only one class is
    present**, and the module docstring carries why for both. Every rate here is a function
    of the four counts and of nothing else, so a record cannot hold an accuracy its own
    integers do not support.
    """

    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    npv: float
    auc: float | None
    tp: int
    fp: int
    fn: int
    tn: int
    n: int


def assert_disjoint(train_specimens: Sequence[str], eval_specimens: Sequence[str]) -> None:
    """Raise `EvaluationIntegrityError` naming any specimen that is in both lists.

    The caller must supply **the specimen ids the model was trained on and the specimen ids
    about to be scored**, in either order and with as many repeats as it likes: this
    compares names and ignores multiplicity, because a specimen trained on is a specimen
    trained on whether it appears once in the training list or nine times. `evaluate_specimens`
    calls this on its own arguments and `bin/evaluate.py` requires the training list on the
    command line for the same reason — neither list is a fact a reader of a checkpoint can
    reconstruct.

    **The message leads with the word `overlap` and with the intersection alone**, so a
    caller can act on it without reading its own arguments back out of a traceback and
    without removing a specimen that was in only one of the two lists; both lists follow,
    separated, so the caller can see which of them to edit.
    """
    overlap = sorted(set(train_specimens) & set(eval_specimens))
    if overlap:
        raise EvaluationIntegrityError(
            f"overlap between the specimens trained on and the specimens being scored: "
            f"{', '.join(overlap)}. train_specimens = "
            f"{', '.join(sorted(set(train_specimens)))}; eval_specimens = "
            f"{', '.join(sorted(set(eval_specimens)))}. A number computed over a specimen "
            f"the model was trained on is not a measurement of anything. Remove "
            f"{', '.join(overlap)} from --train-specimens, or from the cohort being scored, "
            f"and score a specimen the model has not seen"
        )


def bootstrap_ci(
    values: Sequence[float], n_samples: int, level: float, seed: int
) -> tuple[float, float]:
    """Return the empirical `level` interval of the mean of `values`, by resampling.

    **`n_samples` draws of `len(values)` indices with replacement** from a
    `numpy.random.default_rng(seed)` built from the caller's own seed, and the two bounds are
    the `(1 - level) / 2` and `1 - (1 - level) / 2` quantiles of the resampled means. The
    seed is an argument rather than something read from the configuration because a unit test
    that pays for 2000 resamples on every call is a suite nobody runs; measured here, 2000
    resamples over 2000 values cost 0.035 s, which is why the shipped
    `evaluation.bootstrap_samples` is a run's cost and not a test's.

    **The unit of resampling is whatever the caller passed**, and the module docstring says
    what this project's caller passes: one value per specimen. Two patches of one slide are
    not two observations, so resampling them would report an interval about pixels.

    Measured on ten values at 200 resamples: `level=0.95` gives `(0.545, 0.720375)` at
    `seed=7` and `(0.535, 0.73525)` at `seed=8`; `level=1.0` gives the whole spread
    `(0.505, 0.78)` and `level=0.5` the central half `(0.595, 0.655)`.
    `tests/test_evaluate.py` asserts the relations between those bounds and not the digits, so
    the property under test is what the `level` argument states and not the exact float this
    numpy produced.

    **A sequence with nothing in it is refused**, for the reason `blobcount.train.select_epoch`
    gives: the mean of no value is undefined, and `numpy.quantile` over an empty array raises
    its own message about an empty array rather than about the rule.
    """
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        raise ValueError(
            "bootstrap_ci was given a sequence with no value in it: an interval is a "
            "statement about a mean, and there is no mean of nothing to put one around. "
            "This caller is resampling one value per specimen, so it was handed no "
            "specimen to resample — check the cohort it was given before the interval"
        )
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, sample.size, size=(n_samples, sample.size))
    means = sample[indices].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


@dataclass(frozen=True)
class Intervals:
    """The interval every rate was drawn at, and the conditions it was drawn under.

    **The settings travel with the intervals rather than being read again beside them.** The
    two `evaluation.*` keys have exactly one reader in this project, and it is
    `bootstrap_metric_intervals`; a caller that wanted them for an artifact would otherwise
    read them a second time in `bin/`, where the ledger scan cannot see it and where a change
    to either key would have to be made in two places. `medical_metrics.json` is built out of
    this record, so the numbers and the conditions that produced them come from one call.

    `level` is a fraction and not a percentage, so it is printed and serialized as it is
    configured. `by_metric` carries the six rates and not the AUC, and the module docstring
    says why.
    """

    n_samples: int
    level: float
    by_metric: dict[str, tuple[float, float]]


def bootstrap_metric_intervals(
    per_specimen: Sequence[Metrics], cfg: Config, seed: int
) -> Intervals:
    """Return the `level` interval of each rate, resampling the per-specimen values.

    **`evaluation.bootstrap_samples` and `evaluation.confidence_level` are read here and not
    in `bin/evaluate.py`**, for the reason the module docstring gives: a read in `bin/` is
    outside the source scan in `tests/test_config.py` and the key is then reported as one
    nothing under `blobcount/` reads. Every rate is resampled under the same seed and the
    same level, so one metric's interval is not drawn under different conditions from
    another's printed beside it.

    The six rates are `_INTERVAL_METRICS` and the AUC is not among them: `auc` is `None`
    wherever a specimen carries one class, which is most specimens of a cohort this size, and
    a bootstrap over a `None` is not a number. The AUC is still reported — it is in `Metrics`
    and in both artifacts — it simply carries no interval.

    A caller with one specimen gets an interval of zero width rather than a refusal: the
    resampling is over the values it was given and there is one of them.
    """
    n_samples = int(cfg.get("evaluation.bootstrap_samples"))
    level = float(cfg.get("evaluation.confidence_level"))
    return Intervals(
        n_samples=n_samples,
        level=level,
        by_metric={
            name: bootstrap_ci(
                [float(getattr(record, name)) for record in per_specimen], n_samples, level, seed
            )
            for name in _INTERVAL_METRICS
        },
    )


def quadratic_weighted_kappa(
    y_true: Sequence[int], y_pred: Sequence[int], max_rating: int
) -> float:
    """Return Cohen's kappa with the quadratic weights `((i - j) ** 2) / ((r - 1) ** 2)`.

    The caller must supply **one rating per observation in each of two sequences, in the same
    order, and the width of the rating scale they are on**. `max_rating` is that width and
    not the largest rating present: a caller's scale is a property of the instrument that
    produced the ratings, and a pair that agrees perfectly is perfect agreement on any scale.

    **A `max_rating` of 1 or less is the unweighted rule** — weight 1 on agreement and 0 on
    disagreement, which is Cohen's kappa — because the quadratic divisor is `(1 - 1) ** 2` and
    that is `0`. The module docstring carries the reasoning and the measurements.

    **A rating beyond the scale widens the matrix rather than raising `IndexError`.** The
    matrix is built to `max(max_rating, the largest rating present)`, so a scale a caller
    understated costs them a row rather than an exception, and the extra rows carry no weight
    a rating inside the scale could reach: measured on `([0, 4, 0, 4], [0, 4, 4, 0])`, a
    `max_rating` of 4 and one of 9 both return `0.0`.

    Where the ratings run `0..max_rating` the answers agree exactly with
    `sklearn.metrics.cohen_kappa_score(..., weights="quadratic")`: measured on 1.6.1,
    `0.8` for `([0, 1, 2, 3], [0, 1, 3, 2], 3)`, `-1.0` for
    `([0, 1, 2, 3], [3, 2, 1, 0], 3)`, `0.75` for `([0, 1, 2, 3], [0, 1, 1, 2], 3)` and
    `0.6363636363636364` for the unweighted `([0, 1, 2, 2], [0, 1, 1, 2], 1)`. **That library
    is not called here**: the weights are the brief's and a second implementation of them in
    this project would be a second answer to what a rating is, so the agreement is evidence
    rather than a dependency.

    Two sequences of different lengths are refused by name and by both lengths, rather than
    by a `zip` that stops at the shorter one: pairing up to the shorter would score a cohort
    this function was handed half of, and say nothing about the rest.
    """
    truth = [int(rating) for rating in y_true]
    predicted = [int(rating) for rating in y_pred]
    if not truth:
        raise ValueError(
            "quadratic_weighted_kappa was given a sequence with no rating in it: agreement "
            "is computed over pairs, and there is no pair here to agree or disagree. Pass "
            "one rating per observation in both sequences"
        )
    if len(truth) != len(predicted):
        raise ValueError(
            f"quadratic_weighted_kappa was given two sequences of different lengths: "
            f"{len(truth)} rating(s) and {len(predicted)}. They are one rating per "
            f"observation, so a shorter one would be scored against the ratings the other "
            f"has no observation for"
        )
    size = max(max(max(truth), max(predicted)) + 1, 2)
    if max_rating > 1:
        scale = (max_rating - 1) ** 2
        weights = np.array(
            [[1.0 - ((i - j) ** 2) / scale for j in range(size)] for i in range(size)],
            dtype=float,
        )
    else:
        weights = np.array(
            [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)], dtype=float
        )

    rows = np.zeros(size, dtype=float)
    columns = np.zeros(size, dtype=float)
    joint = np.zeros((size, size), dtype=float)
    count = len(truth)
    for actual, guess in zip(truth, predicted):
        rows[actual] += 1.0 / count
        columns[guess] += 1.0 / count
        joint[actual, guess] += 1.0 / count
    expected = np.outer(rows, columns)

    observed_agreement = float((weights * joint).sum())
    expected_agreement = float((weights * expected).sum())
    if expected_agreement == 1.0 and observed_agreement == 1.0:
        # Every rating the same class, rated and predicted perfectly. The quotient below is
        # `0 / 0` and the module docstring carries what scikit-learn does here instead.
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def _confusion(actual: np.ndarray, predicted: np.ndarray) -> tuple[int, int, int, int]:
    """Return `(tp, fp, fn, tn)` over two equal-length arrays of class indices.

    Counted over the whole array rather than per batch and merged afterwards, which is the
    same answer and one place where the tally can be wrong instead of four. The positive
    class is `_POSITIVE_CLASS`, so every row lands in exactly one of the four buckets and
    their sum is the size of the cohort.
    """
    positive = actual == _POSITIVE_CLASS
    predicted_positive = predicted == _POSITIVE_CLASS
    return (
        int(np.count_nonzero(positive & predicted_positive)),
        int(np.count_nonzero(~positive & predicted_positive)),
        int(np.count_nonzero(positive & ~predicted_positive)),
        int(np.count_nonzero(~positive & ~predicted_positive)),
    )


def _rates(
    tp: int, fp: int, fn: int, tn: int, n: int
) -> tuple[float, float, float, float, float, float]:
    """Return `(accuracy, precision, recall, specificity, f1, npv)` from the four counts.

    **A rate with no denominator is `0.0` rather than a division**, for all six of them, and
    the module docstring says why the state is reachable rather than hypothetical.

    **F1 is derived from precision and recall rather than from `2 * tp / (2 * tp + fp + fn)`
    so that this module and `blobcount.train._rates` answer the same question the same way on
    the same four integers.** Measured on `(tp, fp, fn)`: the two formulas agree exactly at
    `(1, 0, 1)` and at `(0, 2, 0)`, and at `(3, 1, 2)` they differ in the sixteenth digit —
    `0.6666666666666665` through precision and recall against `0.6666666666666666` through
    the counts — which is the rounding of two different divisions rather than a different
    statistic, and is the reason one of the two had to be chosen.
    """
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return accuracy, precision, recall, specificity, f1, npv


def _infer(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(actual, predicted, positive_probability)` over `loader`, with no gradient.

    **One pass over the loader for all three, because the AUC and the four counts have to be
    read off the same rows.** Holding the probabilities of a second pass and the predictions
    of the first would leave the AUC describing a different ordering of the same cohort from
    the counts beside it, and the two would disagree silently whenever the loader reordered
    anything.

    `model` is put into `eval()` mode — no dropout, no batch statistics — and the batches
    are moved to `device`, which the caller resolved from `project.device`.
    """
    model.eval()
    actual: list[int] = []
    predicted: list[int] = []
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            logits = model(inputs.to(device))
            probabilities = torch.softmax(logits, dim=1)
            actual.extend(int(value) for value in targets)
            predicted.extend(int(value) for value in logits.argmax(dim=1).cpu())
            scores.append(probabilities[:, _POSITIVE_CLASS].cpu().numpy())
    # `concatenate` and not `stack`: the last batch of a cohort is the short one whenever the
    # row count is not a multiple of `training.batch_size`, and a stacked array needs every
    # batch to be the same length. The rows come out in loader order either way.
    return (
        np.asarray(actual, dtype=np.int64),
        np.asarray(predicted, dtype=np.int64),
        np.concatenate(scores) if scores else np.zeros(0, dtype=float),
    )


def evaluate_specimens(
    model: nn.Module,
    paths: Sequence[Path],
    labels: Sequence[int],
    specimen_ids: Sequence[str],
    train_specimens: Sequence[str],
    cfg: Config,
) -> Metrics:
    """Score `model` over the cohort handed over and return one `Metrics` for it.

    **The caller must supply one path, one label and one specimen id per dataset row, in the
    same order**, which is the contract `blobcount.splits.loso_folds` states and the one
    `PatchDataset` pairs each path with: `paths[i]` is the patch of row `i`, `labels[i]` is
    its class and `specimen_ids[i]` is the specimen it came from. No length check is applied
    against the other sequences, for the reason `PatchDataset` gives. `specimen_ids` is read
    here for exactly one purpose — the integrity guard — and a caller with no use for it
    passes the ids anyway rather than this function guessing them.

    **`train_specimens` is the specimen list the model was trained on, and the guard runs
    first**: before `cfg` is read, before a patch is opened and before `model` is moved, so a
    run that would score a trained specimen costs a message and not an inference pass.
    `tests/test_evaluate.py` holds that ordering by handing this function a model, a path
    list and a `Config` that each raise on any attribute access.

    **`cfg` is read for the transform, the loader and the device**, and not for the metrics:
    `build_transforms` gives the eval transform, which holds no random operation, so two runs
    of this cohort score the same rendering of every patch. The model is moved to the device
    `project.device` resolves to — the same rule `blobcount.train.train_fold` follows — and
    the batches are moved with it.

    A cohort of no rows is scored and reported as `n == 0` with every rate `0.0` and `auc`
    `None`, rather than refused: the module docstring's zero-denominator rule covers accuracy
    as well, and a refusal here would be a rule about a cohort rather than about the
    arithmetic.
    """
    assert_disjoint(train_specimens, specimen_ids)

    device = torch.device(get_device(cfg))
    model = model.to(device)
    _train_transform, eval_transform = build_transforms(cfg)
    dataset = PatchDataset([Path(path) for path in paths], labels, eval_transform)
    loader = build_loader(
        dataset,
        int(cfg.get("training.batch_size")),
        None,
        False,
        int(cfg.get("training.num_workers")),
        int(cfg.get("project.seed")),
    )

    actual, predicted, probabilities = _infer(model, loader, device)
    tp, fp, fn, tn = _confusion(actual, predicted)
    n = int(actual.size)
    accuracy, precision, recall, specificity, f1, npv = _rates(tp, fp, fn, tn, n)
    auc = (
        float(roc_auc_score(actual, probabilities))
        if set(actual.tolist()) == _BOTH_CLASSES
        else None
    )
    return Metrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        npv=npv,
        auc=auc,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=n,
    )