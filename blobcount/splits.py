"""Leave-one-specimen-out folds: one fold per specimen, and the specimen held out whole.

**The split is by specimen, and that is the whole of this module.** A patch of a
whole-slide image is not an independent observation. Every patch cut from one slide comes
from one patient, one staining batch, one scanner and one 128 um field of view repeated
side by side, so two patches of the same slide agree about the outcome for reasons that
have nothing to do with steatosis. A split that cuts patches at random therefore puts most
of every held-out slide into training, and the score it reports is a measure of how well
the model recognises the staining of a slide it has already seen. That is the class of
number this rebuild exists to remove, and leave-one-specimen-out is the split that removes
it: each fold holds out every patch of one specimen and trains on every patch of the rest,
so the validation score is about a patient the model has never seen. Phase B uses no other
split.

**How many folds there are depends on the registry, not on this module.** `load_registry`
admits three of the six staged specimens for training, the three TCGA-LIHC slides that
carry a recoverable `source_case_id` and a licence, so the shipped cohort gives three
folds of which each trains on two specimens. A caller supplies the ids it has; this module
returns one fold per distinct id and never consults `data/manifest.yaml`, so adding a
specimen to the manifest raises the fold count with no code change and no configuration
key.

**The indices are positions in the caller's array, so the caller must pass one specimen id
per dataset row.** `specimen_of_row` is a per-row column in dataset order — row 0's
specimen, row 1's, and so on — and `train_idx` and `val_idx` index that same order. They
are not positions in a deduplicated specimen list and not positions in the index file on
disk: a caller holding six rows of six distinct specimens gets six folds of one held-out
row each, and a caller holding the same six specimens spread over thousands of patch rows
gets six folds of thousands of held-out rows each. The row count and the fold count are
independent, and a list of six specimens is indistinguishable from six rows of six
specimens from inside this function, so passing the deduplicated list by mistake is not
detectable here and produces folds that validate on one row per specimen. The signature
and this paragraph are the contract. Both index arrays are `np.ndarray` of `int64`, ascending, and
each fold's arrays are independent of every other fold's: masking allocates, so a caller
that reorders or mutates one fold's indices cannot reach another's rows through it.

**The fold order is `sorted()` over the distinct ids, and `fold` numbers that order from
zero.** The order does not depend on the order of the input, on the order a set produced,
or on `PYTHONHASHSEED`: two callers given the same rows in different orders get the same
folds with the same numbers, and `results/fold_summary.csv` is then comparable between
runs. `sorted()` is the ordering, spelled once, rather than "whatever appeared first",
because a fold number written into an artifact has to mean the same specimen on the next
run of the pipeline.

**Fewer than two distinct specimens is refused, as a `SplitError`.** The message states
the rule and the two counts that failed it, because a refusal a caller cannot act on is a
traceback with better manners: holding out the only specimen there is would train on
nothing, so a fold here would not be a measurement of anything, and the call is refused
rather than returned as an empty training set. The count is of *distinct* ids and not of
rows, so no rows at all and several rows of one specimen arrive at the same refusal, and
neither is a state a caller can accidentally train through.

Nothing in this module is caught, logged, or written. There is no file to open and no
patch to read, so there is no specimen, coordinate or exception object to log, and the
folds are returned rather than saved: writing them is the cross-validation loop's
business, because it is the loop that knows the output directory.

Declared interface, in `__all__` order:

    Fold
        fold: int
        held_out: str
        train_idx: np.ndarray
        val_idx: np.ndarray
    SplitError
    loso_folds(specimen_of_row: Sequence[str]) -> list[Fold]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["Fold", "SplitError", "loso_folds"]


class SplitError(Exception):
    """Raised when a set of specimen ids cannot be split into leave-one-specimen-out folds."""


@dataclass(frozen=True, eq=False)
class Fold:
    """One leave-one-specimen-out fold: the specimen held out, and the two index arrays.

    `fold` is the 0-based position of `held_out` in the `sorted()` order of the distinct
    ids, and is stable across calls and across row orderings, so it is usable as a file
    name and as a column in a summary table. `held_out` is the specimen whose rows are in
    `val_idx` and whose absence from `train_idx` is the point of the fold.

    `train_idx` and `val_idx` are `np.ndarray` of `int64` positions in the caller's row
    order, ascending, disjoint, and together every position there is: each row is trained
    on in all folds but one and validated in exactly one. Neither array is a view of
    shared state, so a caller that sorts or mutates one fold's indices leaves every other
    fold's alone.

    **The dataclass keeps identity equality (`eq=False`), and a generated `__eq__` over two
    `np.ndarray` fields is a trap.** It would compare the arrays elementwise, answer with
    an array, and then have that answer converted to a bool — which raises "the truth
    value of an array with more than one element is ambiguous" for every fold of more than
    one row, and quietly compares only the first element for a one-row fold. Both outcomes
    are worse than `False`. Identity equality also leaves `__hash__` as the inherited one,
    so a `Fold` can be put in a set or used as a dict key, which a generated hash over an
    ndarray field could not be.
    """

    fold: int
    held_out: str
    train_idx: np.ndarray
    val_idx: np.ndarray


def loso_folds(specimen_of_row: Sequence[str]) -> list[Fold]:
    """Return one leave-one-specimen-out fold per distinct specimen id, in `sorted()` order.

    The caller must supply **one specimen id per dataset row, in dataset order** — the
    specimen of row 0, row 1, and so on, one entry for every row, repeated where several
    rows come from the same specimen. The returned indices are positions in that same
    order, so a deduplicated list of specimens is the one argument that cannot be used
    here: it would produce one single-row validation set per specimen and train on
    everything else, which is a patch-level split wearing a specimen-level name.

    Each fold holds out every row of one specimen and trains on every row of the rest, and
    `held_out` names that specimen. `fold` numbers the folds from zero in `sorted()` order
    of the distinct ids, so the numbering depends on the id set and not on the order of
    the rows: the same cohort given in a different order yields the same folds with the
    same numbers. Both index arrays are `np.ndarray` of `int64`, ascending, and disjoint,
    and their lengths sum to the number of rows supplied.

    **Fewer than two distinct ids is refused as a `SplitError`**, counting distinct ids
    and not rows: `[]` and `["only", "only"]` are refused alongside `["only"]`, because
    holding out the only specimen there is leaves nothing to train on. The caller cannot
    distinguish those cases by outcome, and none of them is a state a caller can train
    through by accident.
    """
    rows = np.arange(len(specimen_of_row), dtype=np.int64)
    held_out_in_order = sorted(set(specimen_of_row))
    if len(held_out_in_order) < 2:
        raise SplitError(
            "leave-one-specimen-out splits need at least two distinct specimen ids, "
            "because a fold holds out one whole specimen and trains on the rest; this "
            f"caller supplied {len(held_out_in_order)} distinct id(s) over "
            f"{rows.size} row(s). Pass one specimen id per dataset row"
        )
    folds: list[Fold] = []
    for number, held_out in enumerate(held_out_in_order):
        mask = np.array(
            [specimen == held_out for specimen in specimen_of_row], dtype=np.bool_
        )
        folds.append(
            Fold(
                fold=number,
                held_out=held_out,
                train_idx=rows[~mask],
                val_idx=rows[mask],
            )
        )
    return folds