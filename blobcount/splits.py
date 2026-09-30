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

**A `str` and a `bytes` are refused, for the same reason the indices are.** Both satisfy
`Sequence[str]`, so `loso_folds("456")` splits one argument into three specimens called
`"4"`, `"5"` and `"6"` — three folds of one row each, which is the patch-level split
above wearing a specimen-level name, returned as a valid result. `bytes` is worse rather
than different: iterating `b"456"` yields the integers 52, 53 and 54, so `held_out` would
carry an `int` on a dataclass annotated `str`. `str`, `bytes`, `bytearray` and
`memoryview` are excluded by name, and those four are **the character and byte-buffer
sequences a caller already holds without importing anything** — `str` the only sequence
of characters, the other three the byte-buffer types.

**That set is not closed over the standard library, and the tuple is not widened to close
it.** `array.array` is a fifth: `array.array("w", "abc")` iterates `'a'`, `'b'`, `'c'`,
and `array.array("b", b"456")` iterates 52, 53 and 54, both measured
`isinstance(_, Sequence) is True`, and `loso_folds` accepts both today and returns the same
three one-row folds. It is parked to be closed together with the identical hole in
`blobcount.ablation._thresholds`, which refuses these same four names and has the same gap.
What the four buy is that they cover the shapes a caller reaches without an
`import array`; naming them is not the same thing as covering every sequence of
characters, and the difference is recorded here rather than left for a reader to assume
away.

That is the same rule and the same reasoning `blobcount.ablation._thresholds` applies to
`ablation.thresholds`. **Its two forms of the `Sequence` test are named apart here, because
they differ only in their sign and the sign is the whole argument.** The *positive* form,
`isinstance(x, Sequence)`, accepts all four names above — measured `True` for each — so on
its own it would split `'456'` into three specimens exactly as it splits `b"456"`, and only
naming them catches it. This is not a type policy nobody asked for; it is the one shape a
reader writes by accident.

**The *negative* form that module also applies is not repeated here**, and this is the one
place the two predicates differ. Its predicate is
`not isinstance(x, Sequence) or isinstance(x, <the four names>)`, which refuses every
non-`Sequence` on top of the four. Measured: `isinstance(np.array(["a", "b"]), Sequence)`
is `False`, and so is a pandas `Series`, because neither is registered as a `Sequence` — so
the negative form would refuse `ndarray`, `Series`, `set` and `dict`, none of which is a
sequence of characters or of bytes, at the price of refusing the two most natural per-row
id columns a caller holding a DataFrame has. The names to refuse are named here; nothing
else about the argument's type is checked, and no length check is applied against any
expected row count, because with one argument there is nothing to check against and a guess
would refuse a legitimate cohort of six one-patch specimens.

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

# The standard library's sequences **of characters or of bytes** that a caller reaches
# without an import, and the only shapes refused by name: `str` is the only sequence of
# characters, and `bytes`, `bytearray` and `memoryview` are the three byte-buffer types,
# each a sequence of the integers its bytes spell. All four are registered as a
# `collections.abc.Sequence`, so the *positive* test `isinstance(x, Sequence)` accepts
# every one of them and would split `'456'` into three specimens as readily as `b"456"`;
# only naming them catches it.
#
# `range`, `tuple` and `array.array` are correctly absent, and `array.array` is absent for
# a reason the others are not: a tuple or an array of specimen ids is a legitimate per-row
# column and `loso_folds` accepts both, where a `range` is a sequence of numbers and is
# outside `Sequence[str]`. **The cost of that is that an `array.array` of characters or of
# bytes arrives with it** — `array.array("w", "abc")` iterates `'a'`, `'b'`, `'c'` and
# `array.array("b", b"456")` iterates 52, 53 and 54 — so this list is *not* closed over
# the standard library, and it does not claim to be. Parked to be closed alongside the
# identical gap in `blobcount.ablation._thresholds`, which refuses these same four names.
#
# That module's predicate is `not isinstance(x, Sequence) or isinstance(x, <these four>)`,
# and the second clause is the one repeated here. The first is not, because
# `isinstance(np.array(["a", "b"]), Sequence)` is `False` — as it is for a pandas `Series`,
# neither being registered as a `Sequence` — so it would refuse a per-row column of ids
# that arrives in exactly that shape.
_SEQUENCES_OF_CHARACTERS_OR_BYTES = (str, bytes, bytearray, memoryview)


class SplitError(Exception):
    """Raised when specimen ids cannot be split into leave-one-specimen-out folds.

    Two shapes reach it: an argument that is not a column of specimen ids at all — a
    `str`, `bytes`, `bytearray` or `memoryview`, which is a sequence of characters or of
    bytes and splits into one fold per element — and a column carrying fewer than two
    distinct specimens, where holding out the only specimen present would train on
    nothing.
    """


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

    **The dataclass keeps identity equality (`eq=False`), and a generated `__eq__` over
    `np.ndarray` fields cannot answer the question a caller is asking.** Measured on
    CPython 3.13.14: the generated method walks the fields, converts each field's
    comparison to a bool as it goes, and returns the last comparison raw. Comparing two
    folds whose `train_idx` holds more than one row therefore raises `ValueError: The
    truth value of an array with more than one element is ambiguous` **out of the
    comparison itself**, and where every earlier field is short enough for `bool()` to
    succeed, `f1 == f2` evaluates to an `ndarray` rather than to `True` or `False` — which
    raises the same `ValueError` in the `assert` or the `if` the caller wraps it in.
    Neither answer is one a caller can use, and the second is the more treacherous,
    because the comparison looks like it returned something.

    The one shape where the generated method looks harmless is a fold whose every index
    array holds a single row: two size-1 arrays compare to a size-1 boolean, and `bool()`
    of that is defined. That is why the defect is invisible on the smallest possible
    example, and why the tests that pin this use folds of more than one row.

    Identity equality also leaves `__hash__` as the inherited one, so a `Fold` can be put
    in a set or used as a dict key. A generated hash raises `TypeError: unhashable type:
    'numpy.ndarray'` on the first array field, measured on the same interpreter.
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

    **An argument that is a sequence of characters or of bytes is refused, before a single
    row is counted.** `str`, `bytes`, `bytearray` and `memoryview` all satisfy
    `Sequence[str]`, so `loso_folds("456")` would split into three folds of one row each
    called `"4"`, `"5"` and `"6"`, and `loso_folds(b"456")` into three folds held out by
    the integers 52, 53 and 54 — both a patch-level split wearing a specimen-level name,
    returned as a valid result rather than as a refusal. The message names the remedy, and
    the rule is about the shape of the argument and not about how long it is: **no length
    check is applied against any expected row count**, because with one argument there is
    nothing to check against, and a guess would refuse a legitimate cohort of six
    one-patch specimens. The four names are also not the only sequences there are — they
    are the sequences of characters and of bytes, and an argument that is a sequence of
    something else is outside `Sequence[str]` and outside this rule.

    **Fewer than two distinct ids is refused as a `SplitError`**, counting distinct ids
    and not rows: `[]` and `["only", "only"]` are refused alongside `["only"]`, because
    holding out the only specimen there is leaves nothing to train on. The caller cannot
    distinguish those cases by outcome, and none of them is a state a caller can train
    through by accident.
    """
    if isinstance(specimen_of_row, _SEQUENCES_OF_CHARACTERS_OR_BYTES):
        raise SplitError(
            f"loso_folds was given {specimen_of_row!r}, a "
            f"{type(specimen_of_row).__name__}, which is a sequence of characters or of "
            f"bytes and not one specimen id per dataset row: it iterates as "
            f"{len(specimen_of_row)} separate elements — a string as its characters, "
            f"bytes as the integers it spells — so the folds built from it would hold out "
            f"one element each, which is a patch-level split wearing a specimen-level "
            f"name. Pass a list or a tuple of specimen ids, one per dataset row, repeated "
            f"where several rows come from the same specimen"
        )
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