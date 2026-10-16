"""The one predicate for a sequence whose elements are a buffer's atoms.

**What the predicate is for.** Two functions in this project take a value a caller
configured or handed over and must refuse one that is a *character or byte sequence* rather
than the list of values the caller meant. `blobcount.ablation._thresholds` reads
`ablation.thresholds` and swept `'35'` as the two thresholds 51 and 53;
`blobcount.splits.loso_folds` reads one specimen id per row and split `'456'` into three
specimens called `"4"`, `"5"` and `"6"` — a patch-level split wearing a specimen-level name,
returned as a valid result. Each module named `str`, `bytes`, `bytearray` and `memoryview`
and both were written separately, and both were incomplete in the same way.

**The hole, measured, in both modules and live on this machine.**

    _thresholds(<array.array('b', b'35')>)   -> (51, 53)
    loso_folds(array.array('b', b'456'))     -> folds held out 52, 53, 54
    loso_folds(array.array('w', 'abc'))      -> folds held out 'a', 'b', 'c'

`array.array` is a `collections.abc.Sequence` — measured `True` — so the positive test both
modules applied accepted it, and a fourth name in each tuple would have covered one typecode
of it and left the other nine. `blobcount.splits` documented the gap honestly and
`blobcount.ablation` claimed the opposite in prose; this module is what makes both claims
the same claim, which is also what lets `ablation`'s prose be corrected rather than
softened.

**Why it is a type test and not a test of the values it yields, stated because it is the
one design decision here.** The ledger's rule reads "reject any `Sequence` whose iteration
yields byte-valued ints", and for `loso_folds` that is the whole rule: a specimen id is a
string and no column of ints is legitimate. For `_thresholds` it cannot be the rule, because
`ablation.thresholds: [51, 53]` is a perfectly good configured range and measured
`_thresholds([51, 53])` is `(51, 53)` — the very answer the byte array was wrongly giving.
Any test on the yielded values therefore refuses a legitimate configuration in order to
catch an illegitimate one, and the only thing that tells the two apart is the container: a
list holds the values a caller configured, and a typed buffer holds the atoms of its own
bytes. So the predicate names the buffer types once, here, and both modules use it.

**What it covers, exhaustively, so the prose in either module can be corrected rather than
hedged.** `str` is the only sequence of characters. `bytes`, `bytearray` and `memoryview` are
the three byte-buffer types, each a sequence of the integers its bytes spell. `array.array`
is a typed buffer and is *always* one of these: measured over `array.typecodes`, the `b` and
`B` codes iterate byte-valued ints and the `u` and `w` codes iterate single characters, so
no typecode of it is a list of values a caller meant. There is no fifth name to add and no
`Sequence` in the standard library whose iteration yields a buffer's atoms and which is not
in this tuple.

**What it deliberately accepts.** `list`, `tuple` and `range`, including a list of
one-character strings — which is a legitimate column of specimen ids in this project, since
the ids are `"4"`, `"5"` and `"6"` — and a list of small integers, which is a legitimate
threshold range. `numpy.ndarray` is accepted here for the reason `blobcount.splits` records:
`isinstance(np.array([...]), Sequence)` is `False`, so a per-row id column arriving in that
shape is not refused. This predicate does not decide whether a non-`Sequence` is a
reasonable column; the negative-form check that does is `blobcount.ablation._thresholds`'s
own and is unchanged.

Declared interface, in `__all__` order:

    is_character_or_byte_sequence(value: Any) -> bool
"""

from __future__ import annotations

import array
from typing import Any

__all__ = ["is_character_or_byte_sequence"]

# The standard library's sequence types whose elements are the **atoms of a character or
# byte buffer** rather than the values a caller configured. `str` is the only sequence of
# characters; `bytes`, `bytearray` and `memoryview` are the three byte-buffer types, each a
# sequence of the integers its bytes spell; and `array.array` is a typed buffer, which is a
# sequence of one or of the other depending on its typecode and of both never.
#
# **All five, and the point of the module is that there is no sixth.** The two module-local
# tuples this replaces held the first four and missed the fifth, which is a `Sequence` by
# registration and iterates as byte-valued ints for its `b` and `B` typecodes. Naming one
# buffer container here rather than two types in two modules is what makes the prose in
# `blobcount.ablation` sayable as a fact rather than as a hedge.
_ATOMIC_BUFFER_TYPES = (str, bytes, bytearray, memoryview, array.array)


def is_character_or_byte_sequence(value: Any) -> bool:
    """Report whether iterating `value` yields the atoms of a character or byte buffer.

    **The whole of the predicate, and the one line of it is the module's reason to
    exist.** `blobcount.splits.loso_folds` and `blobcount.ablation._thresholds` both take a
    `Sequence` a caller meant as a list of values, and both were wrong about one shape: a
    `Sequence` whose elements are a buffer's atoms reads as a longer list than the caller
    wrote, and every element of it was checked by the same entry checks and accepted — a
    byte range is a range of whole numbers, so the entry checks cannot catch it.

    **A type test and not a test of the yielded values**, which is the decision this
    module makes and the one the docstring above argues: `ablation.thresholds: [51, 53]`
    is a valid configured range and iterates byte-valued ints exactly as
    `array.array('b', b'35')` does, so a value test cannot separate them and would refuse
    a legitimate configuration to catch an illegitimate one.

    A non-`Sequence` — an `int`, a `dict`, a `set`, a generator — is not one of these and
    answers `False`. **Whether such a value is a reasonable argument is not this
    predicate's question**: `blobcount.ablation._thresholds` refuses a non-`Sequence` in its
    own right, and `blobcount.splits` does not, for the reason its docstring gives about
    a caller's per-row id column arriving as a `numpy.ndarray`.
    """
    return isinstance(value, _ATOMIC_BUFFER_TYPES)