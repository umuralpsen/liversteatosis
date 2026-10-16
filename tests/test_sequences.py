"""Tests for the one predicate both sequence-shaped guards use.

`blobcount/sequences.py` is a single function whose reason to exist is that two modules
each excluded the same four names and both were wrong about a fifth. The tests below pin
the predicate itself; `tests/test_splits.py` and `tests/test_ablation.py` pin what each
module does with its answer.

**Two rows are marked red-before and were: they are the hole.** `array.array` is a
`collections.abc.Sequence` — measured `True` — so the positive test both modules applied
accepted it, and a fifth name in either tuple would have covered one typecode and left the
other nine. The prose in `blobcount/ablation.py` claimed the four names were the whole
dividing line while iterating `array.array('b', b'35')` was measured to yield the two
thresholds 51 and 53; `blobcount/splits.py` documented the same hole honestly. One
predicate settles both.
"""

import array
from collections.abc import Sequence

import pytest

from blobcount.ablation import _thresholds
from blobcount.sequences import is_character_or_byte_sequence
from blobcount.splits import SplitError, loso_folds


class _Raw:
    """A stand-in for the one method `_thresholds` reads.

    `Config.__init__` type-checks since 2026-10-16, so no `Config` can carry a value of the
    wrong type for `ablation.thresholds`; this reaches `_thresholds`' own check for a
    caller that is not a `Config`, which is the same position
    `tests/test_ablation.py::_UntypedThresholds` is in and why that class is there.
    """

    def __init__(self, value):
        self._value = value

    def get(self, key, default=None):
        if key != "ablation.thresholds":
            raise AssertionError(f"_thresholds read an unexpected key: {key!r}")
        return self._value


def test_the_four_named_buffer_types_are_covered():
    for value in ("35", b"35", bytearray(b"35"), memoryview(b"35")):
        assert is_character_or_byte_sequence(value), type(value).__name__
        # And the reason each is refused is the one the predicate exists for: iterating
        # it yields a sequence of separate elements.
        assert isinstance(value, Sequence)
        assert len(value) > 1


def test_an_array_of_bytes_is_refused_which_is_the_hole_it_exists_for():
    # **Red before the module.** `_thresholds(<array.array('b', b'35')>)` was measured to
    # return `(51, 53)` and `loso_folds(array.array('b', b'456'))` to return folds held out
    # 52, 53 and 54 — both accepted, both as valid results. A byte range is a range of
    # whole numbers, so every entry check downstream passed it.
    value = array.array("b", b"35")
    assert is_character_or_byte_sequence(value)
    assert list(value) == [51, 53]


def test_an_array_of_wide_characters_is_refused_too():
    # **The other typecode, and the reason a fifth *name* would not have been the fix.**
    # `array.array('w', 'abc')` iterates `'a'`, `'b'`, `'c'` — the same atoms `str` yields,
    # which is why naming `array.array` once covers every typecode of it rather than the
    # two that yield integers.
    value = array.array("w", "abc")
    assert is_character_or_byte_sequence(value)
    assert list(value) == ["a", "b", "c"]


def test_the_plain_containers_of_the_values_a_caller_meant_are_accepted():
    # **The negative space, and the reason the predicate cannot be written over values.**
    # `ablation.thresholds: [51, 53]` is a legitimate configured range and iterates
    # byte-valued ints exactly as the array does; specimen ids in this project are single
    # characters, so a list of them is a legitimate per-row column. Both must survive.
    for value in ([3, 5], (3, 5), range(3, 6), ["4", "5", "6"], (name for name in "456")):
        assert not is_character_or_byte_sequence(value), repr(value)


def test_a_non_sequence_is_not_this_predicates_question():
    # The predicate is about a `Sequence`'s elements, not about whether a value is a
    # sensible argument. `blobcount.ablation._thresholds` refuses a non-`Sequence` in its
    # own right; `blobcount.splits` does not, for the reason its docstring gives about a
    # per-row id column arriving as a `numpy.ndarray`.
    for value in (35, {"3": 5}, {3, 5}, None, 3.5):
        assert not is_character_or_byte_sequence(value), repr(value)


@pytest.mark.parametrize(
    "value", [array.array("b", b"35"), array.array("B", b"35"), array.array("w", "abc")]
)
def test_the_sweep_refuses_an_array_under_its_own_check(value):
    from blobcount.ablation import _thresholds as thresholds

    with pytest.raises(TypeError, match="not a sequence of thresholds"):
        thresholds(_Raw(value))


@pytest.mark.parametrize(
    "value",
    [array.array("b", b"456"), array.array("w", "abc"), array.array("i", [4, 5, 6])],
)
def test_the_split_refuses_an_array(value):
    # The three shapes a typed buffer arrives in: byte-valued ints, characters, and
    # machine ints. The third is the one nobody writes by accident and it is refused for
    # the same reason as the first two — a fixed-width container whose elements are not the
    # values a caller named.
    with pytest.raises(SplitError, match="sequence of characters or of bytes"):
        loso_folds(value)


def test_the_two_modules_agree_because_there_is_one_predicate():
    # **The point of the module, stated as a fact about the codebase.** Both modules import
    # this function and neither holds a tuple of its own, so a sixth buffer type cannot be
    # added to one of them and missed by the other — which is exactly what happened with
    # `array.array` when each module named the four types separately.
    import inspect

    import blobcount.ablation
    import blobcount.splits

    for module in (blobcount.ablation, blobcount.splits):
        source = inspect.getsource(module)
        assert "from blobcount.sequences import" in source, module.__name__
        assert "bytearray, memoryview)" not in source, module.__name__