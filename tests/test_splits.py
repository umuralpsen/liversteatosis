import dataclasses

import numpy as np
import pytest
from blobcount.splits import Fold, SplitError, loso_folds


def test_six_specimens_give_six_folds():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    assert len(folds) == 6


def test_each_specimen_is_held_out_exactly_once():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    assert sorted(f.held_out for f in folds) == ["1", "2", "3", "4", "5", "6"]


def test_train_and_val_are_disjoint_and_complementary():
    folds = loso_folds(["1", "2", "3", "4", "5", "6"])
    for f in folds:
        assert set(f.train_idx) & set(f.val_idx) == set()
        assert len(f.train_idx) + len(f.val_idx) == 6


def test_fold_order_is_deterministic():
    a = loso_folds(["3", "1", "2"])
    b = loso_folds(["1", "2", "3"])
    assert [f.held_out for f in a] == [f.held_out for f in b] == ["1", "2", "3"]


def test_single_specimen_raises():
    with pytest.raises(SplitError, match="at least two"):
        loso_folds(["only"])


# The five tests above are the brief's. They state the count of folds, the coverage of
# the specimens, the partition of the rows, the order of the folds and one refusal. What
# follows pins the parts of the interface they only imply: that an index is a position in
# the caller's own array rather than a position in a deduplicated specimen list, that the
# two arrays are `int64`, that `fold` numbers the sorted order from zero, that the
# refusal covers no rows at all and not only a single specimen, and that comparing two
# folds is not a trap.


def test_val_idx_is_every_position_of_the_held_out_specimen():
    ids = ["b", "a", "b", "a", "c", "b"]
    by_specimen = {f.held_out: f for f in loso_folds(ids)}
    assert by_specimen["b"].val_idx.tolist() == [0, 2, 5]
    assert by_specimen["a"].val_idx.tolist() == [1, 3]
    assert by_specimen["c"].val_idx.tolist() == [4]
    assert by_specimen["b"].train_idx.tolist() == [1, 3, 4]


def test_every_row_is_validated_exactly_once_across_the_folds():
    ids = ["b", "a", "c", "b", "a", "b", "c"]
    folds = loso_folds(ids)
    validated = [int(row) for f in folds for row in f.val_idx]
    assert sorted(validated) == list(range(len(ids)))


def test_fold_numbers_the_sorted_held_out_order_from_zero():
    folds = loso_folds(["c", "a", "b", "a"])
    assert [(f.fold, f.held_out) for f in folds] == [(0, "a"), (1, "b"), (2, "c")]


def test_train_and_val_indices_are_int64_arrays():
    for f in loso_folds(["1", "2", "3"]):
        assert isinstance(f.train_idx, np.ndarray)
        assert isinstance(f.val_idx, np.ndarray)
        assert f.train_idx.dtype == np.int64
        assert f.val_idx.dtype == np.int64


@pytest.mark.parametrize(
    "specimen_of_row",
    [[], ["only", "only"]],
    ids=["no_rows", "one_distinct_specimen_over_several_rows"],
)
def test_fewer_than_two_distinct_specimens_raises(specimen_of_row):
    with pytest.raises(SplitError, match="at least two"):
        loso_folds(specimen_of_row)


def test_a_non_ascii_specimen_id_round_trips_into_its_fold():
    folds = loso_folds(["2", "hasta-ğü-01", "1"])
    assert [f.held_out for f in folds] == ["1", "2", "hasta-ğü-01"]
    assert folds[2].val_idx.tolist() == [1]


def test_a_tuple_of_ids_is_accepted_like_a_list():
    from_tuple = loso_folds(("a", "b"))
    from_list = loso_folds(["a", "b"])
    assert [f.held_out for f in from_tuple] == [f.held_out for f in from_list]
    assert from_tuple[0].val_idx.tolist() == from_list[0].val_idx.tolist()


def test_fold_is_frozen_and_declares_the_four_fields_in_order():
    assert [f.name for f in dataclasses.fields(Fold)] == [
        "fold",
        "held_out",
        "train_idx",
        "val_idx",
    ]
    fold = loso_folds(["a", "b"])[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        fold.fold = 3


# `Fold` carries two `np.ndarray` fields, and the three tests below pin what a caller
# gets from comparing two of them. They are written against folds of MORE THAN ONE ROW on
# purpose. A generated `__eq__` on CPython 3.13.14 walks the fields, converts each
# field's comparison to a bool as it goes, and returns the last comparison raw, so a
# one-row fold is the one shape where the defect is invisible: two size-1 arrays compare
# to a size-1 boolean and `bool()` of that is defined. The earlier version of this test
# used `loso_folds(["a", "b"])[0]`, whose index arrays are both size 1, and so could not
# produce the `ValueError` its name claimed.


def test_comparing_two_folds_with_a_multi_row_train_side_answers_false():
    # `train_idx` is not the last field, so a generated `__eq__` converts its comparison
    # to a bool -- and `bool()` of a two-element boolean array raises `ValueError` out of
    # the comparison itself. `Fold` answers `False`.
    a = loso_folds(["a", "a", "b"])[1]
    b = loso_folds(["a", "a", "b"])[1]
    assert len(a.train_idx) == 2
    assert a is not b
    assert (a == b) is False


def test_comparing_two_folds_with_a_multi_row_val_side_answers_false_not_an_array():
    # `val_idx` is the last field, so a generated `__eq__` returns its elementwise
    # comparison raw: `a == b` would evaluate to `array([True, True])`, which looks like
    # it returned something and raises `ValueError` in whatever boolean context a caller
    # wraps it in. `Fold` answers `False`.
    a = loso_folds(["a", "b", "a"])[0]
    b = loso_folds(["a", "b", "a"])[0]
    assert len(a.val_idx) == 2
    assert (a == b) is False


def test_a_fold_is_hashable_by_identity():
    folds = loso_folds(["a", "b"])
    again = loso_folds(["a", "b"])
    assert folds[0] is not again[0]
    assert len({folds[0], again[0]}) == 2
    assert len({folds[0], folds[0]}) == 1


@pytest.mark.parametrize(
    "specimen_of_row",
    ["456", b"456", bytearray(b"456"), memoryview(b"456")],
    ids=["str", "bytes", "bytearray", "memoryview"],
)
def test_a_sequence_of_characters_or_bytes_is_refused(specimen_of_row):
    # Each of the four is a `Sequence` -- of characters, or of the integers its bytes
    # spell -- and each would split into one fold per element, which is a patch-level
    # split wearing a specimen-level name and would be returned as a valid result.
    with pytest.raises(SplitError, match="one specimen id per dataset row"):
        loso_folds(specimen_of_row)


def test_a_numpy_column_of_ids_is_accepted_like_a_list():
    # The tolerance this guard deliberately keeps: a per-row column of ids arrives from
    # pandas as an `ndarray` or a `Series`, and neither is registered as a
    # `collections.abc.Sequence`, so a predicate that also required membership would
    # refuse the two most natural things a caller has.
    folds = loso_folds(np.array(["b", "a", "b"]))
    assert [f.held_out for f in folds] == ["a", "b"]
    assert folds[1].val_idx.tolist() == [0, 2]
    assert folds[1].train_idx.tolist() == [1]