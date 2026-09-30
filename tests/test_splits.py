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


def test_comparing_two_folds_is_a_plain_answer_and_not_a_value_error():
    # A generated `__eq__` over two `np.ndarray` fields answers with an array, and
    # `bool()` of it raises "the truth value of an array with more than one element is
    # ambiguous" for every fold of more than one row. `Fold` therefore keeps identity
    # equality, so a caller comparing two folds gets `False` instead of that.
    a = loso_folds(["a", "b"])[0]
    b = loso_folds(["a", "b"])[0]
    assert a is not b
    assert (a == b) is False
    assert len({a, b}) == 2