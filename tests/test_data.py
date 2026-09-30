import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    WeightedRandomSampler,
)

from blobcount.config import load
from blobcount.data import (
    PatchDataset,
    build_loader,
    build_sampler,
    build_transforms,
    class_weights,
)

# Every augmentation key set to a value that changes no pixel. `configs/default.yaml`
# configures all eight as non-zero, and an override of `0.0` for a `float` key or `0` for
# an `int` key passes the exact `type() is type()` check in `load`, so a configuration
# with nothing left to augment is one the caller can name without editing the file. What
# it buys is the strongest statement about `build_transforms` there is: with nothing to
# augment, the train transform has to BE the eval transform, so a test can say that in one
# comparison instead of asserting that eight transform objects are present.
_NO_AUGMENTATION = [
    "augmentation.horizontal_flip=false",
    "augmentation.vertical_flip=false",
    "augmentation.rotation_degrees=0",
    "augmentation.color_jitter_brightness=0.0",
    "augmentation.color_jitter_contrast=0.0",
    "augmentation.color_jitter_saturation=0.0",
    "augmentation.color_jitter_hue=0.0",
    "augmentation.affine_translate=0.0",
]


def _write_patch(path, size: int = 64, seed: int = 0):
    """Write one RGB patch of deterministic noise and return the path."""
    random = np.random.default_rng(seed)
    Image.fromarray((random.random((size, size, 3)) * 255).astype("uint8")).save(path)
    return path


def _patches(directory, count: int = 6, size: int = 64):
    """Write `count` distinct patches and return their paths in order."""
    return [_write_patch(directory / f"patch{index}.png", size, seed=index) for index in range(count)]


# The eight labels a fold's column can hold that are not class indices, with the condition
# each one breaks. `2` is absent from the shared list on purpose: it is outside
# `range(2)` and inside `range(3)`, so only the function that was given a class count can
# refuse it, and that asymmetry is what the two tests over the call sites pin. The list is
# the module docstring's enumeration — `0` and not `'0'` and not `true` and not `.inf` or
# `.nan` and not `1.5` and not `-1` and not `2` — with the second half of it split out.
_NOT_A_CLASS_INDEX = [
    ("0", "quoted digit", "not a plain number but a str"),
    (True, "true", "not a plain number but a bool"),
    (False, "false", "not a plain number but a bool"),
    (None, "none", "not a plain number but a NoneType"),
    (1.5, "fraction", "not a whole number"),
    (math.inf, "infinity", "not finite"),
    (-math.inf, "negative infinity", "not finite"),
    (math.nan, "nan", "not finite"),
    (-1, "negative", "outside range(2)"),
]


# ---------------------------------------------------------------------------
# The brief's tests, in the brief's order.
# ---------------------------------------------------------------------------


def test_class_weights_use_only_the_labels_passed():
    w = class_weights([0, 0, 0, 1])
    assert w.tolist() == pytest.approx([4 / 6, 4 / 2])


def test_class_weights_differ_when_given_a_different_subset():
    a = class_weights([0] * 90 + [1] * 10)
    b = class_weights([0] * 50 + [1] * 50)
    assert a[1] > b[1]


def test_sampler_balances_classes():
    labels = [0] * 90 + [1] * 10
    s = build_sampler(labels, enabled=True, seed=42)
    drawn = list(iter(s))
    # The claim is that the draw is BALANCED, so what is measured is the class of each
    # drawn index and not the index itself. The brief's fence wrote `sum(drawn)` and
    # centred it on `0.1`, which are two things at once: `drawn` holds dataset positions,
    # and the position of a patch says nothing about its class. Measured on this stack
    # with these labels and this seed, the mean drawn position is 63.31, and the classes
    # here occupy positions 0-89 and 90-99, so no sampler that balances them can put the
    # mean position within 0.05 of 0.1 — only a sampler that drew almost nothing but the
    # first two patches could, and that is the opposite of what this one is for. What is
    # left is the reading the test's own name states: the fraction of the draw that is
    # class 1, which is 0.1 under no balancing and 0.5 under balancing.
    rate = sum(labels[index] for index in drawn) / len(drawn)
    # 100 draws at p=0.5 have a standard error of 0.05, so 0.15 is three of them. A sampler
    # that did not balance would sit at 0.1, which is eight standard errors away.
    assert abs(rate - 0.5) < 0.15, rate


def test_sampler_disabled_returns_none():
    assert build_sampler([0, 1], enabled=False, seed=42) is None


def test_eval_transform_has_no_randomness():
    _, ev = build_transforms(load())
    img = Image.fromarray((np.random.rand(300, 300, 3) * 255).astype("uint8"))
    assert torch.equal(ev(img), ev(img))


def test_loader_is_deterministic_given_seed(tmp_path):
    paths = []
    for name in ("a.png", "b.png"):
        p = tmp_path / name
        Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(p)
        paths.append(p)
    ds = PatchDataset(paths, [0, 1])
    a = build_loader(ds, 2, None, True, 0, 7)
    b = build_loader(ds, 2, None, True, 0, 7)
    # `tolist()` is what the comparison needs and is why the label is a plain `int`: the
    # brief's fence wrote `[y for _, y in a] == [y for _, y in b]`, and a default collate
    # hands back one tensor per batch, so that is a list of tensors. List equality
    # compares the elements with `PyObject_RichCompareBool`, which short-circuits on
    # identity and otherwise calls `bool()` on the elementwise result — measured on this
    # stack: `RuntimeError: Boolean value of Tensor with more than one value is
    # ambiguous`. The loader is deterministic either way (`torch.equal` on the two label
    # tensors is `True`); the comparison above is what cannot be written with tensors.
    assert [y.tolist() for _, y in a] == [y.tolist() for _, y in b]


def test_loader_shuffle_is_disabled_when_a_sampler_is_supplied():
    ds = PatchDataset([Path("a.png")], [0])
    loader = build_loader(ds, 1, build_sampler([0], True, 1), True, 0, 1)
    assert isinstance(loader.sampler, torch.utils.data.RandomSampler) is False
    assert loader.sampler is not None


# ---------------------------------------------------------------------------
# The seven above are the brief's. They state the per-fold arithmetic, the class
# balance, the disabled sampler, the eval transform's determinism, the loader's seed and
# the sampler overriding shuffle. What follows pins the parts of the interface they only
# imply, and the two rulings that are not in the brief at all: the weight of a class a
# fold holds none of, and the refusal of a label outside the class range.
# ---------------------------------------------------------------------------


def test_class_weights_are_one_float32_entry_per_class():
    assert class_weights([0, 1]).dtype == torch.float32
    assert class_weights([0, 1]).tolist() == pytest.approx([1.0, 1.0])
    assert class_weights([0, 0, 2], num_classes=3).tolist() == pytest.approx([3 / 6, 0.0, 3 / 3])


def test_a_class_the_fold_holds_none_of_weighs_nothing_and_is_never_infinite():
    # This is the state the shipped cohort can reach and not a corner case: the registry
    # admits three specimens for training, a fold trains on two of them, and those two
    # can both be class 0. `total / (num_classes * count[0])` is then a division by zero
    # for class 1, and the two answers are not interchangeable — an `inf` in a tensor a
    # caller may multiply, sum or write into an artifact is a landmine, and this one never
    # multiplies anything, because the class contributes no sample to the loss.
    fold = class_weights([0] * 90)
    assert fold.tolist() == pytest.approx([90 / 180, 0.0])
    assert torch.isfinite(fold).all()
    # And the difference from the pooled vector is the point: the weight the fold is
    # trained under must not be the weight the whole dataset would have given.
    assert class_weights([0] * 90 + [1] * 10)[1] > fold[1]


@pytest.mark.parametrize("label", [-1, 2], ids=["below", "above"])
def test_class_weights_refuse_a_label_outside_the_class_range(label):
    # Left unchecked this surfaces as an index error inside `CrossEntropyLoss`, which
    # attributes a caller's mistake to the loss. The message names the remedy.
    with pytest.raises(ValueError, match="outside range") as refused:
        class_weights([0, 1, label])
    assert "CrossEntropyLoss" in str(refused.value)


def test_class_weights_refuse_a_label_that_is_not_a_whole_class_index():
    # `1.5` is inside `range(2)` and is not a class index, and the message says the second
    # of those things rather than the first: the old one claimed a range it was not
    # outside of, which sends the reader to the class count instead of to the cell.
    with pytest.raises(ValueError, match="not a whole number"):
        class_weights([0, 1.5])


def test_the_sampler_draws_one_sample_per_label_with_replacement():
    labels = [0] * 90 + [1] * 10
    s = build_sampler(labels, enabled=True, seed=42)
    assert isinstance(s, WeightedRandomSampler)
    assert s.num_samples == len(labels)
    assert s.replacement is True
    assert len(s.weights) == len(labels)


def test_the_sampler_seed_decides_the_draw():
    labels = [0] * 90 + [1] * 10
    assert list(iter(build_sampler(labels, True, 42))) == list(iter(build_sampler(labels, True, 42)))
    assert list(iter(build_sampler(labels, True, 42))) != list(iter(build_sampler(labels, True, 43)))


def test_the_eval_transform_is_deterministic_across_calls_and_across_images():
    _, ev = build_transforms(load())
    first = Image.fromarray((np.random.default_rng(1).random((300, 300, 3)) * 255).astype("uint8"))
    second = Image.fromarray((np.random.default_rng(2).random((300, 300, 3)) * 255).astype("uint8"))
    for image in (first, second):
        rendered = ev(image)
        assert torch.equal(rendered, ev(image))
        assert torch.equal(rendered, ev(image))


def test_the_train_transform_varies_between_calls_when_augmentation_is_configured():
    # The counterpart to the test above. "The eval transform has no randomness" is
    # satisfied by a transform that does nothing at all, so it says nothing on its own;
    # this is what makes it a statement about randomness rather than about emptiness.
    train, _ = build_transforms(load())
    img = Image.fromarray((np.random.default_rng(3).random((300, 300, 3)) * 255).astype("uint8"))
    assert len({train(img).numpy().tobytes() for _ in range(5)}) > 1


def test_the_train_transform_is_the_eval_transform_when_nothing_is_configured_to_augment():
    train, ev = build_transforms(load(overrides=_NO_AUGMENTATION))
    img = Image.fromarray((np.random.default_rng(4).random((300, 300, 3)) * 255).astype("uint8"))
    assert torch.equal(train(img), ev(img))


@pytest.mark.parametrize(
    "key", ["augmentation.horizontal_flip", "augmentation.vertical_flip"]
)
def test_each_flip_key_reaches_the_transform_it_names(key):
    # The two booleans are the keys the config module's exact `type() is type()` check
    # exists for, and a `bool` that is coerced through `int()` would still flip. What this
    # pins is that each key reaches its own transform: off leaves the train transform equal
    # to the eval one, and on makes it differ.
    train, ev = build_transforms(load(overrides=_NO_AUGMENTATION))
    img = Image.fromarray((np.random.default_rng(5).random((300, 300, 3)) * 255).astype("uint8"))
    assert torch.equal(train(img), ev(img))

    flipped, ev = build_transforms(load(overrides=[*_NO_AUGMENTATION, f"{key}=true"]))
    assert not torch.equal(flipped(img), ev(img))


@pytest.mark.parametrize("input_size", [224, 64])
def test_both_transforms_render_the_configured_input_size(input_size):
    train, ev = build_transforms(load(overrides=[f"training.input_size={input_size}"]))
    img = Image.fromarray((np.random.default_rng(6).random((300, 300, 3)) * 255).astype("uint8"))
    for rendered in (train(img), ev(img)):
        assert rendered.shape == (3, input_size, input_size)
        assert rendered.dtype == torch.float32


def test_the_eval_transform_normalizes_with_the_imagenet_statistics():
    _, ev = build_transforms(load())
    flat = Image.fromarray(np.full((300, 300, 3), 128, dtype="uint8"))
    rendered = ev(flat)
    # A constant image is its own pixel value in every channel, so the rendered tensor is
    # the normalisation of 128/255 and nothing else. It is the only way to state which
    # constants `Normalize` was given without asserting that two objects exist.
    for channel, (mean, std) in enumerate(zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))):
        assert rendered[channel].mean().item() == pytest.approx((128 / 255 - mean) / std, abs=1e-5)


def test_the_dataset_reports_its_length_and_pairs_each_label_with_its_own_patch(tmp_path):
    paths = _patches(tmp_path, count=3)
    ds = PatchDataset(paths, [0, 1, 1])
    assert len(ds) == 3
    assert [ds[index][1] for index in range(3)] == [0, 1, 1]
    assert all(type(ds[index][1]) is int for index in range(3))


def test_a_dataset_with_no_transform_still_yields_a_tensor(tmp_path):
    # Measured on this stack: `default_collate` raises `TypeError: batch must contain
    # tensors, numpy arrays, numbers, dicts or lists; found <class 'PIL.Image.Image'>`
    # for a list of PIL images. So `transform=None` is not an identity that hands back the
    # image, and the brief's own loader test — which builds a transform-less dataset and
    # iterates it — is what pins it.
    ds = PatchDataset(_patches(tmp_path, count=2), [0, 1])
    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == 0


def test_the_dataset_applies_the_transform_it_is_given(tmp_path):
    _, ev = build_transforms(load(overrides=["training.input_size=16"]))
    paths = _patches(tmp_path, count=2)
    ds = PatchDataset(paths, [0, 1], transform=ev)
    first, second = ds[0][0], ds[1][0]
    assert first.shape == (3, 16, 16)
    # Two different patches have to render differently, or the transform is being handed
    # a constant rather than the image the index named.
    assert not torch.equal(first, second)


def test_a_patch_that_cannot_be_opened_raises_its_own_error(tmp_path):
    ds = PatchDataset([tmp_path / "absent.png"], [0])
    with pytest.raises(OSError):
        ds[0]


def test_the_loader_uses_the_generator_it_is_given_and_not_the_global_one(tmp_path):
    # The two loaders are separated by a deliberate draw from the global generator. A
    # loader whose order came from the global RNG would differ here; one that holds its
    # own `torch.Generator` cannot.
    ds = PatchDataset(_patches(tmp_path, count=6), [0, 1, 0, 1, 0, 1])
    a = build_loader(ds, 2, None, True, 0, 7)
    torch.manual_seed(999)
    torch.randn(64)
    b = build_loader(ds, 2, None, True, 0, 7)
    assert [y.tolist() for _, y in a] == [y.tolist() for _, y in b]


def test_the_loader_honours_the_shuffle_flag_when_no_sampler_is_given(tmp_path):
    ds = PatchDataset(_patches(tmp_path, count=4), [0, 1, 0, 1])
    loader = build_loader(ds, 2, None, False, 0, 7)
    assert isinstance(loader.sampler, SequentialSampler)
    assert [y.tolist() for _, y in loader] == [[0, 1], [0, 1]]


def test_the_loader_uses_the_sampler_it_is_handed(tmp_path):
    ds = PatchDataset(_patches(tmp_path, count=2), [0, 1])
    sampler = build_sampler([0, 1], enabled=True, seed=1)
    loader = build_loader(ds, 2, sampler, True, 0, 1)
    assert loader.sampler is sampler
    assert not isinstance(loader.sampler, RandomSampler)


def test_the_batch_size_reaches_the_loader(tmp_path):
    ds = PatchDataset(_patches(tmp_path, count=5), [0] * 5)
    loader = build_loader(ds, 2, None, False, 0, 7)
    assert loader.batch_size == 2
    assert [images.shape[0] for images, _ in loader] == [2, 2, 1]


def test_the_loader_draws_the_same_order_with_and_without_workers(tmp_path):
    # `training.num_workers: 4` is what the shipped configuration asks for, and the worker
    # path is reached only here. The base seed the workers receive is derived from the same
    # `torch.Generator`, and the order the sampler produced is decided in the parent, so
    # the two label sequences are equal.
    ds = PatchDataset(_patches(tmp_path, count=4), [0, 1, 0, 1], transform=build_transforms(load())[1])
    in_process = build_loader(ds, 2, None, True, 0, 7)
    with_workers = build_loader(ds, 2, None, True, 2, 7)
    assert [y.tolist() for _, y in in_process] == [y.tolist() for _, y in with_workers]


def test_torch_refuses_a_sampler_and_shuffle_together(tmp_path):
    # Not a test of this project: it pins the reason `build_loader` drops the `shuffle`
    # flag rather than honouring it. Measured: `ValueError: sampler option is mutually
    # exclusive with shuffle`. A refactor that passed both through would not be caught by
    # any other test here until a fold trained.
    ds = PatchDataset(_patches(tmp_path, count=2), [0, 1])
    with pytest.raises(ValueError, match="mutually exclusive"):
        DataLoader(ds, batch_size=2, sampler=build_sampler([0, 1], True, 1), shuffle=True)


# ---------------------------------------------------------------------------
# The label predicate. One `_class_index`, three call sites, and the conditions it holds
# are the module docstring's enumeration.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "shape", "condition"), _NOT_A_CLASS_INDEX, ids=[row[1] for row in _NOT_A_CLASS_INDEX]
)
def test_class_weights_refuse_every_label_that_is_not_a_class_index(label, shape, condition):
    with pytest.raises(ValueError, match="is not a class index") as refused:
        class_weights([0, 1, label])
    message = str(refused.value)
    # The condition this label broke, and the two sentences that make the refusal worth
    # reading: why the label matters at all, and what the caller does about it.
    assert condition in message, message
    assert "CrossEntropyLoss" in message
    assert "Fix the label" in message
    # And the enumeration the docstring states, so the two cannot drift apart silently.
    assert "and not '0' and not true and not .inf or .nan and not 1.5 and not -1 and not 2" in message


@pytest.mark.parametrize(
    ("label", "shape", "condition"), _NOT_A_CLASS_INDEX, ids=[row[1] for row in _NOT_A_CLASS_INDEX]
)
def test_the_refusal_never_claims_a_range_the_label_is_inside_of(label, shape, condition):
    # The defect this closes, stated as a property of every message rather than of one
    # label: `int("0")` is `0`, so the range test that stood beside the conversion
    # reported `label '0' is outside range(2)` and sent the reader to the class count.
    # `1.5` is the same false claim with a different value. Only a label that really is
    # outside may be told that it is.
    with pytest.raises(ValueError) as refused:
        class_weights([0, 1, label])
    message = str(refused.value)
    if "outside range" not in condition:
        assert "outside range" not in message, message
    if condition != "not finite":
        assert "not finite" not in message, message


def test_a_non_finite_label_is_refused_by_the_rule_and_not_by_the_conversion():
    # Both of these escaped the old validator for reasons that had nothing to do with the
    # rule, measured on this stack: `int(float("inf"))` raises `OverflowError`, which is
    # not the `ValueError` the module's contract promises, and `int(float("nan"))` raises
    # `ValueError` carrying CPython's `cannot convert float NaN to integer`, which names
    # no rule and offers no remedy. A validator whose refusals are the standard library's
    # is a validator that reports whatever the conversion happened to dislike.
    for label in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError) as refused:
            class_weights([0, 1, label])
        message = str(refused.value)
        assert "not finite" in message, message
        assert "cannot convert float" not in message
        assert "cannot convert float infinity" not in message


def test_a_bool_is_refused_where_every_comparison_would_have_accepted_it():
    # `bool` subclasses `int`, and that is the whole of it: `1 != True` is `False`, so the
    # range test accepted it, and `int(True) == 1`, so the whole-number test did too.
    # `class_weights([0, 1, True])` returned the weights of classes 1 and 0 and said
    # nothing, which is a fold that trains on a class nobody labelled.
    assert (1 == True) and (1 != True) is False
    with pytest.raises(ValueError, match="not a plain number but a bool"):
        class_weights([True, False])


def test_an_int_too_large_to_be_a_float_is_refused_as_a_range_and_not_as_an_overflow():
    # `math.isfinite` converts its argument, so testing every label for finiteness would
    # trade the `OverflowError` above for a second one a layer below it, in the predicate
    # written to end them. There is no such thing as an infinite `int`, so the test is
    # scoped to floats and this label is judged on the only condition that can judge it.
    with pytest.raises(ValueError, match="outside range\\(2\\)"):
        class_weights([0, 1, 10**400])


@pytest.mark.parametrize(
    ("label", "shape", "condition"), _NOT_A_CLASS_INDEX, ids=[row[1] for row in _NOT_A_CLASS_INDEX]
)
def test_the_sampler_refuses_every_label_the_weights_refuse(label, shape, condition):
    # `build_sampler` held no check at all before, so a fold's labels were validated in
    # one of three places depending on which function the caller reached first. The two
    # are the same predicate over the same list now, and the check is before the `enabled`
    # test, so a disabled sampler refuses what an enabled one refuses.
    with pytest.raises(ValueError, match="is not a class index"):
        build_sampler([0, 1, label], enabled=True, seed=1)
    with pytest.raises(ValueError, match="is not a class index"):
        build_sampler([0, 1, label], enabled=False, seed=1)


def test_the_sampler_names_the_lower_bound_because_it_was_given_no_class_count():
    # The asymmetry between the three call sites, stated rather than left to be found at
    # fold time. `build_sampler` has no `num_classes`, so the upper bound is not a
    # condition it can hold and the lower bound is: no class index is below zero under any
    # labelling run. `class_weights` is where `-1` is reported as outside the range.
    with pytest.raises(ValueError, match="below zero, which no class index is"):
        build_sampler([0, -1], enabled=True, seed=1)
    with pytest.raises(ValueError, match="outside range\\(2\\)"):
        class_weights([0, -1])


@pytest.mark.parametrize(
    ("label", "shape", "condition"), _NOT_A_CLASS_INDEX, ids=[row[1] for row in _NOT_A_CLASS_INDEX]
)
def test_the_dataset_refuses_every_label_the_weights_refuse(label, shape, condition, tmp_path):
    # The third of the three, and the one that mattered most: its labels reached
    # `__getitem__`, where the `int()` that stood there truncated whatever it was given.
    with pytest.raises(ValueError, match="is not a class index"):
        PatchDataset(_patches(tmp_path, count=3), [0, 1, label])


def test_the_dataset_does_not_truncate_a_fractional_label(tmp_path):
    # `int(1.5)` is `1`. The conversion that stood in `__getitem__` truncated a fractional
    # label to a class the labelling run never wrote, said nothing, and the only trace was
    # a `labels.csv` cell a reader had to notice by hand. The check is in `__init__` now,
    # so the label never reaches an item.
    with pytest.raises(ValueError, match="not a whole number"):
        PatchDataset(_patches(tmp_path, count=1), [1.5])
    with pytest.raises(ValueError, match="not a class index"):
        PatchDataset(_patches(tmp_path, count=1), [0, "1"])


def test_the_dataset_and_the_sampler_hold_the_same_rule_the_weights_hold(tmp_path):
    # One predicate, three call sites: every label every one of them refuses is refused by
    # the same four conditions, and the only difference between them is which conditions
    # they have the arguments to hold. A `2` is the case that shows it — outside
    # `range(2)`, inside `range(3)` — and it is accepted below by the two that hold no
    # class count and refused by the one that does.
    for labels in ([0, 1], [0.0, 1.0]):
        assert class_weights(labels).tolist() == pytest.approx([1.0, 1.0])
        assert build_sampler(labels, enabled=True, seed=1) is not None
        ds = PatchDataset(_patches(tmp_path, count=2), labels)
        # And the label the dataset hands back is the plain `int` the check returned, which
        # is what `__getitem__`'s return type claims whether the caller wrote `0` or `0.0`.
        assert all(type(label) is int for label in ds.labels)
    assert len(PatchDataset(_patches(tmp_path, count=1), [2]).labels) == 1
    assert build_sampler([0, 2], enabled=True, seed=1) is not None
    with pytest.raises(ValueError, match="outside range\\(2\\)"):
        class_weights([0, 2])


def test_a_quoted_digit_is_refused_because_a_csv_cell_is_a_string(tmp_path):
    # The shape this refusal is most likely to meet, and the reason the message names the
    # type: `labels.csv` is read by `csv.DictReader`, where every cell is a string, so the
    # one mistake this predicate exists for is a caller that skipped the `int()` at the
    # read. `'0'` and `0` differ only in those quotes and the weights were being computed
    # from the difference.
    with pytest.raises(ValueError, match="not a plain number but a str"):
        class_weights(["0", "1"])
    with pytest.raises(ValueError, match="not a plain number but a str"):
        build_sampler(["0", "1"], enabled=True, seed=1)
    with pytest.raises(ValueError, match="not a plain number but a str"):
        PatchDataset(_patches(tmp_path, count=2), ["0", "1"])


# ---------------------------------------------------------------------------
# The loader's shuffle. The order must come from the generator this function seeds, and
# not from a `DataLoader` behaviour that is torch's business and not this interface's.
# ---------------------------------------------------------------------------


def test_the_shuffle_sampler_is_built_here_and_holds_the_loaders_generator(tmp_path):
    # `shuffle=True` is a `RandomSampler` holding the loader's own generator, and the
    # `shuffle` flag this function passes is `False` whatever the caller asked for. The
    # torch installed here hands its `generator` to the sampler it creates for
    # `shuffle=True`, so on this stack the two are indistinguishable in what they produce;
    # what is pinned is the shape, which is what a torch that did not do it would change.
    ds = PatchDataset(_patches(tmp_path, count=6), [0, 1, 0, 1, 0, 1])
    loader = build_loader(ds, 2, None, True, 0, 7)
    assert isinstance(loader.sampler, RandomSampler)
    assert loader.sampler.generator is loader.generator
    assert loader.sampler.generator is not torch.default_generator
    # A loader built with no shuffle has a `SequentialSampler` and no generator at all,
    # which is the branch the flag is not reaching.
    plain = build_loader(ds, 2, None, False, 0, 7)
    assert isinstance(plain.sampler, SequentialSampler)


def test_a_second_epoch_is_a_different_order_and_a_rerun_gives_the_same_one(tmp_path):
    # One generator serves the permutation and the base seed each worker gets, so a second
    # epoch draws again from the same stream: a different order from the first epoch, and
    # the same different order in the next run. `training.epochs` is a configured key, so
    # this is a property the loop depends on rather than an accident of one pass.
    ds = PatchDataset(_patches(tmp_path, count=6), [0, 1, 0, 1, 0, 1])
    a = build_loader(ds, 2, None, True, 0, 7)
    first, second = [y.tolist() for _, y in a], [y.tolist() for _, y in a]
    b = build_loader(ds, 2, None, True, 0, 7)
    assert [y.tolist() for _, y in b] == first
    assert [y.tolist() for _, y in b] == second
    assert first != second
