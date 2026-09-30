"""The patch dataset, the two transforms, and the class weights of one fold.

**The class weights are computed per fold, from that fold's training labels, and that is
the defect this module exists to remove.** `train.py:87` counted the classes of
`all_labels` — every patch in the cohort, pooled — and `train.py:159` handed that one
tensor to `nn.CrossEntropyLoss` in every fold. So the weights a fold trained under were
derived from the label distribution of the specimens that fold was validating on, which is
the same leak the leave-one-specimen-out split exists to close, arriving through the loss
instead of through the split. `class_weights` below takes the labels it is given and
computes `total / (num_classes * count[c])` over those and no others, and the
cross-validation loop hands it a fold's training labels.

**A class the fold holds none of weighs `0.0`, and not `inf`.** This is a reachable state
and not a corner case: `blobcount.registry.usable_for_training` admits three specimens for
training, a fold trains on two of them, and those two can both be class 0. The
quotient is then a division by zero for the class that is absent, and the two answers are
not interchangeable. `inf` never multiplies anything here, because a class with no
samples contributes no term to the loss — but `class_weights` returns a tensor a caller
may multiply, sum, or write into a fold summary, and an `inf` in that tensor is a landmine
the moment it leaves here: it propagates through a mean into a number a reader would
take for a measurement, and a single `nan` in a serialized artifact loads back as a fact
about the run. `0.0` says what is true, that the fold has nothing of that class to
weight, and it is a number every later stage can hold.

**A label outside the class range is refused here rather than inside the loss.** Left
alone it reaches `CrossEntropyLoss` as an out-of-range target and surfaces as an index
error attributed to the loss, which is where a caller goes looking for a defect in this
project's training loop rather than in the column they read. The refusal is a
`ValueError` and not a type of this project's own: the declared interface below declares
none, and a `ValueError` is a type a command line already knows how to report. **No
command line catches it yet, because the training entry point arrives with the
cross-validation loop**, so until then a label outside the range reaches the interpreter
as a traceback out of this module; the message names the remedy, which is the part that
survives the traceback.

**One predicate decides what a label is, and all three functions that take a fold's labels
run it over every one of them.** `class_weights`, `build_sampler` and `PatchDataset` are
the three, and they used to disagree about what a label is: `class_weights` refused what
was outside the range, `build_sampler` counted whatever `Counter` accepted, and
`PatchDataset` passed the label to `int()` in `__getitem__` and truncated a `1.5` to `1`
in silence. So a caller's labels were checked in one of three places, and a caller who had
them wrong learned it from whichever function it happened to call first — which is the
"looks validated and is not" state this module is written to refuse. `_class_index` is the
one answer, it is private, and it runs to completion **before** the `int()` it wraps, so
the conversion is reached only for a value that already has one.

**A class index is a whole number a type can carry, and the refusal names the condition it
broke.** The rule is a type rule and a value rule, in the shape
`blobcount.ablation._thresholds` gives its thresholds and for the reason that module states
— a rule that lives in three places and in a test comment is a rule nobody can hold:

    0, or 0.0, or a numpy integer
    not '0', not True or False, not a float with a fraction, not .inf or .nan, not -1,
    and not 2 when num_classes is 2

**The type rule is `numbers.Integral` or `float`, and it is stated that way rather than as
a list of types it happens to name.** The first version of this predicate tested
`isinstance(label, (int, float))`, and that refused a numpy integer: measured on the numpy
2.2.2 here, `isinstance(np.int64(0), int)` is `False` and so is `isinstance(np.int32(0),
int)`, so `class_weights(np.array([0, 1, 0, 1]))` and `build_sampler(np.array([0, 1, 0, 1]),
True, 1)` raised where both had worked — a caller reading the `label` column of a DataFrame
holds nothing else. `numbers.Integral` is the registration CPython and numpy both agree on
and `float` is the other half of it, so the two rules are **a numpy integer is a class
index; a float is one only when it is finite and exactly an integer.** One sentence, and it
is a contract rather than a list of what happened to be registered.

**A `numpy` array of labels is accepted by all three functions although it is not a
`Sequence`.** Measured: `isinstance(np.array([0, 1]), Sequence)` is `False`, so the declared
`Sequence[int]` does not cover it — and `class_weights(np.array([0, 1, 0, 1]))`,
`build_sampler(np.array([0, 1, 0, 1]), True, 1)` and `PatchDataset(paths, np.array([0, 1]))`
all work, because each of the three iterates what it was given and `_class_index` judges the
values that come out. **This is the same asymmetry `blobcount.splits` states for per-row ids
at `splits.py:78-85`**, where the negative form of a container predicate "would refuse
`ndarray`, `Series`, `set` and `dict` … at the price of refusing the two most natural per-row
id columns a caller holding a DataFrame has" — and it is stated here for the same reason: a
DataFrame column is the most natural source of a fold's labels there is, and this module must
not refuse it.

**`np.float64` is still accepted here and `np.float32` is not, and that is a fact of numpy's
registration rather than a rule — so the rule above is written to absorb it.** `np.float64`
subclasses Python's `float` and `np.float32` does not, measured, which is the same
registration accident as the one the `numbers.Integral` half fixes, pointing the other way.
A width of float is not a different kind of class index, and **`np.float32` is refused by the
type test rather than by the fraction rule**: `int(np.float32(0.5))` is `0` and
`0 != np.float32(0.5)` is `True`, so had the type test let it through the fraction rule
would have caught it anyway, and a caller is told the type it holds rather than a fraction it
does not. `np.float32(0.0)` is therefore the one value a caller can reach that this refuses
for no good reason, and the remedy is a documented `.astype(int)` at that caller's read
rather than a wider rule here — a rule wide enough to admit it would admit a `Fraction` and a
`Decimal` beside it. **`np.bool_` is not an `Integral`** and is refused by the type test, not
by the `bool` guard below.

**A `bool` is refused by a type test and not by the comparison beside it, and the two flip
keys are passed to `p=` as the booleans they are for the same reason.** `bool` subclasses
`int`, so `int(True) == 1` and `1 != True` is `False`: every test a label can be put
through — whole, in range, hashable — accepts it, and `class_weights([0, 1, True])`
returned the weights of classes 1 and 0 and said nothing about it. A `bool` is a
legitimate value for a probability and never a legitimate value for a class, and the two
have to be told apart by type because Python will not. The flip keys stay uncoerced
because the parameter is a probability and `configs/default.yaml` states a `bool` for it,
and because `blobcount.config.load` accepts an override only when its type is exactly the
configured one, so `augmentation.horizontal_flip: 1` is a `ConfigError` at load time and
never reaches this function. `bool` subclasses `numbers.Integral` as well as `int`, so the
`bool` guard is the whole of what keeps it out of the widened type test.

**A non-finite label is refused by the predicate and not by `int()` refusing it on the way
through.** `int(float("inf"))` raises `OverflowError` and `int(float("nan"))` raises
`ValueError` carrying CPython's `cannot convert float NaN to integer`; measured on this
stack, and neither names a rule. The first is not the `ValueError` this module's contract
promises, and the second is a refusal with no remedy in it. The finiteness test is scoped
to `float` in both directions, as in `ablation`: `math.isfinite` converts its argument, so
it raises an `OverflowError` of its own for an `int` too large to be a float — which would
trade the bug this predicate exists for a second one a layer below it — and there is no
such thing as an infinite `int`.

**A `str` is refused, and the refusal does not claim a range it is inside of.** `int("0")`
is `0` and `0 != "0"`, so the range test that stood beside the conversion reported `label
'0' is outside range(2)` — false, and false in the way that costs a reader the most,
because it points at the class count rather than at the cell. This matters here more than
in `ablation`: a label arrives from a `labels.csv` row read by `csv.DictReader`, where
every cell is a string, and a caller who forgot the `int()` at the read hands this function
`'0'`. The refusal names the type by its **qualified** name, `numpy.int64` and not
`int64`, because a traceback out of a worker process is read on its own.

**The range is `class_weights`' condition to hold, and the other two do not hold it.**
`PatchDataset` and `build_sampler` are given no `num_classes` and cannot infer one: a
class the fold holds none of is still a legal label, so the widest label a caller may
legitimately have is a class index it has not named here. They run the same conditions
except the upper bound, and the lower bound with it, which no class index is under, and a
`2` under a two-class labelling run is refused by `class_weights` — where the `num_classes`
that makes `2` wrong is an argument the caller passed. This is stated rather than left to be
found at fold time, and it is not a second opinion that disagrees with the first.

**The transform values were hard-coded in the original and are configured here.**
`train.py:97` built the train transform from a literal `Resize((224, 224))`, two bare
flips, a `RandomRotation(15)`, a `ColorJitter` of four numbers and a `RandomAffine` of
one, and `train.py:108` built the eval transform from a second literal resize. The eight
`augmentation.*` keys and `training.input_size` hold those numbers, so what changed is
that a run can now be changed without editing a file of code. The two flip keys are the
train transform and not the eval one, and the eval transform carries the ImageNet channel
statistics the backbone was fitted with.

**`transform=None` is not an identity that hands the image back.** The declared return is
`(Tensor, int)`, and a `DataLoader` cannot collate a `PIL.Image`: measured on this stack,
`default_collate` raises `TypeError: batch must contain tensors, numpy arrays, numbers,
dicts or lists; found <class 'PIL.Image.Image'>`. So a dataset with no transform renders
`ToTensor` and nothing else, which is the conversion the interface promises and no
resizing and no normalization, because choosing those is what `build_transforms` is for.

**A loader's order comes from the generator it is built with, and the shuffle sampler is
built here rather than left to the `DataLoader` to create.** `build_loader` seeds its own
`torch.Generator` from `seed`, hands it to a `RandomSampler` it constructs, and passes
`shuffle=False` — so two loaders built with the same seed agree whatever has happened to
the global RNG in between, which is the property a fold-by-fold run needs: a run that seeds
once and then builds a loader per fold would otherwise get a different order for every fold
from the global stream it never reseeded. **The sampler is constructed here because the
order must not depend on a library behaviour this project has not declared.** `DataLoader`
accepts a `generator`, and whether it also passes that generator to the `RandomSampler` it
creates for `shuffle=True` is torch's business and not this interface's: in the torch 2.10
installed here it does — measured, and this module's determinism rested on it for a round —
and the pass-through is nowhere in `pyproject.toml`, whose `torch>=2.0` admits resolutions
that do not have it. A red suite on a legal resolution is the wrong failure mode, and
`shuffle=True` *is* `RandomSampler(dataset, generator=generator)`, so spelling it out costs
the loader nothing and needs no floor bump. One generator serves both that permutation and
the base seed each worker gets, so a second epoch of the same loader draws a different
order from the same stream: reproducible across runs, and not a repeat of the first epoch.

**Nothing here is caught, logged, or written.** `PatchDataset.__getitem__` opens one
patch, and a patch file that is not there raises from that read — inside a worker process
where there is no index row to name a specimen or a coordinate for, and where a log line
this project could still read afterwards does not exist. The `OSError` that survives
names the path it could not open, which is the fact a caller needs, and wrapping it would
replace that with a sentence about a rule and lose the path. There is no artifact to
write: the fold summary, the per-fold weight vector and the run log are Task 11's, and
they are what `training.epochs`, `training.learning_rate` and the two `training.patience_*`
keys configure.

Declared interface, in `__all__` order:

    PatchDataset
        PatchDataset.__init__(
            self,
            paths: Sequence[Path],
            labels: Sequence[int],
            transform: Callable | None = None,
        ) -> None
        PatchDataset.__len__(self) -> int
        PatchDataset.__getitem__(self, index: int) -> tuple[torch.Tensor, int]
    build_loader(
        dataset: Dataset,
        batch_size: int,
        sampler: WeightedRandomSampler | None,
        shuffle: bool,
        num_workers: int,
        seed: int,
    ) -> DataLoader
    build_sampler(labels: Sequence[int], enabled: bool, seed: int) -> WeightedRandomSampler | None
    build_transforms(cfg: Config) -> tuple[Callable, Callable]
    class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor
"""

from __future__ import annotations

import math
import numbers
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler
from torchvision import transforms

from blobcount.config import Config

__all__ = [
    "PatchDataset",
    "build_loader",
    "build_sampler",
    "build_transforms",
    "class_weights",
]

# The channel statistics an ImageNet-pretrained backbone was fitted with, and the reason
# the eval transform normalizes at all: `training.architecture` names the backbone and
# `training.pretrained` is what says it arrives with those weights, so a patch rendered
# without these statistics reaches the first convolution in a range that backbone was
# never fitted on. They are module constants rather than arguments because no
# configuration key states them and there is one correct answer, and a second pair in this
# file would be a second place for it to be wrong.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# What a patch becomes when the caller gave no transform. A `ToTensor` and nothing else:
# the declared return is a tensor and the default collate cannot batch anything else, and
# adding a `Resize` or a `Normalize` here would make a transform-less dataset a second,
# differently configured answer to the question `build_transforms` answers. Module scope
# so every `PatchDataset` shares one object, and because it is stateless and pickles for
# `num_workers` above zero.
_TO_TENSOR = transforms.ToTensor()


def _class_index(label: int, num_classes: int | None = None) -> int:
    """Return `label` as a plain `int` class index, refusing anything that is not one.

    **This is the one definition of a label in this module, and all three functions that
    take a fold's labels run it over every label they are given.** The conditions are the
    module docstring's, as is the argument for one predicate over three call sites rather
    than one test per function; what is here is the code that makes both hold.

    `num_classes` is `None` for `PatchDataset` and `build_sampler`, which are given no
    class count and cannot infer one, so the upper bound is not held here for them — the
    module docstring gives why, and `class_weights` is where a `2` under a two-class
    labelling run is refused, because `num_classes` is its argument.

    **A `ValueError` in every case, and one message that names the condition the label
    actually broke rather than one message per condition.** The broken condition comes
    first and the conditions after it, so a quoted digit is told it is a quoted digit and
    not that it is outside a range it is inside of. The type is named by its **qualified**
    name — `numpy.int64` and not `int64` — because a traceback read out of a worker process
    is read on its own and `int64` alone says nothing about where the value came from.

    **A class index is an integer, and `numbers.Integral` is what that means here.** The
    two rules are in the module docstring: the *integral* rule accepts `int` and every
    `numbers.Integral` a caller reaches per row, which is what a numpy integer is; the
    *float* rule accepts a `float` that is finite and exactly a class index, because a
    weight vector computed from a computed label is a weight vector whose provenance nobody
    can check. Both are the same `ablation` shape — one predicate, tests before the
    conversion — with the type test widened from `(int, float)` to `numbers.Integral` and
    `float`, which is the only line that differs from `blobcount.ablation._thresholds:951`
    and the only one the numpy rule needed.
    """
    # One predicate, and every test in it runs to completion *before* the `int()` below, so
    # the conversion is reached only for a value that already has an answer. Measured on
    # this stack, in the order the old code reached them: `int(float("inf"))` raises
    # `OverflowError`, `int(float("nan"))` raises `ValueError` carrying CPython's
    # `cannot convert float NaN to integer`, and `int("0")` is `0`, which is why the range
    # test that stood beside the conversion called a string out of range rather than out of
    # type.
    #
    # **The type test is two rules, not one, and the second is why `numbers` is imported
    # here at all.** `isinstance(np.int64(0), int)` is `False` on this stack and
    # `isinstance(np.int32(0), int)` is `False` too, so the `(int, float)` this predicate
    # used for a round refused a numpy integer a caller reads straight out of a DataFrame
    # column — measured: `class_weights(np.array([0, 1, 0, 1]))` and
    # `build_sampler(np.array([0, 1, 0, 1]), True, 1)` both raised, and both worked before
    # it. `numbers.Integral` is the registration CPython and numpy both agree on, and
    # `bool` subclasses it, so the `bool` guard is what keeps `True` out — which is the
    # condition it was written for and is not weakened by the widening.
    #
    # `float` rather than `numbers.Real` on the other branch, and the reason is the same
    # one: `Real` is `True` of a `Fraction` and of `np.float32` alike, and a fraction is
    # not a class index under any reading. `np.float64` is a `float` subclass and is
    # therefore still accepted here; `np.float32` is not, and that is a float of a different
    # width rather than a different kind of number. The two numpy widths splitting is a
    # registration fact of the library and not a rule, which is why the rule is stated as
    # "`numbers.Integral` or `float`" above instead of as a list of types it happens to
    # name. **`np.bool_` is not an `Integral`** and is refused by the type test rather than
    # by the `bool` guard, which is the one thing about it worth knowing.
    is_number = (
        not isinstance(label, bool) and isinstance(label, (numbers.Integral, float))
    )
    # The finiteness test, scoped to `float` on purpose in both directions: `math.isfinite`
    # converts its argument and raises an `OverflowError` of its own for an `int` too large
    # to be a float, and there is no such thing as an infinite `int`. A `bool` is not asked
    # about, since the type test already refused it and its answer would be `True` anyway.
    is_finite = not isinstance(label, float) or math.isfinite(label)
    # `whole` is `None` for anything the two tests above refused, and every test after this
    # line is guarded on it: `int()` of a `str` is legal, so an unguarded `whole != label`
    # would be one predicate reporting three different mistakes for one value.
    whole = int(label) if is_number and is_finite else None
    problems: list[str] = []
    if not is_number:
        # The qualified name, not `__name__` alone: measured, `type(label).__name__` of a
        # numpy label is `int64`, which tells a reader of a worker's traceback nothing
        # about where the value came from, and `str` is a worse collision with a reader who
        # is holding an `np.str_`. The article agrees with the type and not with the class
        # name, which is why the name is last in the sentence.
        problems.append(
            "not a number a class index can be, but a "
            f"{type(label).__module__}.{type(label).__name__}"
        )
    elif not is_finite:
        problems.append("not finite")
    elif whole != label:
        problems.append("not a whole number")
    # The upper bound is held only where a class count was given, and the lower bound
    # wherever it was not already reported: a `-1` under `class_weights` is outside the
    # range and that is the whole of what is true about it, while the same `-1` at the two
    # call sites with no count is refused for what it is — below zero, which no class index
    # is under any labelling run. One bound per message, so the list never names a
    # condition twice for one value.
    outside = whole is not None and num_classes is not None and not 0 <= whole < num_classes
    if outside:
        problems.append(
            f"outside range({num_classes}), the class indices 0 to {num_classes - 1}"
        )
    elif whole is not None and whole < 0:
        problems.append("below zero, which no class index is")
    if problems:
        # The enumeration is the same conditions in the shape the module docstring states
        # them, with the upper bound's own example added only where there is a class count
        # to be outside of. It is the second half of the message because the first half is
        # what this label did and this is what any label may not do. The numpy examples are
        # in it because the rule that produces this message accepts them, and a caller
        # holding a DataFrame column has them in hand when they read it.
        shapes = (
            "0 or 0.0 or a numpy integer, and not '0' and not true and not a float with a "
            "fraction and not .inf or .nan and not -1"
        )
        if num_classes is not None:
            shapes += f" and not {num_classes}"
        remedy = (
            ", or pass the num_classes that labelling run used"
            if num_classes is not None
            else ""
        )
        raise ValueError(
            f"label {label!r} is not a class index: {'; '.join(problems)}. A label is the "
            f"class a labelling run wrote for a patch, and one that is not reaches "
            f"CrossEntropyLoss as a target it cannot score, which reports the mistake "
            f"against the loss rather than against the column it came from. A class index "
            f"is a whole number a type can carry: an int, a float that is exactly an "
            f"integer, or a numpy integer — {shapes}. Fix the label{remedy}"
        )
    return whole


class PatchDataset(Dataset):
    """One patch per row, with the class labelling gave it, rendered by one transform.

    The caller must supply **one path and one label per dataset row, in the same order**
    — `labels[i]` is the class of `paths[i]`, and the row order is the order
    `blobcount.splits.loso_folds` returns indices into. **No length check is applied
    against the other sequence**, for the reason that module's signature gives: with one
    sequence of paths there is no expected row count to check against, and a guess would
    refuse a legitimate cohort of one-patch specimens. A short `labels` therefore surfaces
    as an `IndexError` from `__getitem__` rather than as a refusal here.

    `transform` is applied to the opened patch when one is given and `ToTensor` alone when
    it is not, so the return is always `(Tensor, int)`. The label is a plain `int` and not
    a numpy scalar or a 0-d tensor: the default collate turns a batch of them into one
    tensor, and a caller holding one label and comparing it against a plain `int` should
    not have to know that anything else was possible.

    **Every label is checked here, in `__init__`, and kept as the plain `int` the check
    returned.** No `num_classes` is passed, so the upper bound is not one of the conditions
    this call makes and the module docstring says why; a `1.5` is refused here rather than
    reaching `__getitem__`, where the `int()` that stood there would have truncated it to
    `1` and said nothing — the fold would then have trained on a class nobody labelled, and
    the only trace would be a `labels.csv` cell a reader had to notice by hand. `labels` is
    declared a `Sequence[int]` and a `numpy` array of labels is accepted anyway, iterated
    row by row; the module docstring gives why, and it is the same rule the other two call
    sites run over the same rows.

    Nothing is resized on the way in and no channel is dropped: `extract.read_patch` wrote
    every patch as the `uint8` `(extraction.patch_size, extraction.patch_size, 3)` array
    the index records, so the pixels here are the pixels that were extracted, and the
    `convert("RGB")` is what an `L`, `RGBA` or palette PNG would have to become for a
    three-channel model rather than a guess about what a channel meant.
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        self.paths = [Path(path) for path in paths]
        self.labels = [_class_index(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # `PIL` is imported inside the read for the reason
        # `blobcount.labeling.load_patch` gives: it is a native wheel, and a module-scope
        # import makes an absent one a collection error for every test in the suite.
        from PIL import Image

        with Image.open(self.paths[index]) as source:
            patch = source.convert("RGB")
        if self.transform is None:
            return _TO_TENSOR(patch), self.labels[index]
        return self.transform(patch), self.labels[index]


def class_weights(labels: Sequence[int], num_classes: int = 2) -> torch.Tensor:
    """Return `total / (num_classes * count[c])` over the labels given, as `float32`.

    **Every count here is a count of the labels this call was handed**, which is the whole
    of the per-fold correction. `train.py:89` built this vector once from the pooled
    cohort and used it in every fold, so each fold's weights were a function of the
    specimens it was validating on. The cross-validation loop hands this function a fold's
    training labels, and the weights that follow cannot mention the held-out specimen,
    because no label of it was passed in. `tests/test_data.py` pins the arithmetic over
    `[0, 0, 0, 1]` and pins that two different subsets of the same cohort give different
    weights.

    A balanced subset gets `1.0` for every class — `total / (num_classes * total/num_classes)`
    is 1.0 — so weighting a loss with it changes nothing, which is the right answer for a
    fold that needs no correction and not a degenerate one. A class the labels hold none of
    gets `0.0` rather than the `inf` the quotient would divide to; the module docstring
    carries why, and the state is reachable in this cohort rather than hypothetical.

    **Every label is checked by `_class_index`, and no weight is computed until every one
    of them has passed**, so a fold's vector is not returned when a fraction of it was a
    label the weight was not really weighting. The conditions are the module
    docstring's; what is added here is the one only this function can hold.

    **The class range is checked here and at neither of the other two call sites, because
    this is the function the `num_classes` was passed to.** A label of `2` is outside
    `range(2)` and inside `range(3)`, so no check that does not have the width can call it
    wrong; `PatchDataset` and `build_sampler` are given no width and refuse a label only
    for what it is. The two are the same predicate asked two questions, and the
    cross-validation loop builds all three from one fold's labels, so a label this refuses
    is a label the fold cannot be trained on rather than one two functions happened to
    allow.

    `num_classes` is the width of the returned vector and the range a label has to fall
    in, so a caller whose labels run used a third class must say so here rather than
    receive a vector it can index out of. `labels` is declared a `Sequence[int]` and a
    `numpy` array of labels is accepted anyway; the module docstring gives why, and the
    rule for one label is the same one the other two call sites run.
    """
    counts = [0] * num_classes
    total = len(labels)
    for label in labels:
        counts[_class_index(label, num_classes)] += 1
    return torch.tensor(
        [total / (num_classes * count) if count else 0.0 for count in counts],
        dtype=torch.float32,
    )


def build_sampler(
    labels: Sequence[int], enabled: bool, seed: int
) -> WeightedRandomSampler | None:
    """Return a balanced `WeightedRandomSampler` over `labels`, or `None` when disabled.

    Each label's weight is the inverse frequency of its class, and a class's weights then
    sum to exactly 1 whatever count it holds, so **every class present in the labels is
    drawn with probability `1 / (the number of classes present)`** — a class holding a
    tenth of a two-class fold's patches is drawn about half the time, and the sampler
    flattens the class distribution rather than reproducing it. That is what `train.py:115`
    did, and it is what `training.use_weighted_sampler` exists to switch off: a caller that
    wants the fold's own proportions reads that key, passes `enabled=False`, and gets
    `None` so that `build_loader` can shuffle the rows itself rather than draw them a
    second time through a sampler that only permutes. **That read is the caller's and it is
    not made here**, because `enabled` is the parameter this function was given and a
    second read of the key inside it would be a value it could honour and did not.

    `num_samples=len(labels)` and `replacement=True` together are what make one epoch the
    same length as the dataset: with replacement, `torch.multinomial` needs a count, and a
    count of the fold's own size is the one that keeps an epoch's worth of steps constant
    across folds whose class counts differ. The `generator` is this function's own, seeded
    from `seed`, because `WeightedRandomSampler` draws from the global RNG when it is given
    none — and a fold that inherited the global stream would draw a different order from a
    fold built before it, which is the same non-reproducibility the loader's own generator
    is there to end.

    The caller must pass the same labels it gave `PatchDataset` and `class_weights` for
    this fold, one per training row: the weights are indexed by position, so a `labels`
    that is not the fold's training labels samples a fold the loss was not fitted for.

    **Every label is checked here too, by the same `_class_index` `class_weights` uses,
    and the labels the weights are counted from are the ones the check returned.** This
    function used to hold no check at all, so a fold's labels were validated in one of
    three places depending on which function the caller happened to reach first; the two
    are now the same predicate over the same list, and the class count is the one condition
    only `class_weights` holds, because only it is given one. **The check runs before the
    `enabled` test and not after it, so a disabled sampler refuses a label an enabled one
    refuses** — the argument is the fold's labels either way, and a caller that gets them
    wrong should hear it from the call it made rather than from the next one. A `numpy` array
    of labels is accepted although it is not a `Sequence`; the module docstring gives why.
    """
    indices = [_class_index(label) for label in labels]
    if not enabled:
        return None
    counts = Counter(indices)
    return WeightedRandomSampler(
        weights=[1.0 / counts[index] for index in indices],
        num_samples=len(labels),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def build_transforms(cfg: Config) -> tuple[Callable, Callable]:
    """Return the `(train_transform, eval_transform)` pair the configuration describes.

    The eval transform is `Resize` to `training.input_size`, `ToTensor`, and the ImageNet
    normalization, and **it holds no random operation at all**: a validation score is a
    measurement of one fixed rendering of each patch, and a transform that flipped or
    jittered a subset of the validation set would make two runs of the same fold disagree
    for a reason that has nothing to do with the model.

    The train transform is the same three with the eight `augmentation.*` keys between the
    resize and the conversion, in that order: the flips, the rotation, the colour jitter
    and the translation act on the `PIL` image, because every one of them is a torchvision
    transform that takes pixels and a later `ToTensor` cannot be undone to try them again.
    `augmentation.affine_translate` is one key and two arguments, and both axes get the
    same value: a histopathology patch has no privileged axis, and translating one axis
    twice as often as the other would be a property of the key this configuration does not
    have. `degrees=0` on that transform is deliberate and not an omission — the rotation
    key is already a rotation, and a second one would apply two angles where the
    configuration states one.

    **The two flip keys are `bool` and reach `p=` uncoerced**, and the module docstring
    gives why. Every other value is converted once, at the read, into a local — so all nine
    keys are read in one block above the two `Compose` calls, and a key added to
    `configs/default.yaml` has one place to be read from rather than one call site per
    transform that uses it.
    """
    size = int(cfg.get("training.input_size"))
    horizontal_flip = cfg.get("augmentation.horizontal_flip")
    vertical_flip = cfg.get("augmentation.vertical_flip")
    degrees = float(cfg.get("augmentation.rotation_degrees"))
    brightness = float(cfg.get("augmentation.color_jitter_brightness"))
    contrast = float(cfg.get("augmentation.color_jitter_contrast"))
    saturation = float(cfg.get("augmentation.color_jitter_saturation"))
    hue = float(cfg.get("augmentation.color_jitter_hue"))
    translate = float(cfg.get("augmentation.affine_translate"))
    eval_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=horizontal_flip),
            transforms.RandomVerticalFlip(p=vertical_flip),
            transforms.RandomRotation(degrees),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.RandomAffine(degrees=0, translate=(translate, translate)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_loader(
    dataset: Dataset,
    batch_size: int,
    sampler: WeightedRandomSampler | None,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Return a `DataLoader` whose order is decided by `seed` and by nothing else.

    **`shuffle` is honoured only when no sampler was supplied, and a sampler is a legal
    substitute for it rather than an addition to it.** torch refuses the two together —
    measured: `ValueError: sampler option is mutually exclusive with shuffle` — because a
    sampler already decides the order and a shuffle on top of it would be a second,
    conflicting claim about the same rows. So the flag is dropped rather than honoured, and
    the loader draws the sampler's order. A `None` sampler with `shuffle=False` gets the
    rows in dataset order, which is what a validation loader wants and what
    `blobcount.splits.val_idx` is built to be read in.

    **`shuffle=True` is a `RandomSampler` this function constructs, not a flag it passes.**
    The module docstring gives the whole of it: a `DataLoader` given `generator=` and
    `shuffle=True` hands that generator to the `RandomSampler` it creates in the torch 2.10
    installed here, and nothing in `pyproject.toml` says a resolution below that would, so
    on such a resolution the order would come from the global stream while the `seed`
    argument still looked honoured. What `DataLoader` builds for a `shuffle` it is not given
    is a `SequentialSampler`, so the two cases below are the flag written out rather than
    interpreted. **One generator serves the permutation and the base seed each worker is
    given**, so a second epoch of one loader draws a different order from the same stream:
    reproducible across runs, and not a repeat of the first epoch.

    `batch_size` and `num_workers` arrive as arguments and are not read from a
    configuration here: `training.batch_size` and `training.num_workers` are read by the
    loop that calls this function, and a second read of either here would be a value this
    module could honour and did not.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = (
        RandomSampler(dataset, generator=generator)
        if sampler is None and shuffle
        else sampler
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        generator=generator,
    )
