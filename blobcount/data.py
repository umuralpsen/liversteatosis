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

**The two flip keys are passed to `p=` as the booleans they are, and are not run through
`int()`.** `bool` subclasses `int`, so `int(True) == 1` and a coercion here would change
no behaviour and hide nothing — it is declined because the parameter is a probability and
`configs/default.yaml` states a `bool` for it, and because `blobcount.config.load` accepts
an override only when its type is exactly the configured one, so
`augmentation.horizontal_flip: 1` is a `ConfigError` at load time and never reaches this
function. `blobcount.ablation._thresholds` refuses a `bool` entry in a threshold list for
the same reason and against the same hazard.

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

**A loader's order comes from the generator it is handed, and not from the global one.**
`build_loader` seeds its own `torch.Generator` from `seed` and passes it to the
`DataLoader`, which draws each worker process's base seed from it and — in the torch 2.10
this project resolves to — hands it to the `RandomSampler` it creates for `shuffle=True`.
So two loaders built with the same seed agree whatever has happened to the global RNG in
between, which is the property a fold-by-fold run needs: a run that seeds once and then
builds a loader per fold would otherwise get a different order for every fold from the
global stream it never reseeded. **That pass-through is a property of the library and is
stated here because this module's determinism rests on it**, and a torch that did not make
it would leave the shuffle order drawn from the global stream while the seed argument
still looked honoured.

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

from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
        self.labels = list(labels)
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
            return _TO_TENSOR(patch), int(self.labels[index])
        return self.transform(patch), int(self.labels[index])


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

    **A label that is not one of the `num_classes` classes is refused here**, as a
    `ValueError` naming the remedy, because the alternative is an out-of-range target
    reaching `CrossEntropyLoss` and surfacing as an index error attributed to the loss. A
    label is also required to be a whole number, and a `1.5` is refused by the same
    message rather than counted as class 1: the labels are written by the labelling run
    from `labeling.blob_threshold` and there is no shape of a labelling run that produces
    one, so a fractional label is a caller mistake and not a rounding to perform.

    `num_classes` is the width of the returned vector and the range a label has to fall
    in, so a caller whose labels run used a third class must say so here rather than
    receive a vector it can index out of.
    """
    counts = [0] * num_classes
    total = len(labels)
    for label in labels:
        whole = int(label)
        if whole != label or not 0 <= whole < num_classes:
            raise ValueError(
                f"label {label!r} is outside range({num_classes}), the class indices 0 to "
                f"{num_classes - 1} that this weight vector has one entry for: a label is "
                f"the class a labelling run wrote for a patch, and one outside the range "
                f"reaches CrossEntropyLoss as a target it cannot score, which reports the "
                f"mistake against the loss rather than against the column it came from. "
                f"Fix the label, or pass the num_classes that labelling run used"
            )
        counts[whole] += 1
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
    """
    if not enabled:
        return None
    counts = Counter(labels)
    return WeightedRandomSampler(
        weights=[1.0 / counts[label] for label in labels],
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

    **The generator is this function's own, seeded from `seed` and never the global one.**
    It is handed to the `DataLoader`, which draws each worker process's base seed from it
    and, in the torch 2.10 this project resolves to, passes it to the `RandomSampler` it
    creates for `shuffle=True`. Both are what make a fold reproducible: the run seeds once
    and builds a loader per fold, and a loader that read the global stream would give every
    fold a different order from a stream nothing reseeded, so a rerun of the same fold
    would not be the same fold. `tests/test_data.py` separates the two loaders by a
    deliberate draw from the global generator, so a loader that fell back to it would fail
    rather than agree by luck.

    `batch_size` and `num_workers` arrive as arguments and are not read from a
    configuration here: `training.batch_size` and `training.num_workers` are read by the
    loop that calls this function, and a second read of either here would be a value this
    module could honour and did not.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,
        generator=generator,
    )
