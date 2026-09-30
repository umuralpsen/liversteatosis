"""Model construction: three backbones, one head width, one dispatch table.

**The head is what this module builds and nothing else is.** `build_model` names a backbone
from `SUPPORTED`, asks torchvision for it with or without its ImageNet weights, and replaces
the one layer that turns a feature vector into a class count. Nothing here is fitted, no
device is chosen and no byte is written: training the result, moving it to the device
`project.device` resolves to and saving it are `blobcount.train`'s, which arrives with the
cross-validation loop, and a constructor that also did those three things would leave a
caller no way to ask for one without the others.

**The three configuration keys behind this factory are `training.architecture`,
`training.num_classes` and `training.pretrained`, and the caller is what reads them.** The
signature takes plain values and not a `Config`, so a test builds a model with no
configuration file at all — which is why every test in `tests/test_models.py` constructs
`pretrained=False` — and `blobcount.train` is the module that holds the configuration and
hands these three over, once per fold.

**A class count is refused here rather than by `nn.Linear`, and that is the only validation
in this module.** Measured on torch 2.10.0: `nn.Linear(512, 0)` builds and warns
`Initializing zero-element tensors is a no-op`, which the `filterwarnings = ["error"]` in
`pyproject.toml` turns into a failed run; `nn.Linear(512, -1)` raises
`RuntimeError: Trying to create tensor with negative dimension -1: [-1, 4]`; and
`nn.Linear(512, 2.0)`, `nn.Linear(512, True)` and `nn.Linear(512, "2")` raise the same
`TypeError: empty() received an invalid combination of arguments`, which names no rule and
no argument. Every one of those four arrives from a hand-edited `configs/default.yaml` —
`blobcount.config.load` checks that a key is declared and nothing about its value — and a
refusal here names the condition the value broke instead of sending the reader into torch's
initialiser, which is the "looks validated and is not" state `blobcount.data._class_index`
is written to end for the same reason.

**The refusal is a `ValueError`, and no command line in the tree catches it yet.** The
declared interface below declares no exception type of this project's own, and a
`ValueError` is one a command line already knows how to report — but `bin/train.py` arrives
with Task 11 and inherits that job, so until it does a mistyped `training.num_classes`
reaches the interpreter as a traceback out of this module. The message naming the remedy is
the part of this that survives the traceback, and the gap is recorded here rather than
discovered at fold time.

**A class count is a positive whole number a type can carry, and a `bool` is not one.**
The type rule is `numbers.Integral`, stated as that and not as a list of types it happens
to name: measured on the numpy 2.2.2 here, `isinstance(np.int64(2), numbers.Integral)` is
`True` and so is `isinstance(np.int32(2), numbers.Integral)`, so a caller reading a count
out of a DataFrame column is not refused — the same widening `_class_index` needed for class
indices and for the same reason. `bool` subclasses `numbers.Integral`, so the guard beside
the type test is the whole of what keeps `True` out: `True` is a width of one to
`nn.Linear` and is an answer to `training.pretrained` to a reader of the file, and those
two must not be the same value. **`np.bool_` is not an `Integral`** and is refused by the
type test rather than by that guard, which is the one thing about it worth knowing. **A
`float` is refused even where it is exactly an integer**, and that is the one rule here
narrower than `_class_index`'s: a class count is a width written in a configuration file and
not a label computed in an array, and `nn.Linear` refuses `2.0` in any case.

**The count a caller passed is kept as the plain `int` the check returned**, for the reason
`blobcount.data.PatchDataset` keeps its labels as one: `nn.Linear` stores `out_features` as
it was handed, and a numpy integer in a head a caller reads back is a type it did not pass.

**An architecture name is matched exactly and case-sensitively, and the refusal lists
`SUPPORTED`.** `"ResNet18"` is unsupported and not quietly accepted, because a
case-insensitive match would train a backbone the reader did not ask for and name the wrong
one in `results/fold_summary.csv`. This is the cheap failure and it is worth having: the
refusal names the three names this project builds and the key to set, so a caller does not
go looking through torchvision for the one it meant. **A name that is not a `str` is refused
by that same refusal** and not by the hash of a `dict` lookup, because `load` accepts any
value for a declared key and a list under `training.architecture` would otherwise raise
`TypeError: unhashable type: 'list'`, which names neither the rule nor the remedy.

**`weights` is the only keyword handed to torchvision, and its value is the string alias
`"DEFAULT"` or `None`.** torchvision deprecated the `pretrained=` spelling and warns for it,
and `pyproject.toml` promotes a warning to a failure, so `weights=` is not a style choice
here. **`pretrained` itself is not checked**, and that is the same argument
`blobcount.data` makes for its two flip keys: `configs/default.yaml` declares a `bool` for
`training.pretrained`, `load` accepts an override only when its type is exactly the
configured one, so `training.pretrained=1` is a `ConfigError` at load time and never
arrives at this function.

**Where a head lives is stated by the row that builds its backbone.** The two ResNets keep
theirs at `.fc` and EfficientNet keeps it at `.classifier[1]` — measured on this stack, a
`Sequential(Dropout, Linear)`, so index 0 is the dropout in front of it and stays — and the
dispatch says so per architecture rather than branching on the name a second time. The
`in_features` of the layer being replaced is read off the model torchvision built rather than
written here, so a head cannot be a different width from the feature vector in front of it.

**A constructor is named rather than imported**, so this module reads the library's own
namespace at call time. That is what lets `tests/test_models.py` replace one torchvision
constructor and assert the argument it was handed — with `weights="DEFAULT"` reaching no
network at all — and it keeps a second name for the same constructor out of this file.

Declared interface, in `__all__` order:

    SUPPORTED
        ("resnet18", "resnet50", "efficientnet_b0")
    build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module
"""

from __future__ import annotations

import numbers
from collections.abc import Callable

import torchvision.models
from torch import nn

__all__ = ["SUPPORTED", "build_model"]


def _replace_resnet_head(model: nn.Module, width: int) -> None:
    """Replace `.fc`, the final linear of a ResNet, with one of `width` outputs.

    The 512-wide or 2048-wide layer is what a ResNet's global average pool hands this one,
    so the head keeps its width and takes only its output count from the caller's class
    count.
    """
    model.fc = nn.Linear(model.fc.in_features, width)


def _replace_efficientnet_head(model: nn.Module, width: int) -> None:
    """Replace `.classifier[1]`, the final linear of an EfficientNet, with one of `width`.

    Measured on this stack the classifier is a `Sequential(Dropout, Linear)` over a
    1280-wide feature vector, so the head is index 1 and index 0 is the dropout in front of
    it. The dropout is left where it is: a head swapped in at index 0 would put a `Linear`
    where the dropout belongs and the model would refuse the width it was just handed.
    """
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, width)


# What each supported architecture is built out of: the name torchvision publishes its
# constructor under, and the one function that replaces the head of the model it returns.
# The constructor is named rather than imported so the dispatch reads `torchvision.models`
# at call time, and the head is a function so where it lives is stated in the row that
# builds the backbone — a fourth architecture is one row and no second branch on the name.
#
# `in_features` is read off the layer each of these replaces rather than written beside the
# row, so a head cannot be a different width from the feature vector in front of it, and
# the widths in `tests/test_models.py` are measured off the built model rather than
# remembered.
_BACKBONES: dict[str, tuple[str, Callable[[nn.Module, int], None]]] = {
    "resnet18": ("resnet18", _replace_resnet_head),
    "resnet50": ("resnet50", _replace_resnet_head),
    "efficientnet_b0": ("efficientnet_b0", _replace_efficientnet_head),
}

# `SUPPORTED` is derived from that table rather than written beside it, so the constant a
# caller checks `training.architecture` against and the dispatch that builds the model
# cannot disagree about which names exist or in what order they are offered.
SUPPORTED = tuple(_BACKBONES)


def _class_count(value: int) -> int:
    """Return `value` as a plain `int` head width, refusing anything that is not one.

    **The conditions are the module docstring's and they run before anything is built**, so
    `nn.Linear` is reached only for a count that already has an answer: `0` and a negative
    count are whole numbers no head can be built for, a `bool` is a `bool` rather than a
    count however much it passes as an `int`, and a `float` is not a width — measured, torch
    refuses `nn.Linear(512, 2.0)` and `nn.Linear(512, "2")` with one `TypeError` naming no
    rule and no argument.

    **A `ValueError` in every case, naming the type by its qualified name.** `data.py`'s
    `_class_index` names a label's type the same way and for the same reason: a traceback
    read on its own says nothing about where the value came from, and `int` alone collides
    with a reader holding an `np.int64`. The rule is stated as a shape rather than as a list
    of types, because a numpy integer is a count and `np.bool_` is not one.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(
            f"num_classes {value!r} ({type(value).__module__}.{type(value).__name__}) is "
            f"not a class count this head can be built for: the last layer of the model is "
            f"nn.Linear(in_features, num_classes), so the count has to be a whole number of "
            f"at least one class — an int or a numpy integer, and not true or false, not 0, "
            f"not -1, not 2.0 and not '2'. Fix: pass the training.num_classes of "
            f"configs/default.yaml, which is the number of classes a labelling run wrote — 2 "
            f"for this project's normal/steatotic labelling"
        )
    return int(value)


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Return `name` with its final linear layer replaced by a head of `num_classes` outputs.

    The caller must pass **a name `SUPPORTED` lists, spelled exactly as `SUPPORTED` spells
    it, and a class count of at least one** — `training.architecture` and
    `training.num_classes` as `blobcount.train` reads them. Anything else is refused here,
    and the refusal names the condition the argument broke and the remedy: a name
    torchvision would build a module for is not a name this project builds.

    The model returned is a torchvision module with its ImageNet weights when `pretrained`
    is true and none when it is false, on the device torchvision built it on — CPU, until
    `blobcount.train` moves it. **The head is the only part that differs from the backbone
    torchvision built**, and the count is the number of logits the forward pass returns per
    patch, so one `224`-square input is enough for a caller to check the width it asked for.

    **Nothing is trained, moved to a device or saved here**, and nothing is caught: a
    `weights="DEFAULT"` call that cannot reach its checkpoint raises from torch's downloader
    naming the URL it could not fetch, which is the fact a caller needs and which this
    module's message would not replace.
    """
    if not isinstance(name, str) or name not in _BACKBONES:
        raise ValueError(
            f"unsupported architecture {name!r} ({type(name).__module__}."
            f"{type(name).__name__}): this project builds the three architectures in "
            f"SUPPORTED — {', '.join(SUPPORTED)} — and nothing else. The name is matched "
            f"exactly and case-sensitively, so 'ResNet18' is not 'resnet18', and the fact "
            f"that torchvision has a model under a name is not what decides this. Fix: set "
            f"training.architecture in configs/default.yaml to one of those three names, or "
            f"pass one of them to build_model"
        )
    width = _class_count(num_classes)
    constructor, replace_head = _BACKBONES[name]
    model = getattr(torchvision.models, constructor)(weights="DEFAULT" if pretrained else None)
    replace_head(model, width)
    return model
