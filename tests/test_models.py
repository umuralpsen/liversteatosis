import numpy as np
import pytest
import torch
import torchvision.models
from torch import nn

from blobcount.config import load
from blobcount.models import SUPPORTED, build_model


def test_resnet18_head_matches_num_classes():
    m = build_model("resnet18", 2, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_resnet50_head_matches_num_classes():
    m = build_model("resnet50", 2, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 2)


def test_efficientnet_b0_head_matches_num_classes():
    m = build_model("efficientnet_b0", 3, pretrained=False)
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, 3)


def test_unknown_architecture_raises():
    with pytest.raises(ValueError, match="unsupported"):
        build_model("vgg16", 2, pretrained=False)


def test_supported_lists_the_configured_default():
    assert load().get("training.architecture") in SUPPORTED


# The five tests above are the brief's, and each one is a statement a caller can act on:
# the head is the width the caller asked for, an architecture this project does not build
# is refused, and the shipped `training.architecture` is one this project builds. What
# follows pins the parts of the interface they leave open: which layer is the head of each
# of the three backbones and how wide it was to begin with, that the constructor each name
# reaches is the one torchvision publishes under that name, that nothing else about the
# model changed, that the names are matched exactly, that a class count is refused by name
# and with a remedy, that `weights` is the only argument handed to torchvision and that its
# value is the string alias this stack resolves, and that two calls give two independent
# models, which is the property the cross-validation loop depends on when it builds one per
# fold.


# The width each backbone's feature vector has, measured on torchvision 0.25.0 rather than
# written from memory. The head is `nn.Linear(in_features, num_classes)` and `in_features`
# is read off the layer being replaced, so a test that asserted a remembered number would
# be asserting this project's memory and not the model it builds.
_RESNET18_FEATURES = 512
_RESNET50_FEATURES = 2048
_EFFICIENTNET_B0_FEATURES = 1280

# The same three numbers keyed by the name that asks for them, so a test can loop over
# `SUPPORTED` without a chain of conditionals and the head it checks is the one belonging to
# the backbone the caller named.
_FEATURES = {
    "resnet18": _RESNET18_FEATURES,
    "resnet50": _RESNET50_FEATURES,
    "efficientnet_b0": _EFFICIENTNET_B0_FEATURES,
}


def test_supported_is_exactly_the_three_names_this_project_builds():
    assert SUPPORTED == ("resnet18", "resnet50", "efficientnet_b0")


def test_build_model_returns_a_torch_module():
    assert isinstance(build_model("resnet18", 2, pretrained=False), nn.Module)


@pytest.mark.parametrize(
    ("name", "head_features"),
    [
        ("resnet18", _RESNET18_FEATURES),
        ("resnet50", _RESNET50_FEATURES),
    ],
)
def test_a_resnet_head_is_a_linear_over_the_backbone_width(name, head_features):
    head = getattr(build_model(name, 3, pretrained=False), "fc")
    assert isinstance(head, nn.Linear)
    assert (head.in_features, head.out_features) == (head_features, 3)


def test_efficientnet_head_is_classifier_one_and_the_dropout_stays():
    m = build_model("efficientnet_b0", 3, pretrained=False)
    # Measured on this stack, the classifier is `Sequential(Dropout, Linear)`, so the head
    # is index 1 and index 0 is the dropout in front of it. A head swapped in at index 0
    # would leave a `Linear` applied after the 1280-wide layer and the model would refuse
    # the input it was just handed, so this states both ends of the sequence.
    assert isinstance(m.classifier[0], nn.Dropout)
    head = m.classifier[1]
    assert isinstance(head, nn.Linear)
    assert (head.in_features, head.out_features) == (_EFFICIENTNET_B0_FEATURES, 3)


def test_only_the_head_differs_from_the_backbone_torchvision_builds():
    reference = torchvision.models.resnet18(weights=None)
    mine = build_model("resnet18", 7, pretrained=False)
    mine_state, reference_state = mine.state_dict(), reference.state_dict()
    assert set(mine_state) == set(reference_state)
    assert {
        key for key in mine_state if mine_state[key].shape != reference_state[key].shape
    } == {"fc.weight", "fc.bias"}


def test_two_calls_give_two_models_that_share_no_parameter():
    first = build_model("resnet18", 2, pretrained=False)
    second = build_model("resnet18", 2, pretrained=False)
    with torch.no_grad():
        first.fc.bias.fill_(1.0)
    assert second.fc.bias.abs().sum().item() > 0.0
    assert all(
        a is not b for a, b in zip(first.parameters(), second.parameters(), strict=True)
    )


# The one property no head-width assertion can reach: that the name in a dispatch row is the
# constructor torchvision publishes. Widths do not catch it — `resnet101` is 2048-wide at
# `.fc` exactly as `resnet50` is, so pointing the `resnet50` row at it builds a model that
# passes every width and every forward pass above and is a different backbone than the one
# `training.architecture` asked for. Measured through `torchvision.models.get_model_weights`,
# the three names resolve to three different checkpoints, so which one a fold starts from is
# decided by the string in the row and by nothing else.
#
# So every torchvision constructor is stubbed at once and the factory is asked for each name
# in turn: the constructor it reached is the row's own name, and nothing outside `SUPPORTED`
# was reached at all. Both `pretrained` values run, so the `weights` argument is carried for
# all three names rather than for the one each of the two tests further down covers.


def _record_every_torchvision_constructor(monkeypatch, called):
    """Replace all three torchvision constructors with stubs that record their keywords.

    The real constructor is captured first and handed back a genuine `weights=None` model, so
    the factory under test still swaps a real head and the head-width assertion beside it is
    a measurement rather than a tautology.
    """
    for attribute in SUPPORTED:
        real = getattr(torchvision.models, attribute)

        def stub(*, _real=real, _name=attribute, **kwargs):
            called.append((_name, kwargs))
            return _real(weights=None)

        monkeypatch.setattr(torchvision.models, attribute, stub)


@pytest.mark.parametrize("pretrained", [True, False])
def test_a_supported_name_reaches_the_torchvision_constructor_of_that_name(
    monkeypatch, pretrained
):
    called = []
    _record_every_torchvision_constructor(monkeypatch, called)

    for name in SUPPORTED:
        model = build_model(name, 3, pretrained=pretrained)
        head = model.fc if name.startswith("resnet") else model.classifier[1]
        assert (head.in_features, head.out_features) == (_FEATURES[name], 3)

    # Each name reached its own constructor exactly once, in `SUPPORTED` order, and nothing
    # else was called at all: a row naming a different backbone of the same width shows up
    # here and in neither the head-width assertion nor the forward pass.
    assert [attribute for attribute, _ in called] == list(SUPPORTED), called
    assert {attribute: kwargs for attribute, kwargs in called} == {
        name: {"weights": "DEFAULT" if pretrained else None} for name in SUPPORTED
    }


# The cheap failure this names: a configuration hand-edited from `resnet18` to `ResNet18`
# is a name torchvision would accept and a name this project does not have, so the refusal
# is the answer and not a silent `KeyError`.


@pytest.mark.parametrize(
    "name",
    ["ResNet18", "RESNET50", "EfficientNet_B0", " resnet18", "resnet18 "],
)
def test_an_architecture_name_is_matched_exactly_and_case_sensitively(name):
    with pytest.raises(ValueError, match="unsupported"):
        build_model(name, 2, pretrained=False)


def test_the_unsupported_message_names_supported_and_every_name_in_it():
    with pytest.raises(ValueError) as caught:
        build_model("vgg16", 2, pretrained=False)
    message = str(caught.value)
    assert "SUPPORTED" in message, message
    for name in SUPPORTED:
        assert name in message, message


# A name that is not a `str` at all is refused by the same refusal rather than by the hash
# of a `dict` lookup. A file holding a list under `training.architecture` is now refused
# earlier, by `load`'s type check, since `KEY_TYPES` declares that key a `str` — so this guard
# is not reachable from a configuration file and is here for the route that still reaches it,
# a `Config` built in memory from any mapping. Without it, `TypeError: unhashable type` would
# name neither the rule nor the remedy.


@pytest.mark.parametrize(
    "name", [None, 18, 2.0, ["resnet18"], {"architecture": "resnet18"}]
)
def test_a_name_that_is_not_a_string_is_refused_by_the_same_refusal(name):
    with pytest.raises(ValueError, match="unsupported") as caught:
        build_model(name, 2, pretrained=False)
    # The type is named by its qualified name, as `blobcount.data._class_index` names a
    # label's: `list` alone says nothing about where the value came from, and a
    # `numpy.str_` collides with a `str` a reader is holding.
    assert f"{type(name).__module__}.{type(name).__name__}" in str(caught.value)


@pytest.mark.parametrize(
    "num_classes", [0, -1, -2, True, False, 2.0, 0.0, "2", None, 2.5, np.float64(2.0)]
)
def test_a_class_count_that_is_not_a_positive_whole_number_is_refused(num_classes):
    with pytest.raises(ValueError, match="num_classes"):
        build_model("resnet18", num_classes, pretrained=False)


def test_the_class_count_refusal_names_the_type_and_the_remedy():
    with pytest.raises(ValueError) as caught:
        build_model("resnet18", True, pretrained=False)
    message = str(caught.value)
    # `bool` is refused by a type test rather than by a comparison, for the reason
    # `blobcount.data._class_index` states: `bool` subclasses `numbers.Integral`, so
    # `True` is a count of one to every check a head width can be put through.
    assert "builtins.bool" in message, message
    assert "training.num_classes" in message, message


def test_the_class_count_refusal_names_the_width_and_the_head_it_builds():
    with pytest.raises(ValueError) as caught:
        build_model("resnet18", 0, pretrained=False)
    message = str(caught.value)
    assert "nn.Linear" in message, message
    assert "0" in message, message


@pytest.mark.parametrize("num_classes", [1, 2, 3, 12, np.int64(2), np.int32(5)])
def test_a_positive_whole_number_is_accepted(num_classes):
    m = build_model("resnet18", num_classes, pretrained=False)
    assert m.fc.out_features == num_classes
    # The head is handed a plain `int`, not the numpy scalar that arrived, for the reason
    # `blobcount.data.PatchDataset` keeps its labels as the plain `int` the check returned.
    assert type(m.fc.out_features) is int
    assert m(torch.randn(1, 3, 224, 224)).shape == (1, num_classes)


def test_the_architecture_is_refused_before_the_class_count_is_read():
    # Both are wrong at once and the name is the mistake a caller made first: it is the
    # one that decides whether there is a head to be given a width at all.
    with pytest.raises(ValueError, match="unsupported"):
        build_model("vgg16", 0, pretrained=False)


# The `pretrained=True` branch is covered by what this project hands torchvision rather
# than by a pretrained model: `configs/default.yaml` says `training.pretrained: true`, so a
# real `weights="DEFAULT"` call would download ImageNet weights into the torch hub cache and
# this suite runs offline. `weights` is the only keyword the factory may pass, so the
# assertion is on the whole mapping: torchvision's legacy `pretrained=` argument is
# deprecated and warns, and `pyproject.toml` turns a warning into a failure.


def test_pretrained_true_hands_torchvision_the_default_weights(monkeypatch):
    called = []
    _record_every_torchvision_constructor(monkeypatch, called)
    build_model("resnet18", 2, pretrained=True)
    assert called == [("resnet18", {"weights": "DEFAULT"})], called


def test_pretrained_false_hands_torchvision_no_weights(monkeypatch):
    called = []
    _record_every_torchvision_constructor(monkeypatch, called)
    build_model("efficientnet_b0", 2, pretrained=False)
    assert called == [("efficientnet_b0", {"weights": None})], called


# The other half of the order of the two refusals: a class count is refused before the
# backbone is built, so a mistyped `training.num_classes` never reaches the
# `weights="DEFAULT"` call. That call downloads, and a run that stops on a configuration
# value should not have moved a checkpoint into the hub cache to find out.


def test_a_bad_class_count_is_refused_before_the_backbone_is_built(monkeypatch):
    called = []
    _record_every_torchvision_constructor(monkeypatch, called)
    with pytest.raises(ValueError, match="num_classes"):
        build_model("resnet18", 0, pretrained=True)
    assert called == []


# The string alias is the one thing this factory relies on that nothing offline above can
# prove: that `"DEFAULT"` resolves to a weight enum on this stack and asks the downloader
# for the ImageNet checkpoint, rather than being refused as an unrecognised spelling. The
# downloader itself is stubbed, so the call ends here and no bytes move. The URL is
# matched loosely because the checkpoint's content hash is torchvision's to change.
#
# `torchvision.models._api` is torchvision 0.25.0's one place every weights enum resolves
# through, and it is a private module. It is imported inside the test rather than at module
# scope so that a torchvision release which moves it fails this one test with a clear
# `ImportError` rather than the whole file.


@pytest.mark.parametrize("name", SUPPORTED)
def test_the_default_weights_alias_resolves_and_reaches_the_downloader(monkeypatch, name):
    from torchvision.models import _api

    reached = []

    def refuse(url, **kwargs):
        reached.append(url)
        raise RuntimeError("this suite does not download weights")

    monkeypatch.setattr(_api, "load_state_dict_from_url", refuse)
    with pytest.raises(RuntimeError, match="does not download"):
        getattr(torchvision.models, name)(weights="DEFAULT")
    assert len(reached) == 1, reached
    assert reached[0].startswith("https://download.pytorch.org/models/"), reached
