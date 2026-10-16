"""Tests for Grad-CAM, the overlay it is drawn on, and the two entry points.

**Nothing here reads the real cohort or a checkpoint a training run wrote.**
The generated data lives outside the checkout — Task 14 moved it under `%LOCALAPPDATA%`
(the `paths.patches` and `paths.labels` of `configs/default.yaml`) — and this suite does
not depend on it: every figure and every
probability below is computed over synthetic patches written into `tmp_path` and a
randomly initialised backbone this file builds, so a property asserted here is a
property of the code and not a measurement of a specimen. A reader must not take any
of it to mean the study has a result; `blobcount.train` and `blobcount.evaluate` say the
same about their own numbers.

**The hooks this module holds are asserted by behaviour wherever a number can tell them
apart, and by the hook table where it cannot.** `tests/test_interpret.py` measures that
`output.register_hook` and `register_full_backward_hook` return **bitwise identical**
gradients for `layer4[-1]` of a resnet18, so a test comparing two maps cannot say which
API produced one; what it *can* say is that the captured gradient is the gradient of the
layer's output tensor and not of its input, because those two differ (measured: maximum
difference `1.8e-3` against a mean absolute input gradient of `3.9e-4`). That is what
`test_the_map_is_the_formula_over_the_gradient_of_the_layer_output` checks, against
`torch.autograd.grad` rather than against a second hook; the hook table is checked
separately by `test_the_only_hook_registered_is_a_forward_hook`.
"""

import csv
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from blobcount.config import DEFAULT_CONFIG_PATH, load
from blobcount.interpret import overlay_from_config
from blobcount.labeling import CLASS_NAMES, LABEL_BY_CLASS, LABEL_COLUMNS
from blobcount.models import build_model

# `extraction.patch_size` in the shipped configuration, and the size a patch the
# extraction wrote carries. `blobcount.labeling.load_patch` refuses anything else, so
# every patch written below is this size whatever `training.input_size` says — the two
# are different questions and the configuration holds both.
_PATCH_SIZE = 256

# The smallest input a resnet18 will take without an average pool collapsing to 1x1
# early enough to make every figure the same colour, and small enough that the suite
# does not pay for a 224-square forward and backward pass per figure.
_TEST_INPUT = 64


# --------------------------------------------------------------------------------------
# The plan's four tests, as written.
# --------------------------------------------------------------------------------------


def test_gradcam_uses_output_hook_and_produces_a_map():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    m = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(m, m.layer4[-1])
    heat, cls, out = cam.generate(torch.randn(1, 3, 224, 224))
    assert heat.shape == (224, 224)
    assert 0.0 <= heat.min() and heat.max() <= 1.0
    assert cls in (0, 1)
    assert out.shape == (1, 2)


def test_gradcam_map_is_not_constant():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    torch.manual_seed(0)
    m = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(m, m.layer4[-1])
    heat, _, _ = cam.generate(torch.randn(1, 3, 224, 224))
    assert heat.std() > 1e-6


def test_overlay_shape_and_range():
    from blobcount.interpret import overlay
    img = np.random.rand(224, 224, 3).astype(np.float32)
    heat = np.random.rand(224, 224).astype(np.float32)
    out = overlay(img, heat, 0.4)
    assert out.shape == (224, 224, 3)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_hooks_are_removable():
    from blobcount.interpret import GradCAM
    from blobcount.models import build_model
    m = build_model("resnet18", 2, pretrained=False)
    g = GradCAM(m, m.layer4[-1])
    g.remove()
    assert g.activations is None


# --------------------------------------------------------------------------------------
# The lifecycle `remove()` has to have: safe to call twice, and automatic on `with`.
# --------------------------------------------------------------------------------------


def test_remove_is_safe_to_call_twice():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))
    assert cam.activations is not None and cam.gradients is not None

    cam.remove()
    assert (cam.activations, cam.gradients) == (None, None)
    cam.remove()  # a second call, which is what a `finally` after an explicit remove does
    assert (cam.activations, cam.gradients) == (None, None), "and it changes nothing"


def test_a_context_manager_removes_its_hooks_on_exit():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    layer = model.layer4[-1]
    before = len(layer._forward_hooks)
    with GradCAM(model, layer) as cam:
        cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))
        assert cam.activations is not None
    assert cam.activations is None and cam.gradients is None
    assert len(layer._forward_hooks) == before, "the context manager left a hook behind"


def test_generate_refuses_after_the_hooks_were_removed():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    cam.remove()
    with pytest.raises(ValueError, match="remove"):
        cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))


def test_the_only_hook_registered_is_a_forward_hook():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    layer = model.layer4[-1]
    # The two APIs return bitwise identical gradients here (measured on torch 2.10.0),
    # so no comparison of two maps can say which one this class uses. This reads the
    # hook table instead, which is the only place the choice is visible. On 2.10.0 a full
    # backward hook writes into the same `_backward_hooks` dict as a plain backward hook
    # and is distinguished by `_is_full_backward_hook`, which starts as `None`.
    assert not layer._forward_hooks and not layer._backward_hooks
    assert layer._is_full_backward_hook is None
    cam = GradCAM(model, layer)
    assert len(layer._forward_hooks) == 1
    assert not layer._backward_hooks and layer._is_full_backward_hook is None, (
        "register_full_backward_hook reports gradients w.r.t. the module's inputs, and "
        "torch's own docstring says it fires with respect to the inputs and that "
        "non-Tensor arguments come back as None; this module takes the gradient on the "
        "output tensor itself"
    )
    cam.remove()
    assert not layer._forward_hooks


# --------------------------------------------------------------------------------------
# The map itself: the formula, the class, and what `generate` leaves behind.
# --------------------------------------------------------------------------------------


def test_the_map_is_the_formula_over_the_gradient_of_the_layer_output():
    from blobcount.interpret import GradCAM

    torch.manual_seed(3)
    model = build_model("resnet18", 2, pretrained=False)
    tensor = torch.randn(1, 3, 224, 224)
    cam = GradCAM(model, model.layer4[-1])
    heat, target_class, _ = cam.generate(tensor)
    captured = cam.gradients.clone()
    cam.remove()

    # The same model and the same input, with the gradient taken by `autograd.grad`
    # against the layer's own output tensor instead of by a hook on it. `generate` set
    # the model to eval and left it there, and a resnet18 in eval mode is deterministic,
    # so this second pass sees the activations the first one saw.
    live: dict = {}
    handle = model.layer4[-1].register_forward_hook(
        lambda module, args, output: live.__setitem__("output", output)
    )
    model.eval()
    logits = model(tensor)
    reference = torch.autograd.grad(logits[0, target_class], live["output"])[0]
    handle.remove()

    assert tuple(captured.shape) == (1, 512, 7, 7)
    # Not the layer's *input* gradient, which differs from this one: measured maximum
    # difference 1.8e-3 against a mean absolute input gradient of 3.9e-4.
    assert torch.allclose(captured, reference, atol=1e-8)

    weights = reference.mean(dim=[2, 3], keepdim=True)
    expected = torch.relu((weights * live["output"].detach()).sum(dim=1, keepdim=True))
    expected = cv2.resize(expected.squeeze().numpy(), (224, 224), interpolation=cv2.INTER_LINEAR)
    expected = (expected - expected.min()) / (expected.max() - expected.min() + 1e-8)
    np.testing.assert_allclose(heat, expected, atol=1e-6)


def test_the_target_class_is_the_prediction_unless_the_caller_names_one():
    from blobcount.interpret import GradCAM

    torch.manual_seed(5)
    model = build_model("resnet18", 2, pretrained=False)
    tensor = torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT)
    # The reference prediction, taken in the mode `generate` will use and before the hook
    # is attached — the hooks outlive `generate`, so a forward pass over this model after
    # one is refused by the hook rather than answered for.
    model.eval()
    with torch.no_grad():
        expected = int(model(tensor).argmax(dim=1))

    cam = GradCAM(model, model.layer4[-1])
    _, predicted, _ = cam.generate(tensor)
    assert predicted == expected

    other = 1 - predicted
    heat, named, out = cam.generate(tensor, target_class=other)
    assert named == other
    assert heat.shape == (_TEST_INPUT, _TEST_INPUT)
    # The map is the one of the named class and not of the predicted one: the two differ,
    # so a `generate` that ignored `target_class` would return the first map twice.
    first, _, _ = cam.generate(tensor, target_class=predicted)
    assert not np.allclose(first, heat)
    cam.remove()
    del out


def test_generate_leaves_gradients_on_the_parameters_it_passed_through():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    assert all(parameter.grad is None for parameter in model.parameters())
    cam = GradCAM(model, model.layer4[-1])
    cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))

    # The caller did not hand `generate` a model to accumulate gradients on, and it is
    # told here rather than in a comment: `generate` zeroes every gradient before its
    # backward and the backward repopulates every parameter in the graph.
    assert model.training is False, "generate puts the model in eval and does not restore it"
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert cam.activations is not None and cam.gradients is not None


def test_generate_refuses_a_batch_that_is_not_one():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    with pytest.raises(ValueError, match="batch"):
        cam.generate(torch.randn(2, 3, _TEST_INPUT, _TEST_INPUT))
    cam.remove()


def test_generate_refuses_a_forward_pass_that_carries_no_graph():
    from blobcount.interpret import GradCAM

    model = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    # torch's own refusal for this state is `RuntimeError: cannot register a hook on a
    # tensor that doesn't require gradient`, measured, which names the condition and not
    # the remedy. A caller who expects a map to need no gradients reaches it.
    with torch.no_grad(), pytest.raises(ValueError, match="no_grad"):
        cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))
    cam.remove()


def test_generate_refuses_a_layer_the_forward_pass_never_reached():
    from blobcount.interpret import GradCAM
    from torch import nn

    model = build_model("resnet18", 2, pretrained=False)
    detached = nn.Conv2d(3, 3, kernel_size=1)
    cam = GradCAM(model, detached)
    with pytest.raises(ValueError, match="never fired"):
        cam.generate(torch.randn(1, 3, _TEST_INPUT, _TEST_INPUT))
    cam.remove()


# --------------------------------------------------------------------------------------
# `overlay`: the range of `alpha`, the two configured keys, and the colour map's name.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_overlay_refuses_an_alpha_outside_the_unit_range(alpha):
    from blobcount.interpret import overlay

    image = np.random.rand(8, 8, 3).astype(np.float32)
    heat = np.random.rand(8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="alpha"):
        overlay(image, heat, alpha)


def test_overlay_at_zero_is_the_image_and_at_one_is_the_colour_map():
    from blobcount.interpret import overlay

    image = np.random.rand(16, 16, 3).astype(np.float32)
    heat = np.random.rand(16, 16).astype(np.float32)
    coloured = (
        cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        / 255.0
    ).astype(np.float32)

    np.testing.assert_array_equal(overlay(image, heat, 0.0), image)
    np.testing.assert_allclose(overlay(image, heat, 1.0), coloured, atol=1e-6)


def test_overlay_reads_the_two_gradcam_keys_from_the_configuration():
    from blobcount.interpret import overlay, overlay_from_config

    image = np.random.rand(16, 16, 3).astype(np.float32)
    heat = np.random.rand(16, 16).astype(np.float32)

    # `gradcam.alpha: 0` is a blend of nothing: the configured value has to be *used*,
    # and a `bin/` read would be outside the scan in `tests/test_config.py` anyway.
    zero = load(overrides=["gradcam.alpha=0"])
    assert np.array_equal(overlay_from_config(image, heat, zero), image)
    hot = overlay_from_config(image, heat, load(overrides=["gradcam.colormap=hot"]))
    jet = overlay_from_config(image, heat, load(overrides=["gradcam.colormap=jet"]))
    assert not np.allclose(hot, jet), "gradcam.colormap is read, not ignored"

    # The shipped values, so the ledger entry this task closes is the shipped one.
    shipped = load()
    assert float(shipped.get("gradcam.alpha")) == 0.4
    assert str(shipped.get("gradcam.colormap")) == "jet"
    np.testing.assert_allclose(
        overlay_from_config(image, heat, shipped), overlay(image, heat, 0.4, colormap="jet")
    )


def test_overlay_refuses_a_colour_map_opencv_does_not_publish():
    from blobcount.interpret import overlay

    image = np.random.rand(8, 8, 3).astype(np.float32)
    heat = np.random.rand(8, 8).astype(np.float32)
    with pytest.raises(ValueError, match="viridian"):
        overlay(image, heat, 0.4, colormap="viridian")


def test_the_normalized_maximum_is_below_one():
    from blobcount.interpret import GradCAM

    torch.manual_seed(11)
    model = build_model("resnet18", 2, pretrained=False)
    cam = GradCAM(model, model.layer4[-1])
    heat, _, _ = cam.generate(torch.randn(1, 3, 224, 224))
    # The epsilon is in the denominator of the normalization, so the maximum lands below
    # one rather than on it: measured 0.9999996423721313 over a map whose range was
    # 0.0254. `overlay` scales by 255 and casts to `uint8`, so a maximum of exactly 1.0
    # would be 255 and of 1.0000004 would be 0 — this is the difference between a map
    # whose hottest pixel is the hottest colour and one that is the coldest.
    assert heat.max() < 1.0
    assert heat.max() > 0.99
    cam.remove()


def test_overlay_clips_an_image_that_is_not_inside_the_unit_range():
    from blobcount.interpret import overlay

    # `alpha=0` so the blend is the image itself, which is what makes the clip the only
    # thing standing between 1.4 and 1.0: at `alpha=0.5` the colour map's own maximum of
    # 0.5 would pull the result back under one and the clip would never be reached.
    image = np.full((8, 8, 3), 1.4, dtype=np.float32)
    heat = np.zeros((8, 8), dtype=np.float32)
    out = overlay(image, heat, 0.0)
    assert out.max() == 1.0
    assert out.min() == 1.0


# --------------------------------------------------------------------------------------
# The two entry points.
# --------------------------------------------------------------------------------------


def _load_cli(monkeypatch, name: str):
    path = Path(__file__).resolve().parents[1] / "bin" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"blobcount_cli_{name}", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _write_image(path: Path, size: int, seed: int = 0) -> Path:
    """Write one RGB PNG of deterministic noise and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(seed)
    Image.fromarray((random.random((size, size, 3)) * 255).astype("uint8")).save(path)
    return path


def _write_checkpoint(path: Path, bias=(10.0, 0.0)) -> Path:
    """Write a real resnet18 state dict whose head forces a known class.

    Zero weights and a large bias make the logits a constant `(bias[0], bias[1])`, so
    the prediction this checkpoint produces is arithmetic rather than a property of a
    random initialisation — and the column that carries it cannot be swapped without
    one of these assertions failing. Nothing is stubbed: the file is a torch archive of
    a real model's parameters, read by `torch.load` and `load_state_dict` as usual.
    """
    model = build_model("resnet18", 2, pretrained=False)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor(bias))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def _cli_config(directory: Path, results: Path, **sections) -> Path:
    """Write a configuration file whose results root is `results`, and return its path.

    `training.input_size` is cut to `_TEST_INPUT` because a resnet18 forward pass over
    224x224 is twelve times the area of one over 64x64 and this suite draws figures
    rather than measuring a model.
    """
    directory.mkdir(parents=True, exist_ok=True)
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data["paths"]["results"] = str(results)
    data["training"].update({"input_size": _TEST_INPUT})
    for section, values in sections.items():
        data[section].update(values)
    path = directory / "config.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def predict_cli(tmp_path, project_scratch, monkeypatch):
    """`bin/predict.py`, three real PNGs, a checkpoint with a known head, and a results
    root that does not exist yet.

    The results root is inside the checkout because the command line refuses a target
    that is not strictly below the project root, and pytest's `tmp_path` is outside it.
    It is *absent*, which is the state the legacy `test_model.py` crashed in.
    """
    module = _load_cli(monkeypatch, "predict")
    images = tmp_path / "images"
    # Written out of order, because a directory scan that returned the filesystem's order
    # would agree with `sorted()` here by accident: the files are created as 2, 0, 1 so a
    # scan is only sorted if the program sorts it.
    for index in (2, 0, 1):
        _write_image(images / f"patch_{index}.png", _TEST_INPUT, seed=index)
    checkpoint = _write_checkpoint(tmp_path / "weights.pth")
    results = project_scratch / "results"
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", _cli_config(tmp_path, results))
    return module, images, checkpoint, results


@pytest.fixture
def labelled_predict_cli(predict_cli, tmp_path, project_scratch, monkeypatch):
    """`predict_cli` plus the handoff the labelling stage would have written.

    **`labels.csv` is the authority on which specimen a patch came from**, and this
    fixture stages it the way `bin/label_patches.py` writes it: one row per patch with
    `slide_id` naming the specimen and `path` relative to the label root. The directory
    tree beside it holds the same files under `normal/` and `steatosis/` and **names no
    specimen in any directory name**, which is the fact `--specimen` has to be built on
    rather than on the layout.

    Five patches over three specimens: 4 holds one steatotic and one normal, 5 one
    steatotic, 6 one steatotic and one normal. `--dir` is pointed at a directory holding
    the three steatotic ones, which is the shape `make predict` uses, so the fixture
    reproduces the finding: two of the three rows are specimens the fold-0 checkpoint
    was fitted on.
    """
    module, images, checkpoint, results = predict_cli
    label_root = tmp_path / "labels"
    rows = []
    for index, (specimen, label, name) in enumerate((
        ("4", 1, "4_100_100.png"),
        ("4", 0, "4_200_200.png"),
        ("5", 1, "5_300_300.png"),
        ("6", 1, "6_400_400.png"),
        ("6", 0, "6_500_500.png"),
    )):
        relative = f"{'steatosis' if label else 'normal'}/{name}"
        (label_root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(images / f"patch_{index % 3}.png", label_root / relative)
        rows.append([name, specimen, label, 7, "0.5", relative])
    handoff = tmp_path / "handoff"
    handoff.mkdir(exist_ok=True)
    with (handoff / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)
    steatosis = label_root / "steatosis"
    return module, steatosis, checkpoint, results, handoff / "labels.csv"


def test_predict_filters_to_the_specimen_the_caller_names(labelled_predict_cli):
    # **The finding.** `make predict` pointed `--dir` at the steatosis tree of every
    # specimen while the fold-0 checkpoint was fitted on two of the three, so 3079 of 3502
    # committed rows were resubstitution over specimens the model had memorised, written
    # into a tracked artifact with no checkpoint, no specimen column and no flag.
    module, steatosis, checkpoint, results, labels = labelled_predict_cli
    assert module.main([
        "--dir", str(steatosis), "--checkpoint", str(checkpoint),
        "--labels", str(labels), "--specimen", "4",
    ]) == 0
    written = _read_csv(results / "predictions.csv")
    assert [Path(row["path"]).name for row in written] == ["4_100_100.png"]


def test_every_prediction_row_says_which_specimen_it_is_and_whether_it_was_trained_on(
    labelled_predict_cli
):
    # **The columns are written whether or not `--specimen` was given.** A run that
    # filters still produces a file somebody will join to `labels.csv`, and the file that
    # tells them which rows are resubstitution without a second command is the one this
    # project was rebuilt to produce. `trained_on` is `false` for specimen 4 because fold 0
    # trains on 5 and 6, and `true` for the other two — the distinction the old file did
    # not carry at all.
    module, steatosis, checkpoint, results, labels = labelled_predict_cli
    assert module.main([
        "--dir", str(steatosis), "--checkpoint", str(checkpoint),
        "--labels", str(labels), "--train-specimens", "5", "--train-specimens", "6",
    ]) == 0
    written = _read_csv(results / "predictions.csv")
    assert len(written) == 3
    for row in written:
        assert row["specimen"] in {"4", "5", "6"}
        assert row["trained_on"] in {"True", "False"}
    by_specimen = {row["specimen"]: row for row in written}
    assert by_specimen["4"]["trained_on"] == "False"
    assert by_specimen["5"]["trained_on"] == "True"
    assert by_specimen["6"]["trained_on"] == "True"
    # **The columns come first and are named, not appended after the probabilities**, so a
    # reader opening the file sees what a row is about before it sees how confident the
    # model was.
    assert list(written[0])[:6] == [
        "path", "specimen", "trained_on", "class_index", "predicted_class", "confidence",
    ]


def test_a_row_the_handoff_does_not_describe_says_so_rather_than_guessing(predict_cli):
    # **`--labels` is optional, and an image outside the handoff is the ordinary case for
    # this program**: it is inference over an image or a directory, and a caller handing
    # it a picture from somewhere else is using it as designed. The specimen column is
    # then empty and `trained_on` is `False` — the empty cell is "not in the handoff",
    # which is a fact, and inventing a specimen from the file name would not be.
    module, images, checkpoint, results = predict_cli
    assert module.main([
        "--dir", str(images), "--checkpoint", str(checkpoint),
        "--train-specimens", "Z",
    ]) == 0
    written = _read_csv(results / "predictions.csv")
    assert len(written) == 3
    for row in written:
        assert row["specimen"] == ""
        assert row["trained_on"] == "False"


def test_naming_a_specimen_without_a_handoff_is_refused(predict_cli, capsys):
    # `--specimen` filters by a fact only the handoff carries, so asking for it without
    # one is a refusal rather than a filter that silently matches nothing.
    module, images, checkpoint, _results = predict_cli
    argv = ["--dir", str(images), "--checkpoint", str(checkpoint), "--specimen", "4"]
    assert module.main(argv) == 2
    assert "--labels" in capsys.readouterr().err


def test_naming_a_specimen_the_handoff_holds_no_patch_of_is_refused(labelled_predict_cli, capsys):
    # The same reason `--specimens` is checked in `bin/evaluate.py`: a filter that
    # matches nothing would write a header and no rows, which is a report of nothing a
    # reader cannot tell apart from a model that predicted nothing.
    module, steatosis, checkpoint, _results, labels = labelled_predict_cli
    argv = [
        "--dir", str(steatosis), "--checkpoint", str(checkpoint),
        "--labels", str(labels), "--specimen", "Z",
    ]
    assert module.main(argv) == 2
    assert "Z" in capsys.readouterr().err


def test_predict_writes_a_row_per_image_and_creates_the_results_directory(
    predict_cli, capsys
):
    module, images, checkpoint, results = predict_cli
    assert not results.exists(), "the directory is what this run has to create"
    assert module.main(["--dir", str(images), "--checkpoint", str(checkpoint)]) == 0

    assert results.is_dir()
    written = _read_csv(results / "predictions.csv")
    assert len(written) == 3
    assert list(written[0])[:6] == [
        "path", "specimen", "trained_on", "class_index", "predicted_class", "confidence",
    ]

    # `sorted()`, so the row order means the same thing on the next run of the pipeline.
    assert [Path(row["path"]).name for row in written] == [
        "patch_0.png", "patch_1.png", "patch_2.png",
    ]
    # The checkpoint's head makes the logits (10.0, 0.0), so class 0 wins by construction
    # and `softmax(10, 0)` is 0.99995 — which is what pins the class-name mapping and
    # says the two probability columns are the ones they claim to be.
    for row in written:
        assert row["class_index"] == "0"
        assert row["predicted_class"] == CLASS_NAMES[0]
        assert float(row["confidence"]) == pytest.approx(float(row["p_0"]))
        assert float(row["p_0"]) > 0.999
        assert float(row["p_0"]) + float(row["p_1"]) == pytest.approx(1.0)

    printed = capsys.readouterr().out
    assert "patch_0.png" in printed and "predictions.csv" in printed


def test_predict_takes_a_single_image_as_well_as_a_directory(predict_cli):
    module, images, checkpoint, results = predict_cli
    assert module.main([
        "--image", str(images / "patch_1.png"), "--checkpoint", str(checkpoint),
    ]) == 0
    written = _read_csv(results / "predictions.csv")
    assert [Path(row["path"]).name for row in written] == ["patch_1.png"]


def test_the_map_is_zero_where_the_summed_contribution_is_negative():
    # **A ReLU is not observable on a randomly initialised backbone, and this test is
    # built rather than stumbled upon.** Measured: for every seed tried, the pre-ReLU sum
    # of a resnet18's channel weights times its activations is non-negative everywhere,
    # and a min-max normalization is invariant under a positive affine map — so a map
    # assembled without the ReLU normalises to the same array and
    # `test_the_map_is_the_formula_over_the_gradient_of_the_layer_output` cannot see the
    # difference. Mutation-measured: dropping the ReLU leaves every other test in this
    # file green. This model is built so the sum changes sign.
    from blobcount.interpret import GradCAM

    torch.manual_seed(0)
    conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
    with torch.no_grad():
        # Channel 0 convolves with +1 everywhere and channel 1 with -1, so the two
        # activations are exact opposites and a gradient of (0, 1) makes the summed
        # contribution `act_1`, which is negative wherever the input is positive.
        conv.weight[0].fill_(1.0)
        conv.weight[1].fill_(-1.0)
    head = nn.Linear(2 * 16, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.weight[0, 16:] = 1.0  # class 0 reads the negated channel and nothing else
    model = nn.Sequential(conv, nn.Flatten(), head)
    image = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)

    cam = GradCAM(model, conv)
    heat, target_class, _ = cam.generate(image)
    cam.remove()

    activation = torch.nn.functional.conv2d(image, conv.weight[0:1], padding=1)
    assert target_class == 0
    assert heat.min() == 0.0
    # The summed contribution is `act_1`, which is `-act_0`, so it is negative wherever
    # the positive channel's activation is positive — and every such cell is at the map's
    # floor. Without the ReLU those cells would be the map's *highest* cells instead,
    # because the minimum is taken over the whole array.
    negative = (activation > 0).squeeze().numpy()
    assert negative.any() and not negative.all()
    assert np.all(heat[negative] == 0.0)
    assert heat.max() > 0.0


def test_the_figure_carries_the_map_it_was_given(gradcam_cli, tmp_path):
    # The map has to reach the file, not only the arguments: a drawing that ignored it
    # would still write a well-formed PNG under the right name, and the CLI tests assert
    # the name and the file and nothing else.
    module, *_ = gradcam_cli
    random = np.random.default_rng(0)
    pixels = random.random((32, 32, 3)).astype(np.float32)
    heatmap = random.random((32, 32)).astype(np.float32)
    assert not np.allclose(heatmap, heatmap[::-1]), "the map has to differ from its flip"
    straight, flipped = tmp_path / "straight.png", tmp_path / "flipped.png"
    module._figure(straight, pixels, heatmap, pixels.copy(), "the same title")
    module._figure(flipped, pixels, heatmap[::-1], pixels.copy(), "the same title")
    assert straight.read_bytes() != flipped.read_bytes()


def test_the_blend_carries_the_blend_it_was_given(gradcam_cli, tmp_path):
    # **The same four-line shape as the map's test, with the other panel's argument
    # flipped.** The third panel has to carry the blend `main` computed: a drawing that
    # substituted the patch, the map or a re-blend of the map would still write a
    # well-formed PNG under the right name, and the CLI tests assert the name and the file
    # and nothing else. **This is what caught the one mutation that survived the first
    # round** — `main` handing `overlay_from_config` a flipped map — and it needed no PNG
    # decoding and no knowledge of the layout: what differs is the argument.
    module, *_ = gradcam_cli
    random = np.random.default_rng(1)
    pixels = random.random((32, 32, 3)).astype(np.float32)
    heatmap = random.random((32, 32)).astype(np.float32)
    # **A random map and not a constant one, and the reason is the assertion rather than
    # the drawing**: a uniform heat map blends to a uniform picture, so flipping it changes
    # nothing and the two files would be identical whatever the drawing did with the blend.
    assert not np.allclose(heatmap, heatmap[::-1]), "the map has to differ from its flip"
    cfg = load()
    straight, flipped = tmp_path / "blend.png", tmp_path / "blend_flipped.png"
    module._figure(
        straight, pixels, heatmap, module.overlay_from_config(pixels, heatmap, cfg), "same"
    )
    module._figure(
        flipped, pixels, heatmap, module.overlay_from_config(pixels, heatmap[::-1], cfg), "same"
    )
    assert straight.read_bytes() != flipped.read_bytes()


def test_the_blend_is_over_the_map_shown_beside_it(gradcam_cli, monkeypatch, tmp_path):
    # **The other half of the property, and the one the figure test cannot reach.** Two
    # separate facts are involved: `_figure` renders the blend it is handed (the test
    # above), and `main` hands it the blend of the map it displays beside. A run that
    # blended one map and displayed another would write a perfectly well-formed figure in
    # which the third panel and the middle panel disagree, and nothing that looks at a PNG
    # as a PNG would notice.
    #
    # **The check is on the arrays, not on the pixels.** `_figure` is wrapped to record
    # what it was handed and then to do its real work, so the assertions are about the two
    # arrays `main` computed — the real `overlay_from_config`, the real map, the real
    # patch — rather than about anything this file drew. The blend a given map must produce
    # is recomputed here with the same configuration the run used, so the test cannot pass
    # by agreeing with a wrong blend.
    module, handoff, _checkpoint, _results = gradcam_cli
    # **The fixture's checkpoint is the wrong one for this test, and using it here is what
    # let M32 survive a first attempt.** `_write_checkpoint` zeroes `fc.weight` so the
    # prediction is arithmetic, and a head with no weights sends **no gradient** to the
    # last stage: measured, the map is identically zero over a whole patch and therefore
    # identical to its own vertical flip, so a run that blended the flipped map and one
    # that did not wrote byte-identical figures. An ordinary initialised head is what a
    # trained model would have, and its map is not flip-invariant: measured over the same
    # patch, `cam max = 1.0000` and `allclose(heat, heat[::-1])` is `False`.
    checkpoint = tmp_path / "ordinary_head.pth"
    torch.save(build_model("resnet18", 2, pretrained=False).state_dict(), checkpoint)
    calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    real_figure = module._figure

    def recording(path, pixels, heatmap, blended, title):
        calls.append((pixels, heatmap, blended))
        real_figure(path, pixels, heatmap, blended, title)

    monkeypatch.setattr(module, "_figure", recording)
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", "2", "--checkpoint", str(checkpoint),
    ]) == 0

    assert len(calls) == 3, "two patches of slide A's normal class and one steatotic"
    cfg = load()
    for pixels, heatmap, blended in calls:
        np.testing.assert_allclose(
            blended, overlay_from_config(pixels, heatmap, cfg), atol=1e-6
        )
        # And the two arrays are of one field of view, which is what makes "over it" a
        # question with an answer: a map at the model's size over a patch at another one
        # could not be blended at all.
        assert heatmap.shape == pixels.shape[:2]


def test_predict_sorts_the_directory_it_scans(predict_cli, monkeypatch):
    # **The ordering policy is not observable through the real filesystem on this
    # machine**, which is why it is pinned this way: measured, `Path.iterdir()` over a
    # directory returns its entries in case-insensitive name order and `sorted()` over
    # the same directory returns the same order, so a scan left unsorted passes every
    # other test in this file — mutation-measured. The listing is handed over in an order
    # a filesystem may produce, and the sort is not.
    module, images, _checkpoint, _results = predict_cli
    listed = list(images.iterdir())
    monkeypatch.setattr(type(images), "iterdir", lambda self: iter(reversed(listed)))
    assert module._images(images, True) == sorted(listed)


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--dir", "not a directory"), ("--image", "not a file")],
)
def test_predict_refuses_a_path_of_the_wrong_kind_for_the_argument(
    predict_cli, capsys, tmp_path, monkeypatch, flag, expected
):
    # Which argument the caller used decides which kind of path is right. A `--dir` naming
    # a file is not a directory of one image and an `--image` naming a directory is not one
    # image, and reading either as the other is a run whose report answers a question the
    # caller did not ask.
    module, images, checkpoint, _results = predict_cli
    wrong = str(images / "patch_0.png") if flag == "--dir" else str(images)
    assert module.main([flag, wrong, "--checkpoint", str(checkpoint)]) == 2
    assert expected in capsys.readouterr().err


def test_predict_refuses_a_results_directory_outside_the_project(
    predict_cli, project_scratch, capsys, tmp_path, monkeypatch
):
    # Re-pointed at the invariant that survives K14, in the one shape this guard's
    # single target allows (K19): `paths.results` is *written inside the checkout*
    # as a junction, and it resolves to a *sibling* of the tree the same
    # configuration says the project owns — `tmp_path/owned-typo`, beside the
    # owned `tmp_path/owned` — so it is neither the owned path itself nor below
    # it, and the guard refuses it after resolving it. The old staging pointed
    # `paths.results` itself at `tmp_path`, which made the refused target one the
    # configuration had just named — owned by definition — and the assertion
    # encoded the superseded invariant "everything outside the checkout is
    # refused". A results directory that resolves beside the owned tree is the
    # hand-edit the guard exists for, and under K19 a value written inside the
    # checkout cannot smuggle the guard to where it points.
    module, images, checkpoint, _results = predict_cli
    outside = tmp_path / "owned-typo"
    if not _directory_link(outside, project_scratch / "results-link"):
        pytest.skip("this platform refused to make a directory link")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(
            tmp_path,
            project_scratch / "results-link",
            paths={"labels": str(tmp_path / "owned" / "labels")},
        ),
    )
    assert module.main(["--dir", str(images), "--checkpoint", str(checkpoint)]) == 2
    err = capsys.readouterr().err
    assert "refusing to write" in err, err
    assert "project root" in err, err
    assert sorted(path.name for path in outside.iterdir()) == [], (
        "a refused run wrote nothing through the link"
    )


def test_predict_refuses_a_directory_holding_no_images(predict_cli, capsys, tmp_path, monkeypatch):
    module, _images, checkpoint, results = predict_cli
    empty = tmp_path / "empty"
    empty.mkdir()
    assert module.main(["--dir", str(empty), "--checkpoint", str(checkpoint)]) == 2
    assert "holds no file carrying" in capsys.readouterr().err
    assert not (results / "predictions.csv").exists()


def _write_labels_csv(directory: Path, rows) -> Path:
    """Write a `labels.csv` under `directory` and return its path.

    The header is `blobcount.labeling.LABEL_COLUMNS` and `path` is written relative to
    the label root and `/`-separated, which is what `blobcount.labeling._write_labels`
    writes and what `blobcount.train.read_labels` resolves against `paths.labels`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "labels.csv"
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_COLUMNS)
        writer.writerows(rows)
    return target


def _handoff_row(slide: str, label: int, relative: str):
    return [f"{slide}_{relative.replace('/', '_').removesuffix('.png')}", slide, label, 7, "0.5",
            relative]


_PATCHES = [
    _handoff_row("A", LABEL_BY_CLASS["normal"], "A/normal_0.png"),
    _handoff_row("A", LABEL_BY_CLASS["steatosis"], "A/steatosis_1.png"),
    _handoff_row("A", LABEL_BY_CLASS["normal"], "A/normal_2.png"),
    _handoff_row("B", LABEL_BY_CLASS["normal"], "B/normal_3.png"),
]


def _directory_link(target: Path, link: Path) -> bool:
    """Make `link` a directory link to `target`; report whether this platform allowed it.

    The same helper `tests/test_evaluate.py` states for its own junction test, kept
    here rather than imported across test modules: `os.symlink` needs a privilege a
    test suite should not assume, and `mklink /J` is the Windows form that needs none.
    `stdout`/`stderr` are discarded rather than captured as text, for the reason the
    sibling file measured: `mklink` echoes the two paths back, the project root holds
    `Masaüstü`, and `subprocess` decodes that with the machine's cp1254 — which raises
    `UnicodeDecodeError` inside the reader thread and surfaces as a warning
    `filterwarnings = ["error"]` turns into a failed run.
    """
    target.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
        return True
    except (OSError, NotImplementedError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        import subprocess

        made = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )
        return made.returncode == 0
    return False


@pytest.fixture
def gradcam_cli(tmp_path, project_scratch, monkeypatch):
    """`bin/gradcam.py`, a handoff over four real patches, and a results root.

    Slide A holds both classes and slide B holds one row of one class, so a run can be
    asked for a slide that is there, for one that is not, and for a `--n` larger than
    the class it is asked to draw. The patches are `extraction.patch_size` squares
    because `blobcount.labeling.load_patch` refuses anything else — the extraction's
    question and not this run's.
    """
    module = _load_cli(monkeypatch, "gradcam")
    label_root = tmp_path / "labels"
    for index, row in enumerate(_PATCHES):
        _write_image(label_root / row[5], _PATCH_SIZE, seed=index)
    handoff = _write_labels_csv(tmp_path / "handoff", _PATCHES)
    checkpoint = _write_checkpoint(tmp_path / "weights.pth")
    results = project_scratch / "results"
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(tmp_path, results, paths={"labels": str(label_root)}),
    )
    return module, handoff, checkpoint, results


def test_gradcam_writes_a_figure_per_sample_and_a_summary_grid(gradcam_cli, capsys):
    module, handoff, checkpoint, results = gradcam_cli
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", "2", "--checkpoint", str(checkpoint),
    ]) == 0

    out = results / "gradcam"
    written = sorted(path.name for path in out.glob("*.png"))
    assert written == [
        "gradcam_normal_1.png", "gradcam_normal_2.png",
        "gradcam_steatosis_1.png", "gradcam_summary.png",
    ]
    for name in written:
        assert (out / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    printed = capsys.readouterr().out
    assert "normal" in printed and "steatosis" in printed
    assert "gradcam_summary.png" in printed


def test_gradcam_refuses_a_slide_the_handoff_holds_no_row_of(gradcam_cli, capsys):
    module, handoff, checkpoint, results = gradcam_cli
    assert module.main([
        "--labels", str(handoff), "--slide", "Z", "--n", "2", "--checkpoint", str(checkpoint),
    ]) == 2
    error = capsys.readouterr().err
    # The words the *cohort* refusal uses, not merely the slide name: with the guard
    # removed the run draws nothing and is refused by the empty-sample check below it,
    # whose message also names the slide — mutation-measured, and this assertion is what
    # tells the two apart.
    assert "carries slide_id 'Z'" in error, error
    assert not (results / "gradcam").exists(), "a refused run wrote nothing"


def _broken_checkpoints(tmp_path: Path, checkpoint: Path) -> dict[str, Path]:
    """Return the three ways a `--checkpoint` cannot be loaded, one path each.

    **All three are real files on disk and all three go through `torch.load` for real** —
    none is a mock and none is a monkeypatch of the library. The first is a path nothing
    wrote; the second is a file that exists and is not a torch archive, which is the shape
    a `--checkpoint` typo produces and the one `_load_weights` exists to name; the third is
    a real state dict whose head is 4-wide against the 2 the configuration declares, which
    is what `load_state_dict` refuses with a size mismatch.
    """
    junk = tmp_path / "junk.pth"
    junk.write_bytes(b"not a torch file at all")
    wrong = tmp_path / "wrong_head.pth"
    wrong_model = build_model("resnet18", 4, pretrained=False)
    torch.save(wrong_model.state_dict(), wrong)
    return {"absent": tmp_path / "absent.pth", "junk": junk, "wrong": wrong}


# What each broken checkpoint's refusal says, as the words to look for. A refusal is
# only useful to a caller who can act on it, so each of these is the fact the arm has to
# carry: which file, or which disagreement.
_BROKEN_EXPECTATION = {
    "absent": "absent.pth",
    "junk": "junk.pth",
    "wrong": "size mismatch",
}


@pytest.mark.parametrize("shape", ["absent", "junk", "wrong"])
def test_predict_refuses_a_checkpoint_it_cannot_load(
    predict_cli, capsys, tmp_path, shape
):
    # **The checkpoint arm, in all three shapes the module docstring promises.** Measured:
    # a missing file raises `FileNotFoundError` (an `OSError`); a file that is not a torch
    # archive raises `pickle.UnpicklingError`, whose bases are `PickleError, Exception,
    # BaseException, object` — neither an `OSError` nor a `ValueError`, so an arm holding
    # only those two would let a `--checkpoint` naming the wrong file reach the caller as a
    # traceback; and weights that do not fit the head raise `RuntimeError: Error(s) in
    # loading state_dict ... size mismatch`. Each has to arrive as a refusal naming the
    # file or the disagreement, and **none of the three may leave a CSV behind**, because a
    # half-written report of a run that refused is worse than no report.
    module, images, checkpoint, results = predict_cli
    broken = _broken_checkpoints(tmp_path, checkpoint)[shape]
    assert module.main(["--dir", str(images), "--checkpoint", str(broken)]) == 2
    error = capsys.readouterr().err
    assert _BROKEN_EXPECTATION[shape] in error, error
    assert not (results / "predictions.csv").exists()


@pytest.mark.parametrize("shape", ["absent", "junk", "wrong"])
def test_gradcam_refuses_a_checkpoint_it_cannot_load(
    gradcam_cli, capsys, tmp_path, shape
):
    # The same three shapes on the figure side, for the reasons the predictor's three
    # comments give. **No figure is written and the hook-bearing model is never built**, so
    # a refused run leaves `results/gradcam/` exactly as the previous one left it.
    module, handoff, _checkpoint, results = gradcam_cli
    broken = _broken_checkpoints(tmp_path, _checkpoint)[shape]
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", "1", "--checkpoint", str(broken),
    ]) == 2
    error = capsys.readouterr().err
    assert _BROKEN_EXPECTATION[shape] in error, error
    assert not (results / "gradcam").exists()


@pytest.mark.parametrize(
    ("section", "key", "value", "expected"),
    [
        ("training", "architecture", "ResNet18", "ResNet18"),
        ("training", "num_classes", 0, "class count"),
    ],
)
def test_predict_refuses_a_configuration_no_model_can_be_built_from(
    predict_cli, capsys, tmp_path, monkeypatch, section, key, value, expected
):
    # **The two refusals `blobcount.models.build_model` makes for itself, reached through
    # this command line rather than through a traceback out of `main`.** They are stdlib
    # types — `models.py` declares no exception type of its own — so this file is where
    # catching them belongs, and the configuration the caller loaded is not the one the
    # run used, which is what makes it a message worth reading.
    module, images, checkpoint, _results = predict_cli
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data[section][key] = value
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    assert module.main(["--dir", str(images), "--checkpoint", str(checkpoint)]) == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize(
    ("section", "key", "value", "expected"),
    [
        ("training", "architecture", "ResNet18", "ResNet18"),
        ("training", "num_classes", 0, "class count"),
    ],
)
def test_gradcam_refuses_a_configuration_no_model_can_be_built_from(
    gradcam_cli, capsys, tmp_path, monkeypatch, section, key, value, expected
):
    module, handoff, checkpoint, _results = gradcam_cli
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    data[section][key] = value
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", broken)
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", "1", "--checkpoint", str(checkpoint),
    ]) == 2
    assert expected in capsys.readouterr().err


@pytest.mark.parametrize("refused", ["0", "-3"])
def test_gradcam_refuses_an_n_below_one_naming_one_it_accepts(gradcam_cli, capsys, refused):
    # **The remedy has to name a value this run would accept, and the refused one has to be
    # named only as the thing that was refused.** Every refusal in this project says what to
    # do next; this one used to say `--n is 0 ... Fix: pass a positive --n — 0 draws one of
    # each`, so the fix a caller followed was the argument they had just been refused. The
    # two assertions are the property and neither is the other's restatement: the first says
    # the refused value is stated up front, the second says it never reappears after
    # `Fix:`.
    module, handoff, checkpoint, _results = gradcam_cli
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", refused, "--checkpoint", str(checkpoint),
    ]) == 2
    error = capsys.readouterr().err
    assert f"--n is {refused}" in error, error
    remedy = error.partition("Fix:")[2]
    assert refused not in remedy, f"the remedy offers the refused value back: {remedy}"
    assert "--n 1" in remedy, f"the remedy names no value this run accepts: {remedy}"


def test_gradcam_releases_the_hooks_when_the_drawing_raises(gradcam_cli, monkeypatch):
    # **The `finally` is the property, and the way to hold it is to make the drawing
    # raise.** `build_model` is replaced with a function returning the real model this
    # test built — not a stub, the same `resnet18` a run would have built — so the hook
    # table below belongs to a model that really did run a forward and backward pass.
    module, handoff, checkpoint, _results = gradcam_cli
    model = build_model("resnet18", 2, pretrained=False)
    monkeypatch.setattr(module, "build_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(
        module,
        "_figure",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("the renderer failed")),
    )
    with pytest.raises(RuntimeError, match="the renderer failed"):
        module.main([
            "--labels", str(handoff), "--slide", "A", "--n", "1",
            "--checkpoint", str(checkpoint),
        ])
    assert not model.layer4[-1]._forward_hooks, "a failed run left a hook on the model"


def test_gradcam_refuses_a_results_directory_outside_the_project(
    gradcam_cli, project_scratch, capsys, tmp_path, monkeypatch
):
    # Re-pointed at the invariant that survives K14. The results directory the
    # configuration names sits inside the checkout, so it is owned; the `gradcam`
    # subdirectory this run would draw its figures into is a **link** that resolves
    # to a *sibling* of the tree the same configuration says the project owns —
    # `tmp_path/owned-typo`, beside the owned `tmp_path/owned` — which is neither
    # the configured path itself nor below it, so the guard refuses it and no figure
    # is written through the link. The old staging pointed `paths.results` itself at
    # `tmp_path`, which made the target one the configuration had just named — owned
    # by definition — and the refusal asserted the superseded invariant "everything
    # outside the checkout is refused". A link resolving to the near-miss beside the
    # owned tree is the mistyped `--out` the guard exists for, and resolving before
    # comparing is what has always made a link unable to point out.
    module, handoff, checkpoint, _results = gradcam_cli
    results = project_scratch / "results"
    results.mkdir()
    outside = tmp_path / "owned-typo"
    if not _directory_link(outside, results / "gradcam"):
        pytest.skip("this platform refused to make a directory link")
    monkeypatch.setattr(
        "blobcount.config.DEFAULT_CONFIG_PATH",
        _cli_config(tmp_path, results, paths={"labels": str(tmp_path / "owned" / "labels")}),
    )
    assert module.main([
        "--labels", str(handoff), "--slide", "A", "--n", "1", "--checkpoint", str(checkpoint),
    ]) == 2
    err = capsys.readouterr().err
    assert "refusing to write" in err, err
    assert "project root" in err, err
    assert sorted(path.name for path in outside.iterdir()) == [], (
        "no figure was written through the link"
    )
