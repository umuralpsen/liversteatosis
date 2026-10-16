"""Grad-CAM over one convolutional block, and the blend a caller draws the map on.

**Nothing in this module has been run against the cohort.** `dataset/labels/` does not
exist, the extraction that would fill it has never been run, and no checkpoint in this
project has ever been trained, so no map this module produces describes a patch of this
study. Every map it has produced in this repository has been over `torch.randn`,
`numpy.random` or a synthetic PNG a test wrote; `bin/gradcam.py`'s `--slide` and
`--checkpoint` have never been given real input.

**The gradient is taken on the layer's output tensor, and the argument is the tensor and
not the number.** `__init__` registers a forward hook, and that hook stores
`output.detach()` and calls `output.register_hook(...)` on the live tensor. Measured on
this stack, the forward hook on `resnet18.layer4[-1]` fires with a single
`(1, 512, 7, 7)` `torch.Tensor` for a `(1, 3, 224, 224)` input — `output.requires_grad`
is `True` and `output.is_leaf` is `False` — and the tensor hook fires during `backward`
with a gradient of the same shape.

**`register_full_backward_hook` is not used, and at this layer the two are not different
numbers.** Measured on the same layer and the same input: that hook reports `grad_input`
and `grad_output`, both `(1, 512, 7, 7)`, and **they are not the same tensor** — the
maximum absolute difference between them is `1.8e-3` against a mean absolute input
gradient of `3.9e-4`. The gradient captured here is bitwise equal to that hook's
`grad_output` (`torch.equal` `True`, maximum difference `0.0`) and **not** equal to its
`grad_input` (`torch.equal` `False`). So nothing here is a numerical fix at this layer,
and no test in `tests/test_interpret.py` claims one: the map is checked against
`torch.autograd.grad` on the layer's output tensor, which is a question the input gradient
would answer differently.

**What the legacy `gradcam.py:53` read is the contract that is being changed.** It
registered `register_full_backward_hook` and took `grad_output[0]` (`gradcam.py:59`), and
torch's own docstring for that hook — read off the installed 2.10.0, not recalled — says
the hook fires "when the gradients are computed with respect to the module inputs", that
`grad_input` "will only correspond to the inputs given as positional arguments and all
kwarg arguments are ignored", that entries "will be `None` for all non-Tensor arguments",
and that the module's forward "will receive a view of each Tensor passed to the Module".
So the tuple this module does not read is documented as the one that fires, with three
stated gaps, while the tensor hook has no such qualifications: **the gradient it captures
is the gradient of the exact tensor whose `detach()` sits in `activations`, by
construction, and not by which half of a two-tuple the reader picked.**

**The map is min-max normalized with an epsilon in the denominator, so its maximum is
strictly below one and a map of one value is all zeros.** Measured: over a map whose
range was `0.0254`, the normalized maximum is `0.9999996423721313`, and a map that is
zero everywhere normalizes to `0.0` without a division warning. That is what lets
`overlay` scale a map by `255` without a value wrapping round a `uint8`.

**`generate` leaves the model in `eval()` and every parameter in the graph carrying a
`.grad`, and a caller has to call `remove()`.** The `.grad` claim is measured, not
assumed: `zero_grad()` defaults to `set_to_none=True` on torch 2.10.0, and after the
backward 62 of 62 resnet18 parameters carry a gradient, so a caller who was accumulating
gradients has lost them. The hooks are still registered afterwards — a forward pass on the
same model captures again — so **`remove()` is the caller's to make, and `bin/gradcam.py`
makes it in a `finally`**. `remove()` is safe to call twice because the two handles are
set to `None` here rather than because `RemovableHandle.remove()` tolerates it: measured
on 2.10.0 a torch handle survives two `remove()` calls, and this module does not depend on
that.

**`overlay` validates `alpha` and refuses a colour map OpenCV does not publish, rather
than repairing either.** `alpha=-0.4` blends in `-0.4 * colour + 1.4 * image` and
`alpha=1.4` blends in `1.4 * colour - 0.4 * image` — a picture the caller did not ask
for, which is the same reason `blobcount.models.build_model` refuses a class count and
`blobcount.data.build_transforms` refuses a mistyped key. One predicate refuses all four
bad values, because `0.0 <= alpha <= 1.0` is `False` for `-0.1`, `1.1`, `nan` and `inf`
alike (measured). The colour map is resolved through OpenCV's own published constants —
measured: `cv2` publishes 22 of them, from `autumn` to `winter`, and no `viridian` — so
the configured `gradcam.colormap` means what it says and a name nobody publishes is a
refusal rather than a black picture.

**`overlay` takes the map `generate` returns, and `overlay_from_config` is where the two
`gradcam.*` keys are read.** `bin/` is a directory of scripts loaded by path rather than a
package, so a read of either key inside a command line is outside the source scan in
`tests/test_config.py` and the key goes on being reported as one nothing under
`blobcount/` reads — the reason `blobcount.train.read_labels` and
`blobcount.evaluate.bootstrap_metric_intervals` exist.

Declared interface, in `__all__` order:

    GradCAM
        GradCAM.__init__(self, model: nn.Module, target_layer: nn.Module) -> None
        GradCAM.activations: torch.Tensor | None
        GradCAM.gradients: torch.Tensor | None
        GradCAM.generate(
            self, input_tensor: torch.Tensor, target_class: int | None = None
        ) -> tuple[np.ndarray, int, torch.Tensor]
        GradCAM.remove(self) -> None
        GradCAM.__enter__(self) -> GradCAM
        GradCAM.__exit__(self, *exc_info: object) -> None
    overlay(
        image: np.ndarray, heatmap: np.ndarray, alpha: float, colormap: str = "jet"
    ) -> np.ndarray
    overlay_from_config(image: np.ndarray, heatmap: np.ndarray, cfg: Config) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from blobcount.config import Config

__all__ = ["GradCAM", "overlay", "overlay_from_config"]

# In the denominator of the normalization and nowhere else, for the reason the module
# docstring gives: a map of one value would otherwise divide by zero, and the measured
# cost of carrying it is that a normalized maximum is `0.9999996423721313` rather than
# `1.0` — below one, which is what `overlay`'s `uint8` scaling wants.
_EPSILON = 1e-8


class GradCAM:
    """Grad-CAM for one convolutional block: its activations, its gradient, and the map.

    **The caller must supply a model and one of its modules, and must call `remove()` when
    it is finished with them.** The module is where the activations are read from and the
    gradient taken of; a block the forward pass never reaches is refused by `generate`
    rather than answered for. `remove()` is the caller's because the hooks outlive the
    call: a model whose forward pass runs after one `GradCAM` has captured again, and
    `bin/gradcam.py` releases its own in a `finally` so a run that failed to draw leaves
    no hook on the model behind it.

    **`activations` and `gradients` are the two tensors the map is made of, and both are
    `None` until a forward pass has produced them.** After `generate` they are the
    detached activation `(batch, channels, height, width)` and the gradient of that exact
    tensor, so a caller holding them can check the formula rather than trust it — which is
    what `tests/test_interpret.py` does, against `torch.autograd.grad`.

    **The model is put into `eval()` by `generate` and is not restored.** A Grad-CAM over
    a model in training mode is a map of a dropout sample and of batch statistics rather
    than of the model, and `blobcount.evaluate._infer` puts the model in `eval()` for the
    same reason; the mode is left as it is because a caller that was going to train next
    says so itself and this class is not that caller's business.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """Store the model and the layer, and register the forward hook on the layer."""
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        # `None` is the state that says "removed", which is what `generate` refuses on and
        # what makes a second `remove()` a no-op without relying on torch's tolerance.
        self._forward_hook: RemovableHandle | None = target_layer.register_forward_hook(
            self._capture
        )
        # The tensor hook belongs to one forward pass and not to this object: a second
        # forward replaces the handle, and the tensor the first one belonged to is gone
        # with its graph.
        self._tensor_hook: RemovableHandle | None = None

    def _capture(self, module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        """Store `output.detach()` and take the gradient of that same tensor.

        **`output.register_hook` and not `register_full_backward_hook`, and the module
        docstring carries the whole of the measurement and of torch's own wording.** The
        activation stored here and the gradient captured there are then gradients and
        values of one tensor by construction, and the map below cannot be assembled from
        two different halves of a block.

        **A forward pass that carries no graph is refused here rather than in
        `generate`,** because this is where it is visible: torch's own refusal for it is
        `RuntimeError: cannot register a hook on a tensor that doesn't require gradient`
        (measured, raised from `Tensor.register_hook`), which names the condition and not
        the remedy. `torch.no_grad()` around `generate` is the way to reach it — a caller
        who expects a map to need no gradients.
        """
        if not output.requires_grad:
            raise ValueError(
                f"the forward pass over {type(module).__name__} produced a tensor that "
                f"does not require a gradient, so there is nothing to take a gradient of: "
                f"Grad-CAM is a backward pass over the activations the forward pass built, "
                f"and torch.no_grad() (or a model whose every parameter has requires_grad "
                f"set to False) builds none. Fix: call generate outside torch.no_grad()"
            )
        self.activations = output.detach()
        self._tensor_hook = output.register_hook(self._capture_gradient)

    def _capture_gradient(self, gradient: torch.Tensor) -> None:
        """Store the gradient of the tensor `_capture` stored the activation of."""
        self.gradients = gradient.detach()

    def generate(
        self, input_tensor: torch.Tensor, target_class: int | None = None
    ) -> tuple[np.ndarray, int, torch.Tensor]:
        """Return `(map, class, logits)` for one image, with the map at the input's size.

        **The caller must supply one image: a `(1, channels, height, width)` tensor of
        the shape `blobcount.data.build_transforms`'s eval transform produces, already on
        the model's device.** A batch of more than one is refused, and not because the
        arithmetic needs a single image: `cam.squeeze()` drops every dimension of size one
        and a batch of two would hand `cv2.resize` a `(2, h, w)` array, which is refused
        by OpenCV with a message about channel counts. `generate` says which of the two it
        is instead.

        **`target_class` is an index into the model's logits and `None` means the model's
        own prediction** — `argmax` over the one row of the batch, so a caller asking for
        the predicted class and a caller naming it get the same answer for the same image.

        **The map is `relu(sum over channels of mean-over-spatial gradient times
        activation)`, resized to the input's height and width with `cv2.INTER_LINEAR`, and
        min-max normalized.** The weights are one number per channel, averaged over the
        two spatial axes and kept as `(1, channels, 1, 1)` so they broadcast over the
        activation beside them; **`relu` then drops every cell whose summed contribution is
        negative**, which is what makes the map a map *for this class* rather than a map of
        the block — and it is not observable on a randomly initialised backbone, because the
        sum is non-negative there for every seed measured and a min-max normalization is
        invariant under a positive affine map. `tests/test_interpret.py` builds the model
        that makes the sum change sign. **The resize is to the size of the tensor the caller
        passed and not to a constant**: `training.input_size` is 224 in
        `configs/default.yaml` and a map at 224 over an image of another size is a map of a
        region that is not there.

        **What this leaves behind is measured and is not the state it found**: the model in
        `eval()`, every parameter in the graph carrying a `.grad` (62 of 62 on a resnet18,
        because `zero_grad()` clears them and the backward refills them), `activations`
        and `gradients` set, and both hooks still registered. The caller releases the last
        with `remove()`.

        **The logits returned are the forward pass's own tensor and not a detached copy**,
        so a caller that converts one to a Python number has to `detach()` it first:
        measured, `float(tensor)` on a tensor that still carries a graph raises
        `UserWarning: Converting a tensor with requires_grad=True to a scalar may lead to
        unexpected behavior`, and `filterwarnings = ["error"]` in `pyproject.toml` turns
        that warning into a failed run. `bin/gradcam.py` carries the `.detach()` and names
        this paragraph beside it.
        """
        if self._forward_hook is None:
            raise ValueError(
                "this GradCAM has had remove() called on it and holds no hooks, so there "
                "is no activation to read and no gradient to weight it with. Build another "
                "one over the model and the layer, or use it as a context manager so the "
                "removal is automatic"
            )
        if input_tensor.dim() != 4 or int(input_tensor.shape[0]) != 1:
            raise ValueError(
                f"generate was given a {tuple(input_tensor.shape)} tensor, and one map is "
                f"made of one image: a batch of more than one has no single argmax, and "
                f"the squeezed map would be a stack of them rather than a map. Fix: pass "
                f"one image at a time, as the (1, channels, height, width) tensor "
                f"blobcount.data.build_transforms produces for a single patch"
            )
        self.model.eval()
        # Cleared first so that a layer whose hook does not fire cannot be answered for
        # with the previous call's tensors: the refusal below has to be about this pass.
        self.activations = None
        self.gradients = None
        output = self.model(input_tensor)
        if target_class is None:
            target_class = int(output.argmax(dim=1))
        self.model.zero_grad()
        output[0, target_class].backward()
        if self.activations is None or self.gradients is None:
            raise ValueError(
                f"the hook on the {type(self.target_layer).__name__} this GradCAM was built "
                f"over never fired, so there is no activation and no gradient to make a "
                f"map from. The layer has to be one the forward pass reaches and the "
                f"backward passes through — on a resnet18 that is "
                f"model.layer4[-1]; a module built outside the model is never called"
            )
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        heatmap = cam.squeeze().detach().cpu().numpy()
        height, width = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        # `cv2` is imported here and not at module scope, for the reason
        # `blobcount.labeling.load_patch` gives for `PIL`: a native wheel at module scope
        # turns an absent one into a collection error for every test in the suite,
        # including the ones that touch no image.
        import cv2

        heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
        low, high = float(heatmap.min()), float(heatmap.max())
        return (heatmap - low) / (high - low + _EPSILON), int(target_class), output

    def remove(self) -> None:
        """Remove both hooks and clear both tensors, and do nothing at all if called again.

        **A second call is a no-op because the handles are dropped here, and not because
        `RemovableHandle.remove()` tolerates being called twice** — measured on torch
        2.10.0 it does, and this module does not depend on it. The reason it is a no-op
        rather than an error is the shape a caller reaches: a `finally` that releases what
        a `with` already released, or two error paths through one function, and an
        `AttributeError` out of a cleanup is a traceback about a run that had already
        finished.

        **The model is left exactly as it is otherwise** — its weights, its device and its
        `eval()` mode are untouched, and the two tensors become `None` rather than being
        zeroed, so a caller can see that they were released.
        """
        if self._forward_hook is not None:
            self._forward_hook.remove()
            self._forward_hook = None
        if self._tensor_hook is not None:
            self._tensor_hook.remove()
            self._tensor_hook = None
        self.activations = None
        self.gradients = None

    def __enter__(self) -> GradCAM:
        """Return this object, having already registered its hooks in `__init__`."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Remove both hooks, whether the block ended or raised."""
        self.remove()


def _colour_map(name: str) -> int:
    """Return the OpenCV colour map constant `name` names, or refuse the name.

    **The table is OpenCV's own and not one written here**: `cv2` publishes 22 of these
    constants, measured, from `autumn` to `winter`, and a second list in this project
    would be a second answer to what a colormap name is and would fall behind the library.
    A name nobody publishes is refused rather than drawn as a black map, which is the same
    ruling `blobcount.models.build_model` applies to an architecture it does not build.

    The attribute is looked up through `getattr` on `cv2` with the name in upper case, so
    the configured `gradcam.colormap: jet` and a `JET` written by hand are the same value;
    anything that is not an integer constant is the refusal.
    """
    import cv2

    code = getattr(cv2, f"COLORMAP_{str(name).upper()}", None)
    if isinstance(code, int):
        return code
    published = ", ".join(
        sorted(
            attribute[len("COLORMAP_"):].lower()
            for attribute in dir(cv2)
            if attribute.startswith("COLORMAP_") and attribute != "COLORMAP_BGRA"
        )
    )
    raise ValueError(
        f"colour map {name!r} is not one OpenCV publishes: this project draws a map with "
        f"the COLORMAP_* constants cv2 declares, which are {published}. Fix: set "
        f"gradcam.colormap in configs/default.yaml to one of those names"
    )


def overlay(
    image: np.ndarray, heatmap: np.ndarray, alpha: float, colormap: str = "jet"
) -> np.ndarray:
    """Return `image` with the map drawn over it, as a `float32` array in `[0, 1]`.

    **The caller must supply an image of the map's own shape and a map normalized into
    `[0, 1]`** — the array `generate` returns, whose maximum is strictly below one by the
    epsilon in its denominator, so `255 * heatmap` cannot wrap round a `uint8`. An image
    of another shape is refused by numpy's broadcasting with a message about shapes,
    which is the fact a caller needs. **An image outside `[0, 1]` is clipped rather than
    refused**: a caller handing over a `uint8` array as floats has made a scale mistake
    this cannot detect, and a picture over-bright by its top end is a better answer than
    no picture.

    **`alpha` is the heat map's weight and is refused outside `[0, 1]` rather than
    clamped**, because `alpha=1.4` blends in a colour the caller did not ask for at a
    weight they did not ask for. One predicate refuses `-0.1`, `1.1`, `nan` and `inf`
    alike (measured: `0.0 <= alpha <= 1.0` is `False` for all four), and the two ends are
    exactly the image and exactly the colour map, which is what makes the blend checkable:
    measured, `overlay(image, heat, 0.0)` is the image bitwise and `overlay(image, heat,
    1.0)` is the colour map.

    **The colour map is applied through OpenCV, which returns BGR, and is converted to
    RGB before the blend** — a JET map left in BGR puts the model's hot region in the
    blue channel, which is a figure that looks wrong rather than one that fails. The
    default is the shipped `gradcam.colormap`, and the result is clipped to `[0, 1]` and
    cast to `float32` so a caller stacking overlays does not accumulate a `float64` copy
    of a picture.
    """
    weight = float(alpha)
    if not (0.0 <= weight <= 1.0):
        raise ValueError(
            f"alpha is {alpha!r}, which is not a weight of a heat map: alpha blends the "
            f"colour map into the image as weight * colour + (1 - weight) * image, so it "
            f"has to be between 0 and 1 inclusive — 0 is the image unchanged, 1 is the "
            f"colour map alone — and not -0.1, not 1.1, not nan and not inf. Fix: pass a "
            f"weight in [0, 1], or set gradcam.alpha in configs/default.yaml, which is "
            f"0.4"
        )
    import cv2

    coloured = cv2.applyColorMap(
        np.uint8(255 * np.asarray(heatmap, dtype=np.float32)), _colour_map(colormap)
    )
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blended = weight * coloured + (1.0 - weight) * np.asarray(image, dtype=np.float32)
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def overlay_from_config(image: np.ndarray, heatmap: np.ndarray, cfg: Config) -> np.ndarray:
    """Return `overlay` at the `gradcam.alpha` and `gradcam.colormap` `cfg` declares.

    **`gradcam.alpha` and `gradcam.colormap` are read here and not in `bin/gradcam.py`**,
    for the reason the module docstring gives: a command line is a script loaded by path
    rather than a package, so a read there is outside the source scan in
    `tests/test_config.py` and the key is then reported as one nothing under `blobcount/`
    reads. The values are used, not acknowledged — the two keys are the whole difference
    between this call and `overlay(image, heatmap, 0.4, colormap="jet")`, and
    `tests/test_interpret.py` holds both by setting `gradcam.alpha=0` and by asking for a
    second map.

    `alpha` arrives as a `float` and `colormap` as a `str`, both converted at the read,
    so the value `overlay` validates is the configured one and not a string the
    configuration happened to parse into.
    """
    return overlay(
        image,
        heatmap,
        float(cfg.get("gradcam.alpha")),
        colormap=str(cfg.get("gradcam.colormap")),
    )
