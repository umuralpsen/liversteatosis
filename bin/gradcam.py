"""Draw a Grad-CAM figure per sample of each class, and one grid of them all.

The seventh entry point of the rebuild, after extraction, labeling, the threshold sweep,
training, evaluation and prediction. It reads the `labels.csv` a previous `label_patches`
run wrote, builds the backbone the configuration names, loads a checkpoint into it, and
writes one figure per sampled patch plus a summary grid into `results/gradcam/`.

    python bin/gradcam.py --labels dataset/labels/labels.csv --slide 2 \
                          --checkpoint results/best_model_fold0.pth --n 4

**A figure is a claim about where the model looked, and one drawn from a model that has
never been trained is a claim about nothing.** `--slide` and `--checkpoint` have never
been given real input: `dataset/labels/` does not exist, the extraction that would fill it
has never been run, and no checkpoint in this project has ever been trained. The
`results/gradcam/` this program writes into holds the legacy `gradcam.py`'s figures today,
made against the weights Task 14 deletes, and this run overwrites the names it uses
rather than refusing the directory — a previous run's figure of a measurement is not
something to refuse, which is the reason `blobcount.ablation.plot` overwrites too.

**The samples are the first `--n` rows of each class in the handoff's own order, and not a
random draw.** The legacy drew four at random per class (`gradcam.py:210`), which is a
figure a second run cannot reproduce and a reader cannot compare with the one from the
last run; `project.seed` would then be doing the work of an index. **The handoff's own row
order is enough to make a run reproducible** — it is fixed by the file the run was given —
so nothing here sorts, filters or draws: the first `n` rows of each class, in the order
`read_labels` returned them. The same reasoning is why `bin/evaluate.py` sorts its cohort
and `bin/predict.py` sorts its directory: the row order of an artifact has to mean the
same thing on the next run of the pipeline.

**A figure is three panels — the patch, the map, and the blend — and the third carries the
prediction the map was made for.** The map is `blobcount.interpret`'s, and the blend is
`overlay_from_config`, so `gradcam.alpha` and `gradcam.colormap` are read under
`blobcount/` where the source scan in `tests/test_config.py` can see them. The panels are
drawn with `matplotlib.use("Agg")` before `pyplot` is imported and every figure is closed
before the next one is opened, for the reasons `blobcount.ablation.plot` gives: a run
writes files and shows no window, and matplotlib counts the figures a process holds open
and warns past twenty of them, which `filterwarnings = ["error"]` turns into a failed run.

**`--labels` is required and `--slide` names a specimen, because a per-class figure needs
a class and the class is what the handoff holds.** A `slide_id` in this project is a
specimen name — `blobcount.labeling.LABEL_COLUMNS` writes it beside every patch and
`blobcount.splits` splits by it — so `--slide 2` is the rows of that specimen and not a
path to a `.svs`. The handoff is read by `blobcount.train.read_labels` and not here, for
the reason `bin/evaluate.py` gives: a copy of the CSV contract in this file would be a
second set of answers to it, and a read of `paths.labels` written into a command line is
outside the source scan in `tests/test_config.py`. **Its precondition is inherited and not
restated as a rule here: a handoff from `bin/label_patches.py --out somewhere` carries
paths relative to *that* directory**, so every row resolves against the wrong root and the
run stops on the first patch with a missing-file `OSError` naming a path that looks right
and is not — refused here rather than traced back, with the same message.

**The patches are read with `blobcount.labeling.load_patch`, which refuses anything that
is not a `uint8 (extraction.patch_size, extraction.patch_size, 3)` array.** This program's
input is a labelling run's output, so the extraction's own rule about what a patch is
holds here and does not need a second definition; `bin/predict.py` reads its caller's
images with `PIL` instead because it has no such provenance to check.

**The artifacts go to `paths.results / "gradcam"`, and nothing here is cleared.** One
figure per sampled patch, named `gradcam_<class>_<n>.png` from the class it was labelled
and its position in that class's rows, and one `gradcam_summary.png` holding the map and
the blend of every sample drawn. A file of the same name is rewritten in place.

**Refusals, in four groups.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the argument, the
configuration or the input rather than about this program:

- **A configuration**, as a `ConfigError`, and this is the one arm that carries two: the
  file `load` refuses, and a `paths.results` whose *value* `Config.path` refuses because
  it is empty. `paths.results` is resolved inside the arm that catches this, for the
  reason `bin/evaluate.py` gives.
- **The output target**, as this program's own check, against both `paths.results` and
  the `gradcam` subdirectory it puts the figures in: a target that exists and is a file
  rather than a directory, and a target that is not below a root this project owns — the
  checkout, or the parent of a configured `paths.*` value.
  **Both paths are tested because both can be the one that is wrong** — a `paths.results`
  that is a regular file makes `results/gradcam` un-creatable, and a `paths.results: .`
  leaves `gradcam/` below the root while putting the results directory *at* it, so a
  guard on the subdirectory alone would write figures into the repository. The predicate
  is `blobcount.extract.is_owned_path`, which follows the configuration rather than the
  checkout alone — a path the configuration names is by definition one this project owns,
  and a rule written against the checkout alone would refuse the destinations the data
  move put outside the repository — and the check runs before the handoff is read and
  before a checkpoint is loaded.
- **The cohort and the count**, as this program's own checks: a `--slide` naming a
  specimen the handoff holds no row of, and a `--n` below one. The first would draw a
  figure of nothing and report a slide it never looked at; the second would write a
  summary grid with no row in it. `argparse` types `--n` as an `int`, so this is a value
  question and not a type one, and it is answered here rather than by the parser.
- **The handoff, the checkpoint and the backbone**, as the groups `bin/evaluate.py` and
  `bin/predict.py` name: `read_labels`' `ConfigError`, `OSError`, `csv.Error` or
  `ValueError`; a patch that is not the array `load_patch` requires; and a checkpoint
  that is not there, not a torch archive, or not one that fits the head — with
  `blobcount.models.build_model`'s own `ValueError` for an architecture this project does
  not build or a class count no head can be built for.

All four groups are printed on stderr behind this program's name, and none of them is a
traceback. **There is no `EXIT_CONTRADICTORY` here**, for the reason `bin/evaluate.py`
states for its own: a figure run has no state a run *completes* with that is a
contradiction rather than a finding, so a status of 1 means the interpreter reported a
traceback, which is a fact about this project's code rather than about the study.

**A bad argument is not one of the four groups, and not one of this program's outcomes at
all.** `argparse` refuses an option this program does not declare, one given no value, a
run given no `--labels` or no `--checkpoint`, a `--slide` given no specimen, and a `--n`
that is not a whole number, by printing its own message and calling `sys.exit(2)` inside
`parse_args`, before the first statement of `main` runs. The parser's status is the same
number as `EXIT_REFUSED` and no code can tell the two apart, so the separation is on
stderr, where the parser names the argument and its usage and none of the four groups names
either.

**One table is printed — one row per class, how many rows of it the slide held and how
many were drawn — and a closing sentence naming the summary grid.** **A class the slide
holds no row of is printed with a zero rather than dropped**, because a slide with no
patch of one class is a fact about the slide and the figure beside it would otherwise be
the only place a reader could learn it.

**A figure drawn here is a visualisation of a measurement, not a measurement.** It says
where the model attended for one class; it says nothing about whether that attention was
right, and no figure of a model that has never been trained is evidence of anything.
"""

from __future__ import annotations

import argparse
import csv
import logging
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from blobcount.config import PROJECT_ROOT, ConfigError, get_device, load
from blobcount.data import build_transforms
from blobcount.extract import is_owned_path
from blobcount.interpret import GradCAM, overlay_from_config
from blobcount.labeling import CLASS_NAMES, LABEL_BY_CLASS, load_patch
from blobcount.models import build_model
from blobcount.train import read_labels

EXIT_OK = 0
EXIT_REFUSED = 2

# Where the figures go and what they are called, under `paths.results`. One place, so the
# writers and the tests that read them back name the same files.
_GRADCAM_DIR = "gradcam"
_SUMMARY_NAME = "gradcam_summary.png"

# How many samples of each class a run draws when `--n` is not given, which is the four
# the legacy `gradcam.py:210` drew at random.
_DEFAULT_N = 4

# The columns of the printed table. The figures are named after the class and the row's
# position in it, so a reader can match a figure to the row it came from.
_TABLE_COLUMNS = ("class", "rows", "drawn", "first figure")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradcam",
        description="Draw a Grad-CAM figure per sample of each class of one slide, and a "
        "summary grid of them.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        type=Path,
        metavar="PATH",
        help="the labels.csv written by bin/label_patches.py",
    )
    parser.add_argument(
        "--slide",
        required=True,
        metavar="ID",
        help="the specimen to draw, as the slide_id the handoff records: rows of no other "
        "slide are read, and a slide the handoff holds no row of is refused",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="PATH",
        help="the weights written by bin/train.py, one results/best_model_fold<n>.pth",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N,
        metavar="N",
        help=f"how many patches of each class to draw, at most; {_DEFAULT_N} otherwise. "
        f"The first N rows of the class in the handoff's order, not a random sample",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/evaluate.py` gives: this is four columns of a
    name and three integers, and `tabulate` is not a dependency this project declares.
    The siblings' `_print_table` is not imported, for the reason
    `bin/ablate_threshold.py` states for its own: `bin/` is a directory of scripts loaded
    by path rather than a package, and reaching into one script's private name would
    couple two entry points that are meant to be read one at a time.
    """
    widths = [
        max([len(str(header))] + [len(str(row[index])) for row in rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _refuse_output_target(results_dir: Path, target: Path) -> str | None:
    """Return why this program will not write its figures under `results_dir`, or None.

    **The same two refusals the two siblings make, against both paths, for the same
    reason.** `bin/ablate_threshold.py` and `bin/evaluate.py` both test the results
    directory and the subdirectory they put their artifacts in, and this one writes five
    figures and a grid into a subdirectory of a directory it does not own. The file test
    runs before the containment test for each path, so a file is reported as the typo it
    is rather than as a containment failure. The containment predicate is
    `blobcount.extract.is_owned_path`, the one the siblings use, and it follows the
    configuration rather than the checkout alone: a path the configuration names is by
    definition one this project owns, and a rule written against the checkout alone
    would refuse the destinations the data move put outside the repository.

    The check returns a message and raises nothing, and it runs before the handoff is
    read and before a checkpoint is loaded, so a mistyped `paths.results` costs a message
    rather than a run.
    """
    for path in (results_dir, target):
        if path.exists() and not path.is_dir():
            return (
                f"refusing to write {path}: it exists and is a file rather than a "
                f"directory, so it is a typo in paths.results and not a previous run"
            )
        if not is_owned_path(path):
            return (
                f"refusing to write {path}: it is not strictly below the project root "
                f"{PROJECT_ROOT} or the parent of a configured paths.* value, so it is "
                f"not a directory this project owns"
            )
    return None


def _load_weights(model, checkpoint: Path) -> None:
    """Load `checkpoint` into `model`, naming the file in a refusal that does not.

    **The same `pickle.UnpicklingError` goes out as the one torch raised**, for the reason
    `bin/evaluate.py` gives: that type is what `main`'s arm catches and a second type
    would be a second answer to what a checkpoint is; only the message is added, because
    the torch message names its own policy and not the path the caller typed.

    **This body is a verbatim third copy, kept rather than shared, and the reason is the
    one `_print_table` above states for its own five.** `bin/` is a directory of scripts
    loaded by path rather than a package: importing a sibling's private name would mean
    reaching into another entry point by path, and `bin/` exists so that each of its files
    can be read one at a time without having read the others. **The fix would be to move
    this into `blobcount/`, where `blobcount.extract.is_owned_path` is the precedent for a
    predicate every entry point shares** — not to import it across `bin/`, which the
    `_print_table` docstring rules out and which this one follows.

    The cost of keeping it is real and is what the tests are for: if a later change to the
    message or the exception lands here and not in the other two copies, each copy is
    covered on its own —
    `tests/test_interpret.py::test_gradcam_refuses_a_checkpoint_it_cannot_load` for this
    one and `tests/test_evaluate.py::test_a_checkpoint_that_is_not_a_torch_file_is_refused`
    for `bin/evaluate.py`'s — so a copy that drifts is a red test in the file that owns it
    rather than a silent difference between two entry points.
    """
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise pickle.UnpicklingError(
            f"{checkpoint} is not a torch archive torch can read into a model: {error}"
        ) from error
    model.load_state_dict(state)


def _target_layer(model):
    """Return the last block of `model`'s last convolutional stage.

    **A ResNet's is `layer4[-1]` and an EfficientNet's is `features[-1]`, and both names
    were measured on this stack** rather than recalled: a resnet18 and a resnet50 have no
    `features` attribute at all, an `efficientnet_b0` has no `layer4`, and the two last
    stages hold `(1, 512, 7, 7)`, `(1, 2048, 7, 7)` and `(1, 1280, 7, 7)` respectively
    for a `224`-square input — which is what makes any one of them a layer a Grad-CAM over
    a patch can be made from. **The names are written here rather than read from
    `blobcount.models`, whose dispatch declares where a *head* lives and says nothing about
    where the last *stage* is**, and adding a column to it would be a change to a module
    this task does not touch.

    A model with neither attribute is a `ValueError` naming them, because a Grad-CAM over
    no layer at all is a map of something other than what this program says it is a map of.
    """
    layer = getattr(model, "layer4", None)
    if layer is None:
        stage = getattr(model, "features", None)
        if stage is None:
            raise ValueError(
                f"this project draws Grad-CAM on resnet18, resnet50 and efficientnet_b0 "
                f"(blobcount.models.SUPPORTED), and {type(model).__name__} has neither a "
                f"layer4 nor a features to take its last block from. Fix: set "
                f"training.architecture in configs/default.yaml to one of those three, "
                f"and load the checkpoint into the model it names"
            )
        layer = stage
    return layer[-1]


def _samples(
    paths: Sequence[Path],
    labels: Sequence[int],
    specimens: Sequence[str],
    slide: str,
    n: int,
):
    """Return `(class name, rows held, rows drawn)` per class, in `CLASS_NAMES` order.

    **The first `n` rows of each class in the order `read_labels` returned them**, and not
    a random sample of them, for the reason the module docstring gives: a figure a second
    run cannot reproduce is a figure a reader cannot compare against the last one.

    **A class the slide holds no row of is present with an empty list rather than
    missing**, and `rows held` is the count *before* the `n` is applied, so the printed
    table can say `4 rows, 2 drawn` and `0 rows, 0 drawn` — a slide with no patches of a
    class is a fact about the slide and silence about it would be the alternative.
    """
    chosen = []
    for name in CLASS_NAMES:
        rows = [
            path
            for path, label, specimen in zip(paths, labels, specimens, strict=True)
            if specimen == slide and label == LABEL_BY_CLASS[name]
        ]
        chosen.append((name, len(rows), rows[:n]))
    return chosen


def _read_patches(samples, patch_size: int) -> list[tuple[str, int, Path, np.ndarray]]:
    """Return `(class, position, path, pixels)` per drawn sample, with every patch read
    first.

    **Every patch is read before any map is made**, so the arm in `main` that catches the
    `ValueError` `load_patch` raises is around this one call and cannot hold a defect in
    the drawing. That is the same split `bin/ablate_threshold.py` makes: count first,
    draw second, so a file it cannot read leaves the previous run's figures where they
    are rather than half-replaced by a new one.

    `position` is the row's place among the drawn rows of its own class, so the figure
    named after it is the same figure on the next run over the same handoff.
    """
    return [
        (name, position + 1, path, load_patch(path, patch_size))
        for name, _held, rows in samples
        for position, path in enumerate(rows)
    ]


def _figure(path: Path, pixels: np.ndarray, heatmap: np.ndarray, blended: np.ndarray,
            title: str) -> None:
    """Write the three-panel figure for one sample and close it.

    **The panels are the patch, the map and the blend, at one size and not three**, so a
    reader comparing them is comparing the same field of view: `main` resizes the patch to
    the size of the tensor it was transformed into, which is the size `generate` resized
    the map to. The colour bar carries the map's scale, which is what makes a map of one
    class readable against a map of the other: both are normalized to `[0, 1]` over their
    own range, so a brighter picture is not evidence of a stronger answer.

    `matplotlib.use("Agg")` is called before `pyplot` is imported, and the figure is
    closed before returning, for the reasons `blobcount.ablation.plot` gives.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (original, heat, superimposed) = plt.subplots(1, 3, figsize=(9.0, 3.2))
    original.imshow(pixels)
    original.set_title("patch")
    original.axis("off")
    drawn = heat.imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
    heat.set_title("Grad-CAM")
    heat.axis("off")
    figure.colorbar(drawn, ax=heat, fraction=0.046, pad=0.04)
    superimposed.imshow(blended)
    superimposed.set_title(title)
    superimposed.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _summary(path: Path, drawn: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> None:
    """Write the grid of every sample drawn: the map beside its blend, one row each.

    **One row per sample and no title per cell**, because the per-sample figure already
    carries the prediction and this grid's question is the comparison between samples: how
    much of the patch each class's map covers, and whether the two ends of one slide look
    alike. A file of the same name is overwritten; nothing here is cleared.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # `squeeze=False` because a one-row grid is a `(1, 2)` array of axes and not a pair of
    # them: `plt.subplots(1, 2)` returns two `Axes` and `axes[row, column]` on those is an
    # `IndexError`, and `--n 1` is a run this program allows.
    figure, axes = plt.subplots(
        len(drawn), 2, figsize=(6.4, 3.2 * len(drawn)), squeeze=False
    )
    for row, (name, heatmap, blended) in enumerate(drawn):
        axes[row, 0].imshow(heatmap, cmap="jet", vmin=0.0, vmax=1.0)
        axes[row, 0].set_ylabel(name)
        axes[row, 1].imshow(blended)
        for column in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    for column, heading in enumerate(("Grad-CAM", "blend")):
        axes[0, column].set_title(heading)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Grad-CAM figures the arguments name and return the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        # Resolved inside the arm, for the reason `bin/evaluate.py` gives: `load` types
        # this key as the declared `str`, and an empty one is a value it makes no claim
        # about, so `Config.path` is what refuses it.
        results_dir: Path = cfg.path("paths.results")
        # Read here rather than on its own arm: `get_device` is `project.device`'s one
        # reader and raises `ConfigError` for a value that is neither `auto`, `cpu` nor
        # `cuda`, which is an answer about the configuration.
        device = torch.device(get_device(cfg))
    except ConfigError as error:
        print(f"gradcam: {error}", file=sys.stderr)
        return EXIT_REFUSED

    target = results_dir / _GRADCAM_DIR
    refused = _refuse_output_target(results_dir, target)
    if refused is not None:
        # Before the handoff is read and before a checkpoint is loaded, so a mistyped
        # `paths.results` costs a message rather than a run. `EXIT_REFUSED` and not a
        # drawing failure: nothing here finished.
        print(f"gradcam: {refused}", file=sys.stderr)
        return EXIT_REFUSED

    if args.n < 1:
        print(
            # **The remedy names values this run accepts and never the one it refused.**
            # `--n` is one whole number of patches to draw per class, so `1` is the least
            # it can be and leaving the argument off is `_DEFAULT_N`; the refused value is
            # named in the first clause, where it is the thing that was refused. The line
            # this replaced interpolated `args.n` into the remedy as well, so `--n 0` was
            # told `Fix: pass a positive --n — 0 draws one of each`, and a caller following
            # the fix passed the argument they had just been refused.
            f"gradcam: --n is {args.n}, and a run draws at least one patch of each class "
            f"or it draws nothing: a summary grid of no rows reports no map at all. Fix: "
            f"pass --n 1 to draw one patch of each class, or --n 2, or leave --n off "
            f"altogether for {_DEFAULT_N}",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        paths, labels, specimens = read_labels(args.labels, cfg)
    except (ConfigError, OSError, csv.Error, ValueError) as error:
        # The handoff, as the module docstring groups it: a file that is not there or
        # cannot be opened, a cell longer than `csv.field_size_limit`, a header that is
        # not the handoff's, a row with no cell for a column this run reads, a `label`
        # that is not a whole number, and a `path` that leaves the label root.
        print(f"gradcam: {error}", file=sys.stderr)
        return EXIT_REFUSED

    held = sorted({specimen for specimen in specimens})
    if args.slide not in held:
        print(
            f"gradcam: no row of {args.labels} carries slide_id {args.slide!r}, so there "
            f"is no patch of that slide to draw: a figure of nothing beside a slide name "
            f"is a claim about a specimen this run never read. The handoff holds "
            f"{', '.join(held)} — draw one of those, or re-run the labelling over a cohort "
            f"that holds the slide",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    try:
        model = build_model(
            cfg.get("training.architecture"),
            int(cfg.get("training.num_classes")),
            # `False` whatever the configuration says, for the reason the module docstring
            # gives: the checkpoint loaded below replaces every parameter.
            False,
        )
        _load_weights(model, args.checkpoint)
        layer = _target_layer(model)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        # The checkpoint, the backbone and the layer, as the module docstring groups them.
        # The catch is around these calls, so it is too narrow to hold a defect in the
        # drawing below, and a `ValueError` from there reaches the interpreter as a
        # traceback.
        print(f"gradcam: {error}", file=sys.stderr)
        return EXIT_REFUSED

    samples = _samples(paths, labels, specimens, args.slide, args.n)
    try:
        patches = _read_patches(samples, int(cfg.get("extraction.patch_size")))
    except (OSError, ValueError) as error:
        # A patch the index names that is not there, or that is not the `uint8` array the
        # extraction wrote. Both are refusals about the input rather than about the
        # drawing, which has not started.
        print(f"gradcam: {error}", file=sys.stderr)
        return EXIT_REFUSED
    if not patches:
        print(
            f"gradcam: slide {args.slide} holds no patch of the class this project labels "
            f"at {', '.join(CLASS_NAMES)}, so there is no figure to draw",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    # Everything below is the run itself. A `ValueError` from the arithmetic of the map,
    # from matplotlib, or from a failed write is deliberately not caught: that is a
    # defect in this project and it reaches the interpreter as a traceback.
    #
    # `PIL` is imported here and not at module scope, for the reason
    # `blobcount.labeling.load_patch` gives: a native wheel at module scope turns an
    # absent one into a collection error for every test in the suite, including the ones
    # that touch no image.
    from PIL import Image

    model = model.to(device)
    _train_transform, transform = build_transforms(cfg)
    cam = GradCAM(model, layer)
    target.mkdir(parents=True, exist_ok=True)
    drawn_rows: list[tuple[str, np.ndarray, np.ndarray]] = []
    try:
        for name, position, path, pixels in patches:
            with Image.fromarray(pixels) as image:
                tensor = transform(image).unsqueeze(0).to(device)
                heatmap, predicted, logits = cam.generate(tensor)
                # **The panels show the patch at the size the model saw it**, and the size
                # is read off the tensor the transform produced rather than read from
                # `training.input_size` a second time, so the two cannot disagree. The
                # resize is PIL's bilinear: measured, `Image.resize((side, side),
                # BILINEAR)` followed by the ImageNet normalisation reproduces the
                # transform's own tensor to within `1.2e-7`, which is float rounding and
                # not a different rendering. The legacy did the same at 224
                # (`gradcam.py:121`), and a map at the model's size over a patch at the
                # extraction's size is a map of a region that is not there.
                side = int(tensor.shape[-1])
                shown = (
                    np.asarray(image.resize((side, side), Image.BILINEAR), dtype=np.float32)
                    / 255.0
                )
                # `.detach()` before the conversion, and not after it: the logits are the
                # forward pass's own tensor, and `float()` on one that still carries a
                # graph raises `UserWarning: Converting a tensor with requires_grad=True
                # to a scalar may lead to unexpected behavior` (measured), which
                # `filterwarnings = ["error"]` in `pyproject.toml` turns into a failed run.
                probability = float(torch.softmax(logits, dim=1)[0, predicted].detach())
            blended = overlay_from_config(shown, heatmap, cfg)
            figure_name = f"gradcam_{name}_{position}.png"
            _figure(
                target / figure_name,
                shown,
                heatmap,
                blended,
                f"{CLASS_NAMES[predicted]} ({probability:.1%})",
            )
            drawn_rows.append((name, heatmap, blended))
            print(
                f"  {name:>10}  {path.name}  ->  {CLASS_NAMES[predicted]} "
                f"({probability:.1%})  ->  {target / figure_name}"
            )
        _summary(target / _SUMMARY_NAME, drawn_rows)
    finally:
        # The hooks are released whether the drawing finished or raised: a figure run that
        # left them behind would keep capturing into a model the caller may go on to use,
        # and `remove()` is safe here even if something below already released them.
        cam.remove()

    rows_of_slide = sum(1 for specimen in specimens if specimen == args.slide)
    _print_table(
        _TABLE_COLUMNS,
        [
            [
                name,
                held,
                sum(1 for drawn in drawn_rows if drawn[0] == name),
                f"gradcam_{name}_1.png" if any(drawn[0] == name for drawn in drawn_rows) else "",
            ]
            for name, held, _rows in samples
        ],
    )
    print(
        f"\n{len(drawn_rows)} figure(s) drawn from {rows_of_slide} row(s) of slide "
        f"{args.slide} in {args.labels} with {args.checkpoint}; every sample is a patch of "
        f"extraction.patch_size rendered by the eval transform, and the first {args.n} "
        f"rows of each class are the ones taken, in the handoff's own order. The grid of "
        f"them is {target / _SUMMARY_NAME}. A figure says where the model attended for one "
        f"class and nothing about whether that attention was right; these figures are a "
        f"function of that checkpoint and of the handoff this run was pointed at"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
