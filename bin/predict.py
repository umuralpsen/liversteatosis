"""Predict on one image or a directory of them and write the rows as a CSV.

The sixth entry point of the rebuild, after extraction, labeling, the threshold sweep,
training and evaluation. It builds the backbone the configuration names, loads a
checkpoint into it, runs the eval transform over one image or over every image in a
directory, and writes `results/predictions.csv` with one row per image.

    python bin/predict.py --image dataset/patches/normal_0.png \
                          --checkpoint results/best_model_fold0.pth
    python bin/predict.py --dir dataset/patches --checkpoint results/best_model_fold0.pth

**`results/` is created when it is absent, and that is the crash the legacy
`test_model.py` had.** The legacy wrote `results/test_predictions.png` and never created
the directory it was writing into: measured here, `Figure.savefig` into a directory that
does not exist raises `FileNotFoundError: [Errno 2] No such file or directory:
'.../results/test_predictions.png'`, naming the path and nothing about which run made it.
So the directory is created here, before the first row is written and after the guard
below has said the target is one this project owns — a write is a write wherever it lands
and the guard is about *where*, not about whether.

**The output target is guarded before an image is opened and before the checkpoint is
read**, with the same two refusals its sibling entry points make and the same predicate,
`blobcount.extract.is_owned_path`: a target that exists and is a file rather than a directory,
and a target that is not below a root this project owns — the checkout, or the parent of a
configured `paths.*` value. `paths.results` is a hand-edited
string in `configs/default.yaml` and this program writes a row per image into it, so a
`paths.results: /` is how a study's predictions end up somewhere this repository is not
responsible for. **Both refusals are the whole of the guard**, and a file at
`paths.results` is reported as the typo it is before the containment test runs for it.

**`--image` and `--dir` are one argument in two spellings, and the parser refuses
neither, both or an unknown option itself.** They are a mutually exclusive group marked
required, so `argparse` prints its own message and calls `sys.exit(2)` inside `parse_args`
before the first statement of `main` runs. Its status is the same number as
`EXIT_REFUSED` and no code can tell the two apart, so the separation is on stderr, where
the parser names the argument and its usage and none of the four groups below names
either.

**Refusals, in four groups.** A run that completed is `EXIT_OK = 0`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the argument, the
configuration or the input rather than about this program:
- **A configuration**, as a `ConfigError`, and this is the one arm that carries two: the
  file `load` refuses — missing, unreadable, not a mapping, a key the code does not
  declare, a required key the file omits, or a value for a declared key whose type this
  project does not read it as — and a `paths.results` whose *value* `Config.path` refuses
  because it is empty. `paths.results` is resolved inside the arm that catches this, for
  the reason `bin/evaluate.py` gives.
- **The output target**, as this program's own check, on `paths.results` alone: this
  program writes one file into the results directory itself rather than into a
  subdirectory of it, which is the single-path shape `bin/train.py` already guards. A
  `paths.results: .` is refused here for the same reason it is refused there — the
  directory would be the project root itself, which `is_owned_path` answers rather than a
  string comparison.
- **The checkpoint and the backbone**, as an `OSError`, a `pickle.UnpicklingError`, a
  `RuntimeError` or a `ValueError`: a checkpoint that is not there; one that is not a
  torch archive, which `torch.load` refuses with `UnpicklingError` — measured, its bases
  are `PickleError, Exception, BaseException, object`, so it is neither an `OSError` nor
  a `ValueError`; one whose weights do not fit the head the configuration describes, which
  `load_state_dict` refuses with `RuntimeError: Error(s) in loading state_dict ... size
  mismatch`; and a `training.architecture` this project does not build, or a
  `training.num_classes` no head can be built for, which is
  `blobcount.models.build_model`'s own `ValueError`.
- **The images**, as refusals of this program's own: a `--image` naming something that is
  not a file, a `--dir` naming something that is not a directory, and a `--dir` holding no
  file this program reads. Which argument the caller used decides which kind of path is
  right, so the two are refused rather than read as the other.
- **The specimen filter**, as two more refusals of this program's own: a `--specimen` with
  no `--labels`, where there is nothing to match against and a filter that matched nothing
  would write a header and no rows; and a `--specimen` the handoff records no patch under,
  which is the same shape of nothing reported as a prediction. Both are about the argument,
  and both arrive before a checkpoint is loaded. **A handoff that cannot be read is a
  fourth**, and it is not one of this program's own: it is the `ConfigError`, `OSError`,
  `csv.Error` or `ValueError` out of `blobcount.train.read_labels`, grouped exactly as
  `bin/evaluate.py` groups the same reader's.

**An image that cannot be opened is not one of the four, and is deliberate.** `PIL`
raises an `OSError` — measured, `PIL.UnidentifiedImageError` subclasses `OSError` — and it
names the full resolved path, which is the whole fact a caller needs and better than a
sentence about a rule. **The CSV is written after the last image is scored, so a run that
could not open one writes no file at all** and the previous run's `predictions.csv` stays
where it was; a reader is never left with a header and a partial set of rows.
**A file of the same name is rewritten in place, and the file holds the last run's rows
and not an accumulation of them.** Two runs over two directories leave the second one's
rows, which is the same rule `blobcount.ablation.plot` follows for a figure: this is a
report of a measurement, and a previous run's report of a different measurement is not
something to merge into or to refuse. **The printed table is what says which run wrote
the file**, and it is rendered from the same rows the file is written from.

**The backbone is built with `pretrained=false` whatever the configuration says**, for
the reason `bin/evaluate.py` gives: a checkpoint replaces every parameter of the model a
moment later, so honouring `training.pretrained: true` would download the ImageNet
weights of a backbone whose weights are about to be overwritten, over the network, on
every run.

**The pixels are read with `PIL` and not with `blobcount.labeling.load_patch`, and the two
answer different questions.** `load_patch` refuses anything that is not a
`uint8 (extraction.patch_size, extraction.patch_size, 3)` array, which is the right rule
for the extraction's own tree and the wrong one for a caller's image file: a patch this
pipeline did not write, resized by whoever handed it over, is still an image somebody
asked about. Every patch *this pipeline* wrote is read by `load_patch`, which is why
`bin/gradcam.py` — which has no choice, its input is a labelling run's output — uses it
and this one does not.

**The rows are `sorted()` by path, and the CSV says which class each image was called.**
The row order of an artifact has to mean the same thing on the next run of the pipeline,
and `glob` over a directory returns whatever the filesystem hands back; the class column
is `blobcount.labeling.CLASS_NAMES[class_index]`, read from the labelling run's own table
rather than written out a second time in this file. The probability columns are named
`p_<index>` from the width of the logits rather than from a count of classes written here,
so a head of a different width produces a CSV whose columns describe it.

**Every row carries the specimen it came from and whether the checkpoint was fitted on
it, and this is the second thing to change about this program's output.** The measured
defect it fixes: `make predict` pointed `--dir` at the steatosis directory of all three
admitted specimens while the fold-0 checkpoint was fitted on two of them, so 3079 of the
3502 committed rows were specimens the model had already seen — resubstitution, written
into a tracked artifact, and the CSV named no checkpoint, no specimen and nothing else.
`--specimen` filters the run to the specimen the fold held out, `--train-specimens` says
which cohort the checkpoint saw, and the two columns make the file self-describing for
anyone who joins it back to `labels.csv` whether or not either was given.

**`labels.csv` is the authority on which specimen a patch came from, and the directory
layout says nothing.** The labelling stage writes `normal/<patch_id>.png` and
`steatosis/<patch_id>.png` and names no specimen in any directory name, so `--specimen` is
a join against the handoff and not a walk of the tree — which is why it requires `--labels`
and refuses without it rather than matching nothing. The handoff is read through
`blobcount.train.read_labels`, the package's one reader, for the reason the module
docstring of `bin/evaluate.py` gives about not copying the CSV contract into `bin/`; the
cost is that every row resolves against the configured `paths.labels`, so a handoff written
with `--out somewhere` resolves against the wrong root and is refused.

**A row the handoff does not describe carries an empty `specimen` and `trained_on` false.**
This program is inference over an image or a directory and a caller's own picture belongs
to no specimen; the empty cell says exactly that, where a specimen guessed from a file name
would be an invention.

**One table is printed — one row per image — and a closing sentence that says what the
numbers are and where they were written.** The table is rendered out of the same rows the
CSV is written from, so the printed report and the artifact cannot disagree.
`tests/test_interpret.py` asserts the printed cells against the file.

**There is no `EXIT_CONTRADICTORY` here, and that is a decision rather than an omission.**
The two entry points that define `1` do so for a study-level failure — a slide that
yielded no patches, a labelled tree that contradicted itself — and both are states a run
*completes* with. A prediction over a directory a caller named is not one of them: the
run reports what the model said about the images it was given, and an image the model
calls normal that a clinician would not is the finding the run exists to show. So this
file defines two codes, and a status of 1 means the interpreter reported a traceback,
which is a fact about this project's code rather than about the study. `main`'s *return
value* is the contract, and an exception never returns one.

**Nothing here has been run against this study.** `dataset/labels/` does not exist, the
extraction that would fill it has never been run, and no checkpoint in this project has
ever been trained, so `--image`, `--dir` and `--checkpoint` have never been given real
input. Every run in this repository has been over synthetic PNGs written by
`tests/test_interpret.py` against a randomly initialised backbone.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

from blobcount.config import PROJECT_ROOT, Config, ConfigError, get_device, load, resolved
from blobcount.data import build_transforms
from blobcount.extract import is_owned_path
from blobcount.labeling import CLASS_NAMES
from blobcount.models import build_model
from blobcount.train import read_labels

EXIT_OK = 0
EXIT_REFUSED = 2

# Where the artifact goes and what it is called, under `paths.results`. One place, so the
# writer and the tests that read it back name the same file.
_PREDICTIONS_NAME = "predictions.csv"

# The sidecar carrying the provenance and the resolved parameters for the CSV, which
# cannot carry them inline: a new column shifts every row a reader indexes by position,
# and a leading `#` line becomes the *header* — `csv.DictReader` reads the first line as
# the header whatever it holds, so a comment line would demote the real header to a row
# and every lookup by name would return nothing.
_PROVENANCE_SIDECAR = "predictions.provenance.json"

# The sentence the artifacts carry about what their numbers are. **It names the two
# columns that decide how a row is read**, because a prediction file is the easiest of
# this project's artifacts to over-read: a row is the model's opinion about one patch,
# and whether that patch came from a specimen the model was fitted on is the difference
# between a measurement and a memorisation.
_PROVENANCE = (
    "written by bin/predict.py from the images this run was pointed at and the weights "
    "named by `checkpoint`, so every row is a function of those two inputs and of the "
    "resolved parameters under `config`. `specimen` is what the `labels` handoff records "
    "for the patch and is empty for an image the handoff does not describe; `trained_on` "
    "marks a row whose specimen the caller stated the checkpoint was fitted on, and such a "
    "row is resubstitution over a specimen the model has already seen and not a measurement "
    "of one it has not. The probabilities are the softmax of the head this checkpoint "
    "carries, and they are the model's confidence and not a measurement of anything"
)

# The columns every row carries before the probabilities. `specimen` and `trained_on` are
# new and they are the point of the column list: **a prediction file that does not say
# which specimen a row came from, or whether the checkpoint was fitted on it, is a file
# whose rows can be read as a measurement of anything.** Resampling over specimens is the
# correct unit for this project — patches from one slide are not independent observations
# — and the same reasoning applies here: a row scored on a specimen the model was trained
# on is a resubstitution number and not a result, and the file has to be able to say so
# without the reader re-running the pipeline to find out.
#
# The probabilities are named `p_<class index>` and appended after them, so the header
# describes the head the checkpoint actually has.
_COLUMNS = (
    "path",
    "specimen",
    "trained_on",
    "class_index",
    "predicted_class",
    "confidence",
)

# What a `--dir` is scanned for, as lower-case suffixes compared case-insensitively with
# `Path.suffix.lower()`, so a `.PNG` is found beside a `.png`: a directory scan needs a
# rule for which files in it are images, and this one is the extensions of the formats this
# project produces and receives. **It is a filter and not a test of what is inside** —
# whether a file is an image is `PIL.open`'s answer, and a file carrying one of these
# suffixes that is not one raises out of the read rather than being skipped, which is the
# refusal the module docstring states.
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Predict on one image or on every image in a directory and write "
        "results/predictions.csv.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image",
        type=Path,
        metavar="PATH",
        help="the single image to predict on",
    )
    source.add_argument(
        "--dir",
        type=Path,
        metavar="PATH",
        help="a directory to predict over; every file in it carrying one of "
        f"{', '.join(_IMAGE_SUFFIXES)}, in sorted order",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        metavar="PATH",
        help="the weights written by bin/train.py, one results/best_model_fold<n>.pth",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        metavar="PATH",
        help="the labels.csv written by bin/label_patches.py, which is the authority on "
        "which specimen each patch came from; required by --specimen and optional "
        "otherwise, since a caller's own image belongs to no specimen",
    )
    parser.add_argument(
        "--specimen",
        metavar="ID",
        help="score only the patches this handoff records under this specimen id. The "
        "label tree names no specimen in any directory - the layout is normal/ and "
        "steatosis/ and nothing else - so this filter is a join against labels.csv "
        "rather than a walk of the tree, which is why it needs --labels.",
    )
    parser.add_argument(
        "--train-specimens",
        action="append",
        default=None,
        metavar="ID",
        help="a specimen this checkpoint was fitted on; repeatable. Recorded in the "
        "trained_on column of every row, and stated rather than inferred for the same "
        "reason bin/evaluate.py requires it: a row scored over a trained specimen is "
        "resubstitution, and this program cannot know the cohort from a state dict.",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/evaluate.py` gives: this is four columns of a
    path, two short fields and a float, and `tabulate` is not a dependency this project
    declares. The siblings' `_print_table` is not imported, for the reason
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


def _refuse_output_target(results_dir: Path) -> str | None:
    """Return why this program will not write `predictions.csv` under `results_dir`.

    **The same two refusals `bin/evaluate.py` makes, on one path rather than two.** That
    program writes two files into `results/evaluation` and tests the results directory and
    the subdirectory; this one writes one file into the results directory itself, so
    there is no second path to test — the shape `bin/train.py` guards.

    **The file test runs before the containment test, so a file is reported as the typo
    it is.** A `paths.results` that is a regular file makes the `mkdir` below raise
    `NotADirectoryError`, and the project root itself is not strictly below itself, which
    `is_owned_path` answers rather than a string comparison here: "is this ours" has one
    implementation in this project and not one per guard. The roots it enforces are the
    ones the project owns — the checkout, plus the parent of every configured `paths.*`
    value, the set `blobcount.extract.allowed_roots` derives from the configuration at
    call time — because a path the configuration names is by definition one this project
    owns, and a rule written against the checkout alone would refuse the destinations
    the data move put outside the repository.

    The check returns a message and raises nothing, and it runs before an image is opened
    and before a checkpoint is loaded, so a mistyped `paths.results` costs a message rather
    than an inference pass.
    """
    if results_dir.exists() and not results_dir.is_dir():
        return (
            f"refusing to write {results_dir}: it exists and is a file rather than a "
            f"directory, so it is a typo in paths.results and not a previous run"
        )
    if not is_owned_path(results_dir):
        return (
            f"refusing to write {results_dir}: it is not strictly below the project root "
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

    The cost of keeping it is real and is what the tests are for:
    `tests/test_interpret.py::test_predict_refuses_a_checkpoint_it_cannot_load` covers this
    copy on its own, so one that drifts is a red test in the file that owns it rather than
    a silent difference between two entry points.
    """
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise pickle.UnpicklingError(
            f"{checkpoint} is not a torch archive torch can read into a model: {error}"
        ) from error
    model.load_state_dict(state)


def _images(named: Path, is_directory: bool) -> list[Path]:
    """Return the one image `--image` named, or every image in the directory `--dir` named.

    **Which argument the caller used decides which kind of path is right, and the wrong one
    is refused rather than read as the other.** A `--dir` naming a file is not a directory
    of one image; it is a path where one end of the argument is not spelled the way this
    program reads it, and answering it as a single image would be a run whose table says
    "1 image" for a directory the caller asked about. The two refusals are `--image` naming
    something that is not a file and `--dir` naming something that is not a directory.

    **A `--dir` holding no file this program reads is refused too**, rather than answered
    with nothing: a run that predicted nothing and wrote a CSV with a header and no rows is
    a report of nothing that a later reader cannot tell apart from a run over a directory
    of unreadable files.

    **`sorted()`, so the rows mean the same thing on the next run**, which is the same
    reason `bin/evaluate.py` sorts its cohort: a row order that depends on the filesystem
    is not a row order a second reader can reproduce.
    """
    if is_directory:
        if not named.is_dir():
            raise ValueError(
                f"--dir was given {named}, which is not a directory: this argument names a "
                f"directory to predict over, and a file under it is not a directory. Fix: "
                f"pass --image for a single file, or point --dir at the directory "
                f"bin/label_patches.py wrote its patches into"
            )
        found = sorted(
            item
            for item in named.iterdir()
            if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not found:
            raise ValueError(
                f"{named} holds no file carrying one of "
                f"{', '.join(_IMAGE_SUFFIXES)}, so there is nothing to predict on: a "
                f"prediction over an empty directory would write a CSV with a header and "
                f"no rows. Fix: point --dir at the directory bin/label_patches.py wrote "
                f"its patches into, which is paths.patches from the configuration"
            )
        return found
    if not named.is_file():
        raise ValueError(
            f"--image was given {named}, which is not a file: this argument names one "
            f"image, and a directory is not one. Fix: pass --dir for a directory, or check "
            f"the path --image was given"
        )
    return [named]


def _specimens_of(index: Path, cfg: Config) -> dict[str, str]:
    """Return `{patch file name: specimen id}` for every row of a handoff.

    **`labels.csv` is the authority and the directory layout is not.** The labelling stage
    writes `normal/<patch_id>.png` and `steatosis/<patch_id>.png` and names no specimen in
    any directory, so there is nothing in a tree walk to read a specimen from; the
    `slide_id` column is the only place the mapping exists. The key is the file name,
    which is the `patch_id` verbatim — the labelling stage writes patches under their own
    id — so a join on the name is exact and needs no directory to agree about anything.

    **The handoff is read by `blobcount.train.read_labels`, not here.** `bin/` is a
    directory of scripts loaded by path rather than a package, so a copy of the CSV
    contract in this file would be a second set of answers to it, and a read of
    `paths.labels` in `bin/` would be outside the source scan in `tests/test_config.py`,
    which would then keep reporting a declared key nothing under `blobcount/` reads. The
    cost of that is the one `bin/evaluate.py` names: rows resolve against the configured
    `paths.labels`, so a handoff written with `--out somewhere` resolves against the wrong
    root and every patch of it is reported missing.
    """
    paths, _labels, specimen_ids = read_labels(index, cfg)
    return {path.name: specimen for path, specimen in zip(paths, specimen_ids, strict=True)}


def _load_image(path: Path):
    """Return the `PIL.Image` at `path`, as RGB.

    **`PIL` is imported inside the read**, for the reason `blobcount.labeling.load_patch`
    gives: a native wheel at module scope turns an absent one into a collection error for
    every test in the suite, including the ones that touch no image.

    `.convert("RGB")` is not optional and nothing is resized: the eval transform resizes
    to `training.input_size` and `Normalize` is given three means, so a greyscale or RGBA
    file handed over as it is fails inside the transform with a message about tensor
    shapes rather than about the file. Measured: a greyscale patch raises `RuntimeError:
    output with shape [1, 64, 64] doesn't match the broadcast shape [3, 64, 64]` and an
    RGBA one `RuntimeError: The size of tensor a (4) must match the size of tensor b (3)
    at non-singleton dimension 0`, and neither names the image the caller passed.
    """
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB")


def _predict(
    model,
    transform,
    images: Sequence[Path],
    device: torch.device,
    specimen_of: dict[str, str],
    trained_on: Sequence[str],
) -> list[dict]:
    """Return one row per image, each holding the class and the whole probability vector.

    **One pass with `torch.no_grad()`**, because there is nothing here to differentiate:
    the map in `blobcount.interpret` needs the graph and a prediction does not, which is
    the difference between this call and `GradCAM.generate`.

    **The model is put into `eval()`**, so the rows are a function of the weights rather
    than of a dropout sample, for the reason `blobcount.evaluate._infer` gives.

    **Two columns are filled from what the run knows and not from what the file looks
    like.** `specimen` is the handoff's `slide_id` for a patch the handoff holds and the
    empty string for one it does not — this program is inference over an image or a
    directory, and a caller's own picture belongs to no specimen, so an empty cell is the
    true answer and a specimen guessed from a file name would not be. `trained_on` is
    `True` only where a specimen is both known and among the ones the caller stated the
    checkpoint was fitted on, so an unknown specimen is `False` rather than an accidental
    claim.

    The probability columns are named from the width of the logits this run produced, so
    the row describes the head the checkpoint has rather than a head this file assumes.
    """
    trained = set(trained_on)
    rows: list[dict] = []
    model.eval()
    with torch.no_grad():
        for path in images:
            image = _load_image(path)
            tensor = transform(image).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(tensor), dim=1)[0]
            predicted = int(probabilities.argmax())
            specimen = specimen_of.get(path.name, "")
            rows.append(
                {
                    "path": str(path),
                    "specimen": specimen,
                    "trained_on": bool(specimen) and specimen in trained,
                    "class_index": predicted,
                    "predicted_class": CLASS_NAMES[predicted],
                    "confidence": f"{float(probabilities[predicted]):.6f}",
                    **{
                        f"p_{index}": f"{float(value):.6f}"
                        for index, value in enumerate(probabilities)
                    },
                }
            )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prediction the arguments name and return the exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        # Resolved inside the arm, for the reason `bin/evaluate.py` gives: `load` types
        # this key as the declared `str`, and an empty one is a value it makes no claim
        # about, so `Config.path` is what refuses it and outside this arm that refusal
        # would leave a traceback.
        results_dir: Path = cfg.path("paths.results")
        # Read here rather than on its own arm: `get_device` is `project.device`'s one
        # reader and it raises `ConfigError` for a value that is neither `auto`, `cpu`
        # nor `cuda`, which is an answer about the configuration.
        device = torch.device(get_device(cfg))
    except ConfigError as error:
        print(f"predict: {error}", file=sys.stderr)
        return EXIT_REFUSED

    refused = _refuse_output_target(results_dir)
    if refused is not None:
        # Before an image is opened and before a checkpoint is loaded, so a mistyped
        # `paths.results` costs a message rather than an inference pass. `EXIT_REFUSED`
        # and not a prediction failure: nothing here finished, so there is no run whose
        # outcome is being reported.
        print(f"predict: {refused}", file=sys.stderr)
        return EXIT_REFUSED

    named = args.image if args.image is not None else args.dir
    try:
        images = _images(named, args.dir is not None)
    except ValueError as error:
        # The `--image`/`--dir` arm: an argument whose path is not the kind that argument
        # names, and a directory holding no image. Both are this program's own refusals
        # and both are about the argument, which is where they belong.
        print(f"predict: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # **The handoff is read before the checkpoint is loaded and before an image is opened,
    # and it is read at all only when it is needed**: `--specimen` filters by a fact only
    # the handoff carries, and without `--labels` there is nothing to filter on. A
    # `--specimen` without it is refused here rather than matched against nothing, because
    # a filter that finds no rows writes a header and no rows and calls that a prediction.
    specimen_of: dict[str, str] = {}
    if args.labels is not None or args.specimen is not None:
        if args.labels is None:
            print(
                f"predict: --specimen was given {args.specimen!r} and no --labels, and "
                "the specimen a patch came from is recorded only in labels.csv: the "
                "label tree names no specimen in any directory, so there is nothing for a "
                "filter to match against. Fix: pass --labels the handoff "
                "bin/label_patches.py wrote, or drop --specimen",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        try:
            specimen_of = _specimens_of(args.labels, cfg)
        except (ConfigError, OSError, csv.Error, ValueError) as error:
            # The same four types `bin/evaluate.py` catches around its own call into the
            # same reader, for the same reasons: a handoff that is not there, cannot be
            # opened, carries a cell `csv.field_size_limit` refuses, or is not the header
            # this reader expects; and `ConfigError` arrives out of it because
            # `read_labels` resolves every row against `paths.labels`. The catch is around
            # this one call and nothing else.
            print(f"predict: {error}", file=sys.stderr)
            return EXIT_REFUSED

    if args.specimen is not None:
        wanted = args.specimen
        if wanted not in set(specimen_of.values()):
            print(
                f"predict: --specimen was given {wanted!r} and {args.labels} records no "
                f"patch under it; the handoff holds {sorted(set(specimen_of.values()))}. "
                "A filter that matched nothing would write a CSV with a header and no rows, "
                "which is a report of nothing rather than a prediction. Fix: name a "
                "specimen this handoff holds, or drop --specimen",
                file=sys.stderr,
            )
            return EXIT_REFUSED
        images = [path for path in images if specimen_of.get(path.name) == wanted]

    try:
        model = build_model(
            cfg.get("training.architecture"),
            int(cfg.get("training.num_classes")),
            # `False` whatever the configuration says, for the reason the module
            # docstring gives: the checkpoint loaded below replaces every parameter.
            False,
        )
        _load_weights(model, args.checkpoint)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        # The checkpoint and the backbone, as the module docstring groups them. The catch
        # is around these two calls, so it is too narrow to hold a defect in the
        # inference below, and a `ValueError` from there reaches the interpreter as a
        # traceback — which is the honest report of a defect in this project.
        print(f"predict: {error}", file=sys.stderr)
        return EXIT_REFUSED

    # Everything below is the run itself. An image that cannot be opened and a failed
    # write are deliberately not caught: the `OSError` an image read raises names the
    # full resolved path, which is the whole fact a caller needs.
    _train_transform, transform = build_transforms(cfg)
    model = model.to(device)
    rows = _predict(
        model, transform, images, device, specimen_of, args.train_specimens or ()
    )

    # **The directory is created here and not before the images are read**, so a `--dir`
    # that cannot be listed leaves no empty `results/` behind. It is created with
    # `parents=True` because `paths.results` may name a nested path, and with
    # `exist_ok=True` because the previous run's predictions are beside this one's.
    results_dir.mkdir(parents=True, exist_ok=True)
    columns = [*_COLUMNS, *sorted(key for key in rows[0] if key not in _COLUMNS)]
    with (results_dir / _PREDICTIONS_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    # The sidecar, for the reason `_PROVENANCE_SIDECAR` gives. It names the CSV it covers
    # so a reader holding one file knows what the other is about, and it carries the
    # training cohort the caller stated — which is the fact a reader needs most and the
    # one `trained_on` alone does not give, because the column says which rows are
    # affected and not which specimens are.
    (results_dir / _PROVENANCE_SIDECAR).write_text(
        json.dumps(
            {
                "artifacts": [_PREDICTIONS_NAME],
                "provenance": _PROVENANCE,
                "checkpoint": str(args.checkpoint),
                "train_specimens": list(args.train_specimens or ()),
                "scored_specimen": args.specimen,
                "config": resolved(
                    cfg,
                    (
                        # **No `project.seed`, and that is the rule rather than an omission
                        # of interest:** this program seeds nothing. There is no `set_seed`
                        # call in this file, and inference over a fixed checkpoint and a
                        # fixed set of images is seed-independent — no sampling, no
                        # augmentation, no shuffled loader. The key was recorded here
                        # until 2026-10-16, which invited a reader to believe it had a
                        # bearing on the numbers beside it. Pinned by
                        # `tests/test_interpret.py::test_the_prediction_sidecar_records_no_
                        # parameter_this_run_did_not_read`.
                        "project.device",
                        "training.architecture",
                        "training.num_classes",
                        "training.input_size",
                    ),
                ),
            },
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )

    _print_table(
        columns,
        [[row[column] for column in columns] for row in rows],
    )
    counted: dict[str, int] = {}
    for row in rows:
        counted[row["predicted_class"]] = counted.get(row["predicted_class"], 0) + 1
    tally = ", ".join(f"{count} {name}" for name, count in sorted(counted.items()))
    resubstituted = sum(1 for row in rows if row["trained_on"])
    known = sum(1 for row in rows if row["specimen"])
    cohort = (
        f", {resubstituted} of them over a specimen this checkpoint was fitted on"
        if resubstituted
        else ""
    )
    unknown = (
        f"; {len(rows) - known} row(s) are not in the handoff and carry no specimen"
        if known != len(rows)
        else ""
    )
    print(
        f"\n{len(rows)} image(s) predicted with {args.checkpoint}: {tally}{cohort}{unknown}. "
        f"The rows are in {results_dir / _PREDICTIONS_NAME}, one per image in the order "
        f"above, and the probabilities are the softmax of the head this checkpoint carries. "
        f"Every row is a function of that checkpoint and of the images this run was pointed "
        f"at; the `specimen` column is what labels.csv records for the patch and is empty "
        f"for an image the handoff does not describe, and `trained_on` marks a row whose "
        f"specimen the checkpoint was fitted on, which is resubstitution and not a "
        f"measurement"
    )
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
