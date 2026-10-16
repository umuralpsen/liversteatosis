"""Labeling: the count that decides a label, and the resolution the count was made at.

`label.py` wrote a patch into `dataset/train/<class>/` by copying the file, and it
decided the class with a blob count taken at one resolution for a set of slides read at
two. Both are fixed here, and the two fixes are separate: one is about the state of the
output directory and the other is about the argument `count_blobs` is given.

**The output directory is cleared, not written into.** Re-running labeling after
changing the threshold left every file the previous run had put in the other class
directory, so one patch could sit in `normal/` and in `steatosis/` at once and a
directory listing could no longer say which label a file carried. `label_patches` calls
`prepare_output` on `out_dir` before it writes anything, so a run starts from a tree it
owns. That call is the whole of the fix and it is the reason this function takes `out_dir`
rather than a caller passing a directory it has already filled. `prepare_output` carries
its own refusals: the project root, `paths.slides`, `paths.manifest`, anything holding
either, anything not strictly below the project root, and a target that is a file are all
refused rather than deleted, and the refusal is an `OutputPathError` that reaches the
caller.

**Each patch is counted at the resolution it was read at.** `patches.csv` records the
achieved `mpp` per patch, and `count_blobs` requires that number as its third argument
with no default, because the area filter is in square microns and the conversion to
pixels is the only thing that makes it mean the same area on a 40x and a 20x slide. A
global constant here is the original defect wearing a different hat: it produces a
number, it produces the same number for every slide, and nothing about the run says it
was guessed. `per_slide_mpp` below is a *summary* of each slide's achieved resolution
for the printed record, and it is not the value any count was made at; the per-row value
is the one that reaches `count_blobs`.

**The gray threshold is a percentile of the slide, not of the patch.** `params_from_config`
derives one from the patch in hand through `gray_threshold_from_patch`, and that function
is superseded for this purpose and is not called here: a percentile of a patch moves with
the patch, so a threshold derived that way is a different number for every patch in a
slide and the count then measures the patch's own spread rather than the material. The
threshold used here is the `blobs.gray_percentile` of the `tissue_gray` column of that
slide's own index rows, so it is one number per slide and the `BlobParams` carrying it is
built once per slide. The value it thresholds against is the *background* level of each
patch, since `tissue_gray` is what `extract.tissue_gray_threshold` recorded, so this
figure is a slide-wide scale for the cut and not a measurement of any one patch; the
ablation in the next task is what measures what it costs.

A slide with no tissue has no rows in the index and therefore no group here, so the
per-slide percentile is taken over a non-empty array by construction and cannot be the
`NaN` that a percentile over nothing produces. The blank slide is detected and reported
at extraction, which is where it is detectable; this function cannot see a slide that
wrote no patches.

`load_patch` is the single seam through which this module reads an image file, and the
tests substitute it rather than staging patches. It is also a precondition check: the
array a patch file holds must be `extraction.patch_size` square, three channels of
`uint8`, which is what `extract.read_patch` wrote and what `count_blobs` requires, and
anything else raises rather than being resized, converted or counted. A silently
resampled patch is a patch whose physical field of view is no longer the one the index
records, and nothing downstream could detect it. A failure to write is likewise not
caught: it leaves a tree that `labels.csv` does not describe, and stopping is the honest
response to that.

**`label_patches` is where a bad input stops being reportable as one.** `read_index` and
`load_patch` are public and keep raising what they raise: a `ValueError` for a header
that is not the `patches.csv` header and for a patch array of the wrong shape, an
`OSError` for a path that is not there, and a `csv.Error` for a cell longer than
`csv.field_size_limit`. `label_patches` wraps those two reads and the conversion of the
two numeric columns — which is the step that turns a `None` or a `not-a-number` into a
`TypeError` or a `ValueError` — and raises `LabelingInputError` with the original attached
as its `__cause__`. That is the whole of the separation, and it is placed here because
this is the boundary: a `ValueError` from `count_blobs`, from `prepare_patch`, from
`np.percentile` on a `blobs.gray_percentile` this project range-checks nowhere, or from
this run's own arithmetic is not a statement about what the caller pointed at, so it is
not wrapped and it propagates. The command line then reports the first kind as a refusal
and lets the second out as a traceback, because a refusal is an answer about the argument
the user passed and a traceback is the answer about this code.

`labels.csv` is written with `path` relative to `out_dir`, separated by `/` whatever the
platform separator is, and it is the only handoff from this task to training, the ablation
and the evaluation. Those three read it and resolve each row against the configured label
root, rather than re-deriving a label from the directory layout — which is the thing the
clearing above exists to make possible.

Declared interface, in `__all__` order:

    LabelingInputError
    LabelingStats
        blob_threshold: int
        per_slide: dict[str, dict[str, int]]
        per_slide_mpp: dict[str, float]
        per_slide_gray_threshold: dict[str, float]
        duplicates_removed: int
    label_patches(index_path: Path, out_dir: Path, cfg: Config) -> LabelingStats
    load_patch(path: Path, size: int) -> np.ndarray
    read_index(path: Path) -> list[dict]
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from blobcount.blobs import BlobParams, count_blobs, prepare_patch
from blobcount.config import Config, resolved
from blobcount.extract import INDEX_COLUMNS, prepare_output

# The keys this stage reads and that decide what a patch's label is: the cut point, and
# the seven blob-counter parameters behind it. **These eight and not the whole
# configuration**, because every table in the README's ablation section is a function of
# exactly these numbers and an artifact recording the cut point alone cannot be
# re-examined — a reader who wants to know why specimen 4's mean blob count is 2.63 would
# have the output of the rule and none of its inputs. Each is read through `resolved`, so
# it is recorded here exactly as a `cfg.get` literal records it.
_CONFIG_KEYS = (
    "labeling.blob_threshold",
    "blobs.area_um2_min",
    "blobs.area_um2_max",
    "blobs.circularity_min",
    "blobs.gray_percentile",
    "blobs.morph_kernel_size",
    "blobs.morph_iterations",
    "blobs.stain_normalization",
)

# The sentence the artifact carries about what its numbers are, in the artifact rather
# than only in this module's docstring where a reader of the file alone would not see it.
# **It names what the labels are and are not**, because that is the one fact about a
# labels.csv a reader cannot work out for themselves and the one everything downstream
# rests on: every label is a morphological blob count made at the configured threshold
# under `config` below, and not a pathologist's grade, so the ceiling of every number
# computed from this file is agreement with that counter.
_PROVENANCE = (
    "written by blobcount.labeling.label_patches from the patches named by the index it "
    "read, so every label in labels.csv and every count in this document is a function of "
    "those patches, of the per-slide gray threshold derived from their recorded "
    "tissue_gray values, and of the resolved parameters under `config` — which is what the "
    "run actually used and not the shipped file. Every label is a morphological blob count "
    "made at that threshold and not a pathologist's grade, so the ceiling of anything "
    "computed from this file is agreement with a hand-tuned counter"
)

__all__ = [
    "LabelingInputError",
    "LabelingStats",
    "label_patches",
    "load_patch",
    "read_index",
]

LOGGER = logging.getLogger(__name__)

# The two class directories, and the label each one carries in `labels.csv`. The
# directory name is the class name, so a consumer that reads the tree and one that reads
# the CSV cannot disagree about what a file is; `label_patches` writes both from this
# table and the duplicate scan reads the same two names.
CLASS_NORMAL = "normal"
CLASS_STEATOSIS = "steatosis"
CLASS_NAMES = (CLASS_NORMAL, CLASS_STEATOSIS)
LABEL_BY_CLASS = {CLASS_NORMAL: 0, CLASS_STEATOSIS: 1}

# The column order `labels.csv` pins, and the keys every row of it carries. The header
# is the contract the three downstream tasks read, so it is written from one constant
# rather than assembled at the call site.
LABEL_COLUMNS = ("patch_id", "slide_id", "label", "blob_count", "mpp", "path")

# The file name of a patch inside a class directory, and the name `extract_slide` gave
# the same patch in the directory the index was written for. The index column is
# `patch_id` and `extract_slide` fills it with the file name including its extension
# (`2_1016_0.png`), so for an index this pipeline writes the two are the same string and
# this function is the identity. An index whose `patch_id` carries no extension names no
# file on disk, and the destination is required to be an image file, so the extension is
# completed. The *same* name is used to find the source and to name the destination, so
# the two cannot come to disagree about which file a label belongs to.
_PATCH_SUFFIX = ".png"


class LabelingInputError(Exception):
    """Raised when what the index named is not the data a labeling run needs.

    It covers the six things a caller can point at and be wrong about: an index that is
    not there, one that cannot be opened, one carrying a cell longer than
    `csv.field_size_limit`, a header that is not the `patches.csv` header, a cell that is
    not the number its column records, and a patch file that is not the `uint8`
    `(patch_size, patch_size, 3)` array `count_blobs` requires. What it deliberately does
    not cover is a defect in `label_patches` itself. **Only three things are wrapped: the
    read of the index, the conversion of the two numeric columns, and the read of each
    patch file**, so a `ValueError` from `np.percentile` on a configured percentile, from
    the counting or from the normalization propagates as itself rather than reaching a
    caller as a bad `--index`. Six refusals out of three operations, and the count was
    four until `csv.Error` joined the first: `csv.DictReader` raises it on a cell over
    `csv.field_size_limit` rather than truncating the cell, it is neither an `OSError` nor
    a `ValueError`, and a catch around the read that omitted it let it reach a caller that
    then exited 1 — which `bin/label_patches.py` defines as `EXIT_CONTRADICTORY`, so a
    malformed index would have been reported to anyone reading the exit code as the labels
    contradicting themselves. The original exception is the `__cause__` of every
    instance, because the message alone says the run refused and the traceback behind it
    says which read refused.

    **`blobcount.ablation.ablate` raises it for a seventh shape: an index that carries no
    rows.** That one is not produced by wrapping — an index with a header and no rows is
    refused before the conversion and before `sweep`, because the sweep's pooled ratio is
    a division by a count of nothing and the same `ValueError` would have left a command
    line with a traceback. It is named here so that the list above is not read as closed
    to every caller of the type, and it is raised there rather than here because this
    module's own contract is about labelling.
    """


@dataclass(frozen=True)
class LabelingStats:
    """What one labeling run produced, and the parameters every label was made under.

    `per_slide` is keyed by slide id and then by class, and always carries both class
    keys with a zero for one of them where no patch landed there: a class absent from the
    table reads as "not measured" and a class present with a zero reads as "measured,
    none found", and only the second is a fact about the run. `n` is the number of rows
    labeled for that slide, which is `normal + steatosis`.

    `per_slide_mpp` and `per_slide_gray_threshold` are the per-slide summaries written to
    `labeling_stats.json` beside the counts, so that a label can be traced to the
    resolution the slide was read at and to the cutoff its count was binarised at. The
    mpp here is the **mean** of that slide's recorded values and is a summary only: two
    patches of one slide can differ in the fourth decimal, and the value that decided
    each count is the one in its own index row and in its own `labels.csv` row.

    `duplicates_removed` is a diagnostic and not the mechanism. `prepare_output` empties
    the tree before anything is written, so on a clean run it is 0 and the scan that
    fills it finds nothing. It counts the copies deleted from the class directories after
    the write, and a non-zero value means the tree held a file under a name this run
    labelled somewhere else, which is the state the clearing exists to prevent.

    **A non-zero value is a regression sentinel and not a reachable outcome of this
    function.** The clearing is unguarded — `shutil.rmtree` raises rather than returning
    a tree it could not empty — and the row loop writes each name into exactly one class
    directory exactly once, so there is no path through `label_patches` that produces a
    name under two of them. The field exists to report the one regression that would:
    a second write path into a tree this run did not clear, whether by removing the
    `prepare_output` call, by making the clearing conditional, or by having something
    other than this function write into `out`. `bin/label_patches.py` turns a non-zero
    value into exit 1 for that reason and says so in its own docstring.
    """

    blob_threshold: int
    per_slide: dict[str, dict[str, int]]
    per_slide_mpp: dict[str, float]
    per_slide_gray_threshold: dict[str, float]
    duplicates_removed: int


def read_index(path: Path) -> list[dict]:
    """Return the rows of the patch index at `path`, as the strings the file holds.

    Read as UTF-8 with `newline=""`, for the reason `extract.write_index` writes it that
    way: the slide ids in this project are specimen names, and a locale-encoded read on
    a machine whose code page is Turkish mangles the non-ASCII ones. The values are left
    as strings rather than coerced here, because the caller converts each column to the
    type it needs and a single `float()` over a row would turn a patch id into a number.
    A line with fewer fields than the header carries is not rejected here: `csv.DictReader`
    fills the missing ones with `None`, and `label_patches` is what reports that.

    The header must carry every name in `extract.INDEX_COLUMNS`. It is checked here
    rather than left to the first `row["mpp"]` halfway through a run, because a truncated
    or reordered index is a defect in the extraction that produced it and the failure
    belongs on the file that has it, not on the labeling that read it.

    A cell longer than `csv.field_size_limit` raises `csv.Error` from `csv.DictReader`
    rather than being truncated, and that propagates as itself: nothing here raises a
    `csv.Error` of its own and nothing here catches one. `label_patches` is the boundary,
    and it is what turns it into a `LabelingInputError`.
    """
    target = Path(path)
    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in INDEX_COLUMNS if name not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(
                f"patch index {target} is missing the column(s) {', '.join(missing)}; "
                f"an index written by this project carries {', '.join(INDEX_COLUMNS)}"
            )
        return list(reader)


def load_patch(path: Path, size: int) -> np.ndarray:
    """Return the patch image at `path` as a `(size, size, 3)` `uint8` array.

    This is the only place in the module that opens an image file, so a test can
    substitute the pixels a run sees by replacing this one name. It is a seam and not an
    indirection: everything downstream of it is the same array whichever way the pixels
    arrived.

    The array must be exactly `size` on both axes, three channels, `uint8`, and anything
    else raises `ValueError` naming the file, the shape and the dtype it found. Nothing is
    resized and no channel is dropped, because both would change the patch silently: a
    resized patch covers a different physical area from the one its index row records, and
    a fourth channel is not information a three-channel count can use, so the honest
    response to either is to refuse the file rather than to guess what it meant. `size` is
    `extraction.patch_size`, which is what `extract.read_patch` resized every accepted
    patch to before writing it, so on an index this pipeline produced the condition holds
    and the check costs one comparison per patch.

    `PIL` is imported inside the read for the reason `cv2` is imported inside
    `blobcount.blobs.count_blobs`: it is a native wheel, and a module-scope import makes
    an absent one a collection error for every test in the suite, including the ones that
    touch no image.
    """
    from PIL import Image

    target = Path(path)
    with Image.open(target) as image:
        patch = np.array(image)
    if patch.shape != (size, size, 3) or patch.dtype != np.uint8:
        raise ValueError(
            f"patch {target} is a {patch.dtype} array of shape {patch.shape}, not a "
            f"uint8 ({size}, {size}, 3) one; it was not written by extract.read_patch "
            f"at extraction.patch_size={size}, and it is not resized or converted here"
        )
    return patch


def _patch_file_name(patch_id: str) -> str:
    """Return the file name a `patch_id` names, completing the extension when absent.

    See `_PATCH_SUFFIX` for why this exists and why the same result is used to find the
    source patch and to name the labelled copy.
    """
    return patch_id if patch_id.endswith(_PATCH_SUFFIX) else patch_id + _PATCH_SUFFIX


class _RowNumbers(NamedTuple):
    """The two numeric cells of one index row, converted.

    `tissue_gray` and `mpp` are the only two columns the run treats as numbers. Both are
    filled once, by `_row_numbers`, and every value below reads them from here rather
    than from the cell again.
    """

    tissue_gray: float
    mpp: float


def _row_numbers(row: Mapping[str, str | None]) -> _RowNumbers:
    """Return the `tissue_gray` and `mpp` of one index row, both converted to `float`.

    **This conversion is the only place in the module where a cell becomes a number, and
    the only `try` in it is the one around the two `float()` calls.** The `TypeError` is
    for the cell a truncated index line leaves `None` — `csv.DictReader` fills a short
    row's missing fields with `None` rather than raising on the line — and the `ValueError`
    is for a cell that is text where its column records a number. Both are statements
    about the file the caller pointed at, so both are raised as a `LabelingInputError`
    with the original attached.

    Nothing else is inside it, and that is what keeps the boundary true in both
    directions: a `TypeError` or a `ValueError` from `np.percentile`, from `np.mean`,
    from `count_blobs` or from a write is not a statement about the index, so it is not
    wrapped and it reaches the caller as itself.
    """
    try:
        return _RowNumbers(float(row["tissue_gray"]), float(row["mpp"]))
    except (TypeError, ValueError) as error:
        # Logged with the specimen, the coordinates this index row records for the patch
        # and the exception object, before the refusal is raised: the rule every caught
        # exception in this project is held to, and a cell that is text where its column
        # records a number is exactly the case where the line is wanted and is most often
        # missing. `row.get` rather than `row[...]` because the row may be the short one
        # that made this fire, where the trailing columns are `None` rather than absent —
        # a handler that raised here would replace the cell's own error with one of its
        # own, which is the one thing a log line must not do.
        LOGGER.error(
            "slide %s: reading the cell for patch %s at (%s, %s) raised %r",
            row.get("slide_id"), row.get("patch_id"), row.get("x"), row.get("y"), error,
        )
        raise LabelingInputError(
            f"the patch index carries a cell that is not the number its column records: "
            f"{error}"
        ) from error


def _slide_gray_threshold(values: Sequence[float], percentile: float) -> float:
    """Return the `percentile` of one slide's `tissue_gray` column.

    `values` is every recorded background level for the slide, so the result is a
    slide-wide cutoff rather than a per-patch one, and every patch of the slide is
    binarised at the same number. `values` is never empty: the caller groups the index by
    slide and a group exists only because a row put a patch there, so there is no
    percentile over an empty array here to produce the `NaN` that propagates into every
    comparison downstream of one.
    """
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _write_labels(out: Path, rows: Sequence[dict]) -> None:
    """Write `rows` to `out / "labels.csv"` under the pinned `LABEL_COLUMNS` header.

    UTF-8, `newline=""` and slide ids written as data, for the reasons
    `extract.write_index` gives: a non-ASCII slide id has to survive a Turkish-locale
    machine, and an id carrying a comma or a quote has to survive the round trip. The
    header is written even for no rows, so an empty run leaves a readable file rather
    than a missing one.
    """
    with (out / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LABEL_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _remove_duplicates(out: Path, labelled: Mapping[str, Path]) -> int:
    """Delete every labelled copy of a patch that is not the one this run wrote.

    A file name found under more than one class directory is a patch carrying both
    labels, which is the state `prepare_output` exists to prevent: the run wrote it once,
    and the other copy is a previous threshold's. The copy this run wrote is the one the
    `labels.csv` row names, and it is the one that is kept, so the file tree and the CSV
    keep describing the same thing. A name under exactly one class directory is left
    alone: it is a patch whose label did not change, and there is nothing to reconcile.

    **That keeping holds only because `labelled` is keyed by the resolved file name, the
    same string this scan groups on.** The `patch_id` column carries the name
    `extract_slide` wrote, which for an index this pipeline produced already ends in
    `.png`, so the two keys coincide there; for the extension-absent form `A_0_0` they do
    not, and a `labelled` keyed by `patch_id` looks `A_0_0.png` up in a map holding
    `A_0_0`, gets `None`, and deletes **both** copies — including the one `labels.csv`
    names, so the CSV goes on describing a file that is no longer there. `label_patches`
    therefore keys it by `name`, and the scan's guarantee is stated for every form of the
    column rather than for the one that happens to be written today. A name this run
    never labelled has no entry, so every copy under it is foreign and every one is
    deleted; the scan never has to decide between two copies it did not write.

    Returns the number of files deleted, which is 0 on a run whose clearing worked.
    """
    by_name: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        directory = out / class_name
        if not directory.is_dir():
            continue
        for item in sorted(directory.glob(f"*{_PATCH_SUFFIX}")):
            by_name.setdefault(item.name, []).append(item)

    removed = 0
    for name, found in by_name.items():
        if len(found) < 2:
            continue
        wanted = labelled.get(name)
        for item in found:
            if item == wanted:
                continue
            LOGGER.error(
                "patch %s is at %s as well as at the file this run labelled, so the "
                "duplicate is deleted and the run reported as contradictory",
                name, item,
            )
            item.unlink()
            removed += 1
    return removed


def label_patches(index_path: Path, out_dir: Path, cfg: Config) -> LabelingStats:
    """Label every patch the index at `index_path` names and write the run into `out_dir`.

    `out_dir` is emptied by `prepare_output` before anything is written, so a run never
    inherits a file from a previous threshold, and then filled with one directory per
    class, one `labels.csv` row per patch and one `labeling_stats.json` recording the
    parameters the labels were made under. See the module docstring for why the tree is
    cleared and why the count is per row rather than per slide.

    A patch is `steatosis` when its blob count is **at or above**
    `labeling.blob_threshold` and `normal` below it, and the count is
    `count_blobs(patch, params, mpp)` with `mpp` the row's own recorded resolution and
    `params` the `BlobParams` of the row's slide. `prepare_patch` is applied between the
    read and the count so that a run with `blobs.stain_normalization` set counts the same
    pixels the key is about; with the key false it returns the array it was handed, so the
    default path is unchanged.

    `patch_id` is taken as the name of the file to read and to write, so a label follows
    its file. An index that names the same patch twice is labeled once and the repeat is
    logged: the second row would be a second file for one patch, and one `labels.csv` row
    per patch is what the three downstream tasks read. The comparison is on the *resolved*
    name, so `2_1016_0` and `2_1016_0.png` in one index are the same patch and the second
    is the repeat, not a second file.

    Nothing per patch is caught but the read. A patch file that is not the shape the count
    needs raises in `load_patch` and a cell that is not a number raises at the conversion
    of it — the two numeric columns are converted once per row, before the first patch is
    read, so a bad cell costs no write — and both arrive here as a `LabelingInputError`
    carrying the original as its `__cause__`. A count that raises, a normalization that
    raises, a percentile outside [0, 100] and a write that fails are not caught: they are
    either a defect in this module or a tree that `labels.csv` would not describe, and
    stopping is the honest response to both.

    **A failure of `prepare_output` is not caught either**, and is not turned into a
    `LabelingInputError`: it is the refusal that keeps `--force` from deleting the slides,
    the manifest or the repository, and it reaches the caller as an `OutputPathError` so
    the command line can report it and exit non-zero rather than write into a directory it
    should not have. That is also the answer for a target that is a file: `prepare_output`
    refuses it rather than handing `shutil.rmtree` a regular file to fail on.
    """
    index = Path(index_path)
    out = Path(out_dir)
    # The index is read before the tree is cleared, so an index that cannot be read
    # leaves the previous run's labels on disk rather than deleting them on its way to
    # reporting why it could not be read.
    try:
        rows = read_index(index)
    except (OSError, ValueError, csv.Error) as error:
        # `csv.Error` is in the tuple because `csv.DictReader` raises it on a cell
        # longer than `csv.field_size_limit` rather than truncating the cell, and it is
        # neither an `OSError` nor a `ValueError`. It is a file the caller pointed at,
        # so it belongs with the two.
        raise LabelingInputError(f"the patch index could not be read: {error}") from error
    prepare_output(out)

    threshold = int(cfg.get("labeling.blob_threshold"))
    percentile = float(cfg.get("blobs.gray_percentile"))
    patch_size = int(cfg.get("extraction.patch_size"))
    area_um2_min = float(cfg.get("blobs.area_um2_min"))
    area_um2_max = float(cfg.get("blobs.area_um2_max"))
    circularity_min = float(cfg.get("blobs.circularity_min"))
    morph_kernel_size = int(cfg.get("blobs.morph_kernel_size"))
    morph_iterations = int(cfg.get("blobs.morph_iterations"))

    # The two numeric columns are converted here, once per row, and the values are read
    # from the converted list everywhere below. Converting at each point of use instead
    # would mean a second conversion of a cell already checked, and two conversions of
    # one cell are two places to come to disagree about what the index said.
    numbers = [_row_numbers(row) for row in rows]
    # Positions rather than the row dictionaries, so the grouping indexes `numbers` as
    # well and the two stay in step. `grouped` is built after the conversion so a row
    # whose cell is not a number is reported before anything is grouped around it.
    grouped: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        grouped.setdefault(str(row["slide_id"]), []).append(position)

    # The `BlobParams` below, the `np.percentile` it calls and the `np.mean` beside it
    # are outside every `try`, and that is the point: `blobs.gray_percentile` is a
    # configured value this project range-checks nowhere, and `np.percentile` is the only
    # thing that refuses one outside [0, 100]. Wrapped, that refusal would reach the
    # caller as a statement about the index, which is a file with nothing wrong with it.
    # It leaves as the `ValueError` it is.
    params = {
        slide_id: BlobParams(
            area_um2_min=area_um2_min,
            area_um2_max=area_um2_max,
            circularity_min=circularity_min,
            gray_threshold=_slide_gray_threshold(
                [numbers[position].tissue_gray for position in slide_rows], percentile
            ),
            morph_kernel_size=morph_kernel_size,
            morph_iterations=morph_iterations,
        )
        for slide_id, slide_rows in grouped.items()
    }
    per_slide_gray_threshold = {
        slide_id: item.gray_threshold for slide_id, item in params.items()
    }
    per_slide_mpp = {
        slide_id: float(np.mean([numbers[position].mpp for position in slide_rows]))
        for slide_id, slide_rows in grouped.items()
    }
    per_slide = {
        slide_id: {name: 0 for name in CLASS_NAMES} | {"n": 0}
        for slide_id in grouped
    }

    from PIL import Image

    labelled: dict[str, Path] = {}
    label_rows: list[dict] = []
    for position, row in enumerate(rows):
        patch_id = str(row["patch_id"])
        name = _patch_file_name(patch_id)
        if name in labelled:
            LOGGER.warning(
                "index %s names patch %s more than once; the first row keeps the label "
                "and the repeat is not counted again",
                index, name,
            )
            continue
        slide_id = str(row["slide_id"])
        mpp = numbers[position].mpp
        # The read of the file the index names, and nothing else. `prepare_patch` and
        # `count_blobs` sit outside the try on purpose, so a defect in either of them is
        # not reported as a patch the index named wrongly. The two exceptions here are
        # the two a corrupt patch file produces: an `OSError` for a path that is not a
        # readable image — `UnidentifiedImageError` for a text file, an empty one, a bad
        # chunk CRC and absurd dimensions, plain `OSError` for a truncated one and a
        # corrupt IDAT, `PermissionError` for a directory — and the `ValueError`
        # `load_patch` raises itself for an array that is not the `uint8` `(patch_size,
        # patch_size, 3)` one the count needs. A `TypeError` is deliberately not in the
        # tuple: no corrupt-file shape produces one, so it can only be a defect in this
        # project, and reporting that as "the patch the index names could not be used"
        # would send the caller to fix a file that is fine and exit 2 on it.
        try:
            source = load_patch(index.parent / name, patch_size)
        except (OSError, ValueError) as error:
            raise LabelingInputError(
                f"the patch the index names could not be used: {error}"
            ) from error
        patch = prepare_patch(source, cfg)
        count = count_blobs(patch, params[slide_id], mpp)
        class_name = CLASS_STEATOSIS if count >= threshold else CLASS_NORMAL

        target = out / class_name / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(patch).save(target)
        # Keyed by `name`, not by `patch_id`, for the reason `_remove_duplicates` gives:
        # the map is looked up by the file name the scan groups on, and the two coincide
        # only for the extension-present form of the column.
        labelled[name] = target

        per_slide[slide_id][class_name] += 1
        per_slide[slide_id]["n"] += 1
        label_rows.append({
            "patch_id": patch_id,
            "slide_id": slide_id,
            "label": LABEL_BY_CLASS[class_name],
            "blob_count": count,
            "mpp": mpp,
            "path": target.relative_to(out).as_posix(),
        })

    _write_labels(out, label_rows)
    duplicates_removed = _remove_duplicates(out, labelled)

    stats = LabelingStats(
        blob_threshold=threshold,
        per_slide=per_slide,
        per_slide_mpp=per_slide_mpp,
        per_slide_gray_threshold=per_slide_gray_threshold,
        duplicates_removed=duplicates_removed,
    )
    (out / "labeling_stats.json").write_text(
        json.dumps(
            {
                "provenance": _PROVENANCE,
                "blob_threshold": stats.blob_threshold,
                "per_slide": stats.per_slide,
                "per_slide_mpp": stats.per_slide_mpp,
                "per_slide_gray_threshold": stats.per_slide_gray_threshold,
                "duplicates_removed": stats.duplicates_removed,
                "config": resolved(cfg, _CONFIG_KEYS),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return stats
