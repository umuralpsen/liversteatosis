"""Label the patches a patch index names, and report the run.

The entry point for the second stage of the pipeline. It reads the `patches.csv` a
previous `extract_patches` run wrote, labels every patch in it against the blob count
made at that patch's own achieved resolution, and writes the labelled tree, the
`labels.csv` handoff and the `labeling_stats.json` record into the configured label
directory.

    python bin/label_patches.py --index dataset/patches/patches.csv
    python bin/label_patches.py --index dataset/patches/patches.csv --threshold 7
    python bin/label_patches.py --index ... --out /tmp/relabel

`--threshold` overrides `labeling.blob_threshold` for the run, which is how the ablation
sweep is driven from the command line without a second configuration file. `--out`
overrides `paths.labels`, and the run **always** clears the directory it is given: a
labeling run that wrote into a previous run's tree would leave a patch in both class
directories, and the clearing is what makes a single label per patch a property of the
run rather than a coincidence about the dataset. That makes the argument a deletion, so
it goes through `blobcount.extract.prepare_output`, which refuses the project root,
`paths.slides`, `paths.manifest`, anything containing either, anything not strictly
below the project root, and a target that is a file; a refusal is reported and exits 2
rather than raised.

Two tables are printed. The first is the per-slide class count, which is the number the
study reports. The second is the per-slide achieved resolution, printed beside it because
a per-slide class count means nothing on its own: a slide labelled 90% steatotic at one
resolution and one labelled 20% at another is a magnification difference before it is a
disease difference, and the resolution is what tells the two apart.

Three outcomes, three codes. A run that completed is 0. A refusal is 2, and a refusal is
always an answer about what the caller pointed at. Six of them can reach it, in three
groups, and nothing else does:

- **A configuration.** One that cannot be loaded — missing, unreadable, not a mapping, a
  key the code does not declare, a required key the file does not declare — and a
  `paths.labels` that `load` accepted because it validates nothing past the key's
  presence but `Config.path` refuses, because it is not a path string or is empty. Both
  arrive as a `ConfigError`.
- **An output target.** An `--out` or a `paths.labels` that `prepare_output` will not
  clear: the project root, `paths.slides` or `paths.manifest` themselves, a directory
  holding either, anything not strictly below the project root, and an existing path that
  is a file rather than a directory. It arrives as an `OutputPathError`.
- **The index and the patches it names.** An index that cannot be read — it is not there,
  it cannot be opened, or it carries a cell longer than `csv.field_size_limit` — or whose
  header is not the one this pipeline writes; a cell of that index that is not the number
  its column records; and a patch file the index names that is not the `uint8`
  `(patch_size, patch_size, 3)` array `count_blobs` needs. All three arrive as a
  `LabelingInputError`, which `label_patches` raises around the two reads that can fail
  that way and around the conversion of the two numeric columns, and around nothing else,
  so a defect inside it is not reported as a bad `--index`.

All six are reported on stderr and none of them is a traceback.

**`--threshold` is not one of the six.** Its argument is typed `int` by the parser, so a
value that is not an integer never reaches `main`: the parser refuses it and exits 2 of
its own accord. There is no run in which a threshold is applied that the configuration
would not take, because the only value the run can produce for
`labeling.blob_threshold` is an `int` and that is what the key is configured as.

**A defect inside `label_patches` is never a refusal.** A `ValueError` from the counting,
from the normalization, from `np.percentile` on a `blobs.gray_percentile` outside [0, 100]
that this project range-checks nowhere, or from a write is not caught here and leaves the
interpreter to print a traceback, which is the honest report of a defect in this project.
The process status it exits with is 1, the same status the sentinel below uses, so a
caller that has to tell the two apart reads stderr: the sentinel names the contradiction it
found, and a traceback names a file and a line. `main`'s *return value* is the contract,
and an exception never returns one.

**There is a third constant, `EXIT_CONTRADICTORY = 1`, and it is a regression sentinel
rather than a reachable outcome.** `label_patches` empties `out` before it writes and then
writes each patch name into exactly one class directory exactly once, so a name under two
of them is not a state this program can reach. The code exists to report the regression
that would make it reachable — a second write path into a tree the run did not clear, from
removing the clearing, from making it conditional, or from anything else writing into
`out` — and to fail the run rather than hand a caller a dataset whose own guard says it is
unsound. Naming it here is deliberate: a code in the module's exit contract that no path
produces has to be declared as a sentinel, or a caller reading the header cannot tell it
apart from a contradiction it should have been told about. `LabelingStats`'s
`duplicates_removed` field is the input to it and its own docstring says the same thing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from blobcount.config import ConfigError, load
from blobcount.extract import OutputPathError
from blobcount.labeling import (
    CLASS_NORMAL,
    CLASS_STEATOSIS,
    LabelingInputError,
    label_patches,
)

EXIT_OK = 0
EXIT_CONTRADICTORY = 1
EXIT_REFUSED = 2

_COUNT_COLUMNS = ("slide_id", "normal", "steatosis", "n")
_RESOLUTION_COLUMNS = ("slide_id", "achieved_mpp", "gray_threshold")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label_patches",
        description="Label the patches a patch index names, counting blobs at each "
        "patch's own achieved resolution.",
    )
    parser.add_argument(
        "--index",
        required=True,
        type=Path,
        metavar="PATH",
        help="the patches.csv written by bin/extract_patches.py",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        metavar="N",
        help="override labeling.blob_threshold for this run",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        metavar="PATH",
        help="label output directory; paths.labels from the configuration otherwise. "
        "The directory is deleted and recreated, so nothing else may be named here.",
    )
    return parser


def _print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    """Print `rows` under `headers` as a plain aligned table.

    No table library, for the reason `bin/extract_patches.py` has none: this is three or
    four columns of integers, one float and one identifier, and `tabulate` is not a
    dependency this project declares.
    """
    widths = [
        max([len(str(header))] + [len(str(row[index])) for row in rows])
        for index, header in enumerate(headers)
    ]
    print("  ".join(str(header).ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[index]) for index, cell in enumerate(row)))


def _count_rows(per_slide: Mapping[str, Mapping[str, int]]) -> list[list[object]]:
    return [
        [slide_id, counts[CLASS_NORMAL], counts[CLASS_STEATOSIS], counts["n"]]
        for slide_id, counts in per_slide.items()
    ]


def _resolution_rows(
    per_slide_mpp: Mapping[str, float],
    per_slide_gray_threshold: Mapping[str, float],
) -> list[list[object]]:
    return [
        [slide_id, f"{per_slide_mpp[slide_id]:.4f}", f"{per_slide_gray_threshold[slide_id]:.1f}"]
        for slide_id in per_slide_mpp
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Run labeling over the index the arguments name and return the process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        overrides = [] if args.threshold is None else [f"labeling.blob_threshold={args.threshold}"]
        cfg = load(overrides=overrides)
        # Resolved here rather than after the `load` above, so a `paths.labels` that
        # `Config.path` refuses — not a path string, or an empty one — is caught by the
        # same `ConfigError` arm. Outside it, that value left a traceback and the process
        # exited 1, which is `EXIT_CONTRADICTORY` and told a caller the labels had
        # contradicted themselves over a hand-edited configuration string.
        out_dir: Path = args.out if args.out is not None else cfg.path("paths.labels")
    except ConfigError as error:
        # A configuration that is missing, unreadable or not a mapping, one that does not
        # declare a key the run reads, and a `paths.labels` `load` accepted because it
        # validates nothing past the key's presence but `Config.path` refused. The
        # override the threshold produces cannot be the cause: the parser has already
        # typed it as the `int` the key is configured as, and `--out` has already been
        # typed as a `Path`.
        print(f"label_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        stats = label_patches(args.index, out_dir, cfg)
    except OutputPathError as error:
        # The refusal from `prepare_output`: the target is the project root, the slides,
        # the manifest, something holding either, not below the project root at all, or an
        # existing path that is a file rather than a directory. It is caught rather than
        # raised because it is an answer about the argument the user passed, and a
        # traceback is not an answer.
        print(f"label_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except LabelingInputError as error:
        # The six things the caller pointed at and got wrong: an index that is not there,
        # cannot be opened, carries a cell longer than `csv.field_size_limit`, or is not
        # the `patches.csv` header; a cell of it that is not the number its column
        # records; and a patch file that is not the `uint8` `(patch_size, patch_size, 3)`
        # array `count_blobs` needs. `label_patches` raises this around those two reads
        # and the conversion of the two numeric columns, and around nothing else, so a
        # `ValueError` from anywhere else in it — a `blobs.gray_percentile` outside
        # [0, 100], the counting, the normalization — is not caught here and reaches the
        # caller as a traceback. Which is the point: a refusal is an answer about the
        # argument and a traceback is the answer about our code, and reporting a defect
        # as the first would send the caller to fix an index that is fine. A write that
        # fails still raises for the same reason the module docstring gives: a tree that
        # `labels.csv` does not describe is not something to report and continue past.
        print(f"label_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    _print_table(_COUNT_COLUMNS, _count_rows(stats.per_slide))
    print()
    _print_table(_RESOLUTION_COLUMNS, _resolution_rows(
        stats.per_slide_mpp, stats.per_slide_gray_threshold
    ))
    labeled = sum(counts["n"] for counts in stats.per_slide.values())
    print(f"\n{labeled} patches labelled into {out_dir} at blob threshold "
          f"{stats.blob_threshold}")

    if stats.duplicates_removed:
        # The sentinel the module docstring describes. Unreachable while `label_patches`
        # empties `out` before writing, and kept as the report for the regression that
        # would make it reachable.
        print(
            f"label_patches: {stats.duplicates_removed} patch file(s) were present in "
            f"more than one class directory and were deleted; the labels in {out_dir} "
            f"contradicted themselves, so this run is not a clean one",
            file=sys.stderr,
        )
        return EXIT_CONTRADICTORY
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
