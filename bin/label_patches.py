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
`paths.slides`, `paths.manifest`, anything containing either, and anything not strictly
below the project root; a refusal is reported and exits 2 rather than raised.

Two tables are printed. The first is the per-slide class count, which is the number the
study reports. The second is the per-slide achieved resolution, printed beside it because
a per-slide class count means nothing on its own: a slide labelled 90% steatotic at one
resolution and one labelled 20% at another is a magnification difference before it is a
disease difference, and the resolution is what tells the two apart.

Three outcomes, three codes. A run that completed is 0. A run that found a patch in both
class directories is 1: the tree contradicts itself, and a caller that reads it anyway
would train on a dataset whose own guard says it is unsound. A refusal, from
`prepare_output` or from an unparseable threshold, is 2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from blobcount.config import ConfigError, load
from blobcount.extract import OutputPathError
from blobcount.labeling import CLASS_NORMAL, CLASS_STEATOSIS, label_patches

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
    except ConfigError as error:
        print(f"label_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    out_dir: Path = args.out if args.out is not None else cfg.path("paths.labels")
    try:
        stats = label_patches(args.index, out_dir, cfg)
    except OutputPathError as error:
        # The refusal from `prepare_output`: the target is the project root, the slides,
        # the manifest, something holding either, or not below the project root at all.
        # It is caught rather than raised because it is an answer about the argument the
        # user passed, and a traceback is not an answer.
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
