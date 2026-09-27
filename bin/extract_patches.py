"""Extract patches from the slides the dataset manifest names.

The entry point for the first stage of the pipeline. It resolves the configured
patch output directory, extracts each selected specimen, writes one `patches.csv`
for the run, prints the per-slide counts, and exits non-zero when a slide produced no
patches, so an empty or failed extraction cannot pass for a completed one.

    python bin/extract_patches.py                    # every manifest specimen
    python bin/extract_patches.py --slide 2 --slide 3
    python bin/extract_patches.py --force            # discard a previous run

Without `--force` a non-empty output directory is refused rather than cleared, because
the patches in it are a previous run's and destroying them is not this command's
decision to make. `--force` deletes and recreates it through
`blobcount.extract.prepare_output`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from blobcount.config import load
from blobcount.extract import (
    ExtractionStats,
    extract_slide,
    patches_dir,
    prepare_output,
    write_index,
)
from blobcount.registry import Specimen, load_registry
from blobcount.slides import SlideError

# Three outcomes, three codes. A blank slide is 1, a refusal is 2, and a `SlideError`
# that reached the process would exit 1 as well, so a provenance contradiction and a
# slide that yielded no patches would report the same thing; catching it here keeps
# them apart.
EXIT_OK = 0
EXIT_BLANK_SLIDE = 1
EXIT_REFUSED = 2

_COLUMNS = ("slide_id", "patches_written", "tissue_rejected", "read_failures", "skipped_blank")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract_patches",
        description="Extract patches at the configured target resolution from the "
        "slides the dataset manifest names.",
    )
    parser.add_argument(
        "--slide",
        action="append",
        dest="slides",
        metavar="ID",
        help="extract only this specimen, repeatable; every manifest specimen otherwise",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="delete and recreate the patch output directory, discarding a previous run",
    )
    return parser


def _select(specimens: Sequence[Specimen], wanted: Sequence[str] | None) -> list[Specimen]:
    """Return the specimens to extract, in the order asked for.

    A repeated `--slide` is one request, not two, and a request for an id the
    manifest does not name is refused rather than skipped: a typo in a slide id would
    otherwise be an extraction of the wrong slides, or of none, reported as a
    success.
    """
    if not wanted:
        return list(specimens)
    by_id = {specimen.id: specimen for specimen in specimens}
    missing = [slide_id for slide_id in dict.fromkeys(wanted) if slide_id not in by_id]
    if missing:
        raise LookupError(
            f"the dataset manifest names no specimen {', '.join(repr(item) for item in missing)}"
        )
    return [by_id[slide_id] for slide_id in dict.fromkeys(wanted)]


def _print_table(stats: Sequence[ExtractionStats]) -> None:
    """Print the per-slide counts as a plain aligned table.

    No table library, because the shape is five columns of integers and a boolean
    and `tabulate` is not a dependency this project declares.
    """
    widths = {
        name: max([len(name)] + [len(str(getattr(item, name))) for item in stats])
        for name in _COLUMNS
    }
    print("  ".join(name.ljust(widths[name]) for name in _COLUMNS))
    print("  ".join("-" * widths[name] for name in _COLUMNS))
    for item in stats:
        print("  ".join(str(getattr(item, name)).ljust(widths[name]) for name in _COLUMNS))


def main(argv: Sequence[str] | None = None) -> int:
    """Run extraction over the selected specimens and return the process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load()
    try:
        specimens = _select(load_registry(), args.slides)
    except LookupError as error:
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    out_dir = patches_dir(cfg)
    if args.force:
        prepare_output(out_dir)
    elif out_dir.is_dir() and any(out_dir.iterdir()):
        print(
            f"extract_patches: {out_dir} already holds a previous run; "
            f"re-run with --force to delete and recreate it",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    rows: list[dict] = []
    stats: list[ExtractionStats] = []
    try:
        for specimen in specimens:
            stats.append(extract_slide(specimen, out_dir, cfg, rows=rows))
    except SlideError as error:
        # A slide that contradicts its manifest row is not a slide this run may
        # extract from, and the run stops rather than continuing past the record that
        # describes the dataset.
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED
    write_index(rows, out_dir / "patches.csv")

    _print_table(stats)
    print(f"\n{len(rows)} patches written to {out_dir}")
    blank = [item.slide_id for item in stats if item.skipped_blank]
    if blank:
        print(
            f"extract_patches: no patches extracted from {', '.join(blank)}; "
            f"the manifest rows for them and the files they name disagree, or the "
            f"tissue filter rejected every window",
            file=sys.stderr,
        )
        return EXIT_BLANK_SLIDE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
