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
`blobcount.extract.prepare_output`, which refuses a target that is the project root,
that holds the slides or the manifest, that is not below the project root at all, or that
is a file rather than a directory; `paths.patches` is a hand-edited string and `--force`
deletes what it names, so the refusal is reported and exits 2 rather than raised as a
traceback.

A slide that contradicts its manifest row stops the run, and the index for the slides
already extracted is written before it returns. Otherwise the tree would hold
thousands of patches that no `patches.csv` describes, and the refusal on a re-run
would then need `--force`, which deletes what the run just produced.

**Three outcomes, three codes.** A run that completed is `EXIT_OK = 0`. A selected
specimen that yielded no accepted patch is `EXIT_BLANK_SLIDE = 1`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration or the
dataset, never about this program. Nine refusals reach it, in four groups by the exception
that carries them, and nothing else does:

- **A configuration**, as a `ConfigError`. `load` refusing the file: it is not there, its
  top level is not a mapping, it declares a key this project does not, or it omits one the
  project requires. And a `paths.*` that `load` accepted because it validates nothing past
  the key's presence but `Config.path` refused, because it is not a path string or is
  empty: `paths.patches` for this program's own output target, `paths.manifest` inside
  `load_registry`, and `paths.slides` inside the guard `prepare_output` resolves before it
  deletes anything — that guard reads `paths.manifest` as well, but `load_registry` has
  already resolved that key above, so `paths.slides` is the only one that can be the first
  refusal there. Every one of them is refused before a patch is written or a file is
  removed.
- **The dataset manifest**, as a `RegistryError`. A manifest that is not there, cannot be
  read, is not a mapping, holds a `specimens` that is not a list, holds an entry that is
  not a mapping or carries a key this project does not declare, is missing a required
  field, declares a duplicate id, or claims a source it holds no licence for. The load is
  all-or-nothing by design, so one bad row stops the run here rather than extracting a
  dataset that is silently short of a specimen.
- **The selection, and the output tree.** A `--slide` naming an id the manifest does not,
  which arrives as a `LookupError`; and a patch directory that already holds a previous run
  when no `--force` was given to discard it, which is not an exception at all but this
  program's own check that the tree is empty.
- **The output target, and the slide.** A `--force` target `prepare_output` will not
  clear — the project root, `paths.slides` or `paths.manifest` themselves, a directory
  holding either, anything not strictly below the project root, an existing path that is a
  file rather than a directory — which arrives as an `OutputPathError`; and a slide that
  will not open, declares no resolution, carries a resolution the row that records it
  contradicts, or presents a pyramid that cannot be planned against, which arrives as a
  `SlideError`. The run stops at the first such slide and indexes the slides it finished,
  as the paragraph above says.

All nine are printed on stderr behind this program's name, and none of them is a
traceback. The last paragraph below says what is left over, and it is not a fourth
outcome.

**The `cfg.get` reads further down cannot be the source of a refusal.** `extract_slide`
reads `extraction.patch_size`, `extraction.tissue_percentile` and
`extraction.tissue_gray_max` with bare `cfg.get` calls, and the plan in
`blobcount.slides` reads `extraction.target_mpp` and `extraction.step_size`. All five are
in `REQUIRED_KEYS`, so a configuration missing one is refused by `load` before this
program deletes anything. The test that holds that line down is
`test_cli_force_refuses_before_clearing_when_a_key_extract_slide_reads_is_missing` in
`tests/test_extract.py`.

**A mistyped value is not a refusal, and that is a limit rather than a gap.** `load`
checks that a key is declared and nothing at all about its value, so
`extraction.patch_size: big` is a configuration this project accepts and the only thing
that refuses it is the `int()` in `open_info_from_config`. A `ValueError` from a conversion
like that, a `TypeError` or `ValueError` out of planning a read, an `OSError` out of a
write, and a `PermissionError` out of listing or clearing a directory are all left to the
interpreter to report as tracebacks. A catch wide enough to hold them would also hold a
defect in this project's own arithmetic, and reporting a defect as a refusal would send the
caller to fix a configuration that is fine — the ruling `bin/label_patches.py` records for
a `blobs.gray_percentile` outside [0, 100]. The status such a traceback exits with is 1,
the same status `EXIT_BLANK_SLIDE` uses, so a caller that has to tell the two apart reads
stderr: the blank-slide message names the specimens that yielded nothing, and a traceback
names a file and a line. `main`'s *return value* is the contract, and an exception never
returns one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from blobcount.config import ConfigError, load
from blobcount.extract import (
    ExtractionStats,
    OutputPathError,
    extract_slide,
    patches_dir,
    prepare_output,
    write_index,
)
from blobcount.registry import RegistryError, Specimen, load_registry
from blobcount.slides import SlideError

# Three outcomes, three codes. A blank slide is 1, a refusal is 2, and a traceback from a
# defect would exit 1 as well, so a slide that yielded no patches and this program
# breaking report the same status; the module docstring names every path that reaches each
# code and says how a caller tells the two apart.
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

    try:
        cfg = load()
        out_dir = patches_dir(cfg)
    except ConfigError as error:
        # Two refusals, one exception. `load` refusing the file — it is not there, its top
        # level is not a mapping, it declares a key this project does not, it omits one the
        # project requires — and a `paths.patches` that `load` accepted because it
        # validates nothing past the key's presence but `Config.path` refused, because it
        # is not a path string or is empty. Both are answers about the configuration, and
        # both are reached before a patch is written or a file is removed. Neither was
        # wrapped: a `ConfigError` left `main` as a traceback, and the process exited 1,
        # which is `EXIT_BLANK_SLIDE` — so a broken configuration reported to any caller
        # reading the exit code as "no tissue was found in any specimen".
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        specimens = _select(load_registry(), args.slides)
    except ConfigError as error:
        # `paths.manifest`, resolved by `load_registry` off the *default* configuration
        # rather than off the `cfg` above, so the same two refusals `Config.path` makes are
        # reachable on the key that names the dataset. Reported for the reason the arm above
        # is.
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except RegistryError as error:
        # The manifest itself: absent, unreadable, not a mapping, a `specimens` key that is
        # not a list, an entry that is not a mapping, an entry key that is neither a
        # `Specimen` field nor `provenance_note`, a missing or mistyped required field, a
        # duplicate id, or a licence absent while the source is not `unknown`. Loading is
        # all-or-nothing by design, so one bad row stops the run here rather than
        # extracting a dataset that reads as complete while missing a specimen — and the
        # row is fixed in the manifest, which is a report to the caller rather than a
        # traceback out of a program that did nothing wrong.
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except LookupError as error:
        # A `--slide` naming an id the manifest does not name, refused rather than skipped.
        # Already wrapped, and kept as its own arm so the docstring's grouping is the
        # grouping in the code.
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    if args.force:
        try:
            prepare_output(out_dir)
        except OutputPathError as error:
            # The five targets `prepare_output` will not clear: the project root,
            # `paths.slides` and `paths.manifest` themselves, a directory holding either,
            # anything not strictly below the project root, and an existing path that is a
            # file rather than a directory. Each is an answer about the string
            # `paths.patches` was hand-edited to, so it is reported rather than raised.
            print(f"extract_patches: {error}", file=sys.stderr)
            return EXIT_REFUSED
        except ConfigError as error:
            # `prepare_output` resolves `paths.slides` and `paths.manifest` through
            # `_protected_paths` before it deletes anything, so the same two `Config.path`
            # refusals can surface from inside the guard itself, and a `paths.slides: 7`
            # reached them with this arm catching `OutputPathError` only. It is caught
            # here rather than by widening the arm above `prepare_output`, so the delete
            # still happens strictly after every refusal that can precede it: this one
            # raises ahead of `shutil.rmtree`, which is what leaves the previous run on
            # disk.
            print(f"extract_patches: {error}", file=sys.stderr)
            return EXIT_REFUSED
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
        # describes the dataset. The slides already extracted are indexed first: they
        # are on disk, and an unindexed patch tree is a tree a re-run cannot resume
        # without `--force`, which would delete it. The directory does not exist when
        # the first slide is the one that fails, because `extract_slide` raises before
        # it creates anything, and there is then nothing to index.
        print(f"extract_patches: {error}", file=sys.stderr)
        if out_dir.is_dir():
            write_index(rows, out_dir / "patches.csv")
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
