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

**The run writes `results/extraction_summary.json`, and that is the only record it makes
of itself.** Its per-slide counts used to go to stdout and nowhere else, so the
accepted-patch count had no denominator in any artifact and a reader comparing the next
stage's `patches.csv` against it had nothing to compare it to. The summary carries the
per-slide counts, their sum, and the resolved values of the five `extraction.*` keys this
stage read — not the whole configuration, because those five are the ones that decide what
a patch is and the other forty would bury them.

**Three outcomes, three codes.** A run that completed is `EXIT_OK = 0`. A selected
specimen that yielded no accepted patch is `EXIT_BLANK_SLIDE = 1`. A refusal is
`EXIT_REFUSED = 2`, and a refusal is always an answer about the configuration or the
dataset, never about this program. **Nine** refusals reach it — one per arm of `main` that
can refuse, plus this program's own empty-tree check and its own results-target guard, and
with the opening `ConfigError` arm counted twice because it carries two of them. They are:
`load` refusing the file and a `paths.patches` or `paths.results` `Config.path` refuses, on
that one arm; the `ConfigError` around `load_registry`; the `RegistryError` beside it; the
`LookupError` around `_select`; the check that an unforced patch tree is empty; the
`OutputPathError` around `prepare_output`; the `SlideError` around `extract_slide`; and the
`paths.results` guard. In four groups by the exception that carries them, and nothing this
program does does. It was nine before round 6, and the one that left is `paths.slides`
refused from inside the guard, which now arrives on the `OutputPathError` arm below rather
than on an arm of its own:

- **A configuration**, as a `ConfigError`. `load` refusing the file: it is not there, its
  top level is not a mapping, it declares a key this project does not, it omits one the
  project requires, or a value for a declared key has a type this project does not read it
  as — `extraction.patch_size: big` is the shape, and the message names the key, the type
  found and the type read. And a `paths.*` whose *value* `Config.path` refused because it
  is empty: `paths.patches` and `paths.results` for this program's own output targets, and
  `paths.manifest` inside `load_registry`. The last paragraph of this docstring is about the
  four exceptions that are still tracebacks. Every one of the nine is refused before a patch
  is written or a file is removed.
- **The summary's output target, as this program's own check on `paths.results` alone**, for
  the reason every sibling entry point checks the directory it writes into:
  `paths.results` is a hand-edited string in `configs/default.yaml` and this stage writes
  `extraction_summary.json` into it, so a `paths.results: /` is how a study's extraction
  record ends up somewhere this repository is not responsible for. **A run is refused for it
  rather than extracted and not summarised**, which is a deliberate choice and the
  alternative: every other stage in this pipeline would refuse the same configuration on the
  same run, and a pipeline whose first stage succeeds and whose second does not is worse
  than one that stops and says why.
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
  file rather than a directory, and a `paths.slides` or a `paths.manifest` the guard
  cannot resolve from the configuration at all, which it wraps into this same
  `OutputPathError` because a guard that cannot resolve what it protects cannot show the
  target is safe to clear — which arrives as an `OutputPathError`; and a slide that will
  not open, declares no resolution, carries a resolution the row that records it
  contradicts, or presents a pyramid that cannot be planned against, which arrives as a
  `SlideError`. The run stops at the first such slide and indexes the slides it finished,
  as the paragraph above says.

All eight are printed on stderr behind this program's name, and none of them is a
traceback.

**A bad argument is not one of the eight, and not one of this program's outcomes at all.**
`argparse` refuses an option this program does not declare and one given no value, by
printing its own message and calling `sys.exit(2)` inside `parse_args`, before the first
statement of `main` runs. Nothing else reaches it: `--slide` is read as a string and
`--force` is a flag, so neither can fail the parser's own conversion, and a `--slide`
naming an id the manifest does not is refused below with the rest rather than here. The
parser's status is the same number as `EXIT_REFUSED` and no code can tell the two apart, so
the separation is on stderr, where the parser names the argument and its usage and none of
the eight names either. The last paragraph says what else is left over, and that is not an
outcome either.

**The `cfg.get` reads further down cannot be the source of a refusal.** `extract_slide`
reads `extraction.patch_size`, `extraction.tissue_percentile` and
`extraction.tissue_gray_max` with bare `cfg.get` calls, and the plan in
`blobcount.slides` reads `extraction.target_mpp` and `extraction.step_size`. All five are
in `REQUIRED_KEYS`, so a configuration missing one is refused by `load` before this
program deletes anything. The test that holds that line down is
`test_cli_force_refuses_before_clearing_when_a_key_extract_slide_reads_is_missing` in
`tests/test_extract.py`.

**A mistyped value is now a refusal, and it arrives from `load` before this program has
touched anything.** `load` types both of its inputs, so `extraction.patch_size: big` is
refused at load as a `ConfigError` naming the key, the type found and the type read — on the
`ConfigError` arm above, before a slide is opened, a directory is listed, or a byte is
written. That check is a *type* check and nothing more, and the two keys it lets through
divide between them rather than sharing an answer. A `patch_size` of `-1` or `0` is the
declared `int`, passes `load`, and is then refused by
`blobcount.slides.plan_resolution` as a `SlideError` — measured, and that is the right
answer, because a non-positive patch size makes the plan undefined rather than inaccurate. A
`target_mpp` of `.inf` is the declared `float`, passes `load`, and is **not** refused by
`plan_resolution`: measured on this stack it raises `OverflowError: cannot convert float
infinity to integer` out of the `round()` that computes `read_px`, which is the fifth of the
tracebacks enumerated below and not a refusal. So a type check at `load` is not the whole of
what a caller is protected from, and the part it does not cover is named here rather than
left to be found at the first `.inf`. The test is
`test_cli_reports_a_mistyped_configuration_value_as_a_refusal` in `tests/test_extract.py`,
which used to pin the opposite outcome and says in its own prose why the pin moved.

**Five exceptions are still left to the interpreter, and that is the K10 ruling standing.**
A `ValueError` from a conversion inside a value that passed the type check — measured:
`target_mpp: .nan` raises `ValueError: cannot convert float NaN to integer` — an
`OverflowError` out of the same `round()` for a `target_mpp` of `.inf`, a `TypeError` or
`ValueError` out of planning a read, an `OSError` out of the index write, and a
`PermissionError` out of listing or clearing a directory. **`OverflowError` is in that list
because it is a traceback and not a refusal, and the reason it is not caught is that it
arrives from the standard library's own refusal rather than from a rule this project wrote:**
`int()` and `round()` refusing `.inf` names the value and no rule, and adding it to this
arm's catch would turn a defect in this project's own arithmetic — the same `round()` raising
`OverflowError` for a number that is finite and absurd — into a refusal, which is the ruling
`bin/label_patches.py` records for a `blobs.gray_percentile` outside [0, 100]. Nothing was
widened to reach the paragraph above: the refusal there comes from the `load` call whose
`ConfigError` arm already existed, on a value that call already had to look at. A per-patch
read or write that raises is not in that list because `extract_slide` catches both, logs the
exception object with the slide id and the coordinates, and counts the patch as a
`read_failure`. The status such a traceback exits with is 1, the same status
`EXIT_BLANK_SLIDE` uses, so a caller that has to tell the two apart reads stderr: the
blank-slide message names the specimens that yielded nothing, and a traceback names a file
and a line. `main`'s *return value* is the contract, and an exception never returns one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from blobcount.config import PROJECT_ROOT, ConfigError, load, resolved
from blobcount.extract import (
    ExtractionStats,
    OutputPathError,
    extract_slide,
    is_owned_path,
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

# `results/extraction_summary.json`, named here so the writer and the tests that read it
# back name the same file.
_SUMMARY_NAME = "extraction_summary.json"

# The keys this stage reads and that decide what a patch is: the resolution it plans for,
# the window it reads and writes, and the two numbers the tissue filter is. **These five
# and not the whole configuration**, because a reader of the artifact needs to know what
# the accepted patches look like and thirty unrelated numbers would hide that. Each is
# read through `resolved`, so it is recorded here exactly as a `cfg.get` literal records
# it and the source scan in `tests/test_config.py` still sees a reader for every one.
_CONFIG_KEYS = (
    "extraction.target_mpp",
    "extraction.patch_size",
    "extraction.step_size",
    "extraction.tissue_percentile",
    "extraction.tissue_gray_max",
)

# The sentence the artifact carries about what it is, in the artifact rather than in this
# file's docstring where a reader of the file alone would not see it. **It says what the
# tissue filter is and is not**, because the filter is the one thing about an extracted
# patch a reader is most likely to over-read: it is a single global mean-intensity cutoff
# whose sensitivity is unmeasured, and an artifact that recorded `tissue_gray_max` beside a
# per-slide accepted count invites the reader to assume the number separates tissue from
# background.
_PROVENANCE = (
    "written by bin/extract_patches.py from the slides the dataset manifest names, so "
    "every count and every patch in this document and in patches.csv is a function of "
    "those slides, of `config` below, and of nothing else. `patches_written` is the number "
    "of windows the filter accepted; `tissue_rejected` is the number it discarded on a "
    "single global mean-intensity cutoff (`extraction.tissue_gray_max`), which has no "
    "background rejection and whose sensitivity is unmeasured - this document records the "
    "cutoff that was applied, not a measurement of how well it separates tissue from "
    "background"
)


def _document(stats: Sequence[ExtractionStats], cfg, patches: int) -> dict:
    """Return the JSON document this run writes beside `paths.results`.

    **`slides` is the per-slide record and `patches_written` is its sum**, because the
    sum is what the next stage's `patches.csv` has to agree with and a reader comparing
    them should not have to add four numbers up. `n_slides` is the count of records, not
    the count of rows in `patches.csv`, for the reason `bin/ablate_threshold.py::_n_slides`
    gives: a slide that produced nothing has a record here and no row there, and a slide
    with rows but no counted patches is the reverse.
    """
    return {
        "provenance": _PROVENANCE,
        "n_slides": len(stats),
        "patches_written": patches,
        "slides": [
            {
                "slide_id": item.slide_id,
                "patches_written": item.patches_written,
                "tissue_rejected": item.tissue_rejected,
                "read_failures": item.read_failures,
                "skipped_blank": item.skipped_blank,
            }
            for item in stats
        ],
        "config": resolved(cfg, _CONFIG_KEYS),
    }


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


def _refuse_results_target(results_dir: Path) -> str | None:
    """Return why this program will not write its summary under `results_dir`, or None.

    **The same predicate every sibling entry point uses, and for the same reason.** This
    stage wrote nothing outside `paths.patches` until it started writing
    `extraction_summary.json`, and a new output is a new thing a hand-edited
    `paths.results` can put somewhere this repository is not responsible for. The test is
    `blobcount.extract.is_owned_path` rather than a string comparison here, so "is this
    ours" has one implementation in this project rather than one per guard, and it follows
    the configuration rather than the checkout alone — which is what lets a run write to a
    `paths.*` the data move put outside the repository.

    **Only the results directory is tested, and not the file.** `bin/train.py` writes
    files directly into `paths.results` and tests that one path, because there is no second
    directory to test; this one writes one file into it too, so the shape is `bin/train.py`'s
    rather than `bin/evaluate.py`'s.
    """
    if results_dir.exists() and not results_dir.is_dir():
        return (
            f"refusing to write {results_dir}: it exists and is a file rather than a "
            f"directory, so it is a typo in paths.results and not a previous run"
        )
    if not is_owned_path(results_dir):
        return (
            f"refusing to write {results_dir}: it is not strictly below the project root "
            f"{PROJECT_ROOT} or the parent of a configured paths.* value, so it is not a "
            f"directory this project owns"
        )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run extraction over the selected specimens and return the process exit code."""
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = load()
        out_dir = patches_dir(cfg)
        # Resolved inside the arm, for the reason the two lines above are: `load` types
        # the key as the declared `str` and an empty value is a question `load` makes no
        # claim about, so `Config.path` is what refuses it.
        results_dir: Path = cfg.path("paths.results")
    except ConfigError as error:
        # Two refusals, one exception. `load` refusing the file — it is not there, its top
        # level is not a mapping, it declares a key this project does not, it omits one the
        # project requires, or a value for a declared key has a type this project does not
        # read it as — and a `paths.patches` whose *value* `Config.path` refused, because
        # it is empty. The type of that key is `load`'s business now, not this arm's; the
        # empty string is a value question and stays here. Both are answers about the
        # configuration, and both are reached before a patch is written or a file is
        # removed. Neither was wrapped: a `ConfigError` left `main` as a traceback, and the
        # process exited 1, which is `EXIT_BLANK_SLIDE` — so a broken configuration
        # reported to any caller reading the exit code as "no tissue was found in any
        # specimen".
        print(f"extract_patches: {error}", file=sys.stderr)
        return EXIT_REFUSED

    refused = _refuse_results_target(results_dir)
    if refused is not None:
        # Before a slide is opened and before `--force` clears anything: a mistyped
        # `paths.results` costs a message rather than a twenty-minute extraction. The
        # extraction itself is left alone on purpose — refusing this program for a path it
        # can write to but for one it cannot would be a different refusal than the one the
        # guard is making.
        print(f"extract_patches: {refused}", file=sys.stderr)
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
            # The targets `prepare_output` will not clear: the project root,
            # `paths.slides` and `paths.manifest` themselves, a directory holding either,
            # anything not strictly below the project root, an existing path that is a
            # file rather than a directory, and either of those two keys when `Config.path`
            # refuses them. The last of those is the one the guard wraps rather than
            # letting out: `_protected_paths` resolves both keys off the *default*
            # configuration before the delete, so a `paths.slides: 7` refused the target
            # from inside the guard itself, as a `ConfigError` this arm did not catch and
            # this one does now. It is one arm rather than two because the answer is the
            # same either way — this target cannot be shown to be safe to clear — and a
            # caller that had to know which of two types stopped the guard would have to
            # know the guard's internals to report a configuration string. Each refusal is
            # an answer about the string `paths.patches` was hand-edited to, so it is
            # reported rather than raised, and each arrives ahead of `shutil.rmtree`,
            # which is what leaves the previous run on disk.
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

    # **The summary is written after the index and not before it.** The index is the run's
    # record of what exists on disk; a summary written first would report a run whose
    # patches had not been written yet, and a run that raised between the two would leave
    # a summary describing a patch tree no index describes. Both artifacts are written on
    # the path that succeeded, so neither exists without the other.
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / _SUMMARY_NAME).write_text(
        json.dumps(_document(stats, cfg, len(rows)), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

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
