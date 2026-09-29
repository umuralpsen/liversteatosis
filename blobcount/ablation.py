"""Sweep the blob-count threshold, per specimen, and report the shape of what it cuts.

The threshold `labeling.blob_threshold` decides a class, and it was chosen on a pooled
class ratio. **The artifact that chose it is in this repository**, and it says exactly
what it did: `results/label_analysis/label_analysis.json` reports a `balance_ratio` per
threshold for 41,476 patches and no per-slide class ratio at all. At
`labeling.blob_threshold: 5` the pooled ratio is 0.4939 and the balance ratio 0.9757,
the closest to 1.0 of the eight thresholds in that file, and the same file gives a
per-slide mean blob count from 3.16 on slide 6 to 12.99 on slide 1. One pooled
threshold and a fourfold spread of per-slide means is a dataset whose labels depend on
which slide a patch came from, and the eight rows report it as a balance.

That is the error this module exists to make visible rather than to repeat. A pooled
ratio is one number over all the patches, so a set in which one slide reads entirely
steatotic and another entirely normal balances at exactly 0.5, which is the state
`test_pooled_balance_hides_per_slide_spread` names and reproduces. Every number here is
therefore reported per specimen as well as pooled, and the spread of the per-specimen
ratios is reported at every threshold beside the pooled one rather than instead of it.
The figures above are the superseded rule's own counts, kept here as the record of the
choice and not as a measurement of the rule the study now uses.

**The count distribution this sweep cuts was measured once, and it is inflated.**
Task 6 labelled 60 accepted patches of `data/2.svs` at 0.5001 um/px under two rules for
the gray threshold, on the same patches and at the same resolution. The 95th percentile
of the slide's `tissue_gray` column, **the rule in use**, counted **2166** blobs with 57
of the 60 patches non-zero and 86 on the fullest one. The 95th percentile of the patch's
own grayscale, the superseded rule, counted **377**, with 59 non-zero and 19 at most.

The cause is structural rather than a matter of tuning. `tissue_gray` is written by
`blobcount/extract.py:180` as the `extraction.tissue_percentile` of each patch, which
is a **5th** percentile in the shipped configuration, so the column ranges 69.3 to 144.7
on real tissue, and the 95th percentile of that column is **134.4** — below the mean
grayscale of 51 of those 60 patches. At `labeling.blob_threshold: 5` that is 36.1 blobs
per patch. The rule is not degenerate; it is inflated, by `2166 / 377` = **5.75** between
the two totals. (The controller note calls this "about 5.5x"; the two totals are the
measurement and the ratio above is what dividing them gives, so the ratio is what is
written into the artifact and the discrepancy is left visible rather than rounded away.
`RULE_COMPARISON` carries all of it into the JSON, and `bin/ablate_threshold.py` writes
it into every run.)

**So the question this sweep answers is where the distribution bends, not which
threshold balances the classes.** The per-threshold `n_steatosis` is the survival
function of the count distribution, and its knee is the threshold where that curve
turns; a per-specimen ratio is that curve for one specimen. A threshold that balances
the pooled ratio while the per-specimen spread is wide reproduces the defect above with
a different mechanism, and it is available to anyone who reads only the pooled column.

**Whether the configured range reaches the distribution is reported, not assumed.**
`configs/default.yaml` sweeps `[3, 5, 7, 10, 15, 20, 25, 30]`, and against the 60 patches
Task 6 measured, the largest of those is below the mean count of 36.1 and well below the
maximum of 86. **Those two figures are 60 patches of one of six slides, not a property of
the index this sweep runs over**, and that index does not exist on disk yet: Task 4's
extraction has never been run, so there is no `patches.csv` for a sweep to have cut. The
range is therefore not widened on the strength of a 60-patch sample. Widening a range from
a sample is a guess, and guessing a range is the mechanism that produced every wrong
number in the original thesis.

What ships instead is the check the run performs, in `AblationStats.range_coverage`: the
swept min and max against the count distribution's own min, p50, mean and max, a **count
of the patches the swept range leaves outside itself**, a boolean derived from that count,
and a `criterion` string naming the comparison so the boolean is not a claim the reader has
to take on trust. `bin/ablate_threshold.py` writes that under
`threshold_range_coverage` and prints it under the count summary, so a reader of either
learns whether the sweep reached the distribution instead of inferring it from a pooled
ratio printed beside a `blob_count_summary`. Where the range does bracket, the flag says
so and the sweep was adequate without the reader doing the arithmetic; where it does not,
the artifact says which end of the range stops short, by how much, and how many patches
that leaves carrying the same label at every threshold in the sweep.
`ablation.thresholds` is a configured value this module reads, and the run hands the
survival curve to whoever selects the threshold. `blobcount.extract.summarise_tissue_gray`
reports a *different* distribution — the `tissue_gray` background levels themselves — and
the two must not be read as one.

**`ablate` counts the patches again, and reads nothing that a threshold produced.**
`labels.csv` carries a `blob_count` per patch, and reading it would make the sweep a
function of the labelling run it is supposed to inform. So the counts are recomputed
from the index, through the same two seams `label_patches` uses: `read_index`,
`load_patch`, and the per-row `float()` conversion. **A patch is `steatosis` at or
above the threshold**, the same direction `label_patches` uses, so a threshold read
off this sweep and a threshold used by a labelling run mean the same number.

The gray threshold is the one `labeling` derives, and it is *that function* and not a
restatement of it: `_slide_gray_threshold` and `_row_numbers` are imported from
`blobcount.labeling` even though both are private. The alternative is a second
definition of the per-slide percentile in this module, and a percentile of a slide's
background levels is exactly the kind of number that has to have one definition —
`blobcount.blobs` imports `blobcount.extract._grayscale` for the same reason, and a
threshold derived here by a second rule would make this sweep a measurement of a rule
the study does not use. `BlobParams.gray_threshold` is per slide, each count is made
at its own row's `mpp`, and `prepare_patch` is applied between the read and the count
so a run with `blobs.stain_normalization` set counts what that key is about.

**`ablate` writes nothing.** It reads the index and the patches and returns what it
measured; the JSON, the figure and the printed tables are the command line's, because
the output directory is a configuration value the command line resolves. `plot` is the
one exception the plan makes, and it writes the figure it is handed and nothing else.

**`LabelingInputError` is reused rather than redefined.** The refusals here are the six
shapes that module already enumerates — an index that is not there, cannot be opened,
carries a cell over `csv.field_size_limit`, or is not the `patches.csv` header; a cell
that is not the number its column records; and a patch file that is not the `uint8`
`(patch_size, patch_size, 3)` array the count needs — plus **an index that carries no
rows**, which that type's docstring names as this module's seventh shape. They arrive
through the same three wrapped operations and not through a fourth, so a second type would
have to re-enumerate six refusals or subclass an empty one. Those three are the three
`LabelingInputError` documents: the read of the index and the read of each patch here, and
the per-row `float()` conversion in `blobcount.labeling._row_numbers`, every one of them
with the original as the `__cause__`. A `ValueError` from `count_blobs`, from
`np.percentile` on a `blobs.gray_percentile` outside [0, 100] — a value this project
range-checks nowhere — or from the arithmetic of `sweep` and `summarise` is a defect in
this project and is not wrapped, so the command line can tell a refusal from a traceback.

**Every caught exception is logged, and all three catches do it.** The patch read names
the specimen, the coordinates the index records for that patch and the exception object,
which are the three the plan's Global Constraints require. The cell conversion names the
same three, because it has the row they come from. The index read names the index and the
exception object and cannot name coordinates: no row was read, so there are none, and the
specimen the run has is the file. That one is logged in this module because
`blobcount.labeling.read_index` is public and raises what it raises — it is the boundary
here that has the specimen, and an index that cannot be read is the case where a log line
matters most and is most often missing.

**A patch the index names twice is counted once**, as in `label_patches`, and the
repeat is logged: two rows are one file, and reading it twice would read every patch
twice.

**A configured threshold that is not a whole number of blobs is refused** rather than
rounded, because a sweep over 3.5 run as a sweep over 3 is a measurement of a threshold
nobody configured — and a threshold that is not finite is not a count of blobs either, a
question `_thresholds` answers on its own rather than leaving to `int()`, which refuses
a `NaN` and an `inf` by raising two different exceptions. It is a `ValueError` out of
`ablate` and not a
`LabelingInputError`: it is a statement about the configuration, not about the index the
caller pointed at. **It is not a traceback either.** The standing ruling that a
hand-edited `extraction.patch_size` reaches the interpreter unvalidated does not transfer
to a function that exists to reject the value: `extraction.patch_size` is a value this
project chose not to validate, and `_thresholds` is code that was written to validate, and
a validator that validates and then reports by traceback looks like a check without being
one. `bin/ablate_threshold.py` calls `_thresholds` itself before it reads a patch and
returns `EXIT_REFUSED` on its answer; the call inside `ablate` is the same pure conversion
of the same configuration and cannot come to a different one. The command line checks the
selection it was given for the same reason, against the same converted range, and reports
it as a refusal before it reads a patch.

Declared interface, in `__all__` order:

    AblationStats
        AblationStats.count_summary(self) -> dict[str, float | int]
        AblationStats.range_coverage(self) -> dict[str, float | int | bool | str]
        counts: dict[str, int]
        slide_of: dict[str, str]
        per_slide_mpp: dict[str, float]
        per_slide_gray_threshold: dict[str, float]
        per_threshold: dict[int, dict]
        selected: int
    RULE_COMPARISON: dict[str, Any]
    ablate(index_path: Path, cfg: Config, selected: int | None = None) -> AblationStats
    plot(summarised: Mapping[int, Mapping[str, Any]], out_path: Path) -> None
    summarise(
        per_threshold: Mapping[int, Mapping[str, Any]],
        selected: int | None = None,
    ) -> dict[int, dict]
    sweep(
        counts_by_patch: Mapping[str, int],
        slide_of: Mapping[str, str],
        thresholds: Sequence[int],
    ) -> dict[int, dict]
"""

from __future__ import annotations

import csv
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blobcount.blobs import BlobParams, count_blobs, prepare_patch
from blobcount.config import Config
from blobcount.labeling import (
    LabelingInputError,
    _row_numbers,
    _slide_gray_threshold,
    load_patch,
    read_index,
)

__all__ = [
    "AblationStats",
    "RULE_COMPARISON",
    "ablate",
    "plot",
    "summarise",
    "sweep",
]

LOGGER = logging.getLogger(__name__)

# The five names every record of a sweep carries, and the five `summarise` rebuilds from
# one. The spread is computed once, in `sweep`, and copied rather than recomputed here:
# a second definition of "how far apart are the slides" is a second number that can come
# to disagree with the first, and this module's whole argument is that the spread and
# the pooled ratio are two different numbers.
_SWEEP_FIELDS = (
    "n_normal",
    "n_steatosis",
    "pooled_steatosis_ratio",
    "per_slide_ratio",
    "per_slide_ratio_spread",
)

# What `brackets_distribution` means, written into every artifact beside the boolean
# itself, so a reader of `threshold_range_coverage` does not have to have this module
# open to know what `true` claims. It names **both** ends and the statistic each is held
# against, and it is one definition: the code below computes the same two comparisons and
# a test evaluates this string over the numbers the block carries, so the string, the
# boolean and the predicate cannot come to different answers about the same run.
#
# The form is deliberate. Everything after ` iff ` is a comparison over sibling keys, and
# `tests/test_ablation.py` evaluates it in a namespace holding those keys and nothing
# else, which is what makes "the string and the boolean agree" a check rather than a
# promise. A prose criterion would satisfy a reader and fail that test.
_RANGE_CRITERION = (
    "brackets_distribution is true iff swept_min <= count_min and swept_max >= count_max"
)

# The two rules Task 6 measured on the same 60 accepted patches of `data/2.svs` at
# 0.5001 um/px, and what each of them counted. This is a record of a past measurement,
# not a re-derivation of it: nothing here re-counts those patches, and the numbers are
# carried into every artifact this sweep writes so that a reader knows what the counts
# in it were made by. `total_blob_ratio` is the quotient of the two totals, 2166 / 377
# = 5.75, written as a number so that a reader who divides the two totals in the same
# file finds the same value rather than a rounded phrase.
RULE_COMPARISON: dict[str, Any] = {
    "source": "60 accepted patches of data/2.svs at 0.5001 um/px, measured in Task 6",
    "rule_in_use": {
        "rule_in_use": True,
        "rule": "95th percentile of the slide's tissue_gray column",
        "total_blobs": 2166,
        "non_zero_patches": 57,
        "max_per_patch": 86,
    },
    "superseded_rule": {
        "rule_in_use": False,
        "rule": "95th percentile of the patch's own grayscale",
        "total_blobs": 377,
        "non_zero_patches": 59,
        "max_per_patch": 19,
    },
    "total_blob_ratio": 2166 / 377,
}


@dataclass(frozen=True)
class AblationStats:
    """What one ablation measured: the counts, the parameters, and the sweep over them.

    `counts` is keyed by the index's own `patch_id`, which `extract_slide` writes with
    its `.png` extension, and `slide_of` maps the same keys to the slide that patch came
    from. They are reported rather than derived because the survival curve the knee
    question asks about is a statement about these numbers, and an artifact that carried
    only ratios could not be re-examined without re-reading every patch.

    `per_slide_mpp` and `per_slide_gray_threshold` are the parameters the counts were
    made under, one per slide, and both are summaries rather than the values any count
    used: two patches of one slide can differ in the fourth decimal of `mpp`, and the
    value that decided each count is the one in its own index row. They are the same two
    summaries `LabelingStats` carries, over the same derivation.

    `selected` is the threshold the study labels at, and the key of the one record in
    `per_threshold` whose `selected` is true. It defaults to `labeling.blob_threshold`
    because that is the threshold in use, and a selection of anything else is what
    `--selected` is for. **A selection outside `per_threshold` is refused**, by
    `summarise`, because the second panel of the figure is drawn at it and a figure
    with an empty panel reports nothing.
    """

    counts: dict[str, int]
    slide_of: dict[str, str]
    per_slide_mpp: dict[str, float]
    per_slide_gray_threshold: dict[str, float]
    per_threshold: dict[int, dict]
    selected: int

    def count_summary(self) -> dict[str, float | int]:
        """Return the shape of the count distribution the sweep is a sweep of.

        The keys are `count`, `min`, `p50`, `mean` and `max`, the same names
        `blobcount.extract.summarise_tissue_gray` uses for the `tissue_gray` column, and
        the two summaries are about different distributions: that one is the recorded
        background level of each patch and this one is the number of blobs found in it.
        `min` and `max` are whole numbers because a count of blobs is one, and `p50` and
        `mean` are not, because the median of an even number of counts is not. A count of
        36.1 per patch against a swept threshold of 30 is a state the reader has to see,
        not a ratio that hides it.

        There is no percentile over an empty array here to return a `NaN` for, because
        `ablate` refuses an index with no rows at the index boundary, as a
        `LabelingInputError` and not as a `ValueError` out of `sweep` below.
        """
        values = np.asarray(list(self.counts.values()), dtype=np.float64)
        return {
            "count": int(values.size),
            "min": int(values.min()),
            "p50": float(np.percentile(values, 50)),
            "mean": float(values.mean()),
            "max": int(values.max()),
        }

    def range_coverage(self) -> dict[str, float | int | bool | str]:
        """Return the swept range against the count distribution it is a sweep of.

        The keys are `criterion`, `swept_min`, `swept_max`, `count_min`, `count_p50`,
        `count_mean`, `count_max`, `unswept_below`, `unswept_above`, `n_unswept` and
        `brackets_distribution`. The four `count_` keys are read from `count_summary`
        rather than computed again, so the comparison is between the distribution the run
        printed and the range the run swept and not between a number here and a number
        there that can come to disagree.

        **The range swept is the keys of `per_threshold`, not a configured value read a
        second time.** Those keys are what `sweep` was given and `summarise` marked, so
        this cannot report a range the run did not cut at; and `ablate` cannot produce an
        empty `per_threshold`, because `summarise` refuses a selection that is not in it.
        A caller that hands this dataclass an empty one gets a refusal rather than a
        `ValueError` out of `min()` over nothing.

        **`criterion` is the comparison, in the artifact, and the boolean is read from it.**
        It is `_RANGE_CRITERION`, a module constant rather than a phrase built here, and it
        is the answer to "what does `true` mean here" for a reader holding only
        `threshold_ablation.json` and not this module. A file carrying a verdict and no
        criterion admits two readings of that verdict, and only one of them is the one
        this code implements. A test evaluates the string against the numbers in the same
        block and fails if it disagrees with the boolean, so a predicate that moves
        without the string moving is a red test rather than a silent change of meaning.

        **The low end is held against the distribution's `min` and not against its `p50`.**
        A median is the wrong end to hold a range against: a range whose bottom sits above
        the distribution's smallest count but below its median then reports that the sweep
        covered the distribution while the patches below that bottom sit under every
        threshold in the sweep. Counts of 1, 50, 51 and 52 swept at `[40, 60]` are the
        shape of it — `p50` is 50.5, so a bottom of 40 clears a median, and the patch at
        1 is under the whole swept range. **The high end is held against `max`:** a range
        that stops below the largest count leaves the tail above every threshold in it,
        which is the same defect at the other end, and the same argument that rules the
        median out rules anything but the maximum in on the top.

        **`unswept_below`, `unswept_above` and `n_unswept` are counts, and they are the
        reason the boolean is trustworthy rather than a second way of saying it.**
        `unswept_below` is the number of patches counted **below** `swept_min` and
        `unswept_above` the number counted **above** `swept_max`; `n_unswept` is their sum,
        and `brackets_distribution` is `n_unswept == 0`. The count is the primary number
        because a proportion of the cohort is a statement a reader cannot misread the way
        a verdict can: `1 of 4 patches outside the swept range` needs no prose to
        interpret, and `brackets_distribution: false` needs this docstring. The two ends
        are counted separately so the artifact says *which* end stopped short without a
        reader subtracting the two ranges by hand — the arithmetic that produces the
        verdict, done in the code and reported as a number. Nothing here counts a patch
        the sweep did not label: every patch is `normal` or `steatosis` at every threshold
        in the range, and these are the patches whose label is the same at all of them.

        **A `False` here is a finding about the configuration and not a defect.** The
        controller decision is that `ablation.thresholds` is not widened on the strength
        of a 60-patch sample, because the real distribution does not exist yet. So the
        flag reports what the run found and the reader decides what to do about it.
        """
        swept = list(self.per_threshold)
        if not swept:
            raise ValueError(
                "there is no swept threshold to compare a range against; the keys of "
                "`per_threshold` are the range this run swept, and there are none"
            )
        summary = self.count_summary()
        low, high = min(swept), max(swept)
        values = list(self.counts.values())
        below = sum(1 for value in values if value < low)
        above = sum(1 for value in values if value > high)
        return {
            "criterion": _RANGE_CRITERION,
            "swept_min": int(low),
            "swept_max": int(high),
            "count_min": summary["min"],
            "count_p50": summary["p50"],
            "count_mean": summary["mean"],
            "count_max": summary["max"],
            "unswept_below": below,
            "unswept_above": above,
            "n_unswept": below + above,
            "brackets_distribution": bool(below == 0 and above == 0),
        }


def sweep(
    counts_by_patch: Mapping[str, int],
    slide_of: Mapping[str, str],
    thresholds: Sequence[int],
) -> dict[int, dict]:
    """Return, for each threshold, the class distribution pooled and per slide.

    A patch is `steatosis` when its count is **at or above** the threshold and `normal`
    below it, which is the direction `label_patches` labels in, so a threshold read off
    this sweep and a threshold a labelling run used mean the same number.

    Each record carries `n_normal`, `n_steatosis`, `pooled_steatosis_ratio`,
    `per_slide_ratio` — slide id to that slide's own steatosis fraction — and
    `per_slide_ratio_spread`, the max minus the min of those per-slide ratios.

    **The grouping is the whole of this function.** The pooled ratio is one number over
    all the patches and the per-slide ratio is one number per slide, and the second is
    not derivable from the first: 100 patches at 20 on one slide and 100 at 0 on
    another give a pooled ratio of exactly 0.5 with a spread of 1.0, which is the state
    the original code read as a balanced threshold. Every count is therefore attributed
    to its slide before either number is formed, and the two are formed from different
    sums.

    The slides appear in the order their first patch appears in `counts_by_patch`, so
    the keys, the figure and the JSON are stable for a given index rather than ordered
    by whatever a set produced.

    Each threshold is used both as the comparison and as its own key, and nothing here
    rounds one: a caller that passes `3.5` sweeps at 3.5 and keys the record `3.5`.
    `ablate` is what refuses a configured value that is not a whole number, because a
    sweep over 3.5 run as a sweep over 3 is a measurement of a threshold nobody
    configured.

    **The two mappings must agree about every patch.** A count with no slide and a slide
    with no count are each one patch that would be counted in the pooled ratio and in no
    per-slide ratio, or in none of either, and the record would then not add up. Both
    are refused, naming the patch, rather than one of them being dropped silently.

    An empty `counts_by_patch` is refused too: `pooled_steatosis_ratio` is
    `n_steatosis / n` and there is no `n` in an empty map, so a sweep over an index with
    no rows would divide by nothing and leave a command line with a `ZeroDivisionError`.
    `ablate` never reaches this with an empty map — an index with no rows is refused at
    the index boundary, as a `LabelingInputError` a command line has a message for — so
    this is the guard a direct caller of `sweep` meets, and it stays a `ValueError` because
    the two mappings disagree about the study's data rather than about the index file.
    """
    if not counts_by_patch:
        raise ValueError(
            "the sweep was given no patches, so the pooled ratio is a division by an "
            "empty count; an index with no rows is a run that extracted nothing, and "
            "the blank slide is reported where it is detectable, at extraction"
        )
    unassigned = sorted(set(counts_by_patch) - set(slide_of))
    counted_nowhere = sorted(set(slide_of) - set(counts_by_patch))
    if unassigned:
        raise ValueError(
            f"patch {unassigned[0]!r} has a blob count and no slide, and "
            f"{len(unassigned)} patches are in that state, so they would be counted in "
            f"the pooled ratio and in no per-slide ratio and the record would not add up"
        )
    if counted_nowhere:
        raise ValueError(
            f"patch {counted_nowhere[0]!r} is assigned to slide "
            f"{slide_of[counted_nowhere[0]]!r} and has no blob count, and "
            f"{len(counted_nowhere)} patches are in that state, so they would be "
            f"counted in no ratio at all"
        )

    grouped: dict[str, list[int]] = {}
    for patch, count in counts_by_patch.items():
        grouped.setdefault(slide_of[patch], []).append(count)

    out: dict[int, dict] = {}
    for threshold in thresholds:
        per_slide: dict[str, float] = {}
        for slide, values in grouped.items():
            steatotic = sum(1 for value in values if value >= threshold)
            per_slide[slide] = steatotic / len(values)
        total = len(counts_by_patch)
        n_steatosis = sum(
            1 for value in counts_by_patch.values() if value >= threshold
        )
        ratios = list(per_slide.values())
        out[threshold] = {
            "n_normal": total - n_steatosis,
            "n_steatosis": n_steatosis,
            "pooled_steatosis_ratio": n_steatosis / total,
            "per_slide_ratio": per_slide,
            "per_slide_ratio_spread": max(ratios) - min(ratios),
        }
    return out


def summarise(
    per_threshold: Mapping[int, Mapping[str, Any]],
    selected: int | None = None,
) -> dict[int, dict]:
    """Return `sweep`'s records, each marked with whether it is the selected threshold.

    Every record is rebuilt rather than passed through, so the output carries exactly
    the five numbers a sweep produces plus the mark, with the counts as `int` and the
    ratios as `float`, whatever the input held. The spread is **copied** and not
    recomputed from the per-slide ratios: it is defined once, in `sweep`, and a second
    definition here is a second number that could come to disagree with the first
    without anything failing.

    `selected` is the threshold the study labels at, and it is compared against the keys
    it is given, so a selection of `5` marks the record keyed `5` and nothing else. A
    `selected` naming a threshold that was not swept is refused, because the second
    panel of `plot` is drawn at the selected threshold and a record that marks none
    leaves that panel empty. `bin/ablate_threshold.py` checks the same fact for the
    value it was given before it reads a single patch, and reports it as a refusal;
    that is the caller's answer, and this is the invariant behind it.

    `None` is the "no threshold is selected" case and it is not an error here: a
    summary of a whole sweep, with nothing chosen out of it yet, is a legitimate thing
    to look at. `plot` is what needs a selection, and it refuses without one.
    """
    if selected is not None and selected not in per_threshold:
        raise ValueError(
            f"the selected threshold {selected!r} is not one of the swept thresholds "
            f"{sorted(per_threshold)}; a threshold that was not swept has no record to "
            f"mark, and the per-slide figure is drawn at the selected one"
        )
    out: dict[int, dict] = {}
    for threshold, record in per_threshold.items():
        missing = [name for name in _SWEEP_FIELDS if name not in record]
        if missing:
            raise ValueError(
                f"the record for threshold {threshold!r} carries no {', '.join(missing)}; "
                f"`summarise` assembles the output of `sweep` and not a partial record"
            )
        out[threshold] = {
            "n_normal": int(record["n_normal"]),
            "n_steatosis": int(record["n_steatosis"]),
            "pooled_steatosis_ratio": float(record["pooled_steatosis_ratio"]),
            "per_slide_ratio": {
                str(slide): float(value)
                for slide, value in record["per_slide_ratio"].items()
            },
            "per_slide_ratio_spread": float(record["per_slide_ratio_spread"]),
            "selected": threshold == selected,
        }
    return out


def plot(summarised: Mapping[int, Mapping[str, Any]], out_path: Path) -> None:
    """Write the two-panel ablation figure to `out_path` and close it.

    **The left panel is the sweep and the right panel is the selection.** The left one
    plots each slide's own ratio against threshold, which is that slide's survival curve
    for the count distribution, with the pooled ratio as a dashed line beside them and a
    dotted marker at the selected threshold. It is the panel that shows where a slide's
    curve turns, and the two lines crossing is the pooled ratio and the per-slide spread
    telling different stories — which is the whole reason the module exists. The right
    panel is the per-slide bar chart at the selected threshold, with the pooled ratio
    as a horizontal line, so a reader can see the distance between the bar and the line
    for every slide at once.

    Both panels are read from the same `summarised` mapping and neither recomputes a
    ratio, so the figure and the JSON and the printed table cannot disagree.

    **Exactly one record must be marked selected.** Zero leaves the right panel with no
    threshold to draw, and is refused rather than drawn empty. More than one is a summary
    two callers disagree about, and drawing the first of them would pick a winner
    silently, so it is refused too.

    `matplotlib.use("Agg")` is called before `pyplot` is imported, and `pyplot` is
    imported inside the function: a run writes a file and never shows a window, and a
    module-scope import of a native wheel turns an absent one into a collection error
    for every test in the suite, including the ones that touch no figure. The figure is
    closed before returning — matplotlib counts the figures a process holds open and
    warns past `figure.max_open_warning` of them, twenty by default, and
    `filterwarnings = ["error"]` turns that warning into a failure, so an unclosed
    figure is caught instead of accumulating for the rest of the process.

    `out_path`'s parent is created if it is absent, and the figure is overwritten when
    one is already there: this is a report of a measurement, and a previous run's figure
    of the same measurement is not something to refuse. Nothing else is written, and
    the guard on what this project may write into is the command line's, not this
    function's.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds = sorted(summarised)
    if not thresholds:
        raise ValueError("there is no threshold to plot, so the sweep is empty")
    marked = [threshold for threshold in thresholds if summarised[threshold]["selected"]]
    if len(marked) != 1:
        raise ValueError(
            f"{len(marked)} of the {len(thresholds)} swept thresholds are marked "
            f"selected, and the per-slide panel is drawn at exactly one of them; call "
            f"`summarise` with the selected threshold, or with none"
        )
    chosen = marked[0]
    slides = list(summarised[chosen]["per_slide_ratio"])

    figure, (curves, bars) = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for slide in slides:
        curves.plot(
            thresholds,
            [summarised[threshold]["per_slide_ratio"][slide] for threshold in thresholds],
            marker="o",
            markersize=4,
            label=f"slide {slide}",
        )
    curves.plot(
        thresholds,
        [summarised[threshold]["pooled_steatosis_ratio"] for threshold in thresholds],
        linestyle="--",
        color="black",
        linewidth=1.2,
        label="pooled",
    )
    curves.axvline(chosen, color="grey", linestyle=":", linewidth=1.2)
    curves.set_title("per-slide steatosis ratio over threshold")
    curves.set_xlabel("blob count threshold, steatosis at or above")
    curves.set_ylabel("fraction of that slide's patches")
    curves.set_ylim(0.0, 1.05)
    curves.set_xticks(thresholds)
    curves.grid(axis="y", linewidth=0.4, alpha=0.5)
    curves.legend(fontsize="small")

    selected_record = summarised[chosen]
    # Positions and explicit tick labels rather than the slide ids as the bar
    # coordinates: a specimen id here is the string "2" as often as it is "hasta-01", and
    # matplotlib reads a list of strings that all parse as numbers as a numeric axis and
    # says so on stderr. Naming the positions leaves the id a label, which is what it is.
    positions = list(range(len(slides)))
    bars.bar(
        positions,
        [selected_record["per_slide_ratio"][slide] for slide in slides],
        color="#7a9cc6",
    )
    bars.set_xticks(positions, labels=list(slides))
    bars.axhline(
        selected_record["pooled_steatosis_ratio"], linestyle="--", color="black", linewidth=1.2
    )
    bars.set_title(
        f"per-slide ratio at threshold {chosen}, pooled "
        f"{selected_record['pooled_steatosis_ratio']:.3f}, spread "
        f"{selected_record['per_slide_ratio_spread']:.3f}"
    )
    bars.set_xlabel("slide")
    bars.set_ylabel("fraction of that slide's patches")
    bars.set_ylim(0.0, 1.05)
    bars.grid(axis="y", linewidth=0.4, alpha=0.5)

    figure.tight_layout()
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)


def _thresholds(cfg: Config) -> tuple[int, ...]:
    """Return `ablation.thresholds` as whole numbers of blobs, refusing anything else.

    `ablation.thresholds` is the configured value that decides where this sweep looks, so
    a value that is not a whole number of blobs is refused rather than rounded or
    floored: a sweep over 3.5 run as a sweep over 3 is a measurement of a threshold
    nobody configured, and one nobody configured is the same class of number as the
    constant this project keeps refusing to substitute for a measurement.

    **A quoted digit is refused, a YAML boolean is refused, and the refusal message says
    which.** An entry has to be a number: `["3", 5]` fails the test below because `3` and
    `"3"` are not the same value, and the message says so in those words rather than
    leaving a user hand-editing `configs/default.yaml` to infer the rule from which
    exception it produced. **A `true` is refused by a type test rather than by the
    comparison beside it, and that is not pedantry:** `bool` subclasses `int`, so
    `int(True) == 1` and `1 != True` is False, and a `true` in the range would pass the
    whole-number test and be swept as the threshold 1 with nothing said about it. The
    alternative — accepting a quoted digit and converting it — was available and was not
    taken: this key is read by a caller that prints the range it swept, and a range
    printed as `(3, 5)` beside a configuration that says `["3", 5]` is a second thing for
    a reader to reconcile, in an artifact whose whole argument is that two numbers
    disagreeing should be visible rather than reconciled silently.

    **The refusal is a `ValueError` and not a `LabelingInputError`.** It is a statement
    about the configuration and not about the index the caller pointed at, and the
    `LabelingInputError` this module reuses enumerates six shapes that are all about the
    file. A seventh about a configuration value would make that type mean two things.

    **A non-finite float is refused by the same predicate, and not by the standard
    library's refusal reaching through it.** `isinstance(float("inf"), float)` is `True`,
    so a `.inf` passed the type test and arrived at `int()`. The two non-finite values do
    not fail the same way, and the difference is recorded here rather than relied upon:
    `int(float("nan"))` raises `ValueError` and `int(float("inf"))` raises
    `OverflowError`. So NaN was refused and infinity was not, each for a reason that has
    nothing to do with the rule above — NaN's refusal carried CPython's
    `cannot convert float NaN to integer`, which names no rule, and infinity escaped to
    `bin/ablate_threshold.py`, whose catch is `(TypeError, ValueError)`, and left a
    traceback on the status that command line documents as a defect in this project's own
    code, from a hand-edited YAML value. **A value that is not a finite whole number is
    not a whole number, and that is a statement this predicate can make on its own, so it
    makes it** — which is also why fixing it in the command line's catch was not enough:
    a library caller of this function got the traceback either way.

    **`OverflowError` is deliberately *not* added to the command line's catch tuple.** With
    the finiteness test above, this function cannot raise it from this call: `int()` is
    reached only for a value the predicate has already accepted as a finite number. An
    `OverflowError` out of the one call that command line wraps would be a defect in this
    validator rather than a mistyped `configs/default.yaml`, and that file documents a
    defect in its own code as a traceback on purpose, because a traceback names a file and
    a line and a refusal names neither. Catching it would dress a validator bug as a
    refusal on stderr, which is the "looks validated and is not" state this function
    exists to end, and it would contradict the sentence in that file's docstring that its
    catch is wide enough for a hand-edit and too narrow for a defect. The catch holds
    mistyped configuration values, and the finiteness test is what makes that true of every
    one of them.

    **The caller reports it, and it does not reach the interpreter.** The standing limit
    `bin/extract_patches.py` records is that `load` checks that a key is declared and
    nothing about its value, so a conversion this project does not wrap is reported as a
    traceback. That limit covers `extraction.patch_size`, a value this project chose not
    to validate, and it does not transfer to a function written to reject a value: a
    validator that validates and then reports by traceback looks like a check and is not
    one. `bin/ablate_threshold.py` therefore calls this function itself, before it reads a
    patch, and returns `EXIT_REFUSED` on its answer. The catch there is around that one
    call and not around `ablate`, so it is too narrow to hold a defect in the counting.
    The `TypeError` it also catches comes from the key itself not being a sequence —
    `load` accepts that, having validated that the key is declared and nothing about its
    value — and not from an entry, because every entry is checked here, for its type and
    then for its finiteness, before it is converted.
    """
    out: list[int] = []
    for value in cfg.get("ablation.thresholds"):
        # One predicate decides what a whole number of blobs is, and it runs to completion
        # before the `int()` rather than after it, so the conversion below is only ever
        # handed a value that already has an answer and cannot raise on its own.
        #
        # The type test covers the quoted digit, the boolean and the `None`. It cannot be
        # a comparison alone, because `int(True) == 1` and `1 != True` is False: a
        # comparison would sweep a `true` as the threshold 1 in silence.
        is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
        # The finiteness test, and it is scoped to floats on purpose in both directions.
        # `math.isfinite` converts its argument to a float, so `math.isfinite(10 ** 400)`
        # raises an `OverflowError` of its own and testing every entry would trade the bug
        # this predicate was written for a second one a layer below it; and there is no
        # such thing as an infinite `int`, so not asking costs nothing. `3.0` stays on the
        # accepted side because the test is `isfinite` and not `is_integer` — a range a
        # user computed arrives as floats, and refusing those would be the same mistake as
        # refusing a quoted digit.
        is_finite = not isinstance(value, float) or math.isfinite(value)
        # `whole` is `None` for anything the two tests above refused, so the single raise
        # below covers the type, the non-finite value and the fraction together, and
        # `whole != value` is the fraction test alone.
        whole = int(value) if is_number and is_finite else None
        if whole is None or whole != value:
            raise ValueError(
                f"ablation.thresholds holds {value!r}, which is not a whole number of "
                f"blobs: every entry has to be a plain number that is finite, 3 and not "
                f"'3' and not true and not .inf or .nan, because a sweep is cut at the "
                f"count a threshold states and a quoted digit, a rounded fraction or a "
                f"non-finite one would measure a threshold nobody configured"
            )
        out.append(whole)
    return tuple(out)


def ablate(index_path: Path, cfg: Config, selected: int | None = None) -> AblationStats:
    """Count every patch the index names and return the sweep over the configured thresholds.

    `ablation.thresholds` is the range swept and `selected` is the one threshold the
    study labels at, which defaults to `labeling.blob_threshold` because that is the
    threshold in use. `blobs.gray_percentile`, `blobs.area_um2_min`,
    `blobs.area_um2_max`, `blobs.circularity_min`, `blobs.morph_kernel_size` and
    `blobs.morph_iterations` are the counting criterion, and
    `blobs.stain_normalization` reaches the count through `prepare_patch`; all seven are
    the same values `label_patches` reads, so the two stages count the same pixels the
    same way. `extraction.patch_size` is what every patch file must be, and
    `load_patch` refuses a patch that is not.

    **`labels.csv` is not read.** The sweep has to be independent of the threshold that
    produced the labels, so the counts are made again here, through `read_index`,
    `load_patch` and the per-row `float()` conversion, and nothing a labelling run wrote
    reaches this function.

    Each patch is counted at **its own recorded `mpp`**, with the `BlobParams` of **its
    own slide**, and the gray threshold in those parameters is the
    `blobs.gray_percentile` of that slide's `tissue_gray` column — the same derivation
    `label_patches` uses, called rather than restated, for the reason the module
    docstring gives. `params_from_config` is not called: its own docstring records that
    the per-patch gray threshold it derives is superseded.

    The index is read before anything is counted, so an index that cannot be read costs
    no patch read. **An index that carries no rows is refused there, as a
    `LabelingInputError`**, before `_thresholds` is read and before a single cell is
    converted. It is a legitimate state a caller can hit — extraction produced nothing, or
    the wrong index was named — and it deserves the same answer a missing file already
    gets. Refused as a `ValueError` out of `sweep` instead, which is what this function
    used to let happen, it reached a command line that catches only `LabelingInputError`
    and left a traceback, and a status of 1 is what this CLI's own contract defines as
    "a fact about this project's code rather than about the study". The message it produced
    named an empty index, which is a fact about the study's data and not about our code.

    **Only the read of the index and the read of each patch are caught here**, and both
    are re-raised as a `LabelingInputError` with the original as the `__cause__`: the six
    shapes are the ones that type already enumerates, and they are statements about what
    the caller pointed at. The third wrapped operation, the conversion of the two numeric
    cells, is caught in `blobcount.labeling._row_numbers` and reaches here as the same
    type, so the three catches between the index file and this return value are the three
    that type documents. A `ValueError` from `count_blobs`, from the normalization, from
    `np.percentile` on a `blobs.gray_percentile` outside [0, 100] or from the arithmetic
    of `sweep` and `summarise` is a defect in this project and is left to raise, so a
    command line can tell a refusal from a traceback.

    A `ValueError` out of `_thresholds` above is not wrapped either, for a different
    reason: it is a statement about the configuration and not about the index, so
    `LabelingInputError` would be the wrong type for it. It is a `ValueError` and not a
    traceback because `bin/ablate_threshold.py` calls `_thresholds` on the same
    configuration before it reads a patch and reports the refusal as `EXIT_REFUSED`, so
    the call here is a second answer to a question already answered and cannot produce a
    third. Every caught exception is logged first, with the specimen id, the coordinates
    the index records for that patch and the exception object; the index read names the
    index and the exception object, because no row was read and so there are no
    coordinates to name.

    A patch the index names twice is counted once and the repeat is logged, so a
    doubled row does not double a run.

    Nothing is written. The JSON, the figure and the printed tables are the command
    line's, because the output directory is a configuration value it resolves.
    """
    index = Path(index_path)
    try:
        rows = read_index(index)
    except (OSError, ValueError, csv.Error) as error:
        # The same three the labeling boundary catches: the file is not there, cannot be
        # opened, or carries a cell over `csv.field_size_limit` — and a header that is
        # not this pipeline's, which `read_index` raises `ValueError` for. All four are
        # about the file the caller pointed at, and the original is the `__cause__` of
        # every one because the message names the index and the traceback names the read.
        # Logged first, with the index and the exception object. There are no coordinates
        # to name and no specimen but the file itself: the read failed before a row
        # existed, and inventing a row to name would be a log line about a patch the index
        # never carried.
        LOGGER.error("reading the patch index %s raised %r", index, error)
        raise LabelingInputError(f"the patch index could not be read: {error}") from error

    if not rows:
        # Refused here rather than left to `sweep` below, which raises a `ValueError` over
        # an empty count map. Both would report the same fact, and only one of them is a
        # type the command line has an answer for: a `LabelingInputError` is printed with
        # what it refused and returns `EXIT_REFUSED`, and a `ValueError` is a status of 1,
        # which this CLI defines as a fact about this project's code. An index with a
        # header and no rows is a legitimate state — extraction produced nothing, or the
        # wrong index was named — and it gets the same answer as a missing file.
        raise LabelingInputError(
            f"the patch index {index} carries a header and no rows, so there is no patch "
            f"to count and the pooled ratio would be a division by nothing; extraction "
            f"produced nothing, or this is not the index the run was meant to sweep"
        )

    thresholds = _thresholds(cfg)
    if selected is None:
        selected = int(cfg.get("labeling.blob_threshold"))
    percentile = float(cfg.get("blobs.gray_percentile"))
    patch_size = int(cfg.get("extraction.patch_size"))
    area_um2_min = float(cfg.get("blobs.area_um2_min"))
    area_um2_max = float(cfg.get("blobs.area_um2_max"))
    circularity_min = float(cfg.get("blobs.circularity_min"))
    morph_kernel_size = int(cfg.get("blobs.morph_kernel_size"))
    morph_iterations = int(cfg.get("blobs.morph_iterations"))

    # The two numeric columns are converted once per row, before the first patch is
    # read, so a cell that is not a number costs no read and the values below are read
    # from the converted list rather than from the cell again.
    numbers = [_row_numbers(row) for row in rows]
    # Positions rather than row dictionaries, so the grouping indexes `numbers` as well
    # and the two stay in step.
    grouped: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        grouped.setdefault(str(row["slide_id"]), []).append(position)

    # The `BlobParams` below and the `np.percentile` it calls are outside every `try`,
    # and that is the point: `blobs.gray_percentile` is a configured value this project
    # range-checks nowhere and `np.percentile` is the only thing that refuses one
    # outside [0, 100]. Wrapped, that refusal would reach the caller as a statement
    # about the index, which is a file with nothing wrong with it.
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

    counts: dict[str, int] = {}
    slide_of: dict[str, str] = {}
    for position, row in enumerate(rows):
        # `extract_slide` writes the file name into the column, so the id carries its
        # `.png` and the patch is the file the index's own directory holds. No extension
        # is appended: an index that names no file names no patch, and completing the
        # name would count a file this index did not name and file it under a patch id
        # the index never carried.
        #
        # **The two stages resolve one index row to different files, deliberately, and
        # that is stated here rather than left for a reader to discover.**
        # `label_patches` completes a missing `.png` through `labeling._patch_file_name`,
        # so a row of `A_0_0` labels `A_0_0.png`; this module refuses the same row.
        # Both are right for their own question: a labelling run has to name the file it
        # writes beside the index row it read, and a sweep that guessed a file name would
        # be counting a file this index did not name, under an id it never carried. The
        # divergence only shows on an index this pipeline did not write, because
        # `extract_slide` puts the extension in the column.
        patch_id = str(row["patch_id"])
        slide_id = str(row["slide_id"])
        if patch_id in counts:
            LOGGER.warning(
                "index %s names patch %s more than once; the first row keeps its count "
                "and the repeat is not counted again",
                index, patch_id,
            )
            continue
        try:
            source = load_patch(index.parent / patch_id, patch_size)
        except (OSError, ValueError) as error:
            # Logged with the specimen, the coordinates this index records for the patch
            # and the exception object, then refused. The tuple is the two a corrupt
            # patch file produces: an `OSError` for a path that is not a readable image
            # and the `ValueError` `load_patch` raises for an array that is not the
            # `uint8` (patch_size, patch_size, 3) one the count needs. A `TypeError` is
            # deliberately not here, for the reason `label_patches` gives: no corrupt
            # file produces one, so an arm for it could only report a defect in this
            # project as a patch the index named wrongly.
            LOGGER.error(
                "slide %s: reading patch %s at (%s, %s) raised %r",
                slide_id, patch_id, row["x"], row["y"], error,
            )
            raise LabelingInputError(
                f"slide {slide_id}: the patch {patch_id} at "
                f"({row['x']}, {row['y']}) could not be used: {error}"
            ) from error
        patch = prepare_patch(source, cfg)
        counts[patch_id] = count_blobs(patch, params[slide_id], numbers[position].mpp)
        slide_of[patch_id] = slide_id

    return AblationStats(
        counts=counts,
        slide_of=slide_of,
        per_slide_mpp={
            slide_id: float(np.mean([numbers[position].mpp for position in slide_rows]))
            for slide_id, slide_rows in grouped.items()
        },
        per_slide_gray_threshold={
            slide_id: item.gray_threshold for slide_id, item in params.items()
        },
        per_threshold=summarise(sweep(counts, slide_of, thresholds), selected),
        selected=selected,
    )
