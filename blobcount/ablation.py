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
the two totals. (The plan calls this "about 5.5x"; the two totals are the measurement
and the ratio above is what dividing them gives, so the ratio is what is written into
the artifact and the discrepancy is left visible rather than rounded away.
`RULE_COMPARISON` carries all of it into the JSON, and `bin/ablate_threshold.py` writes
it into every run.)

**So the question this sweep answers is where the distribution bends, not which
threshold balances the classes.** The per-threshold `n_steatosis` is the survival
function of the count distribution, and its knee is the threshold where that curve
turns; a per-specimen ratio is that curve for one specimen. A threshold that balances
the pooled ratio while the per-specimen spread is wide reproduces the defect above with
a different mechanism, and it is available to anyone who reads only the pooled column.

**`ablation.thresholds` as configured cannot place that knee, and the run says so.**
`configs/default.yaml` sweeps `[3, 5, 7, 10, 15, 20, 25, 30]`, and the largest of those
is below the mean count of 36.1 and well below the maximum of 86 that the rule in use
produced. Every threshold in the range therefore sits under the bulk of a distribution
that is itself inflated, so the sweep reports ratios in a region where the answer is
decided by the tail rather than by a bend. That is a finding about the configured range
and not a reason to widen it here: `ablation.thresholds` is a configured value and this
module reads it. Widening it until the knee is inside the range is a decision for
whoever selects the threshold, and the run hands them the survival curve to make it on.
`blobcount.extract.summarise_tissue_gray` reports a *different* distribution — the
`tissue_gray` background levels themselves — and the two must not be read as one.

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
`(patch_size, patch_size, 3)` array the count needs — and they arrive through the same
two seams, so a second type would have to re-enumerate six refusals or subclass an
empty one. Only the read of the index and the read of each patch are wrapped, with the
original as the `__cause__`. A `ValueError` from `count_blobs`, from
`np.percentile` on a `blobs.gray_percentile` outside [0, 100] — a value this project
range-checks nowhere — or from `sweep` and `summarise` above is a defect in this
project and is not wrapped, so the command line can tell a refusal from a traceback.
Every caught exception is logged with the specimen, the coordinates the index records
for that patch and the exception object, before the refusal is raised.

**A patch the index names twice is counted once**, as in `label_patches`, and the
repeat is logged: two rows are one file, and reading it twice would read every patch
twice.

**A configured threshold that is not a whole number of blobs is refused** rather than
rounded, because a sweep over 3.5 run as a sweep over 3 is a measurement of a
threshold nobody configured. It is a `ValueError` out of `ablate` and not a
`LabelingInputError`: it is a statement about the configuration, not about the index,
and it is the limit `bin/extract_patches.py` records for a hand-edited scalar — a
`TypeError` or `ValueError` out of a conversion this project does not wrap reaches the
interpreter, which reports it as a traceback. The command line checks the selection it
was given for the same reason and reports it as a refusal before it reads a patch.

Declared interface, in `__all__` order:

    AblationStats
        AblationStats.count_summary(self) -> dict[str, float | int]
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
        `ablate` refuses an index with no rows before it can build one: `sweep` raises
        on an empty count map rather than dividing by a length of nothing.
        """
        values = np.asarray(list(self.counts.values()), dtype=np.float64)
        return {
            "count": int(values.size),
            "min": int(values.min()),
            "p50": float(np.percentile(values, 50)),
            "mean": float(values.mean()),
            "max": int(values.max()),
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

    **The refusal is a `ValueError` and not a `LabelingInputError`.** It is a statement
    about the configuration and not about the index the caller pointed at, and it is
    the limit `bin/extract_patches.py` records for a hand-edited scalar: `load` checks
    that a key is declared and nothing about its value, so a conversion this project
    does not wrap reaches the interpreter, which reports it as a traceback. A `TypeError`
    from a value that is not a number at all is left the same way, and for the same
    reason — a catch wide enough to hold it would also hold a defect in this arithmetic.
    """
    out: list[int] = []
    for value in cfg.get("ablation.thresholds"):
        number = int(value)
        if number != value:
            raise ValueError(
                f"ablation.thresholds holds {value!r}, which is not a whole number of "
                f"blobs; a sweep is cut at the count a threshold states and rounding one "
                f"would measure a threshold nobody configured"
            )
        out.append(number)
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
    no patch read. **Only the read of the index and the read of each patch are caught**,
    and both are re-raised as a `LabelingInputError` with the original as the
    `__cause__`: the six shapes are the ones that type already enumerates, and they are
    statements about what the caller pointed at. A `ValueError` from `count_blobs`, from
    the normalization, from `np.percentile` on a `blobs.gray_percentile` outside
    [0, 100] or from the arithmetic of `sweep` and `summarise` is a defect in this
    project and is left to raise, so a command line can tell a refusal from a traceback.
    A `ValueError` out of `_thresholds` above is not wrapped either, for a different
    reason: it is a statement about the configuration and not about the index, and
    `bin/ablate_threshold.py` records that limit for the same reason
    `bin/extract_patches.py` does. Every caught exception is logged first, with the
    specimen id, the coordinates the index records for that patch and the exception
    object.

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
        raise LabelingInputError(f"the patch index could not be read: {error}") from error

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
