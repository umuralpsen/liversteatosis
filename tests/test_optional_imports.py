"""The suite's own optional-dependency guards, asserted rather than assumed.

This module is not a package-module test. It is the one cross-cutting property of the
test suite that no single module can assert about itself: that every
`pytest.importorskip` call in `tests/` states which exception it is skipping on.

**The measured failure this exists to prevent.** A native binding that is absent is
usually absent with `ModuleNotFoundError`, and `pytest.importorskip` handles that by
skipping. When it raises a *plain* `ImportError` — which is what a wrapper that fails
to load a DLL does, and what `openslide` does on a machine with the Python package and
without the system library — `importorskip` treats it as a deprecation and raises
`PytestDeprecationWarning`. Measured on the pytest this suite runs, with a module whose
import raises plain `ImportError`:

    PytestDeprecationWarning: Module 'boommod' was found, but when imported by pytest
    it raised: ImportError(...). In pytest 9.1 this warning will become an error by
    default. You can fix the underlying problem, or alternatively overwrite this
    behavior and silence this warning by passing exc_type=ImportError explicitly.

Under `filterwarnings = ["error"]` — which is how this suite runs — that warning is a
**failure, not a skip**, so a machine without the OpenSlide binary library fails at
collection time on the module whose import raised, instead of skipping the tests that need
it: the exact outcome the guard exists to prevent, reached through the guard. And from
pytest 9.1 it is an error regardless of the filter, so this is a dated debt rather than a
stylistic one.

**The fix is one keyword on each call and the assertion is a scan, because the defect
is the absence of a keyword at a call site** — there is nothing to invoke and no return
value to assert on when openslide *is* installed, which is the machine this suite was
written on. A scan over the source is the only assertion that holds on both machines.
"""

import re
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent

# The call as it appears in source, spanning the lines it is written over: eight of the
# ten calls in this suite put their arguments one per line, and a single-line pattern
# would have found two of them and reported a suite that is nearly clean.
#
# The two alternatives are "closes on this line" and "closes on a line of its own", and
# both are needed. A plain `.*?\)` is not: three of the calls carry a reason string
# containing a parenthesis inside the literal — "…has no wheel for)" — and a non-greedy
# match stops there, before the `exc_type` it was looking for, so the call is reported
# as missing the keyword it has. Counting parentheses by depth would also work and would
# need a string-literal-aware scanner to do it honestly; these two shapes are every call
# this suite actually contains, and the count test below is what says so.
_IMPORTSKIP = re.compile(r"pytest\.importorskip\(((?:[^\n]*?\)|.*?\n\s*\)))", re.S)

# This module is excluded from its own scan, because it quotes the call it is looking
# for — in the pattern, in a docstring and in a failure message — and a scan that found
# its own text would either match forever or need the three exclusions spelled out here.
# A call added *to this file* would therefore go unchecked; there are none, and this is
# the cheapest way to say so once rather than three times in the pattern.
_THIS_MODULE = Path(__file__).name


def _calls() -> list[tuple[str, str]]:
    """Return `(file, argument text)` for every `pytest.importorskip` call in the suite."""
    found: list[tuple[str, str]] = []
    for path in sorted(TESTS_DIR.rglob("test_*.py")):
        if path.name == _THIS_MODULE:
            continue
        for match in _IMPORTSKIP.finditer(path.read_text(encoding="utf-8")):
            found.append((path.name, " ".join(match.group(1).split())))
    return found


def test_the_suite_does_import_something_optional_so_the_scan_is_not_vacuous():
    # **Red before anything else, because a scan over zero call sites passes for free.**
    # This project's whole defect class is a check that cannot fail; a guard test that
    # finds nothing to guard is the same shape. So the count is asserted rather than
    # assumed: ten calls, five of them for `openslide` and one each for `cv2` and
    # `PIL.Image`. A guard silently dropped from a file lowers this number, and the
    # assertion is what notices.
    calls = _calls()
    assert len(calls) == 10, [f"{name}: {arguments}" for name, arguments in calls]
    assert sum(1 for _n, arguments in calls if '"openslide"' in arguments) == 5


def test_every_importorskip_states_the_exception_it_skips_on():
    # The assertion, one call at a time so a failure names the file and the call. The
    # message quotes the call because the whole defect is a missing keyword inside it,
    # and a message saying only "missing exc_type" leaves a reader to find which of the
    # calls in a file it was.
    missing = [
        f"{name}: pytest.importorskip({arguments})"
        for name, arguments in _calls()
        if "exc_type=" not in arguments
    ]
    assert not missing, (
        "pytest.importorskip called without exc_type, which turns a skip into a "
        f"PytestDeprecationWarning and so into a failure under filterwarnings=error:\n"
        + "\n".join(missing)
    )