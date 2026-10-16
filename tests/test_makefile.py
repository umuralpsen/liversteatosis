"""The Makefile's facts, asserted rather than assumed.

`Makefile` is code. It is the single command the README names as the reproduction of
every number in this repository, so a line that silently misparses a configured value is
a defect in the reported results, not a convenience.

**The defect, stated once.** `LAYOUT` is built by one `$(shell ...)` that prints the three
configured directories and the admitted specimen ids, and the four variables below it are
cut out of that string with `$(word N,$(LAYOUT))`. Make splits a word list on
whitespace, so the transport *is* the delimiter: a `paths.patches` written as
`/data/my slides/patches` is two words, `PATCHES` becomes `/data/my`, and every recipe
downstream reads a directory that does not exist. The ids are safe — a specimen id
carries no space — and the three paths are not, and they are the ones that carry 19 GB of
generated data and the tree this repository refuses to commit.

**The fix is a placeholder inside the record, and one `$(subst)` per field.** The shell
prints every space in the three directories as `@` and the records stay whitespace-
separated, so a record carries no whitespace and is one word to `$(word N,...)`; each
field converts its own placeholder back to a space, which is the one place a space inside
a path is restored. **Not `|`, and this paragraph used to say so.** A `|` between records
was tried and dropped: make splits on whitespace and on nothing else, so it would not have
separated two records, it would have glued them into one word — the one thing
`$(word N,...)` cannot take apart again.

**Three tests, because the claim has three ends, and the third was missing until
2026-10-16.** The shape test runs everywhere and asserts the transport in the source text,
so the property is not vacuously green on a machine with no `make` on it — the same reason
`tests/test_optional_imports.py` asserts its own call count. The two live ones run the real
`make` over a synthetic layout carrying a space in every field; both skip where `make` is
not installed and record the reason rather than hiding it. The first reads make's
*variables*, and on that alone it was satisfied while `make label` still handed the shell
`--index C:/data/my patches/patches/patches.csv`, which `sh` split into three arguments.
The second runs the real recipes — `PYTHON` pointed at a program that prints its own argv
— so it sees the word splitting the shell does and the exact `argv` each stage is given.

**The derived variables are recursively assigned (`=`, not `:=`) on purpose.** That is
what lets a live test include the real Makefile, override `LAYOUT` with a synthetic
layout, and read the real derivations: with `:=` the value is frozen at parse time and the
probe would measure make's `:=` rather than this project's expression. `INDEX` and
`HANDOFF` were `:=` until the same date, one line below the fix, and the live test above
could not see it: they came back holding the *real* configuration's paths.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from blobcount.config import PROJECT_ROOT

MAKEFILE = PROJECT_ROOT / "Makefile"
MAKEFILE_TEXT = MAKEFILE.read_text(encoding="utf-8")


def _make() -> str | None:
    """Return a `make` to run, or None.

    `shutil.which` first, and then the MSYS2 install this project was reproduced on and
    which the README names as the prerequisite — `C:\\msys64\\usr\\bin\\make.exe` is not
    on `PATH` on the machine that wrote this suite. Both are asked for rather than only
    the first, because a test that skips on the machine that ran the pipeline is a test
    that never runs on the machine that reported the numbers.
    """
    found = shutil.which("make")
    if found:
        return found
    candidate = Path("C:/msys64/usr/bin/make.exe")
    return str(candidate) if candidate.is_file() else None


def _shell_argument() -> list[str]:
    """Return the `SHELL=` argument for the `make` `_make()` found, or nothing.

    **GNU Make runs every recipe through a POSIX shell, and on a Windows `PATH` the
    MSYS2 `make.exe` cannot find one**: measured, `make: echo: No such file or
    directory` and status 127, on the bare `make` that was otherwise present and
    working. The shell lives beside the `make` that needs it, so it is named
    explicitly and only when that shell is actually there — a make on a POSIX machine
    keeps the one it has.
    """
    make = _make()
    if make is None:
        return []
    sh = Path(make).with_name("sh.exe" if Path(make).suffix else "sh")
    return [f"SHELL={sh.as_posix()}"] if sh.is_file() else []


# The character the shell prints a space as. It is the whole of the transport: make
# splits a word list on whitespace and on nothing else, so a record carrying a space is
# two records, and the substitution below is what makes a record carry none.
_SPACE_WRITTEN_AS = "@"

# One row per derived variable: the name, and the field index it cuts out of `LAYOUT`.
_PATH_FIELDS = {"PATCHES": "1", "LABELS": "2", "RESULTS": "3"}


def test_the_layout_transport_is_not_whitespace_separated():
    # **Always runs, and fails on the Makefile as it is today.** The live test below is
    # the claim that matters; this one is what keeps the claim from being untested on a
    # machine with no `make`, and it is written against the two facts the fix consists
    # of: the shell writes every space in the three directories as the placeholder, and
    # each path variable converts the placeholder back through `$(subst` after it has
    # cut its field out.
    shell_line = next(
        line for line in MAKEFILE_TEXT.splitlines() if line.startswith("LAYOUT :=")
    )
    for required, what in (
        (f"'{_SPACE_WRITTEN_AS}'", "the character a space is written as"),
        ("replace", "the substitution that writes spaces as the placeholder"),
    ):
        assert required in shell_line, (
            f"the LAYOUT shell command does not carry {what} ({required}). "
            f"The line is:\n{shell_line}"
        )

    wrong = []
    for name, index in _PATH_FIELDS.items():
        pattern = re.compile(
            rf"^{name}\s*=\s*\$\(subst\s+{re.escape(_SPACE_WRITTEN_AS)},\s*,\s*"
            rf"\$\(word\s+{index},\s*\$\(LAYOUT\)\)\)$",
            re.M,
        )
        if not pattern.search(MAKEFILE_TEXT):
            wrong.append(name)
    assert not wrong, (
        "these variables cut a field out of LAYOUT without putting the space back, so "
        f"a configured path containing a space is cut in two: {wrong}. Each must read "
        f"exactly `{next(iter(_PATH_FIELDS))} = $(subst {_SPACE_WRITTEN_AS}, ,"
        "$(word N,$(LAYOUT)))`."
    )


@pytest.mark.skipif(_make() is None, reason="no GNU make on PATH or at C:/msys64/usr/bin")
def test_a_configured_path_containing_a_space_survives_the_transport(tmp_path):
    # **The live claim.** A layout whose first three fields all contain a space, printed
    # with the separator the real Makefile uses, included into a probe that then reads
    # the real derived variables. `PATCHES` must come back with its space intact, and
    # the specimen ids must still cut into `HELD_OUT` and `TRAIN_SPECIMENS` — a fix that
    # changed the separator without re-splitting the ids would pass the first assertion
    # and break `make evaluate`.
    layouts = {
        "PATCHES": "C:/data/my patches/patches",
        "LABELS": "C:/data/my labels/labels",
        "RESULTS": "C:/data/my results/results",
    }
    # Exactly what the real `LAYOUT` line prints: three records and three ids, all
    # space-separated, every space inside a path written as the placeholder.
    synthetic = " ".join(
        value.replace(" ", _SPACE_WRITTEN_AS) for value in layouts.values()
    ) + " 4 5 6"
    probe = tmp_path / "probe.mk"
    # **Every echo is single-quoted, and that is not decoration.** The values under test
    # contain spaces, which is the whole point, so an unquoted `echo patches=$(PATCHES)`
    # hands the shell two words and the shell answers `C:/data/my: No such file or
    # directory` — a failure of the probe, not of the Makefile. Quoting also makes the
    # assertion below exact, because nothing can be re-tokenised on the way back.
    probe.write_text(
        "include " + MAKEFILE.as_posix() + "\n"
        f"override LAYOUT := {synthetic}\n"
        "print:\n"
        "\t@echo 'patches=$(PATCHES)'\n"
        "\t@echo 'labels=$(LABELS)'\n"
        "\t@echo 'results=$(RESULTS)'\n"
        "\t@echo 'held_out=$(HELD_OUT)'\n"
        "\t@echo 'trained=$(TRAINED)'\n"
        "\t@echo 'flags=$(TRAIN_SPECIMENS)'\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [_make(), "-f", str(probe), *_shell_argument(), "print"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    printed = dict(
        line.strip().strip("'").split("=", 1)
        for line in done.stdout.splitlines()
        if "=" in line
    )
    assert printed["patches"] == layouts["PATCHES"], done.stdout
    assert printed["labels"] == layouts["LABELS"], done.stdout
    assert printed["results"] == layouts["RESULTS"], done.stdout
    # The ids still cut, which is the half of the transport that was never broken and
    # the half a separator change can silently break.
    assert printed["held_out"] == "4", done.stdout
    assert printed["trained"].split() == ["5", "6"], done.stdout
    assert printed["flags"] == "--train-specimens 5 --train-specimens 6", done.stdout


# The six recipes that hand a path to a program, and the argv the shell must build for
# each. `|` joins because it cannot occur in an argument here and it keeps a value
# carrying a space as one string in the comparison below.
#
# **The ids are in this table and are not quoted, and that is the rule rather than an
# oversight**: a specimen id carries no space (`Makefile`, on the ids), so `--slide 4`
# reaches the shell as two arguments and `--train-specimens 5` as four, which is what
# the program parses. The three directory-derived values are the ones a space can reach,
# so each of them is one argument here whether or not the configuration it came from
# happens to hold one.
_PATH_RECIPES = {
    "label": (
        "bin/label_patches.py|--index|C:/data/my patches/patches/patches.csv"
    ),
    "ablate": (
        "bin/ablate_threshold.py|--index|C:/data/my patches/patches/patches.csv"
    ),
    "train": "bin/train.py|--labels|C:/data/my labels/labels/labels.csv",
    "evaluate": (
        "bin/evaluate.py|--labels|C:/data/my labels/labels/labels.csv"
        "|--checkpoint|C:/data/my results/results/best_model_fold0.pth"
        "|--specimens|4|--train-specimens|5|--train-specimens|6"
        "|--selection-cohort|4"
    ),
    "predict": (
        "bin/predict.py|--dir|C:/data/my labels/labels/steatosis"
        "|--checkpoint|C:/data/my results/results/best_model_fold0.pth"
        "|--labels|C:/data/my labels/labels/labels.csv"
        "|--specimen|4|--train-specimens|5|--train-specimens|6"
    ),
    "gradcam": (
        "bin/gradcam.py|--labels|C:/data/my labels/labels/labels.csv"
        "|--slide|4|--checkpoint|C:/data/my results/results/best_model_fold0.pth"
    ),
}


def _argv_stub(path: Path) -> str:
    """Write a program that prints the arguments it was handed, joined by `|`.

    It stands in for the `bin/*.py` a recipe names, reached through the route a real run
    reaches it: `RUN` expands to `$(PYTHON)`, so a `PYTHON` of `<interpreter> <this file>`
    puts the stub where the stage program would be and leaves the recipe line itself — the
    Makefile's own, unedited — as the thing under test. `sys.executable` rather than the
    word `python`, because the recipe runs under the MSYS2 shell and whether that shell's
    `PATH` names this interpreter is a property of the machine rather than of the
    Makefile. `|` cannot occur in an argument here, so it is a joiner that keeps a value
    carrying a space as one string in the comparison below.
    """
    path.write_text(
        "import sys\n"
        "print('|'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    return f'"{Path(sys.executable).as_posix()}" "{path.as_posix()}"'


@pytest.mark.parametrize("target", sorted(_PATH_RECIPES))
@pytest.mark.skipif(_make() is None, reason="no GNU make on PATH or at C:/msys64/usr/bin")
def test_a_configured_path_containing_a_space_reaches_the_shell_as_one_argument(
    target, tmp_path
):
    # **The half the test above cannot reach.** `test_a_configured_path...` reads make's
    # variables, so it stops one step before the defect it names: the shell is a separate
    # program with its own word splitting, and `$(INDEX)` arriving intact in make's memory
    # says nothing about what `sh` receives. Both halves were open at once on the real
    # Makefile. `INDEX` and `HANDOFF` were simply-expanded, so they froze when the file was
    # read — before `override LAYOUT` below is seen — and came back holding the *real*
    # configuration's paths; and no recipe quoted them, so the shell split
    # `--index C:/data/my patches/patches/patches.csv` into `--index`, `C:/data/my`,
    # `patches/patches/patches.csv`. Verified by probe on that file: the variable printed
    # correctly, the shell received it in two pieces.
    #
    # `PYTHON` is overridden in the probe rather than the recipes being copied into it, so
    # the command line under test is the one in the Makefile. A probe with its own `print:`
    # target holding a hand-written `--index $(INDEX)` would assert the probe's quoting and
    # not the Makefile's, which is the shape of check that let the original stand.
    layouts = {
        "PATCHES": "C:/data/my patches/patches",
        "LABELS": "C:/data/my labels/labels",
        "RESULTS": "C:/data/my results/results",
    }
    synthetic = " ".join(
        value.replace(" ", _SPACE_WRITTEN_AS) for value in layouts.values()
    ) + " 4 5 6"
    python = _argv_stub(tmp_path / "argv.py")
    # Overrides written into the probe rather than given on the command line: a command-line
    # value carrying spaces has to survive the MSYS2 runtime's own argv parsing before make
    # ever sees it, and a second parser between the test and the thing under test is one
    # more thing to be wrong about for no gain.
    probe = tmp_path / "probe.mk"
    probe.write_text(
        "include " + MAKEFILE.as_posix() + "\n"
        f"override LAYOUT := {synthetic}\n"
        f"override PYTHON := {python}\n",
        encoding="utf-8",
    )
    done = subprocess.run(
        [_make(), "-f", str(probe), *_shell_argument(), target],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == _PATH_RECIPES[target], (
        f"the shell did not receive the arguments `make {target}` intends.\n"
        f"expected: {_PATH_RECIPES[target]}\n"
        f"received: {done.stdout.strip().splitlines()[-1]!r}\n"
        "A value carrying a space arrives as two arguments when the recipe does not "
        "quote it, and a simply-expanded INDEX or HANDOFF holds the real configuration "
        "rather than this layout."
    )