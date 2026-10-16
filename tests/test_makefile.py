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

**The fix is a delimiter that is not whitespace and one `$(subst)` per field.** The
shell prints `|`.join(...), so a path may hold any character except `|` and still be one
word to `$(word N,...)`; each field then converts its own single separator back to a
space, which is the one place the substitution has to happen and the one place a space
inside a path is restored.

**Two tests, because there are two claims.** The live one runs the real `make` over a
synthetic layout carrying a space in every field, which is the claim that can fail; it
skips where `make` is not installed, and the reason is recorded rather than hidden. The
shape test below it runs everywhere and asserts the transport is not whitespace-separated,
so the property is not vacuously green on a machine with no `make` on it — the same
reason `tests/test_optional_imports.py` asserts its own call count.

**The derived variables are recursively assigned (`=`, not `:=`) on purpose.** That is
what lets the live test include the real Makefile, override `LAYOUT` with a synthetic
layout, and read the real derivations: with `:=` the value is frozen at parse time and
the probe would measure make's `:=` rather than this project's expression.
"""

import re
import shutil
import subprocess
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