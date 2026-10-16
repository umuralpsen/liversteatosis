import shutil
import sys
import tempfile
from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path

# **`sklearn.metrics` is imported here, before anything below it can pull torch in, and the
# line must not be moved down.** Measured on this machine, with each of these as the whole
# body of a one-test file run under pytest: `import torch` followed by
# `import sklearn.metrics` prints `Windows fatal exception: access violation` and every run;
# `import sklearn.metrics` followed by `import torch`, and `import torch` followed by
# `import pyarrow` or `import pandas` or `import numpy`, are all clean. **The chain is
# `sklearn.metrics` -> `sklearn.utils._array_api` -> `sklearn/utils/fixes.py:20`
# (`import pandas as pd`) -> pandas' own `pandas.compat.pyarrow` -> `import pyarrow`, and it
# is pyarrow's C extension load that faults once torch is already resident in the process.**
#
# **Two reasons this line belongs here rather than in the module that needs it.** `conftest.py`
# is imported by pytest before any test module, so it is the only place in this suite that can
# guarantee the order; and `pyarrow` is an **undeclared transitive** — it is in neither
# `pyproject.toml` nor `requirements.txt`, it arrives only because `pandas` is declared and
# `pandas` reaches for it — so whether this suite runs at all depends on whether an undeclared
# package happens to be installed on the machine. Delete this line and the fault comes back as
# a mystery on a machine that has pyarrow and stays silent on one that does not.
#
# **A recovered native fault prints once per process and every test still passes**, which is
# the second cost and the reason this is not merely cosmetic: a suite that emits one on every
# run has no way to distinguish it from a *real* fault later in the same run, because both
# print the same shape and both are swallowed the same way. That is how the junction test in
# `tests/test_evaluate.py` was deleted once already, on the grounds that the environment fault
# made it flaky. It was put back; the environment fault is fixed here instead.
import sklearn.metrics  # noqa: F401  (the import IS the point; see above)

# **The order the import above produced, recorded rather than described.** The fault is in the
# sequence and a sequence is gone by the time a test runs, so the state that sequence left is
# what is asserted: `sklearn.metrics` is loaded and torch is not, which is exactly the order
# that measures clean. Deleting the preload makes this `False` and
# `tests/test_evaluate.py::test_sklearn_was_imported_before_torch` goes red — the remedy is
# pinned, not only commented.
SKLEARN_IMPORTED_BEFORE_TORCH = "sklearn.metrics" in sys.modules and "torch" not in sys.modules

import numpy as np
import pytest

from blobcount.config import PROJECT_ROOT, _reset_default_config

# The scratch root inside the checkout. `prepare_output` refuses a target that is not
# strictly below the project root, so a test that exercises the real guard cannot use
# pytest's basetemp, which lives outside the project. The directory is git-ignored and
# removed after each test.
PROJECT_SCRATCH = PROJECT_ROOT / ".pytest-tmp"

SyntheticPatch = namedtuple("SyntheticPatch", "array disc_centres disc_radius")

_PATCH_SIZE = 256
_DISC_RADIUS = 12
_BACKGROUND = 180
_DISC_VALUE = 255
_DISC_CENTRES = ((64, 64), (128, 90), (200, 170))


@pytest.fixture(autouse=True)
def reset_default_config() -> Iterator[None]:
    """Drop the cached default config around every test.

    `default_config()` hands out one mutable process global. Without this, the
    first test that calls `Config.set` on it leaks that change into every test
    after it, in this order and every later run.
    """
    _reset_default_config()
    yield
    _reset_default_config()


@pytest.fixture
def synthetic_patch() -> SyntheticPatch:
    """A 256x256x3 uint8 patch: mid-gray tissue with three white discs."""
    rows, columns = np.mgrid[0:_PATCH_SIZE, 0:_PATCH_SIZE]
    array = np.full((_PATCH_SIZE, _PATCH_SIZE, 3), _BACKGROUND, dtype=np.uint8)
    for centre_x, centre_y in _DISC_CENTRES:
        disc = (columns - centre_x) ** 2 + (rows - centre_y) ** 2 <= _DISC_RADIUS**2
        array[disc] = _DISC_VALUE
    return SyntheticPatch(array, _DISC_CENTRES, _DISC_RADIUS)


@pytest.fixture
def tmp_output_tree(project_scratch):
    """A `project_scratch` with the `patches/` and `labels/` output subdirectories."""
    (project_scratch / "patches").mkdir()
    (project_scratch / "labels").mkdir()
    return project_scratch


@pytest.fixture
def project_scratch() -> Iterator[Path]:
    """A fresh directory inside the project root, removed when the test ends.

    `prepare_output` refuses to delete anything that is not strictly below the project
    root, because the patch directory it is handed comes from a hand-edited
    `paths.patches` and `--force` deletes it. A test that calls the real `prepare_output`
    therefore has to place its output inside the checkout rather than in pytest's
    basetemp, which is outside it. The name is `tempfile.mkdtemp`'s, not the test's,
    because a test node name carries characters that are not valid in a path.
    """
    PROJECT_SCRATCH.mkdir(exist_ok=True)
    scratch = Path(tempfile.mkdtemp(dir=PROJECT_SCRATCH))
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
