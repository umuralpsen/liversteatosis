import shutil
import tempfile
from collections import namedtuple
from collections.abc import Iterator
from pathlib import Path

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
