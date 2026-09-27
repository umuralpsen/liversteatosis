from collections import namedtuple
from collections.abc import Iterator

import numpy as np
import pytest

from blobcount.config import _reset_default_config

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
def tmp_output_tree(tmp_path):
    """A `tmp_path` with the `patches/` and `labels/` output subdirectories."""
    (tmp_path / "patches").mkdir()
    (tmp_path / "labels").mkdir()
    return tmp_path
