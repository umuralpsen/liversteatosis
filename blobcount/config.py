"""Validated configuration for the liver steatosis pipeline.

`load()` is the single entry point: it reads a YAML file, applies ``key=value``
overrides, rejects undeclared keys and missing required keys, and returns a
`Config` whose keys can be read with dot notation.

Two independent checks keep configuration and code in agreement, and neither
substitutes for the other:

- `KNOWN_KEYS` is the declaration of what the code may read. A key present in a
  configuration file, or introduced by an override, that is absent from
  `KNOWN_KEYS` is a hard error at load time. The schema stops an undeclared key
  from reaching a run; it says nothing about a declared key that nothing reads.
  `tests/test_config.py` asserts that `KNOWN_KEYS` and `configs/default.yaml`
  hold the same key set, in both directions.
- `load` reads no key on the caller's behalf, because a caller asks for keys
  after `load` has already returned, so read tracking can never be complete at
  load time. A test instead scans the source of every module under `blobcount/`
  and requires each declared key to appear there as a `cfg.get("...")` or
  `cfg.path("...")` literal. The source scan stops a declared key that no code
  reads from passing unnoticed.

Declared interface:

    load(path: Path | None = None, overrides: Sequence[str] = ()) -> Config
    default_config() -> Config
    get_device(cfg: Config | None = None) -> str
    set_seed(seed: int | None = None, cfg: Config | None = None) -> None
    Config.get(self, key: str, default: Any = _UNSET) -> Any
    Config.set(self, key: str, value: Any) -> None
    Config.path(self, key: str) -> Path
    Config.unknown_keys(self) -> set[str]
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "KNOWN_KEYS",
    "PROJECT_ROOT",
    "REQUIRED_KEYS",
    "default_config",
    "get_device",
    "load",
    "set_seed",
]


class ConfigError(Exception):
    """Raised when a configuration is unreadable, incomplete, or undeclared."""


class _Unset:
    """Sentinel type telling `Config.get` that the caller passed no default."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "_UNSET"


_UNSET = _Unset()


def _find_project_root(start: Path | None = None) -> Path:
    """Walk up from `start`, or from this module, to the holding pyproject.toml."""
    module = Path(__file__).resolve() if start is None else Path(start).resolve()
    for candidate in module.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        f"project root not found: no pyproject.toml in {module} or any parent directory, "
        "so the expected configuration file 'configs/default.yaml' cannot be located; "
        "run from a checkout of the project or pass an explicit path to load()"
    )


PROJECT_ROOT: Path = _find_project_root()
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "default.yaml"

REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "project.seed",
        "project.device",
        "paths.slides",
        "paths.patches",
        "paths.labels",
        "paths.results",
        "paths.manifest",
        "extraction.target_mpp",
        "extraction.patch_size",
        "extraction.step_size",
        "blobs.area_um2_min",
        "blobs.area_um2_max",
        "blobs.circularity_min",
        "blobs.gray_percentile",
        "labeling.blob_threshold",
        "training.architecture",
        "training.num_classes",
    }
)

KNOWN_KEYS: frozenset[str] = frozenset(
    {
        "project.name",
        "project.seed",
        "project.device",
        "paths.slides",
        "paths.patches",
        "paths.labels",
        "paths.results",
        "paths.manifest",
        "extraction.target_mpp",
        "extraction.patch_size",
        "extraction.step_size",
        "extraction.tissue_percentile",
        "extraction.tissue_gray_max",
        "blobs.area_um2_min",
        "blobs.area_um2_max",
        "blobs.circularity_min",
        "blobs.gray_percentile",
        "blobs.morph_kernel_size",
        "blobs.morph_iterations",
        "blobs.stain_normalization",
        "labeling.blob_threshold",
        "ablation.thresholds",
        "training.architecture",
        "training.pretrained",
        "training.num_classes",
        "training.input_size",
        "training.batch_size",
        "training.epochs",
        "training.learning_rate",
        "training.patience_early_stop",
        "training.patience_lr_scheduler",
        "training.lr_scheduler_factor",
        "training.use_weighted_sampler",
        "training.num_workers",
        "augmentation.horizontal_flip",
        "augmentation.vertical_flip",
        "augmentation.rotation_degrees",
        "augmentation.color_jitter_brightness",
        "augmentation.color_jitter_contrast",
        "augmentation.color_jitter_saturation",
        "augmentation.color_jitter_hue",
        "augmentation.affine_translate",
        "evaluation.bootstrap_samples",
        "evaluation.confidence_level",
        "gradcam.alpha",
        "gradcam.colormap",
    }
)

_SECTIONS: frozenset[str] = frozenset(key.split(".", 1)[0] for key in KNOWN_KEYS)


def _iter_keys(data: dict[str, Any], prefix: str = "") -> list[str]:
    """Flatten nested mappings into dotted keys, leaves become key ends."""
    keys: list[str] = []
    for name, value in data.items():
        key = f"{prefix}{name}"
        if isinstance(value, dict):
            keys.extend(_iter_keys(value, f"{key}."))
        else:
            keys.append(key)
    return keys


def _lookup(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    """Walk nested mappings along a dotted key and report whether it exists."""
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def _validate_key(key: str, origin: str) -> None:
    section = key.split(".", 1)[0]
    if section not in _SECTIONS:
        raise ConfigError(
            f"unknown section '{section}' in {origin}: '{key}' is not a valid configuration key"
        )
    if key not in KNOWN_KEYS:
        raise ConfigError(
            f"unknown key '{key}' in {origin}: add it to blobcount.config.KNOWN_KEYS if code reads it"
        )


def _parse_override(item: str) -> tuple[str, str, Any]:
    """Split a ``key=value`` override and parse the value so types survive."""
    key, separator, raw = item.partition("=")
    if not separator:
        raise ConfigError(f"override must have the form 'key=value', got '{item}'")
    key = key.strip()
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    return key, raw, value


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"cannot set '{key}': '{part}' is not a section")
        node = child
    node[parts[-1]] = value


class Config:
    """Dotted-key view over loaded configuration data.

    Every `get`, `set`, and `path` call is recorded, so `unknown_keys()` can
    report which declared keys no code has asked for. `load` records nothing
    itself, so on a freshly loaded config `unknown_keys()` is every declared key.
    """

    def __init__(self, data: dict[str, Any], *, source: Path | None = None) -> None:
        self._data = data
        self._source = Path(source) if source is not None else None
        self._read: set[str] = set()

    def _keys(self) -> list[str]:
        """Return every dotted key currently declared by the loaded data."""
        return sorted(_iter_keys(self._data))

    def get(self, key: str, default: Any = _UNSET) -> Any:
        """Return the value at `key`.

        A bare `get(key)` raises `ConfigError` for a key the data does not
        declare. `get(key, default)` returns `default` instead, including when
        `default` is `None`, so an optional key reads the way it is written.
        """
        self._read.add(key)
        found, value = _lookup(self._data, key)
        if found:
            return value
        if default is not _UNSET:
            return default
        raise ConfigError(f"unknown key '{key}' in {self._describe()}")

    def set(self, key: str, value: Any) -> None:
        """Write `value` at the dotted `key`, under the same check `load` applies."""
        _validate_key(key, self._describe())
        _set_dotted(self._data, key, value)
        self._read.add(key)

    def path(self, key: str) -> Path:
        """Return the value at `key` as a concrete platform `Path`.

        Environment variables and `~` are expanded first. An absolute value is
        returned as it stands; a value still relative afterwards is resolved
        against the project root. A POSIX-absolute value on a non-POSIX platform
        is read as an absolute path on the current drive, which is the only
        interpretation that yields a `Path` able to open, create, or stat.
        """
        raw = self.get(key)
        if not isinstance(raw, str):
            raise ConfigError(
                f"key '{key}' must hold a path string, got {type(raw).__name__} in {self._describe()}"
            )
        expanded = os.path.expanduser(os.path.expandvars(raw))
        if not expanded.strip():
            raise ConfigError(f"key '{key}' holds an empty path in {self._describe()}")
        native = Path(expanded)
        return native if native.is_absolute() else PROJECT_ROOT / native

    def unknown_keys(self) -> set[str]:
        """Return declared keys that were never requested through get, set, or path."""
        return set(self._keys()) - self._read

    def _describe(self) -> str:
        return str(self._source) if self._source is not None else "in-memory configuration"


def load(path: Path | None = None, overrides: Sequence[str] = ()) -> Config:
    """Load configuration from `path`, apply `overrides`, and validate the result.

    `path` defaults to `configs/default.yaml` inside the project. Each override
    is a `key=value` string whose value is parsed with `yaml.safe_load` so that
    types survive; a value that is not valid YAML, such as `%VAR%/out`, is kept
    as a string. An override whose parsed type differs from the type already
    configured is rejected, so a mistyped boolean cannot silently invert a run.

    No key is read here. Callers request keys after this returns, so marking them
    read now would make `unknown_keys()` unconditionally empty.
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must hold a mapping at the top level: {source}")

    parsed: list[tuple[str, str, Any]] = []
    for item in overrides:
        key, raw, value = _parse_override(item)
        _validate_key(key, f"override '{key}={raw}'")
        parsed.append((key, raw, value))
    for key, raw, value in parsed:
        found, existing = _lookup(data, key)
        if found and type(value) is not type(existing):
            raise ConfigError(
                f"override '{key}={raw}' has type {type(value).__name__}, "
                f"but the configured value is {type(existing).__name__}"
            )
        _set_dotted(data, key, value)

    declared = _iter_keys(data)
    for key in declared:
        _validate_key(key, str(source))
    missing = sorted(REQUIRED_KEYS - set(declared))
    if missing:
        raise ConfigError(f"missing required keys in {source}: " + ", ".join(missing))

    return Config(data, source=source)


_DEFAULT_CONFIG: Config | None = None


def default_config() -> Config:
    """Return the process-wide `Config` built from `configs/default.yaml`.

    `get_device` and `set_seed` use it when they are given no config. It is
    built on first use, so a caller that changes it with `Config.set` changes
    what those two functions see.
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load()
    return _DEFAULT_CONFIG


def set_seed(seed: int | None = None, cfg: Config | None = None) -> None:
    """Seed `random`, `numpy`, and `torch` for reproducible runs.

    `seed` wins when given; otherwise the seed is `project.seed` read from `cfg`,
    or from `configs/default.yaml` when `cfg` is None.

    The `PYTHONHASHSEED` assignment below does not affect the running
    interpreter: Python reads that variable once, at startup, so setting it here
    is a no-op for this process. It is kept so that a child process launched
    later in the run inherits the seed. Hash determinism for the current process
    has to come from the launcher's environment.
    """
    import numpy as np
    import torch

    if seed is None:
        seed = int((cfg if cfg is not None else default_config()).get("project.seed"))
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(cfg: Config | None = None) -> str:
    """Return the configured device, probing CUDA when `project.device` is `auto`.

    `project.device` is read from `cfg`, or from `configs/default.yaml` when
    `cfg` is None. A value that is neither `auto`, `cpu`, nor `cuda` raises
    `ConfigError` rather than being handed to `torch.device` unchecked.
    """
    import torch

    device = str((cfg if cfg is not None else default_config()).get("project.device"))
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ConfigError(f"project.device must be 'auto', 'cpu', or 'cuda', got '{device}'")
    return device
