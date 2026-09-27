"""Validated configuration for the liver steatosis pipeline.

`load()` is the single entry point: it reads a YAML file, applies ``key=value``
overrides, rejects undeclared keys and missing required keys, and returns a
`Config` whose keys can be read with dot notation.

The declaration of what the code reads lives in `KNOWN_KEYS`. A key that is
present in a configuration file but absent from `KNOWN_KEYS` is a hard error at
load time, so a configuration key can never drift away from the code that uses
it.
"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

__all__ = [
    "Config",
    "ConfigError",
    "DEFAULT_CONFIG_PATH",
    "KNOWN_KEYS",
    "PROJECT_ROOT",
    "REQUIRED_KEYS",
    "get_device",
    "load",
    "set_seed",
]


class ConfigError(Exception):
    """Raised when a configuration is unreadable, incomplete, or undeclared."""


def _find_project_root() -> Path:
    module = Path(__file__).resolve()
    for candidate in module.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return module.parent.parent


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
    report which declared keys no code has asked for.
    """

    def __init__(
        self,
        data: dict[str, Any],
        *,
        source: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._data = data
        self._source = Path(source) if source is not None else None
        self._project_root = Path(project_root) if project_root is not None else PROJECT_ROOT
        self._read: set[str] = set()

    @property
    def source(self) -> Path | None:
        return self._source

    @property
    def project_root(self) -> Path:
        return self._project_root

    def keys(self) -> list[str]:
        """Return every dotted key currently declared by the loaded data."""
        return sorted(_iter_keys(self._data))

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at `key`, or `default` when `default` is not None."""
        self._read.add(key)
        found, value = _lookup(self._data, key)
        if found:
            return value
        if default is not None:
            return default
        raise ConfigError(f"unknown key '{key}' in {self._describe()}")

    def set(self, key: str, value: Any) -> None:
        """Write `value` at the dotted `key`, creating sections as needed."""
        parts = key.split(".")
        if not all(parts):
            raise ConfigError(f"invalid key '{key}'")
        _set_dotted(self._data, key, value)
        self._read.add(key)

    def path(self, key: str) -> Path:
        """Return the value at `key` as a path.

        Environment variables and `~` are expanded first. A value that is still
        relative is resolved against the project root. On a non-POSIX platform a
        configured POSIX-absolute value is returned as a `PurePosixPath` so the
        configured string is preserved instead of being resolved against the
        Windows project root.
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
        if native.is_absolute():
            return native
        if PurePosixPath(expanded).is_absolute():
            return PurePosixPath(expanded)
        return self._project_root / native

    def unknown_keys(self) -> set[str]:
        """Return declared keys that were never requested through get, set, or path."""
        return set(_iter_keys(self._data)) - self._read

    def _describe(self) -> str:
        return str(self._source) if self._source is not None else "in-memory configuration"


def load(path: Path | None = None, overrides: Sequence[str] = ()) -> Config:
    """Load configuration from `path`, apply `overrides`, and validate the result.

    `path` defaults to `configs/default.yaml` inside the project. Each override
    is a `key=value` string whose value is parsed with `yaml.safe_load` so that
    types survive; a value that is not valid YAML, such as `%VAR%/out`, is kept
    as a string.
    """
    source = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not source.is_file():
        raise ConfigError(f"configuration file not found: {source}")
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"configuration file must hold a mapping at the top level: {source}")

    parsed = []
    for item in overrides:
        key, raw, value = _parse_override(item)
        _validate_key(key, f"override '{key}={raw}'")
        parsed.append((key, value))
    for key, value in parsed:
        _set_dotted(data, key, value)

    for key in _iter_keys(data):
        _validate_key(key, str(source))
    missing = sorted(REQUIRED_KEYS - set(_iter_keys(data)))
    if missing:
        raise ConfigError(f"missing required keys in {source}: " + ", ".join(missing))

    config = Config(data, source=source)
    for key in config.keys():
        config.get(key)
    for key in config.keys():
        if key.startswith("paths."):
            config.path(key)
    return config


def set_seed(seed: int | None = None) -> None:
    """Seed `random`, `numpy`, and `torch` for reproducible runs.

    When `seed` is None the value of `project.seed` from the default
    configuration is used.
    """
    import numpy as np
    import torch

    if seed is None:
        seed = int(load().get("project.seed"))
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Return the configured device, probing CUDA when `project.device` is `auto`."""
    import torch

    device = str(load().get("project.device"))
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ConfigError(f"project.device must be 'auto', 'cpu', or 'cuda', got '{device}'")
    return device
