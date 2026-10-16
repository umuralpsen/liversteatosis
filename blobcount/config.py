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
- `KEY_TYPES` is the declaration of what each declared key is read *as*, and it
  is checked on **both** of `load`'s two input paths. A file whose value for a
  declared key has a type other than the one declared is refused at load, and so
  is an override — the same `type() is` comparison and the same one widening of
  an `int` under a declared `float`, in one place. Before this, the check covered
  overrides only, which made `training.pretrained: 1` a silent inversion: a truthy
  `int` reaching the pretrained path with no error anywhere, reachable by editing
  one character in a git-tracked file. The declared types are **not** read from
  `configs/default.yaml`, because `load()` reads that file by default and would
  therefore measure it against itself and accept every hand-edit to it;
  `tests/test_config.py` asserts the table and the file agree for all 45 keys in
  both directions so the two cannot drift.
- `load` reads no key on the caller's behalf, because a caller asks for keys
  after `load` has already returned, so read tracking can never be complete at
  load time. A test instead scans the source of every module under `blobcount/`
  and requires each declared key to appear there as a `cfg.get("...")` or
  `cfg.path("...")` literal. The source scan stops a declared key that no code
  reads from passing unnoticed.

Declared interface, in `__all__` order:

    Config
        Config.__init__(self, data: dict[str, Any], *, source: Path | None = None) -> None
        Config.get(self, key: str, default: Any = _UNSET) -> Any
        Config.set(self, key: str, value: Any) -> None
        Config.path(self, key: str) -> Path
        Config.unknown_keys(self) -> set[str]
    ConfigError
    DEFAULT_CONFIG_PATH: Path
    KEY_TYPES: dict[str, type]
    KNOWN_KEYS: frozenset[str]
    PROJECT_ROOT: Path
    REQUIRED_KEYS: frozenset[str]
    default_config() -> Config
    get_device(cfg: Config | None = None) -> str
    load(path: Path | None = None, overrides: Sequence[str] = ()) -> Config
    set_seed(seed: int | None = None, cfg: Config | None = None) -> None
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
    "KEY_TYPES",
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

# What a run cannot be trusted to substitute a default for. A key read with a bare
# `cfg.get` on a path that has already deleted something belongs here, because this is the
# only place its absence is refused *before* that delete: `extraction.tissue_percentile`
# and `extraction.tissue_gray_max` are read by `extract_slide` after a `--force` run has
# emptied the patch tree, and `blobs.morph_kernel_size`, `blobs.morph_iterations` and
# `blobs.stain_normalization` are read by `label_patches` and `prepare_patch` after
# `prepare_output` has emptied the label tree. A configuration that omitted one used to
# pass `load`, take the previous run's artifacts with it, and then raise past a command
# line that had no answer for it.
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
        "training.architecture",
        "training.num_classes",
    }
)

KNOWN_KEYS: frozenset[str] = frozenset(
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

# The type each declared key is read as, and the whole of the file-value check. This is a
# declaration in code rather than a reading of `configs/default.yaml`, and the reason is the
# one that made the file-only version useless: `load()` with no `path` reads
# `configs/default.yaml`, so comparing a file's values against that same file accepts every
# hand-edit to it — `training.pretrained: 1` would agree with the `1` it just became and be
# called correct. The measured defect is exactly that edit, in exactly that file. A table here
# is refused by it, and `tests/test_config.py` asserts the table and the file hold the same
# type for every one of the 45 keys in both directions, so the two cannot drift apart and the
# shipped file cannot become the yardstick it is measured against.
#
# The types are the ones a caller reads the value as, not the ones YAML spells: `true` in the
# file is a `bool` here, and a `bool` is written `bool` rather than `int` so that the exact
# comparison below refuses the integer 1 — `bool` subclasses `int`, and `1 != True` is False,
# so an `isinstance`-shaped check would let `augmentation.horizontal_flip: 1` through as a
# truthy integer and invert a run on a hand-edit.
#
# `ablation.thresholds` is declared `list` and nothing more. What a list *holds* is
# `blobcount.ablation._thresholds`'s business, on its own reasoning, and it refuses the
# entries it refuses today; a second opinion here would be two rules for one key.
#
# An `int` where a `float` is declared is widened, on the file path exactly as on the override
# path and for the same reason: `blobs.area_um2_max: 125` states the same limit as the
# configured `125.0`, so it is the same number and refusing it would be a rule about spelling
# rather than about type. This is the only pair the check admits, it is a widening and never a
# narrowing, and it is the one that keeps `evaluation.confidence_level: 1` mean 100% rather
# than being refused — which is a value question, and this check makes no value claims.
KEY_TYPES: dict[str, type] = {
    "project.seed": int,
    "project.device": str,
    "paths.slides": str,
    "paths.patches": str,
    "paths.labels": str,
    "paths.results": str,
    "paths.manifest": str,
    "extraction.target_mpp": float,
    "extraction.patch_size": int,
    "extraction.step_size": int,
    "extraction.tissue_percentile": int,
    "extraction.tissue_gray_max": int,
    "blobs.area_um2_min": float,
    "blobs.area_um2_max": float,
    "blobs.circularity_min": float,
    "blobs.gray_percentile": int,
    "blobs.morph_kernel_size": int,
    "blobs.morph_iterations": int,
    "blobs.stain_normalization": bool,
    "labeling.blob_threshold": int,
    "ablation.thresholds": list,
    "training.architecture": str,
    "training.pretrained": bool,
    "training.num_classes": int,
    "training.input_size": int,
    "training.batch_size": int,
    "training.epochs": int,
    "training.learning_rate": float,
    "training.patience_early_stop": int,
    "training.patience_lr_scheduler": int,
    "training.lr_scheduler_factor": float,
    "training.use_weighted_sampler": bool,
    "training.num_workers": int,
    "augmentation.horizontal_flip": bool,
    "augmentation.vertical_flip": bool,
    "augmentation.rotation_degrees": int,
    "augmentation.color_jitter_brightness": float,
    "augmentation.color_jitter_contrast": float,
    "augmentation.color_jitter_saturation": float,
    "augmentation.color_jitter_hue": float,
    "augmentation.affine_translate": float,
    "evaluation.bootstrap_samples": int,
    "evaluation.confidence_level": float,
    "gradcam.alpha": float,
    "gradcam.colormap": str,
}


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


def _check_value_type(key: str, value: Any, origin: str) -> Any:
    """Return `value` as the type `KEY_TYPES` declares for `key`, or refuse it.

    **The comparison is `type(value) is expected`, and that shape is the whole of the rule.**
    `bool` subclasses `int`, so `1 != True` is `False` and an `isinstance`-shaped check would
    accept `training.pretrained: 1` as a truthy boolean — the silent inversion this check
    exists to prevent, reachable by editing one character in a git-tracked file. `is` on the
    two types has no such gap.

    **An `int` under a declared `float` is widened rather than refused**, and it is the only
    pair admitted: `blobs.area_um2_max: 125` and the configured `125.0` are the same number,
    so refusing it would be a rule about how the limit was spelled. This mirrors what the
    override path has always done, and it is what keeps the two paths from disagreeing about
    the same value.

    **Nothing else is checked, deliberately.** There is no finiteness test, no range, no sign
    and no allow-list of shapes: a value that is the declared type and is nonsense passes,
    because the job of a wrong *type* is to be caught where the type is known and the job of
    a wrong *value* belongs to the function that reads it and knows what it means. This is why
    the message names the type and stops there.
    """
    expected = KEY_TYPES[key]
    if type(value) is expected:
        return value
    if expected is float and type(value) is int:
        return float(value)
    raise ConfigError(
        f"key '{key}' in {origin} has type {type(value).__name__}, but this project reads "
        f"it as {expected.__name__}; supply a value of type {expected.__name__} for '{key}' "
        f"(configs/default.yaml states one)"
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
        """Write `value` at the dotted `key`, under both of the checks `load` applies.

        The key is checked against `KNOWN_KEYS` and the value's type against `KEY_TYPES`,
        through the same `_check_value_type` the file path uses, so the two routes a caller
        can write a value into a configuration cannot disagree about what a declared key
        holds. That was not true of `set` before: it validated the key and not the value, so
        `default_config().set("training.pretrained", 1)` inverted a run in a process holding a
        long-lived config, in exactly the way `training.pretrained: 1` written into the file
        used to.

        **The value is checked before it is written, so a refused `set` leaves the
        configuration exactly as it was** — the same order `load` uses, and the reason a
        caller can retry with a corrected value without reloading. There is no source path
        to name beyond `self._describe()`, so the refusal identifies its origin as the file
        this config was loaded from, or as an in-memory configuration when there is no file,
        which is the same string every other refusal on a `Config` uses.
        """
        _validate_key(key, self._describe())
        _set_dotted(self._data, key, _check_value_type(key, value, self._describe()))
        self._read.add(key)

    def path(self, key: str) -> Path:
        """Return the value at `key` as a concrete platform `Path`.

        Environment variables and `~` are expanded first. An absolute value is
        returned as it stands; a value still relative afterwards is resolved
        against the project root. A POSIX-absolute value on a non-POSIX platform
        is read as an absolute path on the current drive, which is the only
        interpretation that yields a `Path` able to open, create, or stat.

        **The two refusals here are about the value and not about its type, and the
        non-string arm is no longer reachable from a command line.** `load` types every
        `paths.*` key as `str`, so a number under one is a `ConfigError` naming the key at
        load, before any `Config` exists; both CLIs resolve their path keys inside the arm
        that catches it. The arm stays because a `Config` is constructible in memory from any
        mapping and need not come from a file — `Config({"paths": {"slides": 7}})` is the
        shape, and `tests/test_config.py` builds it. The blank-string arm is reachable either
        way: an empty string is the declared type and `load` makes no claim about values.
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
    as a string.

    **Both of this function's inputs are type-checked, and neither is checked
    loosely.** A value is compared to the type `KEY_TYPES` declares for its key
    with `type(value) is declared`, on the file path and on the override path, so
    a mistyped boolean cannot silently invert a run: `bool` subclasses `int`, and
    `1 != True` is `False`, so an `isinstance`-shaped test would accept
    `augmentation.horizontal_flip=1` as a truthy integer. The one exception is an
    `int` where a `float` is declared, which is widened because
    `blobs.area_um2_max=125` states the same limit as the configured `125.0`; it
    is the only pair admitted, and it is a widening and never a narrowing.

    **This is a type check and not a value check**, and the difference is the point.
    There is no finiteness test, no range, no sign and no per-key allow-list: a
    `confidence_level` of `1` is the declared `float` and passes, a negative
    `patch_size` is the declared `int` and passes, and a threshold list holding
    `'3'` is the declared `list` and passes. What a value has to *mean* is the
    reading function's business and stays there — `blobcount.ablation._thresholds`
    refuses a non-finite threshold, `blobcount.ablation` refuses a percentile
    outside [0, 100], and neither of those moves here.

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
            if type(existing) is float and type(value) is int:
                value = float(value)
            else:
                raise ConfigError(
                    f"override '{key}={raw}' has type {type(value).__name__}, "
                    f"but the configured value is {type(existing).__name__}; "
                    f"supply a value of type {type(existing).__name__} for '{key}'"
                )
        _set_dotted(data, key, value)

    declared = _iter_keys(data)
    for key in declared:
        _validate_key(key, str(source))
    # The file path is checked against `KEY_TYPES`, which is the same check the override path
    # has always made against the configured value, applied to the other of `load`'s two
    # inputs. It runs after the key check and before the missing-key check, so a file that is
    # wrong in both ways is reported for the key it got wrong first, and so an undeclared key
    # is named as one rather than as a type.
    for key in declared:
        _set_dotted(data, key, _check_value_type(key, _lookup(data, key)[1], str(source)))
    missing = sorted(REQUIRED_KEYS - set(declared))
    if missing:
        raise ConfigError(f"missing required keys in {source}: " + ", ".join(missing))

    return Config(data, source=source)


_DEFAULT_CONFIG: Config | None = None


def default_config() -> Config:
    """Return the process-wide `Config` built from `configs/default.yaml`.

    `get_device` and `set_seed` use it when they are given no config. It is
    built on first use, so a caller that changes it with `Config.set` changes
    what those two functions see. `_reset_default_config` drops it.
    """
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = load()
    return _DEFAULT_CONFIG


def _reset_default_config() -> None:
    """Drop the cached default config, so the next `default_config` rebuilds it.

    `default_config` is a mutable process global with no invalidation of its
    own, so a `Config.set` on it outlives whatever made it. This is the hook
    that ends that; the test suite calls it between tests.
    """
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG = None


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
