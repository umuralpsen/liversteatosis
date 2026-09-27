"""The dataset registry: the gate that decides which specimens may be trained on.

`data/manifest.yaml` is the record of what each whole-slide image is and where it
came from. `load_registry` reads it into `Specimen` values, and `usable_for_training`
and `excluded` split them by rule: a specimen enters training only when its record
carries both a `source_case_id` and a `licence`. The rule reads the record, never a
list of slide numbers, so a slide of unknown origin added to the manifest later is
excluded for the same reason as one already known to lack provenance.

`slide_path` is resolved against the manifest's own directory at load time, so a
manifest and its slides can be moved together and a caller never builds a path by
string concatenation. Slides are not opened while the registry loads: the manifest is
the record, and the smoke tests load it before any slide is staged.

Entry keys outside the `Specimen` fields, `provenance_note` among them, are written
for a human reader and are dropped here, because the dataclass is the interface
Tasks 4, 6, and 12 import field by field.

Declared interface, in `__all__` order:

    Specimen
        id: str
        source: str
        source_case_id: str | None
        scanner: str | None
        mpp: float | None
        objective: int | None
        stain: str | None
        acquisition_date: str | None
        label_source: str | None
        label_detail: str | None
        licence: str | None
        ethics: str | None
        slide_path: Path
    RegistryError
    excluded(specimens: Sequence[Specimen]) -> list[Specimen]
    load_registry(path: Path | None = None) -> list[Specimen]
    usable_for_training(specimens: Sequence[Specimen]) -> list[Specimen]
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from blobcount.config import default_config

__all__ = [
    "RegistryError",
    "Specimen",
    "excluded",
    "load_registry",
    "usable_for_training",
]

_UNKNOWN_SOURCE = "unknown"
_REQUIRED_TEXT_FIELDS = ("id", "source", "slide_path")
_OPTIONAL_TEXT_FIELDS = (
    "source_case_id",
    "scanner",
    "stain",
    "acquisition_date",
    "label_source",
    "label_detail",
    "licence",
    "ethics",
)


class RegistryError(Exception):
    """Raised when a dataset manifest is unreadable, incomplete, or self-contradictory."""


@dataclass(frozen=True)
class Specimen:
    """One whole-slide image and everything the manifest knows about its origin.

    Every field except `id`, `source`, and `slide_path` is nullable, because a slide
    of unknown origin is still a slide. An empty field is a fact about the record,
    not a default: `None` in `source_case_id` means the case was never identified.
    `slide_path` is absolute, so a caller can open it without knowing where the
    manifest lives.
    """

    id: str
    source: str
    source_case_id: str | None
    scanner: str | None
    mpp: float | None
    objective: int | None
    stain: str | None
    acquisition_date: str | None
    label_source: str | None
    label_detail: str | None
    licence: str | None
    ethics: str | None
    slide_path: Path


def _describe_type(value: Any) -> str:
    return type(value).__name__


def _required_text(entry: dict[str, Any], field: str, manifest: Path) -> str:
    if field not in entry or entry[field] is None:
        raise RegistryError(f"manifest {manifest} has a specimen entry missing required field '{field}'")
    value = entry[field]
    if not isinstance(value, str):
        raise RegistryError(
            f"manifest {manifest} field '{field}' must be a string, "
            f"got {_describe_type(value)} ({value!r}); quote the value if it is a number"
        )
    if not value.strip():
        raise RegistryError(f"manifest {manifest} field '{field}' is empty")
    return value


def _optional_text(entry: dict[str, Any], field: str, manifest: Path) -> str | None:
    if field not in entry or entry[field] is None:
        return None
    value = entry[field]
    if not isinstance(value, str):
        raise RegistryError(
            f"manifest {manifest} field '{field}' must be a string or null, "
            f"got {_describe_type(value)} ({value!r}); quote the value if it is a number"
        )
    return value


def _optional_float(entry: dict[str, Any], field: str, manifest: Path) -> float | None:
    if field not in entry or entry[field] is None:
        return None
    value = entry[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegistryError(
            f"manifest {manifest} field '{field}' must be a number or null, "
            f"got {_describe_type(value)} ({value!r})"
        )
    return float(value)


def _optional_int(entry: dict[str, Any], field: str, manifest: Path) -> int | None:
    if field not in entry or entry[field] is None:
        return None
    value = entry[field]
    if isinstance(value, bool) or not isinstance(value, int):
        raise RegistryError(
            f"manifest {manifest} field '{field}' must be an integer or null, "
            f"got {_describe_type(value)} ({value!r})"
        )
    return int(value)


def _parse_entry(entry: Any, position: int, manifest: Path) -> Specimen:
    if not isinstance(entry, dict):
        raise RegistryError(
            f"manifest {manifest} specimen entry {position} must be a mapping, "
            f"got {_describe_type(entry)}"
        )
    for field in _REQUIRED_TEXT_FIELDS:
        _required_text(entry, field, manifest)
    identifier = _required_text(entry, "id", manifest)
    source = _required_text(entry, "source", manifest)
    licence = _optional_text(entry, "licence", manifest)
    if not licence and source != _UNKNOWN_SOURCE:
        raise RegistryError(
            f"manifest {manifest} specimen '{identifier}' has no licence while its source is "
            f"'{source}'; a specimen of unrecorded origin must declare "
            f"source: {_UNKNOWN_SOURCE!r} so the training rule can exclude it by record"
        )
    values: dict[str, Any] = {field: _optional_text(entry, field, manifest) for field in _OPTIONAL_TEXT_FIELDS}
    values["id"] = identifier
    values["source"] = source
    values["mpp"] = _optional_float(entry, "mpp", manifest)
    values["objective"] = _optional_int(entry, "objective", manifest)
    values["slide_path"] = (manifest.parent / _required_text(entry, "slide_path", manifest)).resolve()
    return Specimen(**values)


def load_registry(path: Path | None = None) -> list[Specimen]:
    """Read a dataset manifest and return one `Specimen` per entry, in file order.

    `path` defaults to the file named by the `paths.manifest` configuration key, so
    the manifest has one location in the project and this function names none.

    `RegistryError` is raised for a manifest that is not a mapping, a `specimens`
    key that is not a list, an entry that is not a mapping, a missing or wrongly
    typed required field, a duplicate `id`, or a `licence` that is absent while
    `source` is not `unknown`. The last rule is the load-time half of the training
    rule: a specimen that claims a source it cannot license is a contradiction, and
    a specimen that declares `source: unknown` is admitted here so that
    `excluded` can report it.
    """
    manifest = Path(path) if path is not None else default_config().path("paths.manifest")
    if not manifest.is_file():
        raise RegistryError(f"manifest not found: {manifest}")
    try:
        document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise RegistryError(f"manifest is unreadable: {manifest}: {error}") from error
    if not isinstance(document, dict):
        raise RegistryError(
            f"manifest must hold a mapping at the top level, got {_describe_type(document)}: {manifest}"
        )
    entries = document.get("specimens")
    if not isinstance(entries, list):
        raise RegistryError(
            f"manifest key 'specimens' must hold a list, got {_describe_type(entries)}: {manifest}"
        )
    specimens: list[Specimen] = []
    first_seen: dict[str, int] = {}
    for position, entry in enumerate(entries):
        specimen = _parse_entry(entry, position, manifest)
        if specimen.id in first_seen:
            raise RegistryError(
                f"manifest {manifest} has a duplicate specimen id '{specimen.id}' "
                f"at entries {first_seen[specimen.id]} and {position}"
            )
        first_seen[specimen.id] = position
        specimens.append(specimen)
    return specimens


def _is_provenanced(specimen: Specimen) -> bool:
    return bool(specimen.source_case_id) and bool(specimen.licence)


def usable_for_training(specimens: Sequence[Specimen]) -> list[Specimen]:
    """Return the specimens cleared for training, in input order.

    A specimen is cleared when its record carries both a `source_case_id` and a
    `licence`. Nothing else decides membership, and in particular no identifier is
    special: two slides of unknown origin are excluded the same whether they are
    numbered 1 and 2 or 7 and 9.
    """
    return [specimen for specimen in specimens if _is_provenanced(specimen)]


def excluded(specimens: Sequence[Specimen]) -> list[Specimen]:
    """Return the specimens cleared for pipeline smoke tests only, in input order.

    The complement of `usable_for_training`: a slide whose case or licence was never
    recorded. It stays loadable, because a smoke test needs a slide to run the
    pipeline end to end, and it stays reported, because an excluded slide that goes
    unmentioned is how unprovenanced data reaches a training set.
    """
    return [specimen for specimen in specimens if not _is_provenanced(specimen)]
