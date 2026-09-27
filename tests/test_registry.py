import re
from dataclasses import fields

import pytest
from blobcount.config import default_config
from blobcount.registry import RegistryError, Specimen, load_registry, usable_for_training, excluded


def test_loads_six_entries():
    reg = load_registry()
    assert len(reg) == 6
    assert {s.id for s in reg} == {"1", "2", "3", "4", "5", "6"}


def test_tcga_entries_carry_recovered_case_ids():
    reg = {s.id: s for s in load_registry()}
    assert reg["4"].source == "TCGA-LIHC"
    assert reg["4"].source_case_id == "TCGA-2V-A95S"
    assert reg["5"].source_case_id == "TCGA-DD-AAEH"
    assert reg["6"].source_case_id == "TCGA-GJ-A6C0"


def test_unknown_source_is_excluded_from_training():
    reg = load_registry()
    excluded_ids = {s.id for s in excluded(reg)}
    assert excluded_ids == {"1", "2", "3"}


def test_usable_set_is_non_empty():
    assert len(usable_for_training(load_registry())) == 3


def test_entry_without_licence_is_rejected(tmp_path):
    bad = tmp_path / "manifest.yaml"
    bad.write_text(
        "specimens:\n"
        "- id: X\n  source: local\n  source_case_id: X1\n  slide_path: x.svs\n",
        encoding="utf-8",
    )
    with pytest.raises(RegistryError, match="licence"):
        load_registry(bad)


def test_duplicate_ids_rejected(tmp_path):
    bad = tmp_path / "manifest.yaml"
    body = (
        "specimens:\n"
        "- id: X\n  source: local\n  source_case_id: A\n  licence: L\n  slide_path: a.svs\n"
        "- id: X\n  source: local\n  source_case_id: B\n  licence: L\n  slide_path: b.svs\n"
    )
    bad.write_text(body, encoding="utf-8")
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(bad)


def test_non_ascii_id_round_trips(tmp_path):
    p = tmp_path / "manifest.yaml"
    p.write_text(
        "specimens:\n"
        "- id: \"hasta-ğü-01\"\n  source: local\n  source_case_id: C\n"
        "  licence: L\n  slide_path: c.svs\n",
        encoding="utf-8",
    )
    assert load_registry(p)[0].id == "hasta-ğü-01"


# The seven tests above are the brief's. What follows guards the rules they state
# only indirectly: that the manifest ships with the package, that it describes
# the slides actually on disk, that the header's own claims still match the
# files, that exclusion follows the record rather than a list of slide numbers,
# and that a malformed record is refused loudly instead of quietly dropped.


def _write_manifest(tmp_path, body):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(body, encoding="utf-8")
    return manifest


def _skip_without_slides():
    """Skip rather than fail where a slide cannot be opened, and say which reason holds.

    Two independent things can be missing, and they call for opposite remedies, so
    this distinguishes them instead of reporting one generic skip. `openslide` is
    a native binding, not a pure-Python package: `pyproject.toml` declares
    `openslide-python`, and the shared library it binds arrives as the
    `openslide-bin` wheel on Windows and macOS, which has no Linux build, so a
    Linux runner needs a system `libopenslide0` installed by the image. The import
    is therefore attempted here, inside the helper, rather than at module scope: a
    module-scope import would make one absent native library a collection *error*
    for every test in this file, including all the ones that never open a slide
    and would otherwise pass anywhere.

    The slides are the second reason. `.gitignore` holds `data/` and `*.svs`, so
    the images are out of the repository on purpose and a fresh clone has the
    manifest but no slides. A test that opens a slide is meaningful only where the
    slides are staged, and failing on a clone that was never given 30 GB of WSI
    would be a false alarm about the manifest.

    Returns the imported `openslide` module for the caller to open slides with.
    """
    openslide = pytest.importorskip(
        "openslide",
        reason="openslide cannot be imported, so no .svs can be opened; install the "
        "native library (the openslide-bin wheel on Windows and macOS, a system "
        "libopenslide0 on Linux, which openslide-bin has no wheel for)",
    )
    manifest = default_config().path("paths.manifest")
    if not any(manifest.parent.glob("*.svs")):
        pytest.skip(
            f"no .svs beside {manifest}: the slides are git-ignored by design, "
            f"so a fresh clone carries the manifest and no image files"
        )
    return openslide


_QUICKHASH_LINE = re.compile(r"^#\s+(\S+\.svs)\s+([0-9a-f]{64})$")


def _recorded_quickhashes():
    manifest = default_config().path("paths.manifest")
    recorded = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        match = _QUICKHASH_LINE.match(line)
        if match is not None:
            recorded[match.group(1)] = match.group(2)
    return recorded


# The header's field-provenance table, one row per `Specimen` field. A row is any
# three-space-indented comment line that is not a quickhash line: the field name is
# the first token, the marking is the last. The marking is not matched here on
# purpose, so a misspelled one is parsed as the row it is rather than vanishing
# from the table and leaving the check below with nothing to complain about.
_PROVENANCE_ROW = re.compile(r"^#\s{3}(\S+)\s+.+\s+(\S+)$")
_PROVENANCE_MARKINGS = ("read", "derived", "inferred", "assigned")


def _recorded_provenance_rows():
    manifest = default_config().path("paths.manifest")
    rows = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if _QUICKHASH_LINE.match(line) is not None:
            continue
        match = _PROVENANCE_ROW.match(line)
        if match is not None:
            rows.append((match.group(1), match.group(2)))
    return rows


def test_missing_required_field_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  licence: L\n",
    )
    with pytest.raises(RegistryError, match="missing required field 'slide_path'"):
        load_registry(manifest)


def test_manifest_that_is_not_a_mapping_is_rejected(tmp_path):
    manifest = _write_manifest(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(RegistryError, match="mapping at the top level"):
        load_registry(manifest)


def test_specimens_key_that_is_not_a_list_is_rejected(tmp_path):
    manifest = _write_manifest(tmp_path, "specimens:\n  id: X\n")
    with pytest.raises(RegistryError, match="must hold a list"):
        load_registry(manifest)


def test_missing_manifest_file_is_rejected(tmp_path):
    with pytest.raises(RegistryError, match="manifest not found"):
        load_registry(tmp_path / "absent.yaml")


def test_unreadable_manifest_is_reported_as_a_registry_error(tmp_path):
    manifest = _write_manifest(tmp_path, "specimens: [unclosed\n")
    with pytest.raises(RegistryError, match="unreadable"):
        load_registry(manifest)


def test_unquoted_numeric_id_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: 7\n  source: local\n  source_case_id: A\n"
        "  licence: L\n  slide_path: a.svs\n",
    )
    with pytest.raises(RegistryError, match="'id' must be a string"):
        load_registry(manifest)


def test_wrong_typed_measurement_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  source_case_id: A\n"
        "  licence: L\n  mpp: quarter\n  slide_path: a.svs\n",
    )
    with pytest.raises(RegistryError, match="'mpp' must be a number"):
        load_registry(manifest)


def test_slide_path_resolves_against_the_manifest_directory(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  source_case_id: A\n"
        "  licence: L\n  slide_path: slides/a.svs\n",
    )
    slide_path = load_registry(manifest)[0].slide_path
    assert slide_path == tmp_path / "slides" / "a.svs"
    assert slide_path.is_absolute()


def test_manifest_location_comes_from_the_configuration_key(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  source_case_id: A\n"
        "  licence: L\n  slide_path: a.svs\n",
    )
    default_config().set("paths.manifest", str(manifest))
    assert [s.id for s in load_registry()] == ["X"]


def test_exclusion_follows_the_record_and_not_the_slide_number(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n"
        "- id: '1'\n  source: TCGA-LIHC\n  source_case_id: C1\n"
        "  licence: L\n  slide_path: a.svs\n"
        "- id: '7'\n  source: unknown\n  slide_path: b.svs\n"
        "- id: '9'\n  source: unknown\n  source_case_id: C9\n  slide_path: c.svs\n",
    )
    specimens = load_registry(manifest)
    assert {s.id for s in usable_for_training(specimens)} == {"1"}
    assert {s.id for s in excluded(specimens)} == {"7", "9"}


def test_excluded_and_usable_partition_the_input(tmp_path):
    specimens = load_registry()
    assert len(usable_for_training(specimens)) + len(excluded(specimens)) == len(specimens)
    assert not {s.id for s in usable_for_training(specimens)} & {s.id for s in excluded(specimens)}
    assert {s.id for s in usable_for_training(specimens)} | {s.id for s in excluded(specimens)} == {
        s.id for s in specimens
    }


def test_a_case_id_without_a_licence_is_still_excluded(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: unknown\n  source_case_id: C\n  slide_path: a.svs\n",
    )
    assert excluded(load_registry(manifest))[0].id == "X"


def test_the_reader_only_note_key_is_dropped_and_the_entry_still_loads(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: unknown\n  slide_path: a.svs\n"
        "  provenance_note: free text for a human reader\n",
    )
    assert load_registry(manifest)[0].id == "X"


def test_shipped_manifest_measurements_keep_their_declared_types():
    reg = {s.id: s for s in load_registry()}
    assert reg["1"].mpp == 0.4990
    assert reg["2"].mpp == 0.2520
    assert reg["3"].mpp == 0.2520
    assert reg["4"].mpp == 0.2527
    assert reg["5"].mpp == 0.2525
    assert reg["6"].mpp == 0.2485
    assert reg["1"].objective == 20
    for identifier in ("2", "3", "4", "5", "6"):
        assert reg[identifier].objective == 40
    for specimen in reg.values():
        assert isinstance(specimen.mpp, float)
        assert isinstance(specimen.objective, int)
        assert isinstance(specimen.acquisition_date, str)
        assert isinstance(specimen.label_source, str) or specimen.label_source is None


def test_shipped_manifest_points_at_the_slides_on_disk():
    _skip_without_slides()
    for specimen in load_registry():
        assert specimen.slide_path.is_file(), specimen.slide_path


def test_manifest_header_quickhashes_match_the_slide_files():
    openslide = _skip_without_slides()
    slides = {specimen.slide_path.name: specimen.slide_path for specimen in load_registry()}
    recorded = _recorded_quickhashes()
    assert sorted(recorded) == sorted(slides), (
        "the manifest header must record one quickhash line per slide, shaped "
        f"'#   <file>.svs  <64 hex characters>'; parsed {sorted(recorded)}, "
        f"the manifest names {sorted(slides)}"
    )
    for name, digest in sorted(recorded.items()):
        with openslide.OpenSlide(str(slides[name])) as opened:
            assert opened.properties["openslide.quickhash-1"] == digest, name


def test_manifest_header_provenance_table_covers_every_specimen_field():
    rows = _recorded_provenance_rows()
    recorded = {field for field, _ in rows}
    declared = {field.name for field in fields(Specimen)}
    assert recorded == declared, (
        "the manifest header documents one provenance row per Specimen field, and a "
        f"field added to the dataclass is legal in a manifest without that row; "
        f"rows missing from the header: {sorted(declared - recorded)}; rows in the "
        f"header that are not fields: {sorted(recorded - declared)}"
    )
    assert len(rows) == len(declared), "the header must not repeat a Specimen field"


def test_manifest_header_provenance_markings_are_ones_the_header_declares():
    rows = _recorded_provenance_rows()
    assert rows, "the manifest header carries no field-provenance table to check"
    unknown = sorted({marking for _, marking in rows} - set(_PROVENANCE_MARKINGS))
    assert not unknown, (
        f"the header declares its markings as {_PROVENANCE_MARKINGS}; found {unknown}"
    )


def test_shipped_manifest_records_the_scanner_and_the_inferred_stain():
    reg = {s.id: s for s in load_registry()}
    assert reg["1"].scanner == "Aperio SS1352"
    assert reg["2"].scanner == "Aperio SS1302"
    assert reg["3"].scanner == "Aperio SS1302"
    assert reg["4"].scanner == "Aperio SS1764CNTLR"
    assert reg["5"].scanner == "Aperio SS1763CNTLR"
    assert reg["6"].scanner == "Aperio SS1436CNTLR"
    assert {s.stain for s in reg.values()} == {"H&E"}


def test_shipped_manifest_records_the_acquisition_dates():
    reg = {s.id: s for s in load_registry()}
    assert reg["1"].acquisition_date == "2013-05-23"
    assert reg["2"].acquisition_date == "2013-02-28"
    assert reg["3"].acquisition_date == "2012-01-27"
    assert reg["4"].acquisition_date == "2015-07-01"
    assert reg["5"].acquisition_date == "2014-08-15"
    assert reg["6"].acquisition_date == "2013-05-08"


def test_shipped_manifest_gives_the_tcga_licence_and_ethics():
    reg = {s.id: s for s in load_registry()}
    for identifier in ("4", "5", "6"):
        assert "dbGaP phs000178" in reg[identifier].licence
        assert reg[identifier].ethics == "public, de-identified"
    for identifier in ("1", "2", "3"):
        assert reg[identifier].licence is None
        assert reg[identifier].source_case_id is None


def test_misspelled_entry_key_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  source_case_id: C\n"
        "  licence: L\n  mpp_x: 0.499\n  slide_path: a.svs\n",
    )
    with pytest.raises(RegistryError, match="mpp_x"):
        load_registry(manifest)


def test_extra_unknown_entry_key_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: local\n  source_case_id: C\n"
        "  licence: L\n  mpp: 0.499\n  magnification: 40\n  slide_path: a.svs\n",
    )
    with pytest.raises(RegistryError, match="magnification"):
        load_registry(manifest)


@pytest.mark.parametrize("field", ["licence", "source_case_id"])
def test_whitespace_only_provenance_field_keeps_the_specimen_excluded(tmp_path, field):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n- id: X\n  source: unknown\n"
        f'  source_case_id: "C"\n  licence: "L"\n  {field}: "   "\n  slide_path: a.svs\n',
    )
    specimens = load_registry(manifest)
    assert getattr(specimens[0], field) is None
    assert [s.id for s in excluded(specimens)] == ["X"]
    assert usable_for_training(specimens) == []


def test_whitespace_only_licence_with_a_named_source_is_rejected(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        'specimens:\n- id: X\n  source: local\n  source_case_id: C\n'
        '  licence: "   "\n  slide_path: a.svs\n',
    )
    with pytest.raises(RegistryError, match="no licence"):
        load_registry(manifest)


def test_padded_source_is_compared_as_the_unpadded_escape_hatch(tmp_path):
    """`source: "unknown "` is the escape hatch, padding and all.

    The licence rule at `_parse_entry` compares `source` against the literal
    `"unknown"`. An unstripped value misses that comparison, so a padded escape
    hatch is read as a named source with no licence and the whole manifest is
    rejected for a row that in fact declares one.
    """
    manifest = _write_manifest(
        tmp_path,
        'specimens:\n- id: X\n  source: "unknown "\n  slide_path: a.svs\n',
    )
    specimens = load_registry(manifest)
    assert specimens[0].source == "unknown"
    assert [s.id for s in excluded(specimens)] == ["X"]


def test_padded_slide_path_resolves_to_the_file_that_exists(tmp_path):
    (tmp_path / "a.svs").write_bytes(b"")
    manifest = _write_manifest(
        tmp_path,
        'specimens:\n- id: X\n  source: unknown\n  slide_path: " a.svs "\n',
    )
    slide_path = load_registry(manifest)[0].slide_path
    assert slide_path == tmp_path / "a.svs"
    assert slide_path.is_file()


def test_padded_identifier_does_not_defeat_the_duplicate_check(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n"
        '- id: "X "\n  source: unknown\n  slide_path: a.svs\n'
        '- id: "X"\n  source: unknown\n  slide_path: b.svs\n',
    )
    with pytest.raises(RegistryError, match="duplicate"):
        load_registry(manifest)


def test_one_malformed_row_fails_the_whole_manifest(tmp_path):
    manifest = _write_manifest(
        tmp_path,
        "specimens:\n"
        "- id: GOOD\n  source: local\n  source_case_id: C\n  licence: L\n  slide_path: a.svs\n"
        "- id: BAD\n  source: local\n  slide_path: b.svs\n",
    )
    with pytest.raises(RegistryError, match="'BAD'"):
        load_registry(manifest)
