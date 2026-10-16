import re

import pytest
import yaml

from blobcount.config import (
    KEY_TYPES,
    KNOWN_KEYS,
    PROJECT_ROOT,
    Config,
    ConfigError,
    _find_project_root,
    _iter_keys,
    _lookup,
    _reset_default_config,
    default_config,
    get_device,
    load,
    set_seed,
)

# A read of a declared key looks like `cfg.get("a.b")`, `self.get("a.b")`, or
# `cfg.path("a.b")`. A key the code never reads cannot match.
_KEY_READ = re.compile(r"""\.(?:get|path)\(\s*["']([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)["']""")

# Every declared key that no module under blobcount/ reads yet. The consumer keys that
# arrive with later tasks are the complement, `KNOWN_KEYS` minus this set; the count is
# not written here because a number in a comment is the one thing the equality below
# cannot correct. That equality pins both directions, so this set cannot mask a
# regression in either one: a key that gains a reader is named as a stale entry to
# delete, and a key that loses one is named as an unread declared key.
#
# **`paths.labels` and `paths.results` were read by `bin/train.py` alone and are read by
# `blobcount/train.py` now**, because `bin/` is a directory of scripts loaded by path and a
# read there is outside this scan: `read_labels` resolves every `labels.csv` row against
# `paths.labels` and `run_cross_validation` defaults its output directory to
# `paths.results`, so both keys have a reader under `blobcount/` and consume the value. The
# entry point kept `--out` and the two moved under the package because Task 12 reads the
# same handoff — a reader in each place would be one answer to the CSV contract written
# twice, and a read written into `blobcount/train.py` for no other reason than to clear a
# name in this set would be the fake reader this list exists to catch.
#
# **`evaluation.bootstrap_samples` and `evaluation.confidence_level` were Task 12's two, and
# `blobcount/evaluate.py` reads them** in `bootstrap_metric_intervals` rather than in
# `bin/evaluate.py`, for the same reason: a read in `bin/` is outside this scan. The read
# consumes the value — the interval `medical_metrics.json` records is the configured one —
# and the test that says so is
# `tests/test_evaluate.py::test_the_intervals_are_read_from_the_configuration_rather_than_written_into_the_script`.
#
# **`gradcam.alpha` and `gradcam.colormap` were Task 13's two, and `blobcount/interpret.py`
# reads them** in `overlay_from_config` rather than in `bin/gradcam.py`, for the same
# reason. The read consumes the value — the blend is drawn at the configured weight, in
# the configured colour map — and the test that says so is
# `tests/test_interpret.py::test_overlay_reads_the_two_gradcam_keys_from_the_configuration`,
# which holds both by setting `gradcam.alpha=0` (the blend then *is* the image) and by
# asking for a second map (the two blends then differ).
#
# **`project.name` was the last entry and it was deleted on 2026-10-16**, on the ruling
# that a configuration file declaring unimplemented behaviour is worse than no
# configuration file: no reader was invented for it, and a read written only to clear the
# entry would be the fake reader this list exists to catch. **The set is empty and stays
# a `frozenset`, not `None` and not a deleted name**, because the scan above compares
# against it and a branch that grows a second unreadable key has to fail here rather than
# have nothing to compare against. The merge gate the plan spells out as a shell command
# is spelled out as a test in
# `test_no_declared_key_is_exempt_from_the_reader_scan` below, because that command's
# regex cannot match this literal and raises `AttributeError` instead of reporting.
_KEYS_WITHOUT_READER: frozenset[str] = frozenset()


def _default_yaml_keys() -> set[str]:
    raw = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    return set(_iter_keys(yaml.safe_load(raw)))


def _default_yaml() -> dict:
    """Return the shipped configuration as parsed, for the type assertions to read."""
    raw = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(raw)


def _package_source() -> str:
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((PROJECT_ROOT / "blobcount").rglob("*.py"))
    )


def _cfg_with_device(device: str) -> Config:
    return load(overrides=[f"project.device={device}"])


def test_loads_declared_value():
    cfg = load()
    assert cfg.get("extraction.target_mpp") == 0.5
    assert cfg.get("extraction.patch_size") == 256


def test_unknown_top_level_section_raises():
    with pytest.raises(ConfigError, match="unknown section"):
        load(overrides=["not_a_section.key=1"])


def test_dot_notation_get_and_set():
    cfg = load()
    cfg.set("training.batch_size", 8)
    assert cfg.get("training.batch_size") == 8


def test_missing_required_key_raises(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("project:\n  seed: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key"):
        load(partial)


@pytest.mark.parametrize(
    "key",
    [
        "blobs.morph_kernel_size",
        "blobs.morph_iterations",
        "blobs.stain_normalization",
        "extraction.tissue_percentile",
        "extraction.tissue_gray_max",
    ],
)
def test_every_key_a_run_reads_after_it_deletes_something_is_required(tmp_path, key):
    # `REQUIRED_KEYS` is the only place a missing key can be refused *before* a run
    # deletes anything, so a key read with a bare `cfg.get` on such a path has to be in
    # it. The three `blobs` keys are read by `label_patches` and `prepare_patch` after
    # `prepare_output` has emptied the label tree, and the two `extraction` keys by
    # `extract_slide` after a `--force` run has emptied the patch tree; without them a
    # configuration that omits one took the previous run's artifacts with it and then
    # raised past a command line that had no answer for it.
    data = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    )
    section, name = key.split(".")
    del data[section][name]
    partial = tmp_path / "partial.yaml"
    partial.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigError, match="missing required keys") as refused:
        load(partial)
    assert key in str(refused.value)


def test_known_keys_match_default_yaml():
    declared = _default_yaml_keys()
    assert set(KNOWN_KEYS) - declared == set()
    assert declared - set(KNOWN_KEYS) == set()


def test_the_study_name_is_not_declared_by_this_project(tmp_path):
    # **`project.name` was deleted on 2026-10-16**, and this is the test that holds it
    # deleted. It was the last entry in `_KEYS_WITHOUT_READER`: a declared key that no
    # module reads, kept in the file because the plan made an empty exempt set a merge
    # gate. No reader was invented for it — the ruling is that a configuration file
    # declaring unimplemented behaviour is worse than no configuration file, and that a
    # test written only to clear a name would be the fake reader the source scan above
    # exists to catch.
    #
    # **Three separate declarations, and all three have to go.** The key in the shipped
    # file, the key in `KNOWN_KEYS` and the key in `KEY_TYPES` are the same fact written
    # three times, and a deletion that removed only one of them would leave the file
    # refusing to load or the table declaring a key nothing holds. Hence the three arms.
    assert "project.name" not in KNOWN_KEYS
    assert "project.name" not in KEY_TYPES
    assert "project.name" not in _default_yaml_keys()

    # And the refusal is the one the gate exists for, reached through a file rather than
    # through `Config`, because a configuration that declares an undeclared key has to be
    # a hard error at load and not a key quietly ignored.
    with pytest.raises(ConfigError, match="unknown key 'project.name'"):
        load(_config_with_one_value_edited("project.name", "x", tmp_path))


def test_no_declared_key_is_exempt_from_the_reader_scan():
    # The merge gate, as a test rather than as a shell command. The plan's gate is a
    # `re.search` over this file whose pattern cannot match a set written across lines —
    # measured, it raises `AttributeError: 'NoneType' object has no attribute 'group'`
    # on the shipped file today, so it has never reported anything at all. This asserts
    # the fact it was written to assert: the exempt set is empty, so every declared key
    # has a reader under `blobcount/` or the branch does not satisfy the gate.
    assert _KEYS_WITHOUT_READER == frozenset()


def test_every_declared_config_key_is_read_by_package_source():
    read = set(_KEY_READ.findall(_package_source()))
    assert read <= KNOWN_KEYS, f"code reads undeclared keys: {sorted(read - KNOWN_KEYS)}"
    unread = KNOWN_KEYS - read
    assert unread == _KEYS_WITHOUT_READER, (
        "declared keys that no module under blobcount/ reads: "
        f"{sorted(unread - _KEYS_WITHOUT_READER)}; "
        f"stale entries to delete from _KEYS_WITHOUT_READER: "
        f"{sorted(_KEYS_WITHOUT_READER - unread)}"
    )


def test_load_records_no_reads():
    cfg = load()
    assert cfg.unknown_keys() == set(KNOWN_KEYS)


def test_unknown_keys_reports_a_key_no_code_read():
    cfg = Config({"project": {"seed": 42, "device": "cpu"}})
    assert cfg.unknown_keys() == {"project.seed", "project.device"}
    assert cfg.get("project.seed") == 42
    assert cfg.unknown_keys() == {"project.device"}


def test_get_with_explicit_none_default_returns_none():
    cfg = Config({"project": {"seed": 42}})
    assert cfg.get("project.device", None) is None


def test_get_without_default_raises_for_missing_key():
    cfg = Config({"project": {"seed": 42}})
    with pytest.raises(ConfigError, match="unknown key"):
        cfg.get("project.device")


def test_set_rejects_an_undeclared_key():
    cfg = load()
    with pytest.raises(ConfigError, match="unknown section"):
        cfg.set("not_a_section.key", 1)
    with pytest.raises(ConfigError, match="unknown key"):
        cfg.set("training.never_read_by_code", 1)


# `set` is the second route into a configuration, and it typed nothing at all before this
# group: `_validate_key` and `_set_dotted`, so `default_config().set("training.pretrained",
# 1)` inverted a run in a process holding a long-lived config in exactly the way the file
# path used to. These are the same properties the file path holds down above, reached the
# other way round.


def test_set_refuses_a_value_whose_type_is_not_the_declared_one():
    cfg = load()
    with pytest.raises(ConfigError, match="has type"):
        cfg.set("training.pretrained", 1)


def test_a_bool_declared_key_refuses_an_int_through_set():
    # The measured inversion, on the route that has no file. `int` is truthy, so this used
    # to write a value that `build_model` would read as "train from ImageNet weights" and
    # say nothing at all — the defect the file path was closed for, still open one function
    # to the right.
    cfg = load()
    with pytest.raises(ConfigError) as caught:
        cfg.set("augmentation.horizontal_flip", 1)
    message = str(caught.value)
    assert "has type int" in message, message
    assert "as bool" in message, message


def test_the_set_refusal_names_the_key_the_type_found_and_the_remedy():
    cfg = load()
    with pytest.raises(ConfigError) as caught:
        cfg.set("extraction.patch_size", "big")
    message = str(caught.value)
    # The key and both types, exactly as on the file path: a refusal is a refusal whichever
    # route delivered the value, and a caller reading one has no way to tell which produced
    # it, so the two messages say the same things.
    assert "extraction.patch_size" in message, message
    assert "has type str" in message, message
    assert "as int" in message, message
    assert "supply a value of type int" in message, message


def test_a_refused_set_leaves_the_configuration_unchanged():
    # `set` checks the value before it writes, so a caller can retry with a corrected value
    # without reloading — and a refused `set` does not half-apply. Without this the type
    # check could be added to the wrong side of the write and every refusal would leave the
    # bad value in place.
    cfg = load()
    before = cfg.get("training.pretrained")
    with pytest.raises(ConfigError):
        cfg.set("training.pretrained", 1)
    assert cfg.get("training.pretrained") is before


def test_set_and_load_refuse_the_same_value_the_same_way(tmp_path):
    # One rule, two routes. `set` calls the same `_check_value_type` the file path does, so a
    # value the file refuses cannot be written by hand into a loaded config, and the two
    # routes cannot drift into disagreeing about what a declared key holds.
    with pytest.raises(ConfigError) as from_file:
        load(_config_with_one_value_edited("training.batch_size", "big", tmp_path))
    with pytest.raises(ConfigError) as from_set:
        load().set("training.batch_size", "big")
    file_message = str(from_file.value)
    set_message = str(from_set.value)
    for text in ("has type str", "as int", "training.batch_size"):
        assert text in file_message, file_message
        assert text in set_message, set_message


def test_set_widens_an_int_under_a_declared_float_as_the_file_path_does():
    # The one admitted pair, admitted on both routes. A `set` that refused it would make the
    # two routes disagree about `blobs.area_um2_max = 125`, which the override path accepts.
    cfg = load()
    cfg.set("blobs.area_um2_max", 125)
    value = cfg.get("blobs.area_um2_max")
    assert value == 125.0
    assert type(value) is float


def test_set_accepts_a_value_of_the_declared_type_whatever_it_means():
    # The negative space again, on this route. `set` is a type check and not a value check,
    # so the caller who turns a knob past its range writes it successfully and the reading
    # function refuses it — which is where that refusal belongs, and is what
    # `tests/test_labeling.py` pins for `blobs.gray_percentile: 150`.
    cfg = load()
    cfg.set("blobs.gray_percentile", 150)
    assert cfg.get("blobs.gray_percentile") == 150


@pytest.mark.parametrize(
    "override",
    [
        "augmentation.horizontal_flip=flase",
        "training.pretrained=fals",
        "blobs.stain_normalization=nope",
        "augmentation.horizontal_flip=1",
        "blobs.area_um2_max=true",
    ],
)
def test_mistyped_override_value_is_rejected(override):
    with pytest.raises(ConfigError, match="has type"):
        load(overrides=[override])


def test_mistyped_override_error_names_the_expected_type():
    with pytest.raises(ConfigError) as caught:
        load(overrides=["augmentation.horizontal_flip=flase"])
    message = str(caught.value)
    assert "bool" in message, message
    assert "str" in message, message


def test_int_override_widens_to_float_for_a_float_key():
    cfg = load(overrides=["blobs.area_um2_max=125"])
    value = cfg.get("blobs.area_um2_max")
    assert value == 125.0
    assert isinstance(value, float)


def test_int_override_is_not_accepted_for_an_int_key():
    with pytest.raises(ConfigError, match="has type"):
        load(overrides=["extraction.patch_size=256.0"])


# The next group is the file path, which `load` typed only for overrides before this round.
# The measured defect is in the table below and nowhere else in this file: `load()` reads
# `configs/default.yaml` when given no `path`, so a check that compared a file against that
# same file would have measured `training.pretrained: 1` against the `1` it had just become
# and called it correct. The declared types live in `KEY_TYPES` and these tests pin all four
# things that have to be true of it.


def _config_with_one_value_edited(key, value, tmp_path):
    """Write `configs/default.yaml` with the value at `key` replaced, and return its path.

    The file is written through `yaml.safe_dump` the way a hand-edit reaches `load`, so the
    value arrives as the type PyYAML produces for it rather than as the object a test built.
    """
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    section, name = key.split(".")
    data[section][name] = value
    path = tmp_path / f"config-{name}-is-{type(value).__name__}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_key_types_cover_every_declared_key_and_nothing_else():
    # The table is the whole of the check, so a key missing from it is a key `load` cannot
    # type, and a key in it that nothing declares is a rule about a key that cannot exist.
    assert set(KEY_TYPES) == set(KNOWN_KEYS)


def test_key_types_agree_with_default_yaml_in_both_directions():
    # What stops the table and the shipped file drifting apart. Without this the two could
    # disagree and the file would stop loading at all, or — worse — a rule could be loosened
    # in one and tightened in the other with nothing here noticing which.
    values = {
        key: _lookup(_default_yaml(), key)[1]
        for key in _iter_keys(_default_yaml())
    }
    assert {key: type(value) for key, value in values.items()} == KEY_TYPES


def test_every_declared_key_is_compared_as_a_bool_where_the_file_says_true():
    # `bool` is spelled `bool` and not `int` for the reason the whole comparison is `type() is`:
    # `bool` subclasses `int`, `1 != True` is False, and an `isinstance`-shaped check would
    # accept `augmentation.horizontal_flip: 1` as a truthy integer and invert the flip.
    asserted_bools = sorted(
        key for key in KEY_TYPES if KEY_TYPES[key] is bool
    )
    assert asserted_bools == [
        "augmentation.horizontal_flip",
        "augmentation.vertical_flip",
        "blobs.stain_normalization",
        "training.pretrained",
        "training.use_weighted_sampler",
    ]
    # Four of the five are `true` in the shipped file and `blobs.stain_normalization` is
    # `false`; both are the same declaration, and the point is the *type* — a `false` in a
    # key declared `bool` is what proves `bool` was the type read and not `int`.
    for key in asserted_bools:
        assert _lookup(_default_yaml(), key)[1] is (key != "blobs.stain_normalization"), key


def test_a_bool_declared_key_refuses_an_int_written_in_the_file(tmp_path):
    # The measured defect, and the one the mechanism exists for. `int` is truthy, so this
    # used to take the pretrained path with no error anywhere in the program: a silent
    # inversion reachable by deleting three characters from a git-tracked file.
    with pytest.raises(ConfigError, match="has type"):
        load(_config_with_one_value_edited("training.pretrained", 1, tmp_path))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # One row per declared type, so a rule that fires for `bool` and not for `list` is
        # visible here rather than inferred. `str` has two rows: one under `paths.*` and
        # one under `training.*`, because `Config.path` and a plain `get` treat the same
        # declared type differently.
        ("paths.manifest", 7),
        ("project.seed", "42"),
        ("paths.results", 7),
        ("extraction.patch_size", "big"),
        ("extraction.target_mpp", "half"),
        # A float key is refused a `str` and a `list`, but *not* an `int`: that is the one
        # admitted pair and `test_an_int_in_the_file_widens_to_float_for_a_float_key` below
        # is where it is asserted, so the two tests together are the whole rule.
        ("blobs.area_um2_min", "3.1"),
        ("blobs.gray_percentile", 95.5),
        ("blobs.stain_normalization", 1),
        ("labeling.blob_threshold", "5"),
        ("ablation.thresholds", "35"),
        ("training.architecture", 18),
        ("training.pretrained", 1),
        ("training.learning_rate", "0.0001"),
        ("training.use_weighted_sampler", "yes"),
        ("augmentation.horizontal_flip", 1),
        ("augmentation.rotation_degrees", 15.0),
        ("augmentation.color_jitter_hue", [0.05]),
        ("evaluation.bootstrap_samples", "2000"),
        ("gradcam.colormap", 3),
        ("gradcam.alpha", "0.4"),
    ],
)
def test_a_file_value_whose_type_is_not_the_declared_one_is_refused(key, value, tmp_path):
    with pytest.raises(ConfigError, match="has type"):
        load(_config_with_one_value_edited(key, value, tmp_path))


def test_the_file_refusal_names_the_key_the_type_found_and_the_type_read(tmp_path):
    path = _config_with_one_value_edited("extraction.patch_size", "big", tmp_path)
    with pytest.raises(ConfigError) as caught:
        load(path)
    message = str(caught.value)
    # The key *first*, in the opening clause. The message names the key twice — once as what
    # was found and once in the remedy — so an assertion that only looked for it somewhere
    # would be satisfied by the remedy alone, and the clause identifying the mistake could
    # be dropped without anything here going red.
    assert message.startswith(f"key 'extraction.patch_size' in {path}"), message
    # The type found, then the type read — two clauses, because they are two different facts
    # and a message that named only one would leave a reader guessing which was wrong.
    assert "has type str" in message, message
    assert "but this project reads it as int" in message, message
    # The remedy names the type again *and* the file that states a correct one, so the
    # refusal ends in something a caller can act on rather than in a rule with no next step.
    assert "supply a value of type int for 'extraction.patch_size'" in message, message
    assert "configs/default.yaml" in message, message


def test_a_bool_written_under_an_int_declared_key_is_refused(tmp_path):
    # **The row that makes the comparison's shape observable.** Every other row of the table
    # above is refused under an `isinstance`-shaped check too, because `int` is not a
    # `str`, not a `float` and not a `list`. This one is not: `bool` subclasses `int`, so
    # `isinstance(True, int)` is `True` and an `isinstance` check would read
    # `extraction.patch_size: true` as the integer 1 and build a 1-square patch out of it.
    # `type(value) is expected` has no such gap, and this is the row that proves it.
    with pytest.raises(ConfigError, match="has type"):
        load(_config_with_one_value_edited("extraction.patch_size", True, tmp_path))


def test_the_declared_types_are_not_read_from_the_file_being_validated(
    tmp_path, monkeypatch
):
    # **The tautology, tested rather than asserted in a comment.** If `_check_value_type`
    # read its expected type out of `configs/default.yaml` instead of out of `KEY_TYPES`, then
    # `load()` — which reads `configs/default.yaml` by default — would measure that file
    # against itself and accept every hand-edit to it. `training.pretrained: 1` would agree
    # with the `int` it had just become and be reported correct, which is the measured defect
    # this round exists to close.
    #
    # The test points `PROJECT_ROOT` at a scratch tree whose own
    # `configs/default.yaml` holds the mistyped value, and loads it. A check that measured
    # the file against itself would find `training.pretrained` an `int`, agree, and return
    # a config; the table in this file says `bool` and refuses.
    scratch = tmp_path / "scratch"
    (scratch / "configs").mkdir(parents=True)
    (scratch / "configs" / "default.yaml").write_text(
        (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.setattr("blobcount.config.PROJECT_ROOT", scratch)
    monkeypatch.setattr("blobcount.config.DEFAULT_CONFIG_PATH", scratch / "configs" / "default.yaml")
    assert load().get("training.pretrained") is True

    # Now hand-edit that very file, which is the only way `load()` reaches it.
    data = yaml.safe_load((scratch / "configs" / "default.yaml").read_text(encoding="utf-8"))
    data["training"]["pretrained"] = 1
    (scratch / "configs" / "default.yaml").write_text(
        yaml.safe_dump(data), encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="has type"):
        load()


def test_the_file_refusal_names_the_file_it_came_from(tmp_path):
    path = _config_with_one_value_edited("training.batch_size", "big", tmp_path)
    with pytest.raises(ConfigError) as caught:
        load(path)
    assert str(path) in str(caught.value), str(caught.value)


def test_an_int_in_the_file_widens_to_float_for_a_float_key(tmp_path):
    # The one pair the check admits, and it admits it on this path for the same reason it
    # does on the override path: `125` and `125.0` are the same number, so refusing one
    # spelling would be a rule about writing rather than about type. Measured through the
    # config the caller reads, not through `_check_value_type` directly.
    cfg = load(_config_with_one_value_edited("blobs.area_um2_max", 125, tmp_path))
    value = cfg.get("blobs.area_um2_max")
    assert value == 125.0
    assert type(value) is float


def test_a_float_in_the_file_is_not_narrowed_to_an_int_key(tmp_path):
    # The widening is one-directional. `extraction.patch_size: 256.0` is the wrong type for a
    # key this project reads as an `int`, and admitting it would make the check a comparison
    # of values rather than of types.
    with pytest.raises(ConfigError, match="has type"):
        load(_config_with_one_value_edited("extraction.patch_size", 256.0, tmp_path))


def test_both_input_paths_refuse_the_same_value_the_same_way(tmp_path):
    # The two paths are one rule, so the same mistake is the same refusal whichever way it
    # arrives. Before this, the override arm refused it and the file arm accepted it, and
    # which one a caller hit depended on how the value was written rather than on the value.
    path = _config_with_one_value_edited("training.pretrained", 1, tmp_path)
    with pytest.raises(ConfigError) as from_file:
        load(path)
    with pytest.raises(ConfigError) as from_override:
        load(overrides=["training.pretrained=1"])
    file_message = str(from_file.value)
    override_message = str(from_override.value)
    for text in ("has type int", "training.pretrained"):
        assert text in file_message, file_message
        assert text in override_message, override_message


def test_a_hand_edited_value_of_the_declared_type_is_accepted(tmp_path):
    # **A copy of the shipped file with one value replaced, which is what this test loads** —
    # it does not load `configs/default.yaml` itself, so it says nothing about that file
    # being accepted and used to be named as though it did. The check it does make is the one
    # that matters for the tautology: a value of the declared `str` where the shipped file
    # has a different `str` is accepted, so the type check is not a comparison of values and
    # is not a rule about how a value is spelled.
    #
    # That the shipped file itself loads is covered by every other test in this file that
    # calls `load()` with no argument — `test_loads_declared_value` first among them — and
    # that its types match the table is `test_key_types_agree_with_default_yaml_in_both_
    # directions` above. The stronger claim, that a hand-edit to *that* file is refused, is
    # `test_the_declared_types_are_not_read_from_the_file_being_validated`.
    cfg = load(_config_with_one_value_edited("gradcam.colormap", "x", tmp_path))
    assert cfg.get("gradcam.colormap") == "x"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # A declared type and a nonsense value of it. None of these is refused, and that is
        # the point of the boundary: `load` answers "is this the type I read", and each of
        # these is. What a value has to *mean* belongs to the function that reads it —
        # `ablation._thresholds` refuses a non-finite threshold, `ablation` refuses a
        # percentile outside [0, 100], and neither moved here.
        ("evaluation.confidence_level", 1.0),
        ("evaluation.confidence_level", -1.0),
        ("extraction.patch_size", -1),
        ("extraction.patch_size", 0),
        ("blobs.area_um2_max", 0.0),
        ("training.epochs", -3),
        ("augmentation.rotation_degrees", -15),
        ("training.pretrained", False),
        ("blobs.stain_normalization", False),
    ],
)
def test_a_value_of_the_declared_type_is_accepted_whatever_it_means(key, value, tmp_path):
    # The negative space of the type check, and the assertion a future reader who wants to
    # widen `load` into a value validator has to delete. Without it, "add a range check" is
    # a one-line change with nothing red.
    assert load(_config_with_one_value_edited(key, value, tmp_path)).get(key) == value


def test_a_thresholds_key_that_is_not_a_list_is_refused(tmp_path):
    # Named because `tests/test_ablation.py` asserts the same value reaches
    # `_thresholds`' own check, and it can no longer arrive there from a file. The two tests
    # are one move of one shape: what used to be written out as YAML to get past a check that
    # only covered overrides is now refused earlier, and the reader of either test is pointed
    # at the other.
    with pytest.raises(ConfigError, match="has type"):
        load(_config_with_one_value_edited("ablation.thresholds", b"35", tmp_path))


def _config_missing_a_required_key_and_mistyped(tmp_path):
    """Return a path to a file that is wrong in both ways at once.

    `training.architecture` is dropped, which is a key in `REQUIRED_KEYS`, and
    `training.epochs` is written as a string, which is the wrong type for a key declared
    `int`. A file wrong in one way only cannot say anything about which of two checks runs
    first, which is what the test below needs.
    """
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    del data["training"]["architecture"]
    data["training"]["epochs"] = "ten"
    path = tmp_path / "wrong-in-both-ways.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_a_typed_file_key_refuses_before_a_missing_required_key(tmp_path):
    # **The order of the two checks, pinned on a file that is wrong in both ways.** An
    # earlier version of this test asserted the same thing on a file whose only fault was the
    # type, and its own comment admitted no required key was missing — so moving the type
    # loop below the missing-key check would have left it green and this claim held down by
    # nothing. This file drops a required key *and* mistypes another, so the two checks
    # compete and only one can report.
    #
    # The order is the type check first, and the reason is what a caller does with the
    # answer: the message names the key and the two types and how to fix them, so a reader
    # fixes one line and is done, where a missing-key message names a key whose line is
    # present and correctly spelled and reads as a puzzle. The two faults are independent
    # and both real, so this only says which is reported; it does not say the other is not
    # there. Fixing the first mistake and reloading surfaces the second.
    with pytest.raises(ConfigError) as caught:
        load(_config_missing_a_required_key_and_mistyped(tmp_path))
    message = str(caught.value)
    assert "has type" in message, message
    assert "training.epochs" in message, message
    assert "missing required keys" not in message, message


def test_a_missing_required_key_is_still_refused_when_no_value_is_mistyped(tmp_path):
    # The other direction of the same pair, so neither test can pass while the other order
    # holds: a file that only omits a required key is reported for the omission, which is
    # what `test_missing_required_key_raises` above already says for every required key.
    data = yaml.safe_load((PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    del data["training"]["architecture"]
    path = tmp_path / "missing-only.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required keys"):
        load(path)


def test_override_without_equals_raises():
    with pytest.raises(ConfigError, match="key=value"):
        load(overrides=["training.batch_size"])


def test_override_on_a_non_section_parent_raises(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("training: 3\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not a section"):
        load(partial, overrides=["training.batch_size=8"])


def test_paths_resolve_against_project_root():
    cfg = load()
    resolved = cfg.path("paths.results")
    assert resolved.is_absolute()
    assert resolved.name == "results"


def test_path_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("BLOBCOUNT_TEST_ROOT", str(tmp_path))
    cfg = load(overrides=["paths.results=%BLOBCOUNT_TEST_ROOT%/out"])
    assert cfg.path("paths.results") == tmp_path / "out"


def test_path_of_a_posix_absolute_value_is_a_concrete_path():
    resolved = load(overrides=["paths.results=/content/drive/results"]).path("paths.results")
    assert isinstance(resolved, type(PROJECT_ROOT / "results")), (
        f"expected a concrete Path, got {type(resolved).__name__}"
    )


def test_path_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg = load(overrides=["paths.results=~/out"])
    assert cfg.path("paths.results") == tmp_path / "out"


def test_path_rejects_a_non_string_value():
    cfg = Config({"paths": {"slides": 7}})
    with pytest.raises(ConfigError, match="must hold a path string"):
        cfg.path("paths.slides")


def test_path_rejects_a_blank_value():
    cfg = Config({"paths": {"slides": "   "}})
    with pytest.raises(ConfigError, match="empty path"):
        cfg.path("paths.slides")


def test_project_root_not_found_raises(tmp_path):
    orphan = tmp_path / "orphan" / "package"
    orphan.mkdir(parents=True)
    with pytest.raises(ConfigError, match="pyproject.toml"):
        _find_project_root(orphan)


def test_set_seed_makes_torch_deterministic():
    import torch

    set_seed(42)
    a = torch.randn(4)
    set_seed(42)
    assert torch.equal(a, torch.randn(4))


def test_set_seed_uses_the_config_seed():
    import torch

    set_seed(1234)
    expected = torch.randn(4)
    set_seed(cfg=load(overrides=["project.seed=1234"]))
    assert torch.equal(expected, torch.randn(4))


def test_get_device_returns_known_device():
    assert get_device(_cfg_with_device("cpu")) == "cpu"
    assert get_device(_cfg_with_device("auto")) in {"cpu", "cuda"}
    with pytest.raises(ConfigError, match="project.device"):
        get_device(_cfg_with_device("tpu"))


def test_get_device_falls_back_to_the_default_config():
    assert get_device() in {"cpu", "cuda"}


# The next two tests are one unit and depend on this order. The first mutates the
# process-wide default config; the autouse fixture in conftest.py has to hand the
# second one a config built fresh from configs/default.yaml.


def test_default_config_is_mutated_by_this_test():
    default_config().set("gradcam.colormap", "mutated-by-an-earlier-test")
    assert default_config().get("gradcam.colormap") == "mutated-by-an-earlier-test"


def test_default_config_mutation_does_not_leak_into_the_next_test():
    assert default_config().get("gradcam.colormap") == load().get("gradcam.colormap")
    assert default_config().get("project.seed") == load().get("project.seed")


# The pair above only fails when pytest runs it in definition order; under a
# random-order plugin the leak check would pass vacuously. This test drives the
# reset hook itself, so the guard holds in any order and in isolation.


def test_reset_hook_rebuilds_the_default_config_after_a_mutation():
    mutated = "mutated-by-an-earlier-test"
    default_config().set("gradcam.colormap", mutated)
    assert default_config().get("gradcam.colormap") == mutated

    _reset_default_config()

    assert default_config().get("gradcam.colormap") == load().get("gradcam.colormap")
    assert default_config().get("gradcam.colormap") != mutated
