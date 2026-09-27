import re

import pytest
import yaml

from blobcount.config import (
    KNOWN_KEYS,
    PROJECT_ROOT,
    Config,
    ConfigError,
    _find_project_root,
    _iter_keys,
    get_device,
    load,
    set_seed,
)

# A read of a declared key looks like `cfg.get("a.b")`, `self.get("a.b")`, or
# `cfg.path("a.b")`. A key the code never reads cannot match.
_KEY_READ = re.compile(r"""\.(?:get|path)\(\s*["']([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)["']""")

# Every declared key that no module under blobcount/ reads yet. This task ships
# the loader only, so the 44 consumer keys arrive with later tasks. A key that
# loses its reader must not be added here, because the scan then fails. This set
# is expected to be empty once the pipeline is complete.
_KEYS_WITHOUT_READER: frozenset[str] = frozenset(
    {
        "ablation.thresholds",
        "augmentation.affine_translate",
        "augmentation.color_jitter_brightness",
        "augmentation.color_jitter_contrast",
        "augmentation.color_jitter_hue",
        "augmentation.color_jitter_saturation",
        "augmentation.horizontal_flip",
        "augmentation.rotation_degrees",
        "augmentation.vertical_flip",
        "blobs.area_um2_max",
        "blobs.area_um2_min",
        "blobs.circularity_min",
        "blobs.gray_percentile",
        "blobs.morph_iterations",
        "blobs.morph_kernel_size",
        "blobs.stain_normalization",
        "evaluation.bootstrap_samples",
        "evaluation.confidence_level",
        "extraction.patch_size",
        "extraction.step_size",
        "extraction.target_mpp",
        "extraction.tissue_gray_max",
        "extraction.tissue_percentile",
        "gradcam.alpha",
        "gradcam.colormap",
        "labeling.blob_threshold",
        "paths.labels",
        "paths.manifest",
        "paths.patches",
        "paths.results",
        "paths.slides",
        "project.name",
        "training.architecture",
        "training.batch_size",
        "training.epochs",
        "training.input_size",
        "training.learning_rate",
        "training.lr_scheduler_factor",
        "training.num_classes",
        "training.num_workers",
        "training.patience_early_stop",
        "training.patience_lr_scheduler",
        "training.pretrained",
        "training.use_weighted_sampler",
    }
)


def _default_yaml_keys() -> set[str]:
    raw = (PROJECT_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")
    return set(_iter_keys(yaml.safe_load(raw)))


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
    partial.write_text("project:\n  name: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key"):
        load(partial)


def test_known_keys_match_default_yaml():
    declared = _default_yaml_keys()
    assert set(KNOWN_KEYS) - declared == set()
    assert declared - set(KNOWN_KEYS) == set()


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
    cfg = Config({"project": {"name": "x", "seed": 42}})
    assert cfg.unknown_keys() == {"project.name", "project.seed"}
    assert cfg.get("project.seed") == 42
    assert cfg.unknown_keys() == {"project.name"}


def test_get_with_explicit_none_default_returns_none():
    cfg = Config({"project": {"seed": 42}})
    assert cfg.get("project.name", None) is None


def test_get_without_default_raises_for_missing_key():
    cfg = Config({"project": {"seed": 42}})
    with pytest.raises(ConfigError, match="unknown key"):
        cfg.get("project.name")


def test_set_rejects_an_undeclared_key():
    cfg = load()
    with pytest.raises(ConfigError, match="unknown section"):
        cfg.set("not_a_section.key", 1)
    with pytest.raises(ConfigError, match="unknown key"):
        cfg.set("training.never_read_by_code", 1)


@pytest.mark.parametrize(
    "override",
    [
        "augmentation.horizontal_flip=flase",
        "training.pretrained=fals",
        "blobs.stain_normalization=nope",
    ],
)
def test_mistyped_override_value_is_rejected(override):
    with pytest.raises(ConfigError, match="has type"):
        load(overrides=[override])


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
