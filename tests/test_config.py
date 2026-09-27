import pytest
from blobcount.config import Config, ConfigError, load, set_seed, get_device


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


def test_config_with_no_unread_keys_on_load():
    cfg = load()
    assert cfg.unknown_keys() == set()


def test_paths_resolve_against_project_root():
    cfg = load()
    resolved = cfg.path("paths.results")
    assert resolved.is_absolute()
    assert resolved.name == "results"


def test_path_expands_environment_variables(monkeypatch):
    monkeypatch.setenv("BLOBCOUNT_TEST_ROOT", "/tmp/blobcount")
    cfg = load(overrides=["paths.results=%BLOBCOUNT_TEST_ROOT%/out"])
    assert str(cfg.path("paths.results")) == "/tmp/blobcount/out"


def test_path_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    cfg = load(overrides=["paths.results=~/out"])
    assert cfg.path("paths.results") == tmp_path / "out"


def test_set_seed_makes_torch_deterministic():
    import torch
    set_seed(42)
    a = torch.randn(4)
    set_seed(42)
    assert torch.equal(a, torch.randn(4))


def test_get_device_returns_known_device():
    assert get_device() in {"cpu", "cuda"}
