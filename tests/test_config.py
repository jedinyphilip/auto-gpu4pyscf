"""Settings defaults, persistence and the legacy location."""
import json

import pytest

from auto_gpu4pyscf.config import DEFAULTS, Settings


def test_defaults_when_nothing_saved(tmp_path):
    settings = Settings.load(path=tmp_path / "settings.json")
    assert settings.backend == "docker"
    assert settings.image == "gpu4pyscf:local"
    assert settings.env_dir.endswith("env")


def test_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    settings = Settings(path=path)
    settings.backend = "env"
    settings.image = "gpu4pyscf:test"
    settings.env_dir = tmp_path / "elsewhere"
    settings.save()

    reloaded = Settings.load(path=path)
    assert reloaded.backend == "env"
    assert reloaded.image == "gpu4pyscf:test"
    assert reloaded.env_dir == str(tmp_path / "elsewhere")


def test_unknown_keys_are_dropped(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"backend": "env", "rm": "-rf /"}))
    settings = Settings.load(path=path)
    assert settings.backend == "env"
    assert set(settings.values) == set(DEFAULTS)


def test_corrupt_file_falls_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{{{")
    assert Settings.load(path=path).backend == "docker"


def test_image_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("IMAGE", "gpu4pyscf:from-env")
    assert Settings.load(path=tmp_path / "s.json").image == "gpu4pyscf:from-env"


def test_unknown_backend_is_refused(tmp_path):
    settings = Settings(path=tmp_path / "s.json")
    with pytest.raises(ValueError):
        settings.backend = "podman"


def test_save_creates_the_config_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "settings.json"
    Settings(path=path).save()
    assert path.is_file()
