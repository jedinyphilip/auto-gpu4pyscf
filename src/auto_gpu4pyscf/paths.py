"""Locate the checkout, and the per-user config and state directories.

The tool builds from the Dockerfile and scripts in this repository, so it runs
from a checkout. AUTO_GPU4PYSCF_ROOT overrides the search.
"""
import os
from pathlib import Path

_ENV_ROOT = "AUTO_GPU4PYSCF_ROOT"
APP = "auto-gpu4pyscf"


def repo_root():
    override = os.environ.get(_ENV_ROOT)
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "scripts" / "build.sh").is_file():
            return candidate
    raise FileNotFoundError(
        f"cannot find the checkout from {here}; set {_ENV_ROOT} to the "
        "directory containing scripts/ and docker/"
    )


def scripts():
    return repo_root() / "scripts"


def share():
    return repo_root() / "share"


def dockerfile():
    return repo_root() / "docker" / "Dockerfile"


def _xdg(var, default):
    return Path(os.environ.get(var) or Path.home() / default).expanduser()


def config_dir():
    return _xdg("XDG_CONFIG_HOME", ".config") / APP


def state_dir():
    return _xdg("XDG_STATE_HOME", ".local/state") / APP


def settings_file():
    return config_dir() / "settings.json"


def profile_file():
    return state_dir() / "build-profile.json"


def build_log():
    return state_dir() / "last-build.log"


def default_env_dir():
    return repo_root() / "env"
