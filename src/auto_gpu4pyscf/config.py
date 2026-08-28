"""Persist the backend choice, image tag and environment location."""
import json
import os

from . import paths

DEFAULTS = {
    "backend": "docker",
    "image": "gpu4pyscf:local",
    "env_dir": "",  # empty means <repo>/env
    # Where work runs, independently of which backend holds gpu4pyscf.
    "launcher": "local",
    "slurm": {},
}
BACKENDS = ("docker", "env")
LAUNCHERS = ("local", "slurm")


class Settings:
    def __init__(self, values=None, path=None):
        self.path = path or paths.settings_file()
        self.values = dict(DEFAULTS)
        if values:
            self.values.update({k: v for k, v in values.items() if k in DEFAULTS})

    @classmethod
    def load(cls, path=None):
        path = path or paths.settings_file()
        loaded = {}
        for candidate in (path, _legacy_path()):
            if candidate and candidate.is_file():
                try:
                    with open(candidate) as handle:
                        loaded = json.load(handle)
                    break
                except (OSError, ValueError):
                    continue
        settings = cls(loaded, path=path)
        if os.environ.get("IMAGE"):
            settings.values["image"] = os.environ["IMAGE"]
        return settings

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as handle:
            json.dump(self.values, handle, indent=2)

    @property
    def backend(self):
        return self.values["backend"]

    @backend.setter
    def backend(self, value):
        if value not in BACKENDS:
            raise ValueError(f"unknown backend {value!r}")
        self.values["backend"] = value

    @property
    def image(self):
        return self.values["image"]

    @image.setter
    def image(self, value):
        self.values["image"] = value

    @property
    def launcher(self):
        return self.values["launcher"]

    @launcher.setter
    def launcher(self, value):
        if value not in LAUNCHERS:
            raise ValueError(f"unknown launcher {value!r}")
        self.values["launcher"] = value

    @property
    def slurm(self):
        """Return the Slurm options as a plain dict, see slurm.Options."""
        return self.values.get("slurm") or {}

    @slurm.setter
    def slurm(self, value):
        self.values["slurm"] = dict(value)

    @property
    def env_dir(self):
        return self.values["env_dir"] or str(paths.default_env_dir())

    @env_dir.setter
    def env_dir(self, value):
        self.values["env_dir"] = str(value)


def _legacy_path():
    try:
        return paths.repo_root() / "settings.json"
    except FileNotFoundError:
        return None
