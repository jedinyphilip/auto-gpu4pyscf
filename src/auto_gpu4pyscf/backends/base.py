"""Define the interface both backends provide."""
import os
import shutil
import subprocess


def run(cmd, capture=True, timeout=None, **kwargs):
    return subprocess.run(cmd, capture_output=capture, text=True, timeout=timeout, **kwargs)


class Backend:
    name = ""
    progress_kind = ""

    def __init__(self, settings):
        self.settings = settings
        self._info = None

    def location(self):
        raise NotImplementedError

    def installed(self):
        raise NotImplementedError

    def built_at(self):
        """Return the build timestamp in ISO form, or ''."""
        raise NotImplementedError

    def size(self):
        raise NotImplementedError

    def build_info(self, refresh=False):
        """Return what build_info.py recorded, or {}."""
        if self._info is None or refresh:
            self._info = self._read_build_info()
        return self._info

    def _read_build_info(self):
        raise NotImplementedError

    def build_command(self, ref=None, keep_cache=False):
        """Return (argv, env) for a build, or (None, reason)."""
        raise NotImplementedError

    def run_command(self, script, args):
        """Return (argv, env, cwd) for a script, or (None, reason, None)."""
        raise NotImplementedError

    def uninstall_targets(self):
        """Return [(description, size, remove)] for the uninstall screen."""
        raise NotImplementedError

    def preflight(self):
        """Return the reasons a build would fail here, before starting."""
        return []

    def forget(self):
        self._info = None


def which(tool):
    return shutil.which(tool)


def home_env(base=None):
    env = dict(base or os.environ)
    return env
