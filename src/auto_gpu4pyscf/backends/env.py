"""Run gpu4pyscf from a virtualenv built by scripts/build_env.sh."""
import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .. import paths
from ..ui import spinner
from .base import Backend, which

WINDOWS = os.name == "nt"
# nvcc is absent on purpose: build_env.sh installs one from pip when the host
# has none, or has one too old for the GPU.
REQUIRED_TOOLS = ["cmake", "gfortran", "git"]


class EnvBackend(Backend):
    name = "env"
    progress_kind = "env"

    @property
    def root(self):
        return Path(self.settings.env_dir)

    def python(self):
        return self.root / "venv" / ("Scripts" if WINDOWS else "bin") / (
            "python.exe" if WINDOWS else "python"
        )

    def cuda_root(self):
        return self.root / "cuda"

    def location(self):
        return str(self.root)

    def installed(self):
        return self.python().is_file()

    def built_at(self):
        try:
            stamp = os.path.getmtime(self.root / "build-info.json")
        except OSError:
            return ""
        return datetime.fromtimestamp(stamp, timezone.utc).isoformat()

    def size(self):
        total = 0
        with spinner("measuring the environment"):
            for base, _, files in os.walk(self.root):
                for name in files:
                    try:
                        total += os.lstat(os.path.join(base, name)).st_size
                    except OSError:
                        pass
        return total

    def _read_build_info(self):
        try:
            with open(self.root / "build-info.json") as handle:
                return json.load(handle)
        except (OSError, ValueError):
            return {}

    def missing_tools(self):
        missing = [tool for tool in REQUIRED_TOOLS if which(tool) is None]
        try:
            import venv  # noqa: F401
        except ImportError:
            missing.append("python3-venv")
        return missing

    def preflight(self):
        problems = []
        if WINDOWS:
            problems.append("the env backend is Linux only")
        missing = self.missing_tools()
        if missing:
            problems.append("missing on this host: " + ", ".join(missing))
        return problems

    def build_command(self, ref=None, keep_cache=False):
        env = dict(os.environ)
        env["ENV_DIR"] = str(self.root)
        if ref:
            env["GPU4PYSCF_REF"] = ref
        return [str(paths.scripts() / "build_env.sh")], env

    def runtime_vars(self, base=None):
        """Return what env.sh exports, without needing a shell.

        cupy links against CUDA libraries that live in the environment rather
        than on the system, and an unfindable cuTENSOR downgrades gpu4pyscf to
        cupy contractions silently.
        """
        base = dict(base or os.environ)
        extra = {"CUPY_ACCELERATORS": "cub,cutensor"}
        lib_paths = glob.glob(
            str(self.root / "venv" / "lib" / "python*" / "site-packages" / "cutensor" / "lib")
        )
        cuda_lib = self.cuda_root() / "lib"
        if cuda_lib.is_dir():
            lib_paths.append(str(cuda_lib))
            extra["CUDA_HOME"] = str(self.cuda_root())
            extra["PATH"] = os.pathsep.join([str(self.cuda_root() / "bin"), base.get("PATH", "")])
        if lib_paths:
            if base.get("LD_LIBRARY_PATH"):
                lib_paths.append(base["LD_LIBRARY_PATH"])
            extra["LD_LIBRARY_PATH"] = os.pathsep.join(lib_paths)
        return extra

    def run_command(self, script, args):
        if not self.installed():
            return None, "the environment is not built yet", None
        env = dict(os.environ)
        env.update(self.runtime_vars(env))
        cmd = [str(self.python()), os.path.abspath(script)] + list(args)
        return cmd, env, os.path.dirname(os.path.abspath(script))

    def uninstall_targets(self):
        if not self.root.is_dir():
            return []
        import shutil as _shutil

        def drop():
            _shutil.rmtree(self.root, ignore_errors=True)
            return not self.root.is_dir()

        return [(f"environment {self.root}", self.size(), drop)]
