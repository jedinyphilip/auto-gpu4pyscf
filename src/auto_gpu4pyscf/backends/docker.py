"""Run gpu4pyscf from an image built by scripts/build.sh or build.ps1."""
import json
import os
import sys

from .. import paths
from ..ui import spinner
from .base import Backend, run

BUILDER = "gpu4pyscf-build"
PROBE = "nvidia/cuda:12.8.2-base-ubuntu24.04"
GPU_CANDIDATES = [
    ["--gpus", "all"],
    ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"],
    ["--device", "nvidia.com/gpu=all"],
]
WINDOWS = os.name == "nt"


def docker(*args, **kwargs):
    return run(["docker", *args], **kwargs)


def available():
    from .base import which

    if which("docker") is None:
        return False
    return docker("version", "--format", "{{.Server.Version}}").returncode == 0


class DockerBackend(Backend):
    name = "docker"
    progress_kind = "docker"

    def __init__(self, settings):
        super().__init__(settings)
        self._gpu_args = None
        self._image_env = None

    @property
    def image(self):
        return self.settings.image

    def location(self):
        return self.image

    def _inspect(self):
        result = docker("image", "inspect", self.image, "--format", "{{.Created}}\t{{.Size}}")
        if result.returncode != 0:
            return None
        created, size = result.stdout.strip().split("\t")
        return created, int(size)

    def installed(self):
        return self._inspect() is not None

    def built_at(self):
        found = self._inspect()
        return found[0] if found else ""

    def size(self):
        found = self._inspect()
        return found[1] if found else 0

    def _read_build_info(self):
        # --entrypoint cat: the NVIDIA entrypoint prints a banner that would
        # land in the middle of the JSON.
        with spinner("reading build info out of the image"):
            result = docker("run", "--rm", "--entrypoint", "cat", self.image,
                            "/opt/build-info.json")
        if result.returncode != 0:
            return {}
        try:
            return json.loads(result.stdout)
        except ValueError:
            return {}

    def image_env(self, key):
        """Read one variable from the image config.

        No container needed, and it works on images built before
        build-info.json existed.
        """
        if self._image_env is None:
            self._image_env = {}
            result = docker("image", "inspect", self.image, "--format", "{{json .Config.Env}}")
            if result.returncode == 0:
                try:
                    for item in json.loads(result.stdout) or []:
                        name, _, value = item.partition("=")
                        self._image_env[name] = value
                except ValueError:
                    pass
        return self._image_env.get(key, "")

    def gpu_args(self):
        """Return the flags this host needs to attach a GPU, by probing."""
        if self._gpu_args is None:
            self._gpu_args = False
            with spinner("checking how this host attaches a GPU"):
                for candidate in GPU_CANDIDATES:
                    if docker("run", "--rm", *candidate, self.image, "true").returncode == 0:
                        self._gpu_args = candidate
                        break
        return self._gpu_args or None

    def snap_docker(self):
        return "/snap/" in docker("info", "--format", "{{.DockerRootDir}}").stdout

    def data_root(self):
        return docker("info", "--format", "{{.DockerRootDir}}").stdout.strip()

    def build_command(self, ref=None, keep_cache=False):
        env = dict(os.environ)
        if ref:
            env["GPU4PYSCF_REF"] = ref
        env["IMAGE"] = self.image
        if WINDOWS:
            from .base import which

            shell = which("pwsh") or which("powershell")
            if shell is None:
                return None, "PowerShell not found"
            cmd = [
                shell, "-NoProfile", "-ExecutionPolicy", "Bypass",
                "-File", str(paths.scripts() / "build.ps1"),
                "-Image", self.image,
            ]
            if keep_cache:
                cmd.append("-KeepCache")
            if ref:
                cmd += ["-Ref", ref]
            return cmd, env
        if keep_cache:
            env["KEEP_CACHE"] = "1"
        return [str(paths.scripts() / "build.sh")], env

    def run_command(self, script, args):
        gpu = self.gpu_args()
        if gpu is None:
            return None, ("no working way to attach a GPU to a container on this host "
                          "(tried --gpus, --runtime=nvidia and CDI)"), None
        workdir = os.path.dirname(os.path.abspath(script))
        cmd = ["docker", "run", "--rm", "-i"]
        if sys.stdin.isatty() and sys.stdout.isatty():
            cmd.append("-t")
        cmd += gpu
        cmd += [
            "-v", f"{workdir}:/work", "-w", "/work",
            "-e", "HOME=/tmp",
            "-e", "CUPY_CACHE_DIR=/tmp/.cupy",
            "-e", "PYSCF_TMPDIR=/tmp",
        ]
        if not WINDOWS:
            cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]
        # --entrypoint python3 skips ten lines of NVIDIA banner on every run.
        # The smoke test still goes through the entrypoint for its driver check.
        cmd += ["--entrypoint", "python3", self.image,
                f"/work/{os.path.basename(script)}"] + list(args)
        return cmd, dict(os.environ), workdir

    def uninstall_targets(self):
        targets = []
        found = self._inspect()
        if found:
            targets.append((f"image {self.image}", found[1],
                            lambda: docker("image", "rm", self.image).returncode == 0))
        if BUILDER in docker("buildx", "ls").stdout:
            targets.append((f"leftover builder {BUILDER}", None,
                            lambda: docker("buildx", "rm", BUILDER).returncode == 0))
        if docker("image", "inspect", PROBE).returncode == 0:
            targets.append((f"probe image {PROBE}", 90e6,
                            lambda: docker("image", "rm", PROBE).returncode == 0))
        return targets

    def preflight(self):
        return [] if available() else ["docker is not available"]

    def forget(self):
        super().forget()
        self._gpu_args = None
        self._image_env = None
