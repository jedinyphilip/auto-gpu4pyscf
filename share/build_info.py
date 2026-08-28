"""Record what was built, so the menu can report it without starting anything.

Used by both backends: inside the builder stage (result copied into the image
and read back with `docker run --entrypoint cat`), and by build_env.sh for a
native install. SRC_DIR and BUILD_INFO_OUT say where to look and where to write.
"""
import json
import os
import subprocess
import sys
from importlib import metadata


def sh(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def version(*packages):
    """First of these distributions that is installed.

    The cuda 12 and cuda 13 stacks use different distribution names, and an env
    built against a pip-provided toolkit is a cuda 13 one.
    """
    for pkg in packages:
        try:
            return metadata.version(pkg)
        except Exception:
            continue
    return ""


src = os.environ.get("SRC_DIR", "/opt/gpu4pyscf")
out = os.environ.get("BUILD_INFO_OUT", "/opt/build-info.json")


def cuda_version():
    """CUDA_VERSION is set in the nvidia images; ask nvcc when it is not."""
    if os.environ.get("CUDA_VERSION"):
        return os.environ["CUDA_VERSION"]
    text = sh("nvcc", "--version")
    for token in text.replace(",", " ").split():
        if token.startswith("V") and token[1:2].isdigit():
            return token[1:]
    return ""


archs_file = os.path.join(sys.prefix, "share", "gpu4pyscf", "archs.txt")
archs = []
if os.path.exists(archs_file):
    with open(archs_file) as handle:
        archs = handle.read().split()

info = {
    "gpu4pyscf": version("gpu4pyscf-cuda12x", "gpu4pyscf-cuda13x", "gpu4pyscf"),
    "pyscf": version("pyscf"),
    "cupy": version("cupy-cuda12x", "cupy-cuda13x", "cupy"),
    "cutensor": version("cutensor-cu12", "cutensor-cu13", "cutensor"),
    "git_sha": sh("git", "-C", src, "rev-parse", "HEAD"),
    "git_short": sh("git", "-C", src, "rev-parse", "--short", "HEAD"),
    "git_date": sh("git", "-C", src, "log", "-1", "--format=%cI"),
    "git_subject": sh("git", "-C", src, "log", "-1", "--format=%s"),
    "cuda_arch": os.environ.get("CUDA_ARCH", ""),
    "cuda_version": cuda_version(),
    "archs": archs,
}
with open(out, "w") as fh:
    json.dump(info, fh, indent=2)
print(json.dumps(info, indent=2))
