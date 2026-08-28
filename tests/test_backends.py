"""Backend behaviour that does not need docker or a GPU."""
import os

from auto_gpu4pyscf.backends.env import EnvBackend
from auto_gpu4pyscf.config import Settings


def make_env(tmp_path, with_cuda=True, with_cutensor=True):
    settings = Settings(path=tmp_path / "settings.json")
    settings.backend = "env"
    settings.env_dir = tmp_path / "env"
    root = tmp_path / "env"
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "venv" / "bin" / "python").write_text("")
    if with_cuda:
        (root / "cuda" / "lib").mkdir(parents=True)
        (root / "cuda" / "bin").mkdir(parents=True)
    if with_cutensor:
        (root / "venv" / "lib" / "python3.12" / "site-packages" / "cutensor" / "lib").mkdir(
            parents=True
        )
    return EnvBackend(settings), root


def test_runtime_vars_point_at_cutensor_and_cuda(tmp_path):
    """cupy 14 no longer preloads cuTENSOR, so the loader has to find it."""
    backend, root = make_env(tmp_path)
    variables = backend.runtime_vars({"PATH": "/usr/bin"})
    library_path = variables["LD_LIBRARY_PATH"].split(os.pathsep)
    assert str(root / "venv/lib/python3.12/site-packages/cutensor/lib") in library_path
    assert str(root / "cuda" / "lib") in library_path
    assert variables["CUDA_HOME"] == str(root / "cuda")
    assert variables["CUPY_ACCELERATORS"] == "cub,cutensor"


def test_runtime_vars_keep_an_existing_library_path(tmp_path):
    backend, _ = make_env(tmp_path)
    variables = backend.runtime_vars({"PATH": "/usr/bin", "LD_LIBRARY_PATH": "/opt/mine/lib"})
    assert variables["LD_LIBRARY_PATH"].split(os.pathsep)[-1] == "/opt/mine/lib"


def test_runtime_vars_without_a_pip_toolkit(tmp_path):
    """A host-toolkit build has no env/cuda, so nothing should be set."""
    backend, _ = make_env(tmp_path, with_cuda=False, with_cutensor=False)
    variables = backend.runtime_vars({"PATH": "/usr/bin"})
    assert "CUDA_HOME" not in variables
    assert "LD_LIBRARY_PATH" not in variables


def test_run_command_uses_the_environment_python(tmp_path):
    backend, root = make_env(tmp_path)
    script = tmp_path / "calc.py"
    script.write_text("")
    cmd, env, cwd = backend.run_command(str(script), ["--fast"])
    assert cmd[0] == str(root / "venv" / "bin" / "python")
    assert cmd[1:] == [str(script), "--fast"]
    assert cwd == str(tmp_path)
    assert env["CUPY_ACCELERATORS"] == "cub,cutensor"


def test_run_command_refuses_before_the_env_is_built(tmp_path):
    settings = Settings(path=tmp_path / "settings.json")
    settings.env_dir = tmp_path / "missing"
    cmd, reason, _ = EnvBackend(settings).run_command("x.py", [])
    assert cmd is None
    assert "not built" in reason


def test_build_command_passes_the_environment_directory(tmp_path):
    backend, root = make_env(tmp_path)
    cmd, env = backend.build_command(ref="v1.8.1")
    assert cmd[0].endswith("scripts/build_env.sh")
    assert env["ENV_DIR"] == str(root)
    assert env["GPU4PYSCF_REF"] == "v1.8.1"
