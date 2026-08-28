"""Behaviour of the shell wrappers, exercised by running them."""
import json
import os
import subprocess
import sys

import pytest
from conftest import ROOT

RUN = ROOT / "scripts" / "run.sh"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the .sh wrappers need bash")


def run(args, env=None, **kwargs):
    environment = dict(os.environ, **(env or {}))
    return subprocess.run([str(RUN), *args], capture_output=True, text=True,
                          env=environment, **kwargs)


@pytest.fixture
def fake_env(tmp_path):
    """An environment directory with just the env.sh the wrapper sources."""
    env_dir = tmp_path / "env"
    env_dir.mkdir()
    (env_dir / "env.sh").write_text("export AUTO_GPU4PYSCF_MARKER=sourced\n")
    return env_dir


def test_env_backend_sources_the_environment(fake_env):
    result = run(["bash", "-c", "echo $AUTO_GPU4PYSCF_MARKER"],
                 env={"BACKEND": "env", "ENV_DIR": str(fake_env)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "sourced"


def test_env_backend_runs_in_the_current_directory(fake_env, tmp_path):
    result = run(["bash", "-c", "pwd"], env={"BACKEND": "env", "ENV_DIR": str(fake_env)},
                 cwd=tmp_path)
    assert result.stdout.strip() == str(tmp_path)


def test_env_backend_without_an_environment_says_so(tmp_path):
    result = run(["true"], env={"BACKEND": "env", "ENV_DIR": str(tmp_path / "missing")})
    assert result.returncode == 1
    assert "no environment at" in result.stderr
    assert "build_env.sh" in result.stderr


def test_docker_backend_without_an_image_says_so():
    result = run(["true"], env={"BACKEND": "docker", "IMAGE": "gpu4pyscf:definitely-not-built"})
    assert result.returncode == 1
    assert "no image tagged gpu4pyscf:definitely-not-built" in result.stderr
    assert "build.sh" in result.stderr


def test_backend_comes_from_the_settings_file(fake_env, tmp_path):
    """No BACKEND in the environment: the wrapper reads what the menu saved."""
    config = tmp_path / "config" / "auto-gpu4pyscf"
    config.mkdir(parents=True)
    (config / "settings.json").write_text(
        json.dumps({"backend": "env", "env_dir": str(fake_env)})
    )
    result = run(["bash", "-c", "echo $AUTO_GPU4PYSCF_MARKER"],
                 env={"XDG_CONFIG_HOME": str(tmp_path / "config"), "BACKEND": ""})
    assert result.stdout.strip() == "sourced"
