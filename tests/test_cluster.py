"""Turning work into Slurm jobs: which architecture, and what runs where."""
import pytest

from auto_gpu4pyscf import cluster, slurm
from auto_gpu4pyscf.backends.env import EnvBackend
from auto_gpu4pyscf.config import Settings

PARTITIONS = slurm.parse_sinfo("gpu*|gpu:a100:4|12\ncpu|(null)|64\n")


@pytest.fixture
def settings(tmp_path):
    values = Settings(path=tmp_path / "settings.json")
    values.launcher = "slurm"
    values.env_dir = tmp_path / "env"
    cluster.save_options(values, slurm.Options(partition="gpu", gres="gpu:a100:1"))
    return values


def test_capability_comes_from_the_partition_not_this_machine(settings, monkeypatch):
    """A login node has no GPU, or not the one the jobs land on."""
    monkeypatch.setattr(slurm, "partitions", lambda: PARTITIONS)
    capability, source = cluster.target_capability(settings)
    assert capability == 80
    assert "a100" in source
    assert cluster.build_arch(settings) == "80-real"


def test_a_probe_result_wins_over_the_gres_guess(settings, monkeypatch):
    monkeypatch.setattr(slurm, "partitions", lambda: PARTITIONS)
    settings.values["slurm"]["capability"] = 90
    capability, source = cluster.target_capability(settings)
    assert (capability, source) == (90, "probed")
    assert cluster.build_arch(settings) == "90-real"


def test_unknown_partition_leaves_the_architecture_unset(settings, monkeypatch):
    monkeypatch.setattr(slurm, "partitions", lambda: [])
    assert cluster.target_capability(settings) == (None, "")
    assert cluster.build_arch(settings) == ""


def test_wrap_build_submits_and_pins_the_architecture(settings, monkeypatch):
    monkeypatch.setattr(slurm, "partitions", lambda: PARTITIONS)
    command, env = cluster.wrap_build(["./scripts/build_env.sh"], {"PATH": "/usr/bin"}, settings)
    assert command[0] == "srun"
    assert "--partition=gpu" in command
    assert command[-1] == "./scripts/build_env.sh"
    assert env["CUDA_ARCH"] == "80-real"


def test_wrap_build_is_a_no_op_when_running_locally(settings):
    settings.launcher = "local"
    command, env = cluster.wrap_build(["./x.sh"], {"A": "1"}, settings)
    assert command == ["./x.sh"]
    assert env == {"A": "1"}


def test_run_body_sources_the_environment(settings, tmp_path):
    backend = EnvBackend(settings)
    body = cluster.run_body(backend, str(tmp_path / "calc.py"), ["--fast"], settings)
    assert "source" in body and "env.sh" in body
    assert body.strip().endswith("calc.py --fast")


def test_run_body_quotes_awkward_paths(settings, tmp_path):
    backend = EnvBackend(settings)
    script = tmp_path / "my calc.py"
    body = cluster.run_body(backend, str(script), ["a b"], settings)
    assert "'a b'" in body
    assert "my calc.py" in body


def test_docker_backend_is_refused_on_a_cluster(settings):
    class FakeDocker:
        name = "docker"

    job_id, message = cluster.submit_run(FakeDocker(), "calc.py", [], settings)
    assert job_id is None
    assert "apptainer" in message


def test_status_line_is_silent_when_local(settings):
    settings.launcher = "local"
    assert cluster.status_line(settings) is None
    settings.launcher = "slurm"
    assert "slurm" in cluster.status_line(settings)

