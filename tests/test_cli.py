"""The non-interactive interface the example wrappers use."""
import pytest

from auto_gpu4pyscf import cli


def test_no_arguments_means_the_menu():
    args = cli.parse_args([])
    assert args.command is None
    assert args.backend is None


def test_run_takes_a_script_and_passes_the_rest_through():
    args = cli.parse_args(["run", "calc.py", "--fast", "-n", "4"])
    assert args.command == "run"
    assert args.script == "calc.py"
    assert args.args == ["--fast", "-n", "4"]


def test_backend_and_launcher_overrides():
    args = cli.parse_args(["--backend", "env", "--slurm", "run", "calc.py"])
    assert args.backend == "env"
    assert args.slurm is True
    assert args.local is False


def test_local_and_slurm_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        cli.parse_args(["--local", "--slurm", "run", "calc.py"])


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit):
        cli.parse_args(["--backend", "podman", "status"])


def test_build_flags():
    args = cli.parse_args(["build", "--ref", "v1.8.1", "--keep-cache"])
    assert (args.ref, args.keep_cache) == ("v1.8.1", True)
    assert cli.parse_args(["build"]).ref is None


def test_status_json_flag():
    assert cli.parse_args(["status", "--json"]).as_json is True
    assert cli.parse_args(["status"]).as_json is False


class FakeBackend:
    name = "env"

    def __init__(self, installed=True):
        self._installed = installed
        self.called_with = None

    def installed(self):
        return self._installed

    def run_command(self, script, args):
        self.called_with = (script, args)
        return ["python", script, *args], {}, "/tmp"


def test_run_refuses_a_missing_script(tmp_path, capsys):
    settings = _local_settings(tmp_path)
    code = cli.command_run(FakeBackend(), settings, str(tmp_path / "nope.py"), [])
    assert code == 2
    assert "no such script" in capsys.readouterr().err


def test_run_refuses_when_nothing_is_built(tmp_path, capsys):
    script = tmp_path / "calc.py"
    script.write_text("")
    settings = _local_settings(tmp_path)
    code = cli.command_run(FakeBackend(installed=False), settings, str(script), [])
    assert code == 1
    assert "build" in capsys.readouterr().err


def test_run_passes_arguments_to_the_backend(tmp_path, monkeypatch):
    script = tmp_path / "calc.py"
    script.write_text("")
    settings = _local_settings(tmp_path)
    backend = FakeBackend()
    seen = {}
    monkeypatch.setattr(cli.subprocess, "call",
                        lambda cmd, **kwargs: seen.update(cmd=cmd, kwargs=kwargs) or 0)
    assert cli.command_run(backend, settings, str(script), ["--fast"]) == 0
    assert backend.called_with == (str(script), ["--fast"])
    assert seen["cmd"][-1] == "--fast"


def _local_settings(tmp_path):
    from auto_gpu4pyscf.config import Settings

    settings = Settings(path=tmp_path / "settings.json")
    settings.launcher = "local"
    return settings


def test_run_and_uninstall_are_greyed_out_before_a_build():
    greyed = cli.unavailable(built=False, slurm_ok=True)
    assert greyed == {"2": "nothing built yet", "5": "nothing to remove"}


def test_cluster_stays_reachable_before_a_build():
    """Rebuild reads the partition chosen there, so gating it would deadlock
    the first build on a cluster."""
    assert "4" not in cli.unavailable(built=False, slurm_ok=True)


def test_cluster_is_greyed_out_without_slurm():
    greyed = cli.unavailable(built=True, slurm_ok=False)
    assert greyed == {"4": "no slurm on this host"}


def test_cluster_is_greyed_out_without_slurm_whatever_is_built():
    assert cli.unavailable(built=False, slurm_ok=False)["4"] == "no slurm on this host"


def test_nothing_is_greyed_out_once_built_on_a_cluster():
    assert cli.unavailable(built=True, slurm_ok=True) == {}
