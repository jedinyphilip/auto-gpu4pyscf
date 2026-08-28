"""Build output parsers, replayed against real logs.

Both broke in ways only a real log exposed, so the fixtures are trimmed copies
of actual runs.
"""
import pytest
from conftest import fixture_lines

from auto_gpu4pyscf import progress


def replay(bar, lines):
    """Feed a log through a parser and return (phases, echoed lines)."""
    phases, echoed = [], []
    for line in lines:
        if bar.feed(line):
            echoed.append(line)
        if not phases or phases[-1] != bar.phase:
            phases.append(bar.phase)
    return phases, echoed


def test_docker_phases_in_order():
    bar = progress.DockerProgress()
    phases, _ = replay(bar, fixture_lines("docker-build.log"))
    assert phases == ["pull", "deps", "compile", "install", "export"]


def test_docker_echoes_wrapper_messages_only():
    bar = progress.DockerProgress()
    _, echoed = replay(bar, fixture_lines("docker-build.log"))
    assert any(line.startswith("GPUs") for line in echoed)
    assert any(line.startswith("built ") for line in echoed)
    assert not any(line.startswith("#") for line in echoed), "buildkit noise leaked"


def test_docker_sums_layer_progress():
    """Download progress is summed across layers, not shown per layer."""
    bar = progress.DockerProgress()
    for line in fixture_lines("docker-build.log"):
        bar.feed(line)
    got, want = zip(*bar.layers.values())
    assert len(bar.layers) > 1
    assert sum(want) >= sum(got) > 0
    bar.detail = bar.detail_of("#8 sha256:abcdef123456 1.5GB / 5.1GB 100.0s")
    assert bar.detail.endswith("GB")


def test_docker_detail_does_not_leak_across_phases():
    """A step name from the previous phase used to survive into the next."""
    bar = progress.DockerProgress()
    bar.feed("#11 [builder  2/12] RUN apt-get update && apt-get install -y git")
    assert "apt-get" in bar.detail
    bar.feed("#26 exporting to docker image format")
    assert bar.phase == "export"
    assert bar.detail == ""


def test_docker_ignores_unexpanded_step_definition():
    """The step text names ${jobs} before the shell has expanded it."""
    bar = progress.DockerProgress()
    bar.feed('#17 [builder 8/12] RUN echo "compiling for CUDA_ARCHITECTURES=120-real'
             ' with -j${jobs}"; cmake ...')
    assert "${" not in bar.detail
    bar.feed("#17 0.138 compiling for CUDA_ARCHITECTURES=120-real with -j7")
    assert bar.detail == "CUDA_ARCHITECTURES=120-real with -j7"


def test_env_phases_and_make_percentage():
    bar = progress.EnvProgress()
    phases, echoed = replay(bar, fixture_lines("env-build.log"))
    assert phases[0] == "source"
    assert "compile" in phases
    assert phases == sorted(phases, key=progress.EnvProgress.PHASES.index)
    assert any(line.startswith("PASS") for line in echoed)


def test_env_reports_its_own_percentage_while_compiling():
    bar = progress.EnvProgress()
    bar.feed("compiling for CUDA_ARCHITECTURES=120-real with -j7")
    bar.feed("[ 42%] Building CUDA object gint/CMakeFiles/gint.dir/g2e.cu.o")
    assert bar.within == pytest.approx(0.42)
    assert "Building CUDA object" in bar.detail


def test_fraction_never_reaches_one_and_bar_renders():
    bar = progress.DockerProgress()
    for line in fixture_lines("docker-build.log"):
        bar.feed(line)
        assert 0.0 <= bar.fraction() < 1.0
    rendered = bar.render(elapsed=600, fraction=0.5)
    assert "10:00 elapsed" in rendered
    assert "~10:00 left" in rendered


def test_profile_round_trip_is_per_backend(tmp_path):
    path = tmp_path / "profile.json"
    progress.save_profile("docker", {"pull": 123.4, "deps": 0.5}, path=path)
    progress.save_profile("env", {"compile": 222.0}, path=path)
    docker_profile = progress.load_profile("docker", path=path)
    env_profile = progress.load_profile("env", path=path)
    assert docker_profile["pull"] == 123.4
    # Too short to be a real measurement, so the default is kept.
    assert docker_profile["deps"] == progress.DockerProgress.DEFAULTS["deps"]
    assert env_profile["compile"] == 222.0
    assert set(env_profile) == set(progress.EnvProgress.PHASES)


def test_load_profile_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("{not json")
    assert progress.load_profile("docker", path=path) == progress.DockerProgress.DEFAULTS
