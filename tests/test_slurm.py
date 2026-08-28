"""Slurm parsing, GPU mapping and job scripts.

These run from recorded sinfo and squeue output, so they cover the parsing and
the script shape, not that a scheduler accepts the result.
"""
import pytest

from auto_gpu4pyscf import slurm

SINFO = """\
cpu|(null)|64
gpu*|gpu:a100:4|12
gpu|gpu:a100:4|3
gpu-h100|gpu:nvidia_h100_80gb_hbm3:8|6
gpu-mixed|gpu:v100:2,gpu:t4:4|4
amd|gpu:mi250:4|2
"""

SQUEUE = """\
1234567|gpu4pyscf|RUNNING|12:31|node042
1234568|smoke_test|PENDING|0:00|(Resources)
"""


@pytest.mark.parametrize(
    ("name", "capability"),
    [
        ("a100", 80),
        ("nvidia_a100-sxm4-40gb", 80),
        ("A100_80GB", 80),
        ("a10", 86),  # must not be swallowed by the a100 rule
        ("a40", 86),
        ("nvidia_h100_80gb_hbm3", 90),
        ("h200", 90),
        ("gh200", 90),
        ("l40s", 89),
        ("l4", 89),
        ("v100", 70),
        ("tesla_t4", 75),
        ("NVIDIA GeForce RTX 5070 Ti", 120),
        ("p100", 60),
    ],
)
def test_capability_for_known_cards(name, capability):
    assert slurm.capability_for(name) == capability


def test_unknown_and_amd_cards():
    assert slurm.capability_for("some_future_card") is None
    assert slurm.capability_for("mi300x") is None
    assert slurm.is_amd("mi300x") is True
    assert slurm.is_amd("a100") is False


def test_arch_flag():
    assert slurm.arch_flag(80) == "80-real"


@pytest.mark.parametrize(
    ("gres", "expected"),
    [
        ("gpu:a100:4", [("a100", 4)]),
        ("gpu:a100:4(S:0-1)", [("a100", 4)]),
        ("gpu:4", [("", 4)]),
        ("(null)", []),
        ("", []),
        ("gpu:v100:2,gpu:t4:4", [("v100", 2), ("t4", 4)]),
        ("mps:100", []),
    ],
)
def test_parse_gres(gres, expected):
    assert slurm.parse_gres(gres) == expected


def test_parse_sinfo_merges_partition_lines():
    partitions = {p.name: p for p in slurm.parse_sinfo(SINFO)}
    assert set(partitions) == {"cpu", "gpu", "gpu-h100", "gpu-mixed", "amd"}
    assert partitions["gpu"].default is True
    assert partitions["gpu"].nodes == 15, "one line per node state, summed"
    assert partitions["cpu"].gpus == []
    assert partitions["gpu"].capability == 80
    assert partitions["gpu-h100"].capability == 90


def test_partition_describe():
    partitions = {p.name: p for p in slurm.parse_sinfo(SINFO)}
    assert "sm_80" in partitions["gpu"].describe()
    assert partitions["cpu"].describe() == "no gpus"
    assert "AMD" in partitions["amd"].describe()


def test_parse_squeue():
    jobs = slurm.parse_squeue(SQUEUE)
    assert [job.job_id for job in jobs] == ["1234567", "1234568"]
    assert jobs[0].running is True
    assert jobs[1].running is False
    assert jobs[1].reason == "(Resources)"


def test_options_flags_omit_empty_fields():
    options = slurm.Options(partition="gpu", gres="gpu:a100:1", cpus=8, time="02:00:00")
    flags = options.flags()
    assert "--partition=gpu" in flags
    assert "--gres=gpu:a100:1" in flags
    assert "--cpus-per-task=8" in flags
    assert "--time=02:00:00" in flags
    assert not any(flag.startswith("--account") for flag in flags)
    assert not any(flag.startswith("--mem") for flag in flags)


def test_options_extra_flags_are_passed_through():
    options = slurm.Options(extra=("--qos=high", "--nodelist=node042"))
    assert "--qos=high" in options.flags()


def test_with_gpus_keeps_the_model():
    assert slurm.Options(gres="gpu:a100:1").with_gpus(4).gres == "gpu:a100:4"
    assert slurm.Options(gres="gpu:1").with_gpus(2).gres == "gpu:2"


def test_options_round_trip_through_settings():
    original = slurm.Options(
        partition="gpu", gres="gpu:a100:2", cpus=16, time="04:00:00",
        account="chem", modules=("cuda/12.8", "gcc/13"),
    )
    restored = slurm.Options.from_dict(original.to_dict())
    assert restored == original


def test_from_dict_ignores_unknown_keys():
    options = slurm.Options.from_dict({"partition": "gpu", "nonsense": 1, "cpus": "4"})
    assert options.partition == "gpu"
    assert options.cpus == 4


def test_batch_script_puts_directives_before_any_command():
    """sbatch stops reading #SBATCH lines at the first command."""
    options = slurm.Options(partition="gpu", gres="gpu:a100:1", modules=("cuda/12.8",))
    script = slurm.batch_script(options, "python calc.py", workdir="/scratch/me")
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash"
    directives = [i for i, line in enumerate(lines) if line.startswith("#SBATCH")]
    commands = [
        i for i, line in enumerate(lines)
        if line and not line.startswith("#") and i > 0
    ]
    assert max(directives) < min(commands)
    assert "#SBATCH --chdir=/scratch/me" in lines
    assert "#SBATCH --output=slurm-%j.out" in lines
    assert "module load cuda/12.8" in script
    assert script.rstrip().endswith("python calc.py")


def test_batch_script_without_modules_has_no_module_block():
    script = slurm.batch_script(slurm.Options(), "true")
    assert "module load" not in script


def test_srun_command_wraps_the_argv():
    command = slurm.srun_command(slurm.Options(partition="gpu"), ["./scripts/build_env.sh"])
    assert command[0] == "srun"
    assert command[-1] == "./scripts/build_env.sh"
    assert "--partition=gpu" in command


def test_parse_probe_pairs_names_with_capabilities():
    text = "NVIDIA A100-SXM4-40GB, 8.0\nNVIDIA A100-SXM4-40GB, 8.0\n"
    assert slurm.parse_probe(text) == [("NVIDIA A100-SXM4-40GB", 80)]


def test_parse_probe_ignores_noise():
    assert slurm.parse_probe("srun: job 1 queued\n\n") == []


def test_probe_asks_for_one_gpu_and_a_short_slot(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command

        class Result:
            returncode = 0
            stdout = "NVIDIA A100-SXM4-40GB, 8.0\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(slurm, "_run", fake_run)
    pairs, error = slurm.probe_gpus(slurm.Options(partition="gpu"))
    assert pairs == [("NVIDIA A100-SXM4-40GB", 80)]
    assert error == ""
    assert "--time=00:05:00" in captured["command"]
    assert "--cpus-per-task=1" in captured["command"]
    assert "nvidia-smi" in captured["command"]
    assert "--query-gpu=name,compute_cap" in captured["command"]
