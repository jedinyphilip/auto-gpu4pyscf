"""List partitions, select GPUs, and run work as Slurm jobs.

A login node usually has no GPU, or not the one the jobs land on, so the target
architecture comes from the partition GRES or a probe job, never from the node
this runs on.
"""
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, replace

from . import gpumap

JOB_NAME = "gpu4pyscf"

# GPU identity lives in gpumap, which learns from probe jobs.
squash = gpumap.squash
is_amd = gpumap.is_amd


def capability_for(gpu_name):
    """Return the compute capability as an int, or None if unrecognised."""
    return gpumap.capability_for(gpu_name)


def arch_flag(capability):
    """Convert 80 to '80-real', the CUDA_ARCHITECTURES form cmake wants."""
    return f"{capability}-real"


def parse_gres(gres):
    """Parse a GRES string into [(model, count)]."""
    if not gres or gres in ("(null)", "N/A"):
        return []
    found = []
    for entry in gres.split(","):
        entry = re.sub(r"\(.*?\)", "", entry).strip()
        parts = entry.split(":")
        if not parts or parts[0] != "gpu":
            continue
        if len(parts) == 2:  # gpu:4 -- no model recorded
            name, count = "", parts[1]
        else:
            name, count = parts[1], parts[2]
        try:
            found.append((name, int(count)))
        except ValueError:
            found.append((name, 0))
    return found


@dataclass
class Partition:
    name: str
    gres: str = ""
    nodes: int = 0
    default: bool = False

    @property
    def gpus(self):
        return parse_gres(self.gres)

    @property
    def gpu_name(self):
        found = self.gpus
        return found[0][0] if found else ""

    @property
    def gpus_per_node(self):
        found = self.gpus
        return found[0][1] if found else 0

    @property
    def capability(self):
        return self.capability_with_source()[0]

    def capability_with_source(self):
        if not self.gpu_name:
            return None, ""
        return gpumap.lookup(self.gpu_name)

    def describe(self):
        if not self.gpus:
            return "no gpus"
        name = self.gpu_name or "gpu"
        capability, source = self.capability_with_source()
        if capability:
            arch = f"sm_{capability}" + ("" if source == gpumap.BUILT_IN else "*")
        elif is_amd(name):
            arch = "AMD, not usable"
        else:
            arch = "unknown arch"
        return f"{name or 'gpu'} x{self.gpus_per_node}  ({arch})"


SINFO_FORMAT = "%P|%G|%D"


def parse_sinfo(text):
    """Parse the output of sinfo -h -o '%P|%G|%D'."""
    partitions = {}
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 3:
            continue
        name, gres, nodes = fields[0], fields[1], fields[2]
        default = name.endswith("*")
        name = name.rstrip("*")
        try:
            count = int(nodes)
        except ValueError:
            count = 0
        existing = partitions.get(name)
        if existing:
            # sinfo prints one line per node state.
            existing.nodes += count
            if not existing.gres or existing.gres in ("(null)", "N/A"):
                existing.gres = gres
        else:
            partitions[name] = Partition(name=name, gres=gres, nodes=count, default=default)
    return list(partitions.values())


SQUEUE_FORMAT = "%i|%j|%T|%M|%R"


@dataclass
class Job:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str

    @property
    def running(self):
        return self.state.upper() == "RUNNING"


def parse_squeue(text):
    jobs = []
    for line in text.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 5 and fields[0]:
            jobs.append(Job(*[f.strip() for f in fields[:5]]))
    return jobs


@dataclass
class Options:
    """Hold what to ask Slurm for. Empty fields are not passed."""

    partition: str = ""
    gres: str = "gpu:1"
    cpus: int = 8
    time: str = "01:00:00"
    account: str = ""
    memory: str = ""
    modules: tuple = ()
    extra: tuple = ()
    job_name: str = JOB_NAME

    def flags(self):
        flags = []
        if self.partition:
            flags += [f"--partition={self.partition}"]
        if self.gres:
            flags += [f"--gres={self.gres}"]
        if self.cpus:
            flags += [f"--cpus-per-task={self.cpus}"]
        if self.time:
            flags += [f"--time={self.time}"]
        if self.account:
            flags += [f"--account={self.account}"]
        if self.memory:
            flags += [f"--mem={self.memory}"]
        flags += [f"--job-name={self.job_name}"]
        flags += list(self.extra)
        return flags

    def with_gpus(self, count):
        """Return the same options with a different GPU count."""
        parts = self.gres.split(":") if self.gres else ["gpu"]
        model = parts[1] if len(parts) >= 3 else ""
        gres = f"gpu:{model}:{count}" if model else f"gpu:{count}"
        return replace(self, gres=gres)

    @classmethod
    def from_dict(cls, values):
        known = set(cls.__dataclass_fields__)
        clean = {k: v for k, v in (values or {}).items() if k in known}
        for key in ("modules", "extra"):
            if key in clean:
                clean[key] = tuple(clean[key])
        if "cpus" in clean:
            clean["cpus"] = int(clean["cpus"])
        return cls(**clean)

    def to_dict(self):
        return {
            "partition": self.partition,
            "gres": self.gres,
            "cpus": self.cpus,
            "time": self.time,
            "account": self.account,
            "memory": self.memory,
            "modules": list(self.modules),
            "extra": list(self.extra),
        }


def module_preamble(modules):
    if not modules:
        return ""
    lines = ["# module is a shell function, so this only works under a login shell",
             "if command -v module >/dev/null 2>&1; then"]
    lines += [f"    module load {name}" for name in modules]
    lines += ["fi", ""]
    return "\n".join(lines)


def srun_command(options, command):
    """Wrap an argv in srun, for work watched interactively."""
    return ["srun", *options.flags(), "--unbuffered", *command]


def batch_script(options, body, output="slurm-%j.out", workdir=None):
    """Build a complete sbatch script around an already-quoted body."""
    lines = ["#!/bin/bash"]
    for flag in options.flags():
        lines.append(f"#SBATCH {flag}")
    lines.append(f"#SBATCH --output={output}")
    if workdir:
        lines.append(f"#SBATCH --chdir={workdir}")
    lines += ["", "set -euo pipefail", ""]
    preamble = module_preamble(options.modules)
    if preamble:
        lines.append(preamble)
    lines.append(body)
    lines.append("")
    return "\n".join(lines)


def quote(command):
    return " ".join(shlex.quote(part) for part in command)



def _run(command, **kwargs):
    return subprocess.run(command, capture_output=True, text=True, **kwargs)


def available():
    return all(shutil.which(tool) for tool in ("sinfo", "sbatch", "squeue"))


def inside_job():
    return bool(os.environ.get("SLURM_JOB_ID"))


def version():
    if shutil.which("sbatch") is None:
        return ""
    result = _run(["sbatch", "--version"])
    return result.stdout.strip() if result.returncode == 0 else ""


def partitions():
    result = _run(["sinfo", "-h", "-o", SINFO_FORMAT])
    return parse_sinfo(result.stdout) if result.returncode == 0 else []


def queue(user=None):
    user = user or os.environ.get("USER", "")
    result = _run(["squeue", "-h", "-u", user, "-o", SQUEUE_FORMAT])
    return parse_squeue(result.stdout) if result.returncode == 0 else []


def cancel(job_id):
    return _run(["scancel", str(job_id)]).returncode == 0


def submit(script_text, path):
    """Write the script, submit it, and return (job_id, message)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script_text)
    path.chmod(0o755)
    result = _run(["sbatch", str(path)])
    if result.returncode != 0:
        return None, (result.stderr or result.stdout).strip()
    match = re.search(r"(\d+)", result.stdout)
    return (match.group(1) if match else None), result.stdout.strip()


def parse_probe(text):
    """Parse nvidia-smi name,compute_cap output into [(name, capability)]."""
    pairs = []
    for line in text.splitlines():
        if "," not in line:
            continue
        name, _, capability = line.rpartition(",")
        digits = capability.strip().replace(".", "")
        if name.strip() and digits.isdigit():
            pair = (name.strip(), int(digits))
            if pair not in pairs:
                pairs.append(pair)
    return pairs


def probe_gpus(options, timeout=1800):
    """Ask a compute node for its GPU name and capability together.

    Authoritative, and the only place the two appear in the same output, but it
    queues like any job.
    """
    command = srun_command(
        replace(options, time="00:05:00", cpus=1),
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
    )
    try:
        result = _run(command, timeout=timeout)
    except subprocess.TimeoutExpired:
        return [], "the probe job did not start in time"
    if result.returncode != 0:
        return [], (result.stderr or result.stdout).strip()
    pairs = parse_probe(result.stdout)
    if not pairs:
        return [], "the job produced no usable output"
    return pairs, ""
