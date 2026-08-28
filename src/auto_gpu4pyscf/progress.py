"""Condense build output into one line with elapsed time and an ETA.

Only the wrapper's own messages are echoed; the rest goes to the log. Phase
weights are rewritten after every successful build, per backend, so the
estimate converges on the machine it runs on.
"""
import json
import re
import sys
import time

from . import paths
from .ui import dim, fmt_dur


class Progress:
    PHASES = []
    LABELS = {}
    DEFAULTS = {}
    KIND = ""

    def __init__(self, profile=None):
        self.profile = profile or dict(self.DEFAULTS)
        self.total = sum(self.profile.get(p, self.DEFAULTS[p]) for p in self.PHASES)
        self.start = self.phase_start = time.time()
        self.index = 0
        self.durations = {}
        self.detail = ""
        self.within = None  # 0..1 when the build reports its own percentage
        self.last_draw = 0.0
        self.tty = sys.stdout.isatty()

    def phase_of(self, line):
        return None

    def detail_of(self, line):
        return None

    def echo(self, line):
        return True

    @property
    def phase(self):
        return self.PHASES[self.index]

    def feed(self, line):
        """Consume one output line and return True if it should be printed."""
        phase = self.phase_of(line)
        if phase and self.PHASES.index(phase) > self.index:
            self.durations[self.phase] = time.time() - self.phase_start
            self.index = self.PHASES.index(phase)
            self.phase_start = time.time()
            self.detail = ""  # a step name from the previous phase is noise
            self.within = None
        detail = self.detail_of(line)
        if detail is not None:
            self.detail = detail
        self.draw()
        return self.echo(line)

    def expected(self, phase):
        return self.profile.get(phase, self.DEFAULTS[phase]) or 1

    def fraction(self):
        done = sum(self.expected(p) for p in self.PHASES[: self.index])
        current = self.expected(self.phase)
        within = (
            self.within
            if self.within is not None
            else min(1.0, (time.time() - self.phase_start) / current)
        )
        return min(0.99, (done + within * current) / self.total)

    def render(self, elapsed=None, fraction=None):
        elapsed = time.time() - self.start if elapsed is None else elapsed
        fraction = self.fraction() if fraction is None else fraction
        eta = elapsed * (1 - fraction) / fraction if fraction > 0.02 else self.total
        width = 24
        filled = int(width * fraction)
        return (
            f"  [{'#' * filled}{'.' * (width - filled)}] {fraction * 100:3.0f}%"
            f"  {fmt_dur(elapsed)} elapsed  ~{fmt_dur(eta)} left"
            f"   {self.LABELS[self.phase]}" + (f"  {dim(self.detail)}" if self.detail else "")
        )

    def draw(self, force=False):
        now = time.time()
        if not force and now - self.last_draw < 0.4:
            return
        self.last_draw = now
        line = self.render()
        if self.tty:
            sys.stdout.write("\r\033[K" + line)
            sys.stdout.flush()
        elif force:
            print(line)

    def clear_line(self):
        if self.tty:
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()

    def finish(self):
        self.durations[self.phase] = time.time() - self.phase_start
        self.clear_line()
        return time.time() - self.start


class DockerProgress(Progress):
    KIND = "docker"
    PHASES = ["pull", "deps", "compile", "install", "export"]
    LABELS = {
        "pull": "pulling base images",
        "deps": "installing python dependencies",
        "compile": "compiling CUDA kernels",
        "install": "installing gpu4pyscf",
        "export": "exporting and loading the image",
    }
    # Measured cold on 24 cores and 30 GB.
    DEFAULTS = {"pull": 480, "deps": 200, "compile": 160, "install": 25, "export": 130}
    SIZE = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9}
    LAYER = re.compile(
        r"^#\d+ sha256:(\w{12})\w*\s+([\d.]+)(B|kB|KB|MB|GB) / ([\d.]+)(B|kB|KB|MB|GB)"
    )
    STEP = re.compile(r"^#\d+ \[([a-z]+) +(\d+)/(\d+)\] (.{0,60})")

    def __init__(self, profile=None):
        super().__init__(profile)
        self.layers = {}

    def phase_of(self, line):
        if not line.startswith("#"):
            return None
        if "exporting to docker image format" in line or "importing to docker" in line:
            return "export"
        if "compiling for CUDA_ARCHITECTURES" in line:
            return "compile"
        if "CMAKE_CONFIGURE_ARGS" in line and "pip install" in line:
            return "install"
        step = self.STEP.match(line)
        if step:
            return "pull" if step.group(4).startswith("FROM") else "deps"
        return "pull" if self.LAYER.match(line) else None

    def detail_of(self, line):
        layer = self.LAYER.match(line)
        if layer:
            sha, current, current_unit, total, total_unit = layer.groups()
            self.layers[sha] = (
                float(current) * self.SIZE[current_unit],
                float(total) * self.SIZE[total_unit],
            )
            got = sum(c for c, _ in self.layers.values())
            want = sum(t for _, t in self.layers.values())
            return f"{got / 1e9:.1f} / {want / 1e9:.1f} GB"
        step = self.STEP.match(line)
        if step and self.phase == "deps":
            return step.group(4).split("&&")[0].strip()[:44]
        if "compiling for CUDA_ARCHITECTURES" in line and "${" not in line:
            # The step definition shows ${jobs} unexpanded; the shell's own
            # echo lands a moment later.
            return line.split("compiling for ")[-1].split('"')[0].strip()[:44]
        return None

    def echo(self, line):
        return not line.startswith("#")


class EnvProgress(Progress):
    KIND = "env"
    PHASES = ["source", "deps", "compile", "install", "record"]
    LABELS = {
        "source": "fetching the source",
        "deps": "installing the toolkit and dependencies",
        "compile": "compiling CUDA kernels",
        "install": "installing gpu4pyscf",
        "record": "recording and smoke testing",
    }
    DEFAULTS = {"source": 30, "deps": 300, "compile": 200, "install": 40, "record": 40}
    MAKE = re.compile(r"^\[\s*(\d+)%\]")
    MINE = re.compile(
        r"^(GPUs|CUDA_ARCH|nvcc|environment|gpu4pyscf|toolkit|installing|compiling for|"
        r"built into|running smoke|use it with|note:|GPU |pyscf |cupy |contraction|"
        r"compiled |DF-B3LYP|PASS|FAIL|error:|warning:)"
    )

    def phase_of(self, line):
        if line.startswith("running smoke") or "build-info" in line:
            return "record"
        if "pip install --no-build-isolation" in line or line.startswith("Processing "):
            return "install"
        if line.startswith("compiling for") or self.MAKE.match(line):
            return "compile"
        if line.startswith(("installing", "Collecting", "Downloading")):
            return "deps"
        return None

    def detail_of(self, line):
        make = self.MAKE.match(line)
        if make:
            self.within = int(make.group(1)) / 100.0
            return line.split("]", 1)[-1].strip()[:44]
        if line.startswith("installing "):
            return line[len("installing ") :][:44]
        if line.startswith("Collecting "):
            return line.split()[1][:44]
        return None

    def echo(self, line):
        return bool(self.MINE.match(line))


CLASSES = {cls.KIND: cls for cls in (DockerProgress, EnvProgress)}


def load_profile(kind, path=None):
    """Return phase timings for this backend, defaults where unmeasured."""
    path = path or paths.profile_file()
    try:
        with open(path) as handle:
            saved = json.load(handle).get(kind, {})
    except (OSError, ValueError):
        saved = {}
    cls = CLASSES[kind]
    return {p: float(saved.get(p, cls.DEFAULTS[p])) for p in cls.PHASES}


def save_profile(kind, durations, path=None):
    """Record what this machine took, to calibrate the next estimate."""
    path = path or paths.profile_file()
    try:
        with open(path) as handle:
            everything = json.load(handle)
    except (OSError, ValueError):
        everything = {}
    profile = everything.get(kind, {})
    for phase, seconds in durations.items():
        if seconds > 1:
            profile[phase] = round(seconds, 1)
    everything[kind] = profile
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            json.dump(everything, handle, indent=2)
    except OSError:
        pass
