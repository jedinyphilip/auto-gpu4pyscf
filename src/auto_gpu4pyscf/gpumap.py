"""Map GPU names to compute capabilities, learning as we go.

sinfo reports GRES names and never a compute capability, so the two have to be
bridged. The built-in table is a seed; a GPU seen together with its real
capability, from nvidia-smi or a probe job, is recorded and wins from then on.
Learned entries are exact keys, where the table can only guess by substring.
"""
import json
import re

from . import paths

# Order matters: first hit wins, and "a10" is a substring of "a100".
BUILTIN = [
    ("gb200", 100), ("b200", 100), ("b100", 100),
    ("gb10", 121),
    ("rtxpro6000", 120), ("rtx5090", 120), ("rtx5080", 120),
    ("rtx5070", 120), ("rtx5060", 120),
    ("gh200", 90), ("h200", 90), ("h100", 90),
    ("l40s", 89), ("l40", 89), ("l4", 89),
    ("rtx4090", 89), ("rtx4080", 89), ("6000ada", 89), ("rtxada", 89),
    ("a100", 80), ("a30", 80),
    ("a40", 86), ("a10g", 86), ("a16", 86), ("a10", 86),
    ("rtx3090", 86), ("rtx3080", 86),
    ("a6000", 86), ("a5000", 86), ("a4500", 86), ("a4000", 86),
    ("rtx8000", 75), ("rtx6000", 75), ("quadrortx", 75), ("titanrtx", 75),
    ("rtx2080", 75), ("t4", 75),
    ("v100", 70),
    ("p100", 60), ("p40", 61), ("p4", 61), ("gtx1080", 61), ("titanxp", 61),
    ("k80", 37),
]
AMD = ("mi300", "mi250", "mi210", "mi100", "radeon", "instinct")

BUILT_IN = "built-in"


def squash(name):
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def builtin_capability(name):
    squashed = squash(name)
    for pattern, capability in BUILTIN:
        if pattern in squashed:
            return capability
    return None


def is_amd(name):
    squashed = squash(name)
    return any(marker in squashed for marker in AMD)


class GpuMap:
    """Hold learned name to capability pairs, with the table behind them."""

    def __init__(self, path=None, learned=None):
        self.path = path or paths.state_dir() / "gpu-map.json"
        self.learned = dict(learned or {})

    @classmethod
    def load(cls, path=None):
        path = path or paths.state_dir() / "gpu-map.json"
        try:
            with open(path) as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            data = {}
        entries = data.get("gpus", {}) if isinstance(data, dict) else {}
        clean = {
            key: value
            for key, value in entries.items()
            if isinstance(value, dict) and isinstance(value.get("capability"), int)
        }
        return cls(path=path, learned=clean)

    def save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as handle:
                json.dump({"gpus": self.learned}, handle, indent=2, sort_keys=True)
        except OSError:
            pass

    def record(self, name, capability, source="observed"):
        """Record one observed pair.

        Return the built-in guess it corrects, or None if they agree.
        """
        key = squash(name)
        if not key or not capability:
            return None
        guess = builtin_capability(name)
        self.learned[key] = {
            "name": name,
            "capability": int(capability),
            "source": source,
        }
        return guess if guess is not None and guess != int(capability) else None

    def record_alias(self, alias, name, capability, source="observed"):
        """Record the GRES name and the nvidia-smi name for one card.

        Learning both means a later sinfo lookup on the alias hits an exact key.
        """
        corrections = []
        for candidate in (alias, name):
            if candidate:
                correction = self.record(candidate, capability, source)
                if correction is not None:
                    corrections.append((candidate, correction))
        return corrections

    def lookup(self, name):
        """Return (capability, provenance), learned entries first."""
        entry = self.learned.get(squash(name))
        if entry:
            return entry["capability"], entry.get("source", "learned")
        capability = builtin_capability(name)
        return (capability, BUILT_IN) if capability is not None else (None, "")

    def known(self, name):
        return self.lookup(name)[0] is not None

    def unknown_models(self, gpu_names):
        """Return the names nothing can place, which are the probe candidates."""
        unknown = []
        for name in gpu_names:
            if name and not self.known(name) and not is_amd(name) and name not in unknown:
                unknown.append(name)
        return unknown

    def summary(self):
        return f"{len(self.learned)} learned, {len(BUILTIN)} built-in"


_DEFAULT = None


def default():
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = GpuMap.load()
    return _DEFAULT


def reset():
    """Drop the cached instance."""
    global _DEFAULT
    _DEFAULT = None


def lookup(name):
    return default().lookup(name)


def capability_for(name):
    return lookup(name)[0]


def learn_local(gpus, source="this machine"):
    """Record what nvidia-smi reports on this machine.

    Called on every redraw, so it writes only when something changed.
    """
    gpu_map = default()
    before = dict(gpu_map.learned)
    corrections = []
    for name, capability in gpus:
        digits = squash(capability)
        if digits.isdigit():
            correction = gpu_map.record(name, int(digits), source)
            if correction is not None:
                corrections.append((name, correction))
    if gpu_map.learned != before:
        gpu_map.save()
    return corrections
