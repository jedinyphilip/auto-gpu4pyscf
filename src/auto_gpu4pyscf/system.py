"""Query the machine the tool is running on."""
import shutil
import subprocess


def host_gpus():
    """Return [(name, compute capability)] as nvidia-smi reports them."""
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.strip().splitlines():
        name, _, cap = line.partition(",")
        if name.strip():
            gpus.append((name.strip(), cap.strip()))
    return gpus


def sm_name(compute_cap):
    """Convert '12.0' to 'sm_120'."""
    return "sm_" + compute_cap.replace(".", "")


def disk_free_gb(*candidates):
    for path in candidates:
        if not path:
            continue
        try:
            return shutil.disk_usage(path).free // 2**30
        except OSError:
            continue
    return None
