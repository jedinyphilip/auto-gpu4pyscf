#!/usr/bin/env bash
# Run a command against the built gpu4pyscf, using whichever backend is
# configured: inside the image, or in the environment beside the repo.
#
#   ./run.sh                      interactive shell
#   ./run.sh python3 my_calc.py   run a script from the current directory
#   BACKEND=env ./run.sh ...      override the configured backend
#
# In docker the current directory is mounted at /work; in env the command runs
# where you are, as you, with the environment's python first on PATH.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/.." && pwd)"
IMAGE=${IMAGE:-gpu4pyscf:local}

# Read one key from the menu's settings file, empty if there is none. Falls
# back to the old in-repo location, as the menu does.
setting() {
    python3 - "$1" "${root}" <<'PY' 2>/dev/null || true
import json, os, sys

key, root = sys.argv[1], sys.argv[2]
config = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
for path in (os.path.join(config, "auto-gpu4pyscf", "settings.json"),
             os.path.join(root, "settings.json")):
    try:
        with open(path) as handle:
            print(json.load(handle).get(key) or "")
        break
    except (OSError, ValueError):
        continue
PY
}

backend=${BACKEND:-$(setting backend)}

if [ "${backend:-docker}" = "env" ]; then
    env_dir=${ENV_DIR:-$(setting env_dir)}
    env_dir=${env_dir:-${root}/env}
    if [ ! -f "${env_dir}/env.sh" ]; then
        echo "error: no environment at ${env_dir}" >&2
        echo "       build one with ./scripts/build_env.sh" >&2
        exit 1
    fi
    # shellcheck disable=SC1091
    . "${env_dir}/env.sh"
    exec "${@:-${SHELL:-/bin/bash}}"
fi

# Windows (Git Bash): no path translation, no unix uid to map.
uid_args=(-u "$(id -u):$(id -g)")
host_pwd="${PWD}"
case "${OSTYPE:-}" in
    msys*|cygwin*)
        export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
        host_pwd="$(pwd -W 2>/dev/null || pwd)"
        uid_args=()
        ;;
esac

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "error: no image tagged ${IMAGE}" >&2
    echo "       build one with ./scripts/build.sh" >&2
    exit 1
fi

# --gpus works only where the nvidia hook is on the default runtime; fall back
# to the explicit runtime, then to CDI.
gpu_args=()
for try in "--gpus all" "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all" \
           "--device nvidia.com/gpu=all"; do
    read -ra candidate <<< "${try}"
    if docker run --rm "${candidate[@]}" "${IMAGE}" true >/dev/null 2>&1; then
        gpu_args=("${candidate[@]}"); break
    fi
done
if [ ${#gpu_args[@]} -eq 0 ]; then
    echo "error: no working way to attach a GPU to a container on this host." >&2
    echo "       Tried --gpus, --runtime=nvidia and CDI." >&2
    exit 1
fi

# -t only when there is a terminal, so this works in scripts and pipelines too.
tty_args=(-i)
[ -t 0 ] && tty_args=(-i -t)

exec docker run --rm "${tty_args[@]}" "${gpu_args[@]}" \
    -v "${host_pwd}:/work" -w /work \
    "${uid_args[@]+"${uid_args[@]}"}" \
    -e HOME=/tmp -e CUPY_CACHE_DIR=/tmp/.cupy -e PYSCF_TMPDIR=/tmp \
    "${IMAGE}" "${@:-/bin/bash}"
