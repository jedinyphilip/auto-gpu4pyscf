#!/usr/bin/env bash
# Build gpu4pyscf from source for the GPUs in *this* machine. No arguments.
#
#   ./build.sh              detect, build, smoke-test
#   CUDA_ARCH=90-real ./build.sh    force a target (e.g. building for elsewhere)
#   GPU4PYSCF_REF=v1.4.2 ./build.sh build a tag/branch/SHA instead of master
#   SKIP_TEST=1 ./build.sh          build only
#   KEEP_TOOLKIT=1 ./build.sh       keep nvcc + sources in the image (~3x bigger)
#   DRY_RUN=1 ./build.sh            show what it detected, build nothing
#   KEEP_CACHE=1 ./build.sh         keep the build cache for fast rebuilds
#   FORCE=1 ./build.sh              build even if the disk check says no
set -euo pipefail

# Git Bash rewrites anything that looks like a unix path before handing it to
# docker.exe.
case "${OSTYPE:-}" in
    msys*|cygwin*) export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*' ;;
esac

cd "$(dirname "$0")/.."   # repo root: the build context
IMAGE=${IMAGE:-gpu4pyscf:local}
GPU4PYSCF_REF=${GPU4PYSCF_REF:-master}
PROBE_IMAGE=nvidia/cuda:12.8.2-base-ubuntu24.04
BUILDER=${BUILDER:-gpu4pyscf-build}

# Anything this script pulls, this script removes. Note what was already here.
probe_pre_absent=0
docker image inspect "${PROBE_IMAGE}" >/dev/null 2>&1 || probe_pre_absent=1

# --gpus needs the nvidia hook on the default runtime. Where the toolkit
# registered only the 'nvidia' runtime it is rejected, and CDI wants a device.
GPU_ARGS=()
detect_gpu_args() {
    local img=$1 try
    local -a candidate
    for try in "--gpus all" "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all" \
               "--device nvidia.com/gpu=all"; do
        read -ra candidate <<< "${try}"
        if docker run --rm "${candidate[@]}" "${img}" true >/dev/null 2>&1; then
            GPU_ARGS=("${candidate[@]}")
            return 0
        fi
    done
    return 1
}

# --- 1. what hardware is this? ------------------------------------------------
# "12.0" -> "120". Two sources: the host driver, or a throwaway container for
# hosts with no nvidia-smi outside of containers.
detect_caps() {
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null && return
    detect_gpu_args "${PROBE_IMAGE}" || return 1
    docker run --rm "${GPU_ARGS[@]}" "${PROBE_IMAGE}" \
        nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null
}

# Name a non-NVIDIA card rather than leaving the failure unexplained.
other_vendor_hint() {
    local gpus
    gpus=$(lspci 2>/dev/null | grep -iE 'vga|3d controller|display' | sed 's/.*: //' \
           | grep -vi nvidia) || return 0
    echo "       Non-NVIDIA GPUs present:" >&2
    # shellcheck disable=SC2001  # a per-line prefix is what sed is for
    echo "${gpus}" | sed 's/^/         /' >&2
    echo "       gpu4pyscf is CUDA-only, so it cannot use them -- see the" >&2
    echo "       'Non-NVIDIA GPUs' section of the README." >&2
}

if [ -z "${CUDA_ARCH:-}" ]; then
    caps=$(detect_caps | tr -d ' .\r' | grep -E '^[0-9]+$' | sort -u || true)
    if [ -z "${caps}" ]; then
        echo "error: could not read a compute capability from any GPU." >&2
        echo "       Linux: check 'nvidia-smi' and the NVIDIA container toolkit." >&2
        echo "       Windows: Docker Desktop must use the WSL2 backend (GPUs are" >&2
        echo "       not exposed under the Hyper-V backend)." >&2
        echo "       Or set CUDA_ARCH=<e.g. 120-real> to build blind." >&2
        other_vendor_hint
        exit 1
    fi
    CUDA_ARCH=$(echo "${caps}" | sed 's/$/-real/' | paste -sd';')
else
    caps=$(echo "${CUDA_ARCH}" | tr ';' '\n' | grep -oE '^[0-9]+')
fi
max_cap=$(echo "${caps}" | sort -n | tail -1)

# --- 2. a toolkit that can target it ------------------------------------------
# sm_103 (B300) and sm_121 (RTX PRO / GB10) landed in CUDA 12.9; everything else
# gpu4pyscf supports is covered by 12.8.
case "${max_cap}" in
    103|121|12[2-9]|1[3-9][0-9]) cuda_tag=12.9.1 ;;
    *)                           cuda_tag=12.8.2 ;;
esac
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:${cuda_tag}-devel-ubuntu24.04}
RUNTIME_IMAGE=${RUNTIME_IMAGE:-nvidia/cuda:${cuda_tag}-runtime-ubuntu24.04}
# The compiler only produces the .so files, and dropping it saves ~8 GB.
target=runtime
[ -n "${KEEP_TOOLKIT:-}" ] && target=devel

if [ "${max_cap}" -lt 70 ]; then
    echo "warning: compute capability ${max_cap} is below the 7.0 gpu4pyscf" >&2
    echo "         supports; the build may fail or the kernels may not run." >&2
fi

# --- 3. is there room? --------------------------------------------------------
# The toolkit image, the object files and the exported image all exist at the
# peak. Skipped when df cannot see docker's data root, as under Git Bash.
need_gb=${NEED_GB:-25}
data_root=$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)
avail_gb=""
if [ -n "${data_root}" ]; then
    avail_gb=$(df -BG --output=avail "${data_root}" 2>/dev/null | tail -1 | tr -dc '0-9' || true)
fi

# --- 4. pin the source so rebuilds are cache-correct --------------------------
sha=$(git ls-remote https://github.com/pyscf/gpu4pyscf.git "${GPU4PYSCF_REF}" 2>/dev/null | cut -f1)
[ -n "${sha}" ] || sha=${GPU4PYSCF_REF}   # already a SHA, or offline

echo "GPUs            : compute capability ${caps//$'\n'/, }"
echo "CUDA_ARCH       : ${CUDA_ARCH}"
echo "base image      : ${CUDA_IMAGE}"
echo "runtime image   : ${RUNTIME_IMAGE}"
echo "gpu4pyscf       : ${GPU4PYSCF_REF} -> ${sha}"
echo "image target    : ${target}$([ "${target}" = devel ] && echo ' (keeps nvcc)')"
echo "disk free       : ${avail_gb:-unknown} GB, peak need ~${need_gb} GB"
echo "build cache     : $([ -n "${KEEP_CACHE:-}" ] && echo 'kept (KEEP_CACHE)' || echo "disposable builder '${BUILDER}', removed on success")"
echo

if [ -n "${DRY_RUN:-}" ]; then
    echo "(DRY_RUN) would build with the args above"; exit 0
fi

if [ -n "${avail_gb}" ] && [ "${avail_gb}" -lt "${need_gb}" ] && [ -z "${FORCE:-}" ]; then
    echo "error: only ${avail_gb} GB free on ${data_root}; the build peaks near ${need_gb} GB." >&2
    echo "       What docker is holding right now:" >&2
    docker system df >&2 || true
    echo "       Free some of that, or re-run with FORCE=1 (or NEED_GB=<n>)." >&2
    exit 1
fi

# --- 5. build in a disposable builder, then throw it away ---------------------
# The docker-container driver keeps every intermediate layer and the toolkit
# image in its own container and volume, so removing the builder reclaims all
# of it without touching any other project's cache.
isolated=1
[ -n "${KEEP_CACHE:-}" ] && isolated=0
built=0

builder_exists() {
    docker buildx ls 2>/dev/null | awk '{print $1}' | grep -qx "${BUILDER}"
}

cleanup() {
    # Bash takes the exit status from the last command in an EXIT trap, so
    # without this a failed build reports success.
    local status=$?
    if [ "${isolated}" = 1 ] && builder_exists; then
        if [ "${built}" = 1 ]; then
            freed=$(docker buildx du --builder "${BUILDER}" 2>/dev/null \
                    | awk '/^Total:/ {print $2}' || true)
            echo "cleaning up     : removing builder '${BUILDER}'${freed:+, reclaiming ${freed}}"
            docker buildx rm "${BUILDER}" >/dev/null 2>&1 || true
        else
            # A build that died late should not also lose its object files.
            echo >&2
            echo "note: build cache kept in builder '${BUILDER}' so a re-run resumes." >&2
            echo "      Drop it with: docker buildx rm ${BUILDER}" >&2
        fi
    fi
    if [ "${probe_pre_absent}" = 1 ]; then
        docker image rm "${PROBE_IMAGE}" >/dev/null 2>&1 || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

if [ "${isolated}" = 1 ]; then
    docker buildx create --name "${BUILDER}" --driver docker-container >/dev/null 2>&1 || true
    build_cmd=(docker buildx build --builder "${BUILDER}" --load)
else
    build_cmd=(docker build)
fi

"${build_cmd[@]}" -t "${IMAGE}" --target "${target}" \
    --build-arg CUDA_IMAGE="${CUDA_IMAGE}" \
    --build-arg RUNTIME_IMAGE="${RUNTIME_IMAGE}" \
    --build-arg CUDA_ARCH="${CUDA_ARCH}" \
    --build-arg GPU4PYSCF_REF="${sha}" \
    --build-arg BUILD_LIBXC="${BUILD_LIBXC:-OFF}" \
    -f docker/Dockerfile .
built=1

size=$(docker image inspect "${IMAGE}" --format '{{.Size}}' \
       | awk '{printf "%.1f GB", $1/1e9}')

echo
echo "built ${IMAGE} (${size})"
if [ -z "${SKIP_TEST:-}" ]; then
    echo "running smoke test..."
    if detect_gpu_args "${IMAGE}"; then
        echo "gpu flags       : ${GPU_ARGS[*]}"
        docker run --rm "${GPU_ARGS[@]}" "${IMAGE}" smoke-test
    else
        echo "error: no working way to attach a GPU to a container on this host." >&2
        echo "       Tried --gpus, --runtime=nvidia and CDI. Check that the" >&2
        echo "       NVIDIA container toolkit is installed and configured." >&2
        exit 1
    fi
fi
echo
echo "use it with:  ./run.sh python3 your_script.py"
if [ "${isolated}" = 1 ]; then
    echo "everything else this build created is removed on exit;"
    echo "re-run with KEEP_CACHE=1 if you would rather keep the cache warm."
fi
