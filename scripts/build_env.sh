#!/usr/bin/env bash
# Build gpu4pyscf into a plain environment beside this script -- no docker.
#
#   ./build_env.sh                  detect the GPU, build into ./env, smoke test
#   ENV_DIR=/somewhere ./build_env.sh   put the environment elsewhere
#   CUDA_ARCH=120-real ./build_env.sh   force the target
#   GPU4PYSCF_REF=v1.8.1 ./build_env.sh build a tag/branch/SHA
#   SKIP_TEST=1 ./build_env.sh          build only
#
# Linux only: the build wants nvcc, cmake, gfortran and a unix toolchain,
# which is why the portable answer is the container.
set -euo pipefail

cd "$(dirname "$0")/.."   # repo root
HERE=$PWD
ENV_DIR=${ENV_DIR:-"${HERE}/env"}
SRC="${ENV_DIR}/src"
VENV="${ENV_DIR}/venv"
GPU4PYSCF_REF=${GPU4PYSCF_REF:-master}

if [ "$(uname -s)" != "Linux" ]; then
    echo "error: the env backend is Linux only -- use the docker backend." >&2
    exit 1
fi

# --- 1. does this machine have what it takes?
missing=()
for tool in cmake gfortran git; do
    command -v "${tool}" >/dev/null 2>&1 || missing+=("${tool}")
done
python3 -c 'import venv' >/dev/null 2>&1 || missing+=("python3-venv")
if [ ${#missing[@]} -gt 0 ]; then
    echo "error: missing on this host: ${missing[*]}" >&2
    echo "       Debian/Ubuntu: sudo apt install cmake gfortran git python3-venv" >&2
    exit 1
fi
# nvcc is not in that list: it comes from pip below when the host has none.

# --- 2. what hardware, and can the toolkit target it?
if [ -z "${CUDA_ARCH:-}" ]; then
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | tr -d ' .\r' | grep -E '^[0-9]+$' | sort -u || true)
    if [ -z "${caps}" ]; then
        echo "error: no NVIDIA GPU visible to nvidia-smi." >&2
        exit 1
    fi
    CUDA_ARCH=$(echo "${caps}" | sed 's/$/-real/' | paste -sd';')
else
    caps=$(echo "${CUDA_ARCH}" | tr ';' '\n' | grep -oE '^[0-9]+')
fi
max_cap=$(echo "${caps}" | sort -n | tail -1)

# sm_100 and sm_120 need CUDA 12.8, sm_103 and sm_121 need 12.9.
need_minor=0
[ "${max_cap}" -ge 100 ] && need_minor=8
{ [ "${max_cap}" -eq 103 ] || [ "${max_cap}" -ge 121 ]; } && need_minor=9

# Is the host's nvcc, if any, new enough for this GPU?
host_nvcc=""
if command -v nvcc >/dev/null 2>&1; then
    ver=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    if [ "${ver%%.*}" = "12" ] && [ "${ver##*.}" -ge "${need_minor}" ]; then
        host_nvcc=$(command -v nvcc)
    else
        echo "note: host nvcc ${ver} cannot target sm_${max_cap} (or is not CUDA 12);"
        echo "      a matching toolkit will be installed into the environment."
    fi
fi

# --- 3. source
# Resolve against the remote: a local clone answers with both refs/heads and
# refs/remotes, which is two SHAs.
sha=$(git ls-remote https://github.com/pyscf/gpu4pyscf.git "${GPU4PYSCF_REF}" 2>/dev/null \
      | head -1 | cut -f1)
[ -n "${sha}" ] || sha=${GPU4PYSCF_REF}

echo "GPUs            : compute capability ${caps//$'\n'/, }"
echo "CUDA_ARCH       : ${CUDA_ARCH}"
echo "environment     : ${ENV_DIR}"
echo "gpu4pyscf       : ${GPU4PYSCF_REF} -> ${sha}"
echo "toolkit         : ${host_nvcc:-from pip, into the environment}"
echo

if [ -n "${DRY_RUN:-}" ]; then
    echo "(DRY_RUN) would build with the settings above"; exit 0
fi

mkdir -p "${ENV_DIR}"
if [ -d "${SRC}/.git" ]; then
    git -C "${SRC}" fetch --quiet --all --tags
else
    git clone --quiet --filter=blob:none https://github.com/pyscf/gpu4pyscf.git "${SRC}"
fi
git -C "${SRC}" checkout --quiet --detach "${sha}"

# --- 4. environment
[ -d "${VENV}" ] || python3 -m venv "${VENV}"
"${VENV}/bin/pip" install --quiet --upgrade pip setuptools wheel

CUDA_ROOT="${ENV_DIR}/cuda"

# The wheels install as nvidia/<component>/{bin,include,lib}, but cmake wants
# a single root and nvcc resolves nvcc.profile relative to its own directory
# (TOP = $(_HERE_)/..). So executables are copied, the rest linked.
build_cuda_root() {
    local nv part name
    # The CUDA 13 wheels make "nvidia" a namespace package, so __file__ is None.
    nv=$("${VENV}/bin/python" -c 'import nvidia; print(list(nvidia.__path__)[0])')
    rm -rf "${CUDA_ROOT}"
    mkdir -p "${CUDA_ROOT}/bin" "${CUDA_ROOT}/include" "${CUDA_ROOT}/lib"
    for part in "${nv}"/*/; do
        name=$(basename "${part}")
        if [ "${name}" = "nvvm" ]; then
            ln -sfn "${part%/}" "${CUDA_ROOT}/nvvm"
            continue
        fi
        [ -d "${part}bin" ] && cp -a "${part}bin"/. "${CUDA_ROOT}/bin/"
        [ -d "${part}include" ] && ln -sfn "${part}include"/* "${CUDA_ROOT}/include/" 2>/dev/null
        [ -d "${part}lib" ] && ln -sfn "${part}lib"/* "${CUDA_ROOT}/lib/" 2>/dev/null
        [ -d "${part}nvvm" ] && ln -sfn "${part}nvvm" "${CUDA_ROOT}/nvvm"
    done
    ln -sfn lib "${CUDA_ROOT}/lib64"
    # The linker wants unversioned sonames; the wheels ship only libfoo.so.NN.
    local lib plain
    for lib in "${CUDA_ROOT}/lib"/lib*.so.*; do
        [ -e "${lib}" ] || continue
        plain="${lib%%.so.*}.so"
        [ -e "${plain}" ] || ln -sfn "${lib}" "${plain}"
    done
}

if [ -n "${host_nvcc}" ]; then
    CUDA_HOME=$(dirname "$(dirname "${host_nvcc}")")
    "${VENV}/bin/pip" install --quiet -r "${SRC}/requirements.txt"
    # cupy still dlopens cublas and friends.
    build_cuda_root 2>/dev/null || true
else
    # CUDA 12 is not pip-installable as a compiler: nvidia-cuda-nvcc-cu12 ships
    # ptxas and nothing else. CUDA 13 is, so an environment without a host
    # toolkit becomes a CUDA 13 one, which upstream publishes wheels for.
    echo "installing a CUDA 13 toolkit from pip (no usable host nvcc)"
    "${VENV}/bin/pip" install --quiet "cuda-toolkit[nvcc,cccl,cuobjdump,nvrtc]==13.3.*"
    echo "installing the CUDA 13 python stack"
    "${VENV}/bin/pip" install --quiet "cupy-cuda13x[ctk]"
    # cupy preloads cuTENSOR by exact filename, so install the version its
    # wheel was built against rather than the newest.
    want=$("${VENV}/bin/python" -c "
import glob, json, sys
cfg = glob.glob(sys.argv[1] + '/lib/python*/site-packages/cupy/.data/_wheel.json')
print(json.load(open(cfg[0]))['cutensor']['version'] if cfg else '')" "${VENV}")
    if [ -n "${want}" ]; then
        echo "installing cutensor-cu13==${want} (the version cupy expects)"
        "${VENV}/bin/pip" install --quiet "cutensor-cu13==${want}" \
            || "${VENV}/bin/pip" install --quiet cutensor-cu13
    else
        "${VENV}/bin/pip" install --quiet cutensor-cu13
    fi
    grep -vE '^(cupy-|cutensor-|gpu4pyscf-libxc)' "${SRC}/requirements.txt" \
        > "${ENV_DIR}/requirements-hostcuda.txt"
    "${VENV}/bin/pip" install --quiet -r "${ENV_DIR}/requirements-hostcuda.txt"
    build_cuda_root
    CUDA_HOME="${CUDA_ROOT}"
fi

# cupy 14 dropped the shim that dlopened cuTENSOR by path, and the wheel
# ships libcutensor.so.2 rather than the filename cupy's config names, so
# the loader has to be told where it lives.
CUTENSOR_LIB=$("${VENV}/bin/python" -c "
import glob, sys
found = glob.glob(sys.argv[1] + '/lib/python*/site-packages/cutensor/lib')
print(found[0] if found else '')" "${VENV}")

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUTENSOR_LIB:+${CUTENSOR_LIB}:}${CUDA_ROOT}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
nvcc_ver=$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
echo "nvcc            : ${nvcc_ver}  (${CUDA_HOME})"

# Everything the runtime needs, in one place a script can source.
cat > "${ENV_DIR}/env.sh" <<ENVEOF
# source this to use the environment by hand
export CUDA_HOME="${CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${VENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${CUTENSOR_LIB:+${CUTENSOR_LIB}:}${CUDA_ROOT}/lib:${CUDA_HOME}/lib64:\${LD_LIBRARY_PATH:-}"
export CUPY_ACCELERATORS=cub,cutensor
ENVEOF

# --- 5. compile
# Same reasoning as the Dockerfile: setup.py hardcodes -j8, and nvcc on the rys
# kernels wants ~4 GB per job.
plat=$("${VENV}/bin/python" -c 'import sysconfig; print(sysconfig.get_platform())')
mem_gb=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
jobs=$(python3 -c "import os,sys; print(max(1, min(os.cpu_count(), int(sys.argv[1])//4 or 1)))" "${mem_gb}")
echo "compiling for CUDA_ARCHITECTURES=${CUDA_ARCH} with -j${jobs}"
cmake -S "${SRC}/gpu4pyscf/lib" -B "${SRC}/build/temp.${plat}/gpu4pyscf" \
    -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
    -DCUDAToolkit_ROOT="${CUDA_HOME}" \
    -DCUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DBUILD_LIBXC="${BUILD_LIBXC:-OFF}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MESSAGE_LOG_LEVEL=WARNING
cmake --build "${SRC}/build/temp.${plat}/gpu4pyscf" -j"${jobs}"

CMAKE_CONFIGURE_ARGS="-DCMAKE_CUDA_COMPILER=${CUDA_HOME}/bin/nvcc -DCUDAToolkit_ROOT=${CUDA_HOME} -DCUDA_ARCHITECTURES=${CUDA_ARCH} -DBUILD_LIBXC=${BUILD_LIBXC:-OFF} -DCMAKE_BUILD_TYPE=Release" \
    "${VENV}/bin/pip" install --no-build-isolation "${SRC}"

# --- 6. record what was built
lib=$(find "${VENV}" -name libgint.so | head -1)
mkdir -p "${VENV}/share/gpu4pyscf"
"${CUDA_HOME}/bin/cuobjdump" --list-elf "${lib}" | grep -oE 'sm_[0-9]+' | sort -u \
    > "${VENV}/share/gpu4pyscf/archs.txt"
SRC_DIR="${SRC}" BUILD_INFO_OUT="${ENV_DIR}/build-info.json" CUDA_ARCH="${CUDA_ARCH}" \
    "${VENV}/bin/python" "${HERE}/share/build_info.py" >/dev/null

size=$(du -sh "${ENV_DIR}" 2>/dev/null | cut -f1)
echo
echo "built into ${ENV_DIR} (${size})"
if [ -z "${SKIP_TEST:-}" ]; then
    echo "running smoke test..."
    "${VENV}/bin/python" "${HERE}/share/smoke_test.py"
fi
echo
echo "use it with:  source ${ENV_DIR}/env.sh && python your_script.py"
