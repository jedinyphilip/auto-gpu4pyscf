# gpu4pyscf compiled from source for one specific set of GPUs.
#
# A build step has no GPU access, so build.sh probes the host and passes the
# answer in as CUDA_ARCH. Use it rather than calling docker build directly; the
# context is the repository root, so the paths below are repo-relative.
#
# The nvcc toolkit is ~11 GB and only produces the .so files, so the default
# target is runtime. Use --target devel to keep nvcc in the image.

# CUDA 12.8 is the oldest toolkit that knows sm_120 and still accepts sm_50.
ARG CUDA_IMAGE=nvidia/cuda:12.8.2-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.8.2-runtime-ubuntu24.04

# --- build stage
FROM ${CUDA_IMAGE} AS builder

# CUDA_ARCHITECTURES syntax: "120-real" is SASS for sm_120 only, a bare "80"
# would add compute_80 PTX. The default is what the PyPI wheel ships.
ARG CUDA_ARCH="70-real;80;90-real"
ARG GPU4PYSCF_REF=master
# OFF takes libxc from the wheel, which carries SASS through sm_120.
# ON rebuilds it locally, at 30-60 minutes.
ARG BUILD_LIBXC=OFF

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake make gfortran libopenblas-dev \
        python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv ${VIRTUAL_ENV} && pip install --no-cache-dir -U pip setuptools wheel

# A resolved SHA makes this layer re-run when upstream moves, and hit the
# cache when it has not.
RUN git clone --filter=blob:none https://github.com/pyscf/gpu4pyscf.git /opt/gpu4pyscf
WORKDIR /opt/gpu4pyscf
RUN git checkout --detach ${GPU4PYSCF_REF} && git log -1 --format='building %h %ad %s'

# The set upstream benchmarks against; cupy and cutensor versions are tightly
# coupled. Installing them first leaves only gpu4pyscf to build below.
RUN pip install --no-cache-dir -r requirements.txt

# Compile here rather than in setup.py, which hardcodes -j8. nvcc on the rys
# kernels needs ~4 GB per job, so -j is capped by memory below. The pip install
# that follows re-runs cmake over this directory and finds it up to date.
RUN set -eux; \
    plat="$(python3 -c 'import sysconfig; print(sysconfig.get_platform())')"; \
    mem_gb="$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)"; \
    jobs="$(python3 -c "import os,sys; print(max(1, min(os.cpu_count(), int(sys.argv[1])//4 or 1)))" "${mem_gb}")"; \
    echo "compiling for CUDA_ARCHITECTURES=${CUDA_ARCH} with -j${jobs}"; \
    cmake -S gpu4pyscf/lib -B "build/temp.${plat}/gpu4pyscf" \
        -DCUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DBUILD_LIBXC=${BUILD_LIBXC} \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_MESSAGE_LOG_LEVEL=WARNING; \
    cmake --build "build/temp.${plat}/gpu4pyscf" -j"${jobs}"

RUN CMAKE_CONFIGURE_ARGS="-DCUDA_ARCHITECTURES=${CUDA_ARCH} -DBUILD_LIBXC=${BUILD_LIBXC} -DCMAKE_BUILD_TYPE=Release" \
        pip install --no-cache-dir --no-build-isolation . \
    && rm -rf /opt/gpu4pyscf/build

# cuobjdump lives in the toolkit, so record the architectures now; the runtime
# image reads this file instead.
RUN set -eux; \
    lib="$(find ${VIRTUAL_ENV} -name libgint.so | head -1)"; \
    mkdir -p ${VIRTUAL_ENV}/share/gpu4pyscf; \
    cuobjdump --list-elf "${lib}" | grep -oE 'sm_[0-9]+' | sort -u \
        > ${VIRTUAL_ENV}/share/gpu4pyscf/archs.txt; \
    cat ${VIRTUAL_ENV}/share/gpu4pyscf/archs.txt

# Recorded for the menu to read back later without a GPU.
COPY share/build_info.py /opt/build_info.py
RUN CUDA_ARCH="${CUDA_ARCH}" python3 /opt/build_info.py

# --- fat image, keeps the toolkit
# Only built with --target devel: nvcc, cmake and the checkout stay.
FROM builder AS devel
COPY share/smoke_test.py /opt/smoke_test.py
RUN printf '#!/bin/sh\nexec python3 /opt/smoke_test.py "$@"\n' > /usr/local/bin/smoke-test \
    && chmod +x /usr/local/bin/smoke-test
ENV CUPY_ACCELERATORS=cub,cutensor
WORKDIR /work
CMD ["/bin/bash"]

# --- slim image (default)
FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${CUDA_HOME}/bin:${PATH}" \
    LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# python3 for the venv, libgfortran5 for solvent/mnsol_interface.f90,
# libgomp1 for the OpenMP host code.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 libgfortran5 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Fail here rather than at import time if a stripped-down RUNTIME_IMAGE is
# missing something cupy dlopens.
RUN set -eu; \
    for so in libnvrtc.so libcublas.so libcusolver.so libcufft.so libcurand.so libcusparse.so; do \
        ldconfig -p | grep -q "${so}" || ls /usr/local/cuda/lib64/${so}* >/dev/null 2>&1 \
            || { echo "runtime image is missing ${so}"; exit 1; }; \
    done

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/gpu4pyscf/examples /opt/gpu4pyscf/examples
COPY --from=builder /opt/build-info.json /opt/build-info.json

# A venv is tied to its interpreter, so catch a mismatched RUNTIME_IMAGE here.
# cupy is not imported: libcuda.so.1 is absent during a build.
RUN python3 -V && python3 -c "import pyscf, numpy; print('venv ok, pyscf', pyscf.__version__)"
COPY share/smoke_test.py /opt/smoke_test.py
RUN printf '#!/bin/sh\nexec python3 /opt/smoke_test.py "$@"\n' > /usr/local/bin/smoke-test \
    && chmod +x /usr/local/bin/smoke-test

ENV CUPY_ACCELERATORS=cub,cutensor
WORKDIR /work
CMD ["/bin/bash"]
