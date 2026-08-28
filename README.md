# auto-gpu4pyscf

Build and run [gpu4pyscf](https://github.com/pyscf/gpu4pyscf) compiled for your GPU, on Linux or Windows, in one command.

The published wheels are built for `70-real;80;90-real`. On anything newer than
Hopper: Blackwell included, there is no matching SASS, so the driver
JIT-compiles every kernel from Ampere PTX on first use. This builds for your
exact architecture instead, and tells you plainly whether it succeeded.

It also resolves the dependencies for you, which is the other half of the
problem. cupy, cuTENSOR, libxc and the CUDA toolkit are versioned against each
other, and a mismatch does not announce itself: cuTENSOR fails to load, and
gpu4pyscf quietly contracts tensors with cupy instead, a little slower and with
nothing in the log to explain it. The build takes upstream's pinned set when
the toolkit allows it, installs the matching cuda13 stack when it does not, and
pins cuTENSOR to the exact version cupy's wheel was built against, read out of
cupy itself. The smoke test then prints which engine is actually live, and the
panel below shows the versions that ended up together.

```
  gpu4pyscf -- pyscf on the GPU, compiled for this machine
  ------------------------------------------------------------------
  backend       docker  gpu4pyscf:local
  built         2026-08-27 17:19 (2 min ago)   5.0 GB
  gpu4pyscf     1.8.1   551d9bb 2026-08-27
  compiled      sm_120   native for this GPU   cuda 12.8.2
  stack         pyscf 2.14.0   cupy 13.4.1   cutensor 2.2.0
  gpu           NVIDIA GeForce RTX 5070 Ti (sm_120)
  disk          118 GB free
  ------------------------------------------------------------------
  1  Rebuild    check upstream, then build for this machine
  2  Run        browse for a .py script and run it
  3  Settings   docker or a native env, image tag, paths
  4  Cluster    run builds and jobs through slurm
  5  Uninstall  remove what this tool installed
  q  Quit
```

**Contents**, [quick start](#quick-start) -
[examples](#examples) - [layout](#layout) - [backends](#backends) -
[design notes](#design-notes) - [slurm](#running-on-a-slurm-cluster) -
[development](#development)

## Quick start

| | Linux, WSL2, Git Bash | Windows PowerShell |
|---|---|---|
| menu: all of it | `./menu.sh` | `.\menu.ps1` |
| build only | `./scripts/build.sh` | `.\scripts\build.ps1` |
| run a script | `./scripts/run.sh python3 calc.py` | `.\scripts\run.ps1 python3 calc.py` |
| shell in the build | `./scripts/run.sh` | `.\scripts\run.ps1` |
| native build, no docker | `./scripts/build_env.sh` | not supported |

Requirements: docker (or Docker Desktop with the **WSL2** backend), an NVIDIA
driver, and a python3 on the host for the menu. The CUDA toolkit is not needed
-- both backends bring their own.

A first build takes about 15 minutes cold, most of it pulling the 5 GB CUDA
image; the CUDA compile itself is around 3 minutes for one architecture.

## Examples

[examples/](examples/) holds six of upstream's own examples, each with a
wrapper that runs it menu-less against whatever you have built:

```sh
cd examples && ./00-h2o.sh
```

They also show the non-interactive interface the wrappers use --
`auto-gpu4pyscf run script.py`, `status --json`, `build --ref v1.8.1`: which
works for your own scripts too.

## Layout

```
docker/Dockerfile         three stages: builder, devel (keeps nvcc), runtime
examples/                 upstream examples, each with a menu-less runner
scripts/                  build.sh build.ps1 build_env.sh run.sh run.ps1
share/                    files that go into the image or the environment
  build_info.py           records version, commit and compiled architectures
  smoke_test.py           DF-B3LYP/def2-TZVPP water, plus an architecture check
src/auto_gpu4pyscf/        the menu and the scripted commands
  cli.py                  screens, the main loop, status/build/run
  backends/               docker.py, env.py behind one small interface
  progress.py             build-output parsers, ETA, phase profiles
  upstream.py             GitHub and PyPI comparison
  slurm.py cluster.py     partitions, jobs, and what to submit
  gpumap.py               GPU name to architecture, learned as it goes
  config.py paths.py ui.py system.py
tests/                    pytest, with fixtures cut from real build logs
```

Settings live in `~/.config/auto-gpu4pyscf/settings.json`, build logs and the
timing profile in `~/.local/state/auto-gpu4pyscf/`. Nothing generated is written
into the checkout except `env/`, which is where the native backend lives.

## Backends

| | docker (default) | env |
|---|---|---|
| platforms | Linux and Windows | Linux only |
| CUDA toolkit | in the image | the host's, or pip's |
| CUDA version | 12.8 | 12.8+ from the host, else 13.3 from pip |
| lives in | a docker image (5.0 GB) | `./env` (4.2 GB) |
| peak disk | ~25 GB | ~6 GB |

`env` is a plain virtualenv with gpu4pyscf compiled into it, no container, so
scripts run as you, at native paths, with no bind mounts. Switch in Settings.
Use it by hand with `source env/env.sh && python your_script.py`.

`scripts/run.sh` follows whichever backend is configured, so the same command
works either way. `BACKEND=env` or `BACKEND=docker` overrides it for one run.

On a Slurm cluster, switch the launcher on under **Cluster**: builds run
through `srun` and scripts are submitted with `sbatch`, compiled for the GPU in
the partition you pick rather than whatever the login node has (usually
nothing). [Running on a Slurm cluster](#running-on-a-slurm-cluster) covers it,
including what is and is not tested there.

## Design notes

Why this is shaped the way it is. Everything here was learned by hitting it.

### The GPU is invisible during a docker build

`docker build` gives build steps no GPU access; BuildKit has no `--gpus` --
so nothing inside a `RUN` can see the card, and CMake's
`CUDA_ARCHITECTURES=native` cannot work in a Dockerfile.

The wrapper closes that gap on the host: it reads the compute capability from
`nvidia-smi` (falling back to a throwaway `docker run` probe), turns `12.0` into
`120-real`, picks a CUDA base image that can target it, and passes both in as
build args.

| detected | effect |
|---|---|
| compute capability | `CUDA_ARCH`, e.g. `120-real`; several distinct GPUs give `86-real;120-real` |
| highest capability | base image: CUDA 12.8.2, or 12.9.1 for sm_103 / sm_121+ |
| RAM and cores | build parallelism, capped at one job per 4 GB, nvcc on the Rys integral kernels is a memory hog |
| upstream HEAD | `master` is resolved to a SHA, so the layer cache is correct across rebuilds |

The Dockerfile also compiles before `pip install .` rather than letting
`setup.py` do it: `setup.py` hardcodes `-j8`, and an unbounded `-j$(nproc)`
gets OOM-killed on most workstations. The `pip install` that follows re-runs
cmake over the same directory and finds it up to date: 7 seconds against
156 for the compile.

### Disk, and cleaning up after itself

The toolkit is build-time only. The Dockerfile has three stages: `builder`,
`devel` (that, kept), and `runtime`, the default, which starts from a clean
CUDA runtime image and copies in nothing but the finished `/opt/venv`.

| | download | on disk |
|---|---|---|
| `12.8.2-devel`, build only, discarded | 5.08 GB | ~11 GB |
| `12.8.2-runtime`; final base | 2.09 GB | ~4.6 GB |
| final image | | 5.0 GB |

Builds run inside a **disposable buildx builder** (`docker-container` driver),
so the toolkit image, every intermediate layer and all the object files live in
that builder's own container and volume, not in the image store, and not in
the build cache other projects share. On success the wrapper removes it and
reports the reclaim (7.9 GB on the reference machine). Nothing global is
pruned. A *failed* build keeps its cache so a re-run resumes, and says so.

The cost of that isolation: no cache is shared, so each run re-pulls the base
image, and `--load` briefly holds a second copy of the finished image. That is
the difference between a ~30 GB peak and the ~35 GB the pre-build disk check is
sized against (it refuses below 25 GB, `FORCE=1` overrides).

### Attaching the GPU at runtime

`--gpus all` only works where the NVIDIA hook is wired into the default
runtime. Where the toolkit registered just the `nvidia` runtime, docker rejects
it outright:

```
invoking the NVIDIA Container Runtime Hook directly (e.g. specifying the
docker --gpus flag) is not supported. Please use the NVIDIA Container Runtime
```

So the wrappers probe rather than assume: `--gpus all`, then
`--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all`, then CDI
`--device nvidia.com/gpu=all`, and use whichever starts a container.

### Why the pip toolkit means CUDA 13

The `env` backend compiles on the host, so it needs a compiler. If the host has
an `nvcc` that is CUDA 12 and new enough for the GPU (12.8 for sm_120, 12.9 for
sm_103/sm_121), it is used with upstream's pinned cu12 stack.

Otherwise the toolkit comes from pip: and that means CUDA 13, because **CUDA
12 is not pip-installable as a compiler**: `nvidia-cuda-nvcc-cu12` contains
`ptxas` and nothing else, no `nvcc` driver. `cuda-toolkit[nvcc]==13.3.*` does
ship a complete toolchain, so the environment becomes a CUDA 13 one:
`cupy-cuda13x`, `cutensor-cu13`, and the `gpu4pyscf-libxc-cuda13x` wheel that
gpu4pyscf's own `setup.py` asks for when nvcc reports 13. Upstream builds and
publishes cuda13x wheels for `80;90-real;120-real`.

Do not reach for Ubuntu's `nvidia-cuda-toolkit` to avoid this: it is 12.0.140
on 24.04, which predates sm_120 and cannot compile for a Blackwell card at all.

Two details that are easy to get wrong, both handled in `build_env.sh`:

* `nvcc` finds `nvvm`, headers and libraries relative to its own directory
  (`TOP = $(_HERE_)/..` in `nvcc.profile`), but the wheels install as
  `nvidia/<component>/{bin,include,lib}`. The script stitches a single root at
  `env/cuda`, **copying** the executables in: symlinking them makes nvcc
  resolve `TOP` back into the wheel, where `nvvm` and `include` are not.
* cupy 14 no longer preloads cuTENSOR by absolute path, and `cutensor-cu13`
  ships `libcutensor.so.2` rather than the exact filename cupy's config names.
  Without help the import fails and gpu4pyscf contracts tensors with cupy
  instead, silently. `env/env.sh` and the menu both put the cuTENSOR and CUDA
  library directories on `LD_LIBRARY_PATH`; the smoke test prints which engine
  is actually live.

### Non-NVIDIA GPUs

Not possible, and not a Dockerfile problem: gpu4pyscf is 212 `.cu`/`.cuh` files
(~350k lines) built against CUTLASS 3.4, cuTENSOR, cuBLAS and cuSOLVER, with
warp-level intrinsics throughout. Upstream has no HIP, ROCm or SYCL build --
only a handful of comments anticipating a possible future HIP port. CuPy has
ROCm wheels, so the Python layer would survive, but the kernels are the
blocker.

### Estimating build time

The progress bar weights phases by how long they actually take, not by step
count, pulling 5 GB and running `WORKDIR` are not equal thirds. Estimates
start from timings measured on a cold run and are rewritten into
`~/.local/state/auto-gpu4pyscf/build-profile.json` after every successful build,
per backend, so they converge on the machine in front of them.

## Running on a Slurm cluster

Turn it on under **Cluster** in the menu. Builds then run through `srun` and
scripts are submitted with `sbatch`, using a partition and GPU you choose.

### The login node is not the compute node

The whole premise of this tool is compiling for the GPU you have. On a cluster
that rule needs care: the login node usually has no GPU, and when it has one it
is often not the model your jobs land on. Compiling for it would produce
kernels the compute nodes cannot run, or nothing at all.

So the target architecture never comes from the machine you type on. In order
of preference:

1. **A probe job**; `srun --gres=gpu:1 --time=00:05:00 nvidia-smi
   --query-gpu=name,compute_cap`. Authoritative, and it queues like any job.
2. **The GPU map**, what has already been observed, or failing that a
   built-in table of GPU names.

### The GPU map learns

`sinfo` names the card (`gpu:a100:4`) and never says what architecture it is;
`nvidia-smi` knows the architecture but only runs where the card is. A probe
job is the one place the two appear together, so that is where the mapping is
learned rather than guessed:

* The probe records **both names** for the card: the GRES alias `sinfo` uses
  (`a100`) and the full name `nvidia-smi` reports (`NVIDIA A100-SXM4-40GB`) --
  so later lookups from either direction hit an exact key.
* Running the menu **on any machine with a GPU** records that card too, for
  free, from the `nvidia-smi` call the status panel already makes.
* If an observation **contradicts the built-in table**, the observation wins and
  the disagreement is printed, because it means the table is wrong about that
  card and should be fixed.

The map lives in `~/.local/state/auto-gpu4pyscf/gpu-map.json` and grows as it
goes. The Cluster screen shows what it knows (`gpu map  3 learned, 44
built-in`) and names any GPU model `sinfo` reports that nothing can place, so
you know exactly what is worth probing. In the partition list, an architecture
with a `*` was observed; without one it was guessed from the name.

### What runs where

| | |
|---|---|
| Rebuild | wrapped in `srun` with the chosen partition, GPUs, cpus and time |
| Run | written to an `sbatch` script, submitted, job id reported |
| Cluster > my queue | `squeue -u $USER`, with cancel |

Builds go through a job rather than the login node for two reasons: compiling
these kernels wants several cores and a few GB per job, which is exactly what
login nodes are not for; and the smoke test at the end needs a real GPU.

The generated batch script puts every `#SBATCH` directive before the first
command, `sbatch` stops reading them at the first command, so that ordering
is load-bearing, and there is a test for it. `--chdir` is the script's own
directory and output lands there as `slurm-<jobid>.out`.

### Backends on a cluster

Use the **env** backend; on a shared system you almost certainly cannot use
the other one. Docker grants effective root, so clusters do not hand it to
users; `docker` is usually not installed at all, and where it is, you are not
in the group.

The tool assumes that rather than letting you find out the hard way. Turning
the launcher on offers to switch you to the env backend on the spot, and if you
decline, Rebuild and Run both refuse with the reason instead of failing halfway
through a build.

### Modules

Clusters supply compilers and CUDA through environment modules rather than pip.
Set them under Cluster > module loads (`cuda/12.8 gcc/13`), and they are loaded
at the top of every job, before the environment is sourced, so a module's
`LD_LIBRARY_PATH` survives into the run.

With a CUDA 12.8+ module loaded, `build_env.sh` uses the cluster's own toolkit
and upstream's pinned cu12 stack. Without one it falls back to installing CUDA
13 from pip, which works but downloads a few GB into the environment; on a
cluster the module is usually the better answer.

## Development

```sh
make install     # creates .venv and installs the package with dev extras
make test        # pytest -- works on a bare checkout too, no install needed
make lint        # ruff check + format check
make check       # lint, tests, and a Dockerfile check
```

`make install` builds a project virtualenv rather than installing into the
system python, which recent distributions refuse (PEP 668).

CI runs the same checks on every push, plus `shellcheck` on the shell wrappers
and PSScriptAnalyzer on the PowerShell ones: those cannot be exercised on a
Linux development box, so CI is where they get tested.

The tests cover the parts that broke during development and would break
silently again: the build-output parsers (fixtures are trimmed real logs), the
upstream comparison, settings persistence, the environment's library paths, and
the repository layout the Dockerfile and wrappers depend on.

## License

MIT, see [LICENSE](LICENSE). gpu4pyscf itself is Apache-2.0.
