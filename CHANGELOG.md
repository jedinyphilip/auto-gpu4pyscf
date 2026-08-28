# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-08-28

### Added
- `examples/`: six of upstream's own examples, copied unmodified, each with a
  shell wrapper that runs it without the menu and honours `BOX_ARGS` to pick a
  backend or submit to slurm.
- Non-interactive commands behind the menu: `auto-gpu4pyscf status [--json]`,
  `build [--ref] [--keep-cache]` and `run <script> [args]`, with `--backend`
  and `--local/--slurm` overrides.
- Menu front end (`./menu.sh`, `.\menu.ps1`, or `auto-gpu4pyscf` once installed)
  with a status panel, Rebuild, Run, Settings and Uninstall.
- Two backends: a docker image, and a native environment built beside the
  repository with a CUDA toolkit taken from pip when the host has none.
- `build.sh` / `build.ps1`: detect the GPU's compute capability on the host and
  compile for exactly that, since a docker build cannot see the GPU itself.
- Automatic cleanup: builds run in a disposable buildx builder that is removed
  on success, so nothing outside the finished image survives.
- Live progress with a self-calibrating ETA, per backend.
- Smoke test that reports whether the compiled architectures match the GPU
  present, rather than only whether the calculation ran.

- The GPU map learns: a probe job records the card's GRES alias and its
  nvidia-smi name against the observed compute capability, and any machine the
  menu runs on contributes its own card. Observations override the built-in
  table and say so when they disagree.
- Enabling the slurm launcher offers to switch to the env backend, since
  docker is not available to users on shared clusters.
- Slurm support: a cluster screen for partition and GPU selection, builds
  through `srun`, scripts submitted with `sbatch`, and `squeue`/`scancel` from
  the menu. The compile target comes from the partition's GRES or a probe job,
  never from the login node. Developed without a cluster: parsing and script
  generation are unit-tested, live behaviour is not.

### Notes
- The docker backend uses upstream's pinned CUDA 12 stack. The env backend uses
  it too when the host has a CUDA 12.8+ toolkit, and otherwise installs CUDA 13
  from pip, because CUDA 12 has no pip-installable `nvcc`.
