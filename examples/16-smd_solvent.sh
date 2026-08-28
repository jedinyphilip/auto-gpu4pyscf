#!/usr/bin/env bash
# Solvation free energy in water with the SMD model: gas phase, then solvated.
# Upstream: gpu4pyscf/examples/solvent/16-smd_solvent.py
exec "$(dirname "$0")/_run.sh" "$(dirname "$0")/16-smd_solvent.py" "$@"
