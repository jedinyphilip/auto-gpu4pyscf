#!/usr/bin/env bash
# Density-fitted MP2 correlation energy on top of Hartree-Fock (cc-pVDZ).
# Upstream: gpu4pyscf/examples/post_HF/20-dfmp2.py
exec "$(dirname "$0")/_run.sh" "$(dirname "$0")/20-dfmp2.py" "$@"
