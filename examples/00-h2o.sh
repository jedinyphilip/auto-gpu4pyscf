#!/usr/bin/env bash
# Energy, gradient, Hessian and thermochemistry for water (B3LYP/def2-TZVPP, density fitted). The place to start.
# Upstream: gpu4pyscf/examples/dft/00-h2o.py
exec "$(dirname "$0")/_run.sh" "$(dirname "$0")/00-h2o.py" "$@"
