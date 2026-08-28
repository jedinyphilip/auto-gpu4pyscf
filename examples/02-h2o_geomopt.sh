#!/usr/bin/env bash
# Geometry optimisation of water with geomeTRIC, printing the energy at each step.
# Upstream: gpu4pyscf/examples/geometry_optimization/02-h2o_geomopt.py
exec "$(dirname "$0")/_run.sh" "$(dirname "$0")/02-h2o_geomopt.py" "$@"
