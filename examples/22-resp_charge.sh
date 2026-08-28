#!/usr/bin/env bash
# ESP and RESP atomic charges from a converged density (6-31G*, cartesian).
# Upstream: gpu4pyscf/examples/properties/22-resp_charge.py
exec "$(dirname "$0")/_run.sh" "$(dirname "$0")/22-resp_charge.py" "$@"
