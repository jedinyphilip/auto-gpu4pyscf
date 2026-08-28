#!/usr/bin/env bash
# Shared launcher for the examples: runs a script against the built gpu4pyscf
# without going through the menu, using whichever backend is configured.
#
#   ./00-h2o.sh                             the configured backend
#   BOX_ARGS="--backend env" ./00-h2o.sh    force the native environment
#   BOX_ARGS="--backend docker" ./00-h2o.sh force the image
#   BOX_ARGS="--slurm" ./00-h2o.sh          submit it as a slurm job
#
# Anything after the script name is passed through to python.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "${here}/.." && pwd)"
export PYTHONPATH="${root}/src${PYTHONPATH:+:${PYTHONPATH}}"

python=$(command -v python3 || command -v python || true)
if [ -z "${python}" ]; then
    echo "error: python 3 is required to launch the example (not to run it)." >&2
    exit 1
fi

# BOX_ARGS is deliberately word-split: it carries flags, not a filename.
# shellcheck disable=SC2086
exec "${python}" -m auto_gpu4pyscf ${BOX_ARGS:-} run "$@"
