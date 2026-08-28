#!/usr/bin/env bash
# Menu front end. Runs the package from the checkout, installed or not.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
for py in python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
        exec "$py" -m auto_gpu4pyscf "$@"
    fi
done
echo "error: python 3 is required to run the menu (neither backend is)." >&2
exit 1
