#!/usr/bin/env bash
# Bootstrap the poc-gemma4 environment. Idempotent.
set -euo pipefail
cd "$(cd "$(dirname "$0")" && pwd)"
command -v mise >/dev/null 2>&1 || { echo "ERROR: mise is not installed. brew install mise" >&2; exit 1; }
mise install
[ -d .venv ] || { echo "Creating .venv with mise python 3.12 ..."; mise exec -- python -m venv .venv; }
./.venv/bin/python -m pip install -q --upgrade pip
./.venv/bin/python -m pip install -q -r requirements.txt
mkdir -p reports
echo "poc-gemma4 setup done"
