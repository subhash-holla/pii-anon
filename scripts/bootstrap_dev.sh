#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev,cli,crypto,engines]

echo "Development environment bootstrap complete."
