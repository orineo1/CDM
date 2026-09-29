#!/usr/bin/env bash
# Runs the CPU test suite. Run from anywhere; needs `pip install -e .` once.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pytest tests/ -v --tb=short
