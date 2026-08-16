#!/usr/bin/env bash
# Cluster entry point for the refactor gate. Run from anywhere.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== Tier 0-2: smoke, equivalence, determinism =="
python -m pytest tests/ -v --tb=short
echo
echo "Fast tests passed. Tier 3 (regression vs published numbers) is separate:"
echo "    python scripts/run_regression.py"
