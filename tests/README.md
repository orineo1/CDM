# tests

`bash tests/run_all_tests.sh` (or `pytest tests/`): CPU-only, seconds, tiny untrained models, no downloads. The
same suite runs on GitHub Actions on every push. A green run shows that the code runs and is deterministic, not
that it reproduces the paper's numbers. Those numbers are in each experiment's `results/` (synthetic: `reference/`).

| file | checks |
|---|---|
| `test_smoke.py` | the package imports and a tiny run completes |
| `test_determinism.py` | the same seed gives the same result |
| `test_population.py` | particle populations and refinement (structure only) |
| `test_guards.py` | no module redefines a name it imports |
| `test_peptide_nbead.py` | the peptide simulator's N-bead generalization (GB-FESO parts skipped without their code) |
| `test_sd.py` | the Stable Diffusion experiment's CPU parts (some skipped without its extra dependencies) |
