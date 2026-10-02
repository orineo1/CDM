# tests/

Cheapest first. `tests/run_all_tests.sh` runs all of them.

| file | tier | what it establishes |
|---|---|---|
| `test_smoke.py` | 0 | `cdm` imports and one tiny run completes |
| `test_determinism.py` | 2 | the same seed twice gives the same answer |
| `test_population.py` | — | particle/refinement plumbing, structure only (the file says outright that its values are meaningless) |
| `test_refactor_guards.py` | 0.5 | narrow guards that inspect an object directly, one per defect they pin |
| `test_sd.py` | 0.5 | the Stable Diffusion experiment's CPU parts: its package imports and shared MMD (`run.sh check`), the N = 2,000 evaluation's bookkeeping, the interpolation axes, the saliency change / correlation and the step-cost tables, `export_results.py` |

All of the above are CPU-only, seconds, tiny untrained models, no downloads.

## What a green run does and does not mean

A green run says the code imports, runs and is deterministic on tiny untrained
models. It does **not** say the paper's numbers are reproduced: nothing here
loads a trained checkpoint or compares against a published value. The
published numbers live under `experiments/*/results/` (MNIST, Stable Diffusion) and
`experiments/synthetic/reference/`; reproducing them means running an experiment's
full configuration (`experiments/*/run.sh full`, or the commands in
`experiments/synthetic/README.md`).

## Adding tests here

Anything in `tests/` runs in CI on every push, so keep new tests CPU-only,
seconds, deterministic and download-free. `test_refactor_guards.py` is
deliberately narrow: one guard per defect, not a general expansion of coverage.
