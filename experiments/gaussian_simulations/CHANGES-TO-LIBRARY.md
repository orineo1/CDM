# Changes made outside `experiments/` by Item 9

`experiments/` invokes `scripts/`; it does not duplicate it. One thing a paper experiment
needed was not expressible with the existing flags, so the library gained it.

## `scripts/find_xstar.py` + `losses/functions.py` — `--loss l2`

**Needed by:** `infeasible_target`. Two of its three arms guide with a pointwise
squared-error objective rather than a distributional one.

**Before:** `--loss {mmd,swd}`. No pointwise option existed in `losses/`, although
`losses/__init__.py` describes the package as "Distributional and pointwise losses".

**Change:**
- `losses/functions.py`: added `pointwise_squared_error(model_samples, target_samples)`
  = `((target - model) ** 2).mean()`, row-wise over two equal-length batches.
- `scripts/find_xstar.py`: `--loss` gains the choice `l2`, dispatched in `make_loss`;
  the usage block documents it.

**Blast radius:** additive. `--loss mmd`, which every other experiment uses, is untouched;
the new branch is reachable only by passing the new value. `python -m pytest tests/ -q`:
24 passed before and after.

`losses/functions.py` was subsequently committed as part of `4e0e603` (Item 12), so the
addition now lives in that commit; `scripts/find_xstar.py` remains uncommitted here.

## Things left alone

- `legacy/Optimization_legacy.py::optimize_LGD` accepts a `loss=` argument and never reads
  it. `legacy/` is a frozen reference for the equivalence tests, so it is reported, not
  changed.
- `fig_smc_correction.py`'s single-`--betas` crash (see `SMOKE-RUNS.md`) is in `scripts/`,
  which another item owns.
- `make_table1.py` has no `_ablation` emitter, so `table8_ablations` runs the four arms but
  does not assemble the fragment.
- `train_models.py` emits no timing, so `legacy_training_times/run.sh` times each
  invocation with the shell.
- `SimulationParameters/` has no 5D file.
