# GlobalConditional

Code for **Conditional Distribution Matching (CDM)**: given a frozen conditional
generator `P(Y|X)` and a target distribution `G(Y)`, find an input `x*` whose
induced conditional `P(Y|X=x*)` matches `G`.

## Layout

```
guidance/          THE guided-diffusion optimizer (one implementation)
  core.py            optimize_guided() -- the outer loop
  config.py          GuidanceConfig: every knob, as explicit strategy callables
priors/            unconditional models over X: the diffusion prior P(X)
samplers/          conditional samplers f_phi for P(Y|X): consistency, flow matching
losses/            functions.py (MMD/RBF), circular.py, sliced.py
utils/             nn, dist, evaluation, superposition, encoder_decoder
gcio/              run persistence (results/runs.jsonl)
oz/                CLI pipeline: data generation, training, comparison
tests/             smoke, equivalence, determinism, import shims
scripts/           run_all_tests.sh, run_regression.py
```

Top-level `Diffusion.py`, `LossFunctions.py`, `dist_utils.py` etc. are
**compatibility shims** re-exporting the moved modules, so `import Diffusion`
and pickled checkpoints recording the old paths keep working.

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=...        # notebooks read tokens from the environment
export GITHUB_TOKEN=...
```

## Running the optimizer

```python
from guidance import GuidanceConfig, optimize_guided
out = optimize_guided(model_uncond, model_cond, loss_fn, GuidanceConfig(...))
```

The historical entry points (`optimize_LGD`, `optimize_DPS`, `optimize_LGD_CM`,
`optimize_LGD_cfg`) remain in `Optimization.py` as thin wrappers with unchanged
signatures, each configured to reproduce its original behaviour exactly.

Two functions are deliberately **not** unified, because they are different
algorithms: `optimize_LGD_latent` (Adam on `x_t`; it computes `x_t_minus_1` and
never uses it) and `optimize_DFLOW` (LBFGS over flow matching).

## Tests

```bash
bash scripts/run_all_tests.sh          # smoke, equivalence, determinism, shims
python scripts/run_regression.py --legacy --update-golden   # baseline, once
python scripts/run_regression.py                            # check against it
```

`tests/test_equivalence.py` is the gate: it compares each refactored optimizer
against `Optimization_legacy.py`, a frozen copy of the pre-refactor code, on
fixed seeds. **No behavioural fix should be applied until it passes** -- known
issues are documented in `guidance/config.py` docstrings and in
`../CDM-paper/revision/CHANGES.md`, and are queued as a separate track so that
any change in published numbers is attributable to one commit.

## Results

Every run should call `gcio.results.save_run(...)`; tables are built by reading
the store, never by pasting numbers into a notebook cell. The pre-refactor
pasted literals are archived in `results/legacy_pasted_2d.json`.
