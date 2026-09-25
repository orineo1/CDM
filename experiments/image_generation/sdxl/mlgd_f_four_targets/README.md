# mlgd_f_four_targets

**Produces:** Table 3 (`tab:sd_results`) and Table 15 (`tab:sd_swd`) — MLGD-F against four
distributional targets (balanced, skewed, gender interpolation, age interpolation),
with the Average-prompt and SDEdit-Best baselines.

| | |
|---|---|
| Entry points | `python -m sdxl.run_mlgd_f`, `python -m sdxl.eval_baselines` |
| Config | `config/<scenario>.env` — T=250, t_start=125, ζ=10.0, uniform selection at n_sel/n_cond = 0.5, ControlNet scale 0.5 |
| Full tier | GPU: L40S 48 GB for the optimisation; A100 80 GB for the 2,000-image evaluation |
| Smoke tier | GPU required — see below |
| Check tier | CPU, seconds, no downloads |
| **Tier actually run here** | **`check` only** (2026-09-25, CPU, PASS) |

## Run

```
bash run.sh check                          # CPU, no GPU, no downloads
bash run.sh smoke  balanced                # GPU
bash run.sh full   balanced                # GPU, published configuration
```

Scenarios: `balanced`, `skewed`, `gender_interpolation`, `age_interpolation`.
Set `PYTHON=/path/to/python` if the interpreter on `PATH` is not the project one.

## Reference numbers

`results/table3_sd_results.md`, `results/table15_sd_swd.md` — transcribed from the paper.
**Imported, never reproduced here.** Provenance in `results/PROVENANCE.md`.

`outputs/` is for local runs and is empty. It is never a result.

## What could not be run on this machine, and why

This machine has **no CUDA GPU**, so neither the `smoke` nor the `full` tier was
executed. Both load SDXL Base, SDXL Turbo, ControlNet-Scribble and CLIP ViT-L/14 and
backpropagate through the whole chain; the paper's own runs used an L40S (48 GB).
**This experiment needs an L40S; it cannot be smoke-tested here.** No substitute was run
and no number in `results/` came from this machine.

The sandbox also has no network, so the SD dependencies in `sdxl/requirements.txt`
(`transformers`, `diffusers`, `peft`, `controlnet_aux`, `wandb`) could not be installed,
and the two entry points therefore could not be exercised as far as `--help`.

What *was* verified here, by `bash run.sh check`:

- every one of the 16 rewired intra-package imports resolves, and all 13 modules
  byte-compile (`sdxl/check_imports.py`) — this is what the port could plausibly have
  broken, since upstream used bare sibling imports and a `sys.path` shim;
- the SD MMD routed through the **shared** `losses.functions.MMDLoss` reproduces
  upstream's vendored formula to 1e-6, the estimator is the unbiased U-statistic
  (`MMD²(X,X) < 0`), and gradients flow through the generated samples while the target
  stays detached (`sdxl/check_mmd_equivalence.py`).
