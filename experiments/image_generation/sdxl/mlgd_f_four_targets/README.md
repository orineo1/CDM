# mlgd_f_four_targets

**Produces:** Table 3 (`tab:sd_results`) and Table 15 (`tab:sd_swd`) — MLGD-F against four
distributional targets (balanced, skewed, gender interpolation, age interpolation),
with the Average-prompt and SDEdit-Best baselines.

| | |
|---|---|
| Entry points | `python -m sdxl.run_mlgd_f`, `python -m sdxl.eval_baselines` |
| Config | `config/<scenario>.env` — T=250, t_start=125, ζ=10.0, uniform selection at n_sel/n_cond = 0.5, ControlNet scale 0.5 |
| Full configuration | the paper's settings; GPU: L40S 48 GB for the optimisation, A100 80 GB for the 2,000-image evaluation |
| Reduced configuration (`smoke`) | 12 guided steps, 6 variations; GPU required, loads the same pipelines |
| Static check (`check`) | CPU, seconds, no model downloads |

## Run

```
bash run.sh check                          # CPU, no GPU, no downloads
bash run.sh smoke  balanced                # GPU; reduced configuration, for a quick run
bash run.sh full   balanced                # GPU; the full, published configuration
```

Scenarios: `balanced`, `skewed`, `gender_interpolation`, `age_interpolation`.
Set `PYTHON=/path/to/python` if the interpreter on `PATH` is not the project one.

The `smoke` configuration reproduces no published number; its output is not a result.

## Requirements

Both the `smoke` and the `full` configuration load SDXL Base, SDXL Turbo,
ControlNet-Scribble and CLIP ViT-L/14 and backpropagate through the whole chain, so they
need a CUDA GPU (the paper used an L40S, 48 GB) and network access for the model
downloads. The SD dependencies are in `sdxl/requirements.txt` (`transformers`,
`diffusers`, `peft`, `controlnet_aux`, `wandb`).

`bash run.sh check` needs none of that. It runs two CPU-only checks in seconds:

- `sdxl/check_imports.py` — every intra-package import in `sdxl/` resolves and all 13
  modules byte-compile;
- `sdxl/check_mmd_equivalence.py` — the SD MMD, which is the **shared**
  `losses.functions.compute_mmd`, reproduces the upstream formula to 1e-6, and gradients
  flow through the generated samples while the target stays detached.

## Reference numbers

`results/table3_sd_results.md`, `results/table15_sd_swd.md` — transcribed from the paper.
Provenance in `results/PROVENANCE.md`.

`outputs/` is for local runs and ships empty. It is never a result.
