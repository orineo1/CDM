# four_targets

MLGD-F against four distributional targets, with the Average-prompt and SDEdit-Best baselines, and the saliency
analysis of where MLGD-F edits the scribble. Settings are in `config/<scenario>.env` (T = 250, t_start = 125,
n_cond = 100 (age: 120), Uniform selection with n_sel/n_cond = 0.5, ControlNet scale 0.5) and `config/models.env`
(the pinned model revisions).

## Run

```bash
R=experiments/stable_diffusion/four_targets
bash $R/run.sh check                 # CPU, seconds: imports and the MMD check
bash $R/run.sh smoke balanced        # GPU: 6 guided steps, checks the pipeline end to end
bash $R/run.sh full balanced         # GPU: the full configuration
bash $R/run.sh baselines balanced    # the baseline scribbles (about as long as the run)
bash $R/run.sh eval balanced         # the N = 2,000 evaluation (about 50 min): MMD, SWD, p(male)
bash $R/run.sh saliency balanced     # where MLGD-F edits the scribble (gender scenarios only; see below)
```

Scenarios: `balanced`, `skewed`, `gender_interpolation`, `age_interpolation`, or `all`. Any `run_mlgd_f` flag
after the scenario overrides the config (`bash $R/run.sh help` lists them). `baselines`, `eval` and `saliency`
take `--run_dir <run>`, by default the newest finished full run. `DRY_RUN=1` prints the command. On SLURM:
`sbatch --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00 --wrap "bash $R/run.sh full balanced"`
(with `all`, `--array=0-3` runs one scenario per task).

## The paper's runs

All with `--base_zeta 7.5`. `results/` holds these four runs, and `results/eval_table.md` their N = 2,000 tables.

| scenario | `results/` | flags | SDEdit budget (`--lgd_cm_minutes`) |
|---|---|---|---|
| balanced | `full_BalancedTarget_seed4` | `--seed 4` | 150 |
| skewed | `full_SkewedTarget_seed6` | `--seed 6` | 150 |
| gender_interpolation | `full_GenderInterpolation_seed8` | `--seed 8` | 150 |
| age_interpolation | `full_AgeInterpolation_seed2` | `--seed 2 --age_source min` | 178 |

```bash
bash $R/run.sh full balanced --seed 4 --base_zeta 7.5
bash $R/run.sh baselines balanced --run_dir <run> --lgd_cm_minutes 150
bash $R/run.sh eval balanced --run_dir <run>
bash $R/run.sh saliency balanced --run_dir <run>
```

The SDEdit budget is the MLGD-F run time the paper reports; it is passed explicitly because a run with
deterministic kernels takes longer (see Run time). Each run's exact commands are in its `run_info.txt`,
`baselines/run_info.txt` and `eval/run_info.txt`.

## Saliency: where MLGD-F edits the scribble

`run.sh saliency` analyzes a finished gender run (the paper: the balanced run):

1. **Scribble change**: the signed difference between the source and the MLGD-F scribble, smoothed with
   σ = 2.5 px; a pixel is changed if |difference| ≥ τ = 0.1 (`--sigma`, `--tau`).
2. **Gender saliency**: |∂s/∂x| on the source scribble, with s = cos(CLIP(I(x)), "a photo of a man") −
   cos(CLIP(I(x)), "a photo of a woman"), backpropagated through ControlNet + SDXL Turbo, the VAE decoder and CLIP,
   averaged over 20 seeds (`--n_seeds`).
3. **Saliency vs change** on the source's line pixels: the binned change, Spearman ρ and Pearson r.
4. **Latent check**: the source latent norm against the size of the latent edit, on the 64×64 VAE grid.

```bash
bash $R/run.sh saliency balanced --run_dir <run>      # writes <run>/saliency/, about 6 min on an L40S
bash $R/run.sh saliency balanced --run_dir $R/results/full_BalancedTarget_seed4 --output_dir <dir>   # the paper's scribbles
```

Committed in `results/full_BalancedTarget_seed4/saliency/`: `scribble_change.png` and `saliency_correlation.png`
(the paper's two figures) and `saliency_results.json` with every number: 91.4% of the pixels unchanged, 4.0% added,
4.6% removed; Spearman ρ = 0.53 over 36,668 line pixels; latent Pearson r = 0.03.

## Run time

`run_mlgd_f` uses deterministic GPU kernels by default, so that a run repeats exactly. This roughly doubles its
run time: the paper's runs took about 150 min (age: 178 min) without them. `--no_deterministic` turns them off
(faster, not bit-repeatable). Every run records its own time, so no `../step_cost` run is needed to see it:
`metrics.json` (`optimization_time_sec`, and `elapsed_s` per step, also in `steps.csv`), `eval/eval_results.json`
(`total_minutes`) and `saliency/saliency_results.json` (`minutes`). The committed runs (NVIDIA L40S, deterministic):

| run | optimization | N = 2,000 evaluation |
|---|---|---|
| balanced, skewed, gender interpolation | 4.7–4.8 h (17,032–17,302 s) | 47–49 min |
| age interpolation | 5.7 h (20,527 s) | 47 min |

## Outputs

A run writes `outputs/<tier>_<Scenario>_seed<S>_<id>/`: `steps.csv`, `metrics.json` (args, every step, final MMD,
run time, model revisions), `run_info.txt`, the scribbles and portraits. The later steps add `baselines/`, `eval/`
and `saliency/`. `../export_results.py` copies what the results are read from into `results/<run>/`.

## Reproducibility

Every random draw is seeded from `--seed`, the GPU kernels are deterministic, and `config/models.env` pins every
model to a hub commit. The same GPU model, library versions and seed give the same numbers and images; on other
hardware only the statistics are comparable.
