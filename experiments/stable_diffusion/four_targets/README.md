# four_targets

MLGD-F against four distributional targets, with the Average-prompt and SDEdit-Best baselines. Settings are in
`config/<scenario>.env` (T = 250, t_start = 125, n_cond = 100 (age: 120), Uniform selection with
n_sel/n_cond = 0.5, ControlNet scale 0.5) and `config/models.env` (the pinned model revisions).

## Run

```bash
R=experiments/stable_diffusion/four_targets
bash $R/run.sh check                 # CPU, seconds: imports and the MMD check
bash $R/run.sh smoke balanced        # GPU: 6 guided steps, checks the pipeline end to end
bash $R/run.sh full balanced         # GPU: the full configuration
bash $R/run.sh baselines balanced    # the baseline scribbles (about as long as the run)
bash $R/run.sh eval balanced         # the N = 2,000 evaluation (about 1 h): MMD, SWD, p(male)
bash $R/run.sh saliency balanced     # scribble edits and gradient saliency (gender scenarios only)
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

The SDEdit budget is the MLGD-F run time of the paper's runs. It is passed explicitly because the deterministic
kernels make a run slower. Each run's exact commands are in its `run_info.txt`, `baselines/run_info.txt` and
`eval/run_info.txt`.

## Outputs

A run writes `outputs/<tier>_<Scenario>_seed<S>_<id>/`: `steps.csv`, `metrics.json` (args, every step, final MMD,
model revisions), `run_info.txt`, the scribbles and portraits. The later steps add `baselines/`, `eval/` and
`saliency/`. `../export_results.py` copies what the results are read from into `results/<run>/`.

## Reproducibility

Every random draw is seeded from `--seed`, the GPU kernels are deterministic (`--no_deterministic` turns this off),
and `config/models.env` pins every model to a hub commit. The same GPU model, library versions and seed give the
same numbers and images; on other hardware only the statistics are comparable.
