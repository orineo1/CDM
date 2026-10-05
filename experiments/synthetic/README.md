# Synthetic experiments (Gaussian mixtures)

A Gaussian mixture `P(X, Y)` with known parameters defines a target `G(Y) = P(Y | X = x*)`, so every recovered
`x_hat` is scored exactly: TV and L2 between `P(Y | X = x_hat)` and `G`. Methods: MLGD (conditional-diffusion
sampler), MLGD-F (consistency sampler), their particle versions PMLGD and PMLGD-F, and the oracles `exact` (the
analytic conditional as sampler) and `exactloss` (closed-form MMD).

Run everything from the repository root after `pip install -e .`. Outputs go to `runs/`, `results/` and
`checkpoints/` (git-ignored). Every script has `--help`.

| path | contents |
|---|---|
| `pipeline/` | `generate_data.py` (mixture, target), `train_models.py` (prior, conditional samplers), `search_mog.py` (the search), `check_models.py` (networks vs. the exact mixture), `make_table1.py` (the grid end to end), `fig1_toy_example.py`, `fig_smc_correction.py` (toy example), `search_gbfeso.py`, `make_table1_d256.py` (GB-FESO), `plot_nonstar.py`, diagnostics |
| `slurm/` | SLURM jobs (add `--partition=<gpu partition>` to `sbatch`) and `cluster_setup.txt` |
| `configs/mog/` | fixed mixture parameters (`generate_data.py --params-file`) |
| `reference/` | outputs of the runs behind the paper |

Smoke test (CPU, about 15 min; checks that the code runs, reproduces no number):

```bash
python experiments/synthetic/pipeline/make_table1.py --settings 1x1 --n-samples 500 --epochs 30 --batch-size 256 \
    --nunits 16 --nblocks 1 --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
    --guidance-optim adam --step-size 0.05 --skip-check --device cpu --runs-dir runs/smoke --out results/smoke_table1
```

## Gaussian-mixture grid

`make_table1.py` runs, per scenario `(dim X, dim Y)` in {1, 5, 10}², data generation, training of the prior and
the two conditional samplers (MLP 3 × 128, 10,000 batches of 1,024 fresh draws), `check_models.py` and the search,
then writes the tables. Steps whose outputs exist are skipped, so the variants below reuse the networks. On a
cluster, `slurm/slurm_table1.sh` runs one scenario per array task (`TAG`, `RUNS`, `ARGS`, `OUT` from the environment).

```bash
MT="python experiments/synthetic/pipeline/make_table1.py"
T1="--budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device cuda --runs-dir runs/table1v2"
```

The main runs predate the detached RBF bandwidth, now the default; `--extra=--bandwidth-grad` restores it.

| result in the paper | command | reference |
|---|---|---|
| main table (MLGD, MLGD-F), particle variants, appendix table (‖x̂ − x*‖, relative L²) | `$MT $T1 --tag _adam_refine --extra=--bandwidth-grad --out results/table1_adam_refine` | — |
| ablation: no refinement | `$MT $T1 --refine-steps 0 --tag _adam --extra=--bandwidth-grad --out results/table1_adam` | — |
| ablation: annealed resampling | `$MT $T1 --methods PMLGD PMLGD-F --resample-schedule linear --refine-steps 0 --tag _adam_anneal --extra=--bandwidth-grad --out results/table1_adam_anneal` | — |
| ablation: oracle sampler (the exact conditional) | `$MT $T1 --methods MLGD-X PMLGD-X --refine-steps 0 --tag _adam_oracle --extra=--bandwidth-grad --out results/table1_adam_oracle` | — |
| accuracy of the trained networks | written by the main run (`runs/table1v2/<scenario>/check.json`, `results/table1_adam_refine_models.tex`) | `reference/checks/table1_*.json` |
| kernel bandwidth: detached | `$MT $T1 --tag _detach --out results/table1_detach` | `reference/table1_detach.md` |
| kernel bandwidth: `n_cond` ramp 50 → 250, step decay | `$MT $T1 --methods MLGD MLGD-F --tag _v3 --extra "--n-cond-schedule linear --n-cond-min 50 --step-decay 0.1" --out results/table1_v3` | `reference/table1_v3_ncond_ramp.md` |
| sliced Wasserstein loss at (5, 1) and (10, 1) | `$MT $T1 --settings 5x1 10x1 --methods MLGD MLGD-F --tag _adam_swd_refine --extra "--loss swd" --out results/table1_swd` | `reference/table1_swd_5x1.md` |
| sampler training: 50,000 batches | `$MT $T1 --runs-dir runs/table1v2_long --epochs 50000 --settings 5x5 5x10 10x5 10x10 --methods MLGD MLGD-F --tag _long --out results/table1_long` | `reference/table1_long.md`, `reference/checks/long_*.json` |
| sampler training: 256 hidden units | `$MT $T1 --runs-dir runs/table1v2_wide --nunits 256 --settings 5x5 5x10 10x5 10x10 --methods MLGD MLGD-F --tag _wide --out results/table1_wide` | `reference/table1_wide.md`, `reference/checks/wide_*.json` |
| the (10, 1) cell: 100 restarts | `$MT $T1 --settings 10x1 --methods MLGD MLGD-F MLGD-X --budget 100 --attempts 3 --tag _r100 --out results/table1_10x1_r100` | `reference/table1_10x1_r100.md` |

The appendix table combines `results/table1_adam_refine.tex` with the TV columns of the three ablations.

**(10, 1) with a conditional sampler trained for 50,000 batches:**

```bash
D=runs/table1v2/x10_y1; P=experiments/synthetic/pipeline
python $P/train_models.py --data $D/data.pt --model conditional --architecture diffusion --fresh-samples \
    --epochs 50000 --batch-size 1024 --nunits 128 --nblocks 3 --seed 42 --device cuda --output-dir $D/models/diffusion_50k
python $P/search_mog.py --prior $D/models/uncond/model_unconditional.pt --sampler $D/models/diffusion_50k/model_conditional.pt \
    --target $D/data.pt --algo lgd --n-restarts 10 --n-attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device cuda --out $D/MLGD_50k.json
```

**Guidance step rule** (MLGD-F, three runs per cell, constant step against the Adam-form push):

```bash
for cell in 1x1 1x10 5x5 10x10; do
  for z in 0.03 0.1 0.3 1.0; do $MT $T1 --settings $cell --methods MLGD-F --attempts 3 --refine-steps 0 \
      --guidance-optim sgd  --step-size $z --tag _sgd$z  --out results/steprule_sgd$z;  done
  for z in 0.02 0.05 0.1 0.2; do $MT $T1 --settings $cell --methods MLGD-F --attempts 3 --refine-steps 0 \
      --guidance-optim adam --step-size $z --tag _adam$z --out results/steprule_adam$z; done
done
```

The numbers in "How to read TV" (the TV of `x* + δu` for random directions `u`, and of a random prior draw)
have no script. The trained networks are on Hugging Face:
[`simulations/checkpoints`](https://huggingface.co/anon-submission-cdm/cdm-inverse-design/tree/main/simulations/checkpoints).

## MLGD\* and GB-FESO

MLGD\* (the MLGD\* column of the main table) runs on a wider conditional sampler (diffusion, 256 units, 4 blocks,
trained on fresh draws) with zeroth-order guidance, 30 restarts, refinement with 3,000 samples that keeps its
best iterate, and the 5 lowest-loss restarts re-scored 5 × 5,000. Per cell `C` (e.g. `x10_y1`), five seeds:

```bash
S=experiments/synthetic/slurm; C=x10_y1; R=runs/mlgd_star/$C
DATA=runs/table1v2/$C/data.pt ARCH=diffusion BATCHES=100000 UNITS=256 BLOCKS=4 OUTD=$R/d256 \
  sbatch --job-name=tr_$C --export=ALL $S/slurm_train_sampler.sh
D=runs/table1v2/$C OUTDIR=$R NAME=MLGDstar SAMPLER=$R/d256/model_conditional.pt \
  EXTRA="--n-restarts 30 --guide-grad zo --refine-keep-best --refine-steps 100 --refine-n-cond 3000 \
         --select-top 5 --select-n 5000 --select-repeats 5" \
  sbatch --array=0-4 --job-name=mlgd_star --export=ALL $S/slurm_search.sh       # -> $R/MLGDstar_s42..46.json
```

GB-FESO (Zick & Mendels, arXiv:2607.06421) optimizes `x` directly by Adam through the conditional sampler, with no
prior over `X`. `pipeline/search_gbfeso.py` runs their optimizer loop on our samplers; `cdm/losses/kde_kl.py` is
their loss. It runs on MLGD\*'s samplers, data and budget (30 starts, 250 / 3,000 samples per phase-0 / phase-1
step, the same re-scoring), with their KDE-KL and with our MMD, seeds 42–46:

```bash
for loss in kl mmd; do L=$(echo $loss | tr a-z A-Z); for seed in 42 43 44 45 46; do
  python experiments/synthetic/pipeline/search_gbfeso.py --sampler $R/d256/model_conditional.pt \
    --target runs/table1v2/$C/data.pt --loss $loss --n-restarts 30 --n-attempts 1 --seed $seed \
    --num-samples 250 --num-samples-phase1 3000 --final-n 1000 --select-top 5 --select-n 5000 --select-repeats 5 \
    --device cuda --out runs/gbfeso_d256/$C/GBFESO-${L}_s$seed.json
done; done
cp $R/MLGDstar_s*.json runs/gbfeso_d256/$C/
python experiments/synthetic/pipeline/make_table1_d256.py --runs-dir runs/gbfeso_d256 --data-root runs/table1v2 \
    --out results/table1_gbfeso_d256        # the GB-FESO table and the MLGD* / GB-FESO (KL) columns
```

Reference: `reference/table1_gbfeso_d256.{json,md,tex}`.

## Toy example

```bash
P=experiments/synthetic/pipeline
python $P/fig1_toy_example.py --setting hard --device cuda    # trains the three toy networks, runs the four methods
                                                              # and the beta-sweep: results/fig1_hard{.png,_sim.json,_sweep.npz}
python $P/fig_smc_correction.py --setting hard --betas 0 2 8 16 --n-particles 2000 --n-repeats 10 \
    --max-shift 0.25 --ess-frac 1.0 --bins 80 --device cuda  # twisted-SMC correction: results/fig_smc_hard.{png,json,npz}
python $P/fig1_toy_example.py --setting hard --smc-npz results/fig_smc_hard.npz   # the appendix figure (all panels)
```

The main-text figure is the compact version, redrawn from the same caches (copy `results/fig1_hard_sim.json` and
`_sweep.npz` to `results/fig1_hard_compact_*` first):

```bash
python $P/fig1_toy_example.py --setting hard --smc-npz experiments/synthetic/reference/smc/fig_smc_hard.npz \
    --panels abd --figsize 11 8.5 --font-scale 1.9 --compact --out results/fig1_hard_compact.png
```

`reference/smc/` holds the published SMC populations and the `--max-shift` / `--ess-frac` sweep that chose 0.25 /
1.0. Network check of the toy: `python $P/check_models.py --setting hard`. Cluster: `sbatch slurm/slurm_fig1.sh hard`.

## No-star target

```bash
D=runs/nonstar; P=experiments/synthetic/pipeline
python $P/generate_data.py --params-file experiments/synthetic/configs/mog/nonstar_2d.txt --condition-on 1 \
    --n-samples 5000 --seed 42 --output $D/data.pt
python $P/train_models.py --data $D/data.pt --model unconditional --architecture diffusion --fresh-samples \
    --epochs 20000 --batch-size 1024 --nunits 128 --nblocks 3 --seed 42 --device cuda --output-dir $D/models/uncond
python $P/train_models.py --data $D/data.pt --model conditional --architecture cm --fresh-samples \
    --epochs 20000 --batch-size 1024 --nunits 128 --nblocks 3 --seed 42 --device cuda --output-dir $D/models/cm
for arm in "mmd 250" "l2 250" "l2 1"; do set -- $arm
  python $P/search_mog.py --prior $D/models/uncond/model_unconditional.pt --sampler $D/models/cm/model_conditional.pt \
      --target $D/data.pt --algo lgd --n-restarts 1 --n-attempts 25 --n-mc 1 --init zeros \
      --guidance-optim adam --step-size 0.05 --refine-steps 0 --loss $1 --n-cond $2 \
      --seed 42 --device cuda --out $D/arm_$1_n$2.json
done
python $P/plot_nonstar.py --runs $D     # the figure: results/nonstar/no_star_mmd_vs_l2.png
```

Each JSON holds the 25 `x_hat` and the time per attempt; "% at x_bi" is the fraction closer to `x_bi = -2` than to
`x_uni = 2`.

## Diagnostics (not in the paper)

All use the main-table networks and flags; the usage line is at the top of each script.

| script | what it runs |
|---|---|
| `slurm/slurm_screen.sh` | `search_mog --init screen` (screened start) on the grid |
| `slurm/slurm_zoguide.sh` | zeroth-order guidance on the grid, and at (10, 1) with the exact sampler |
| `slurm/slurm_diag_10x1.sh` | (10, 1) with the `exact` and `exactloss` oracles |
| `slurm/slurm_refine_select.sh` | `refine_endpoints.py`: refinement and re-selection of saved end points at (10, 1) |
| `slurm/slurm_xt_vs_noise.sh` | `diag_xT_vs_noise.py`: start `x_T` against guidance noise at (10, 1) |
| `pipeline/diag_trajectories.py`, `diag_sampler_at_xstar.py`, `check_sampler_accuracy.py`, `summarize_runs.py` | per-trajectory statistics, sampler accuracy at `x*`, run summaries |

`slurm_refine_select.sh` and `slurm_xt_vs_noise.sh` expect the main-table conditional sampler at
`runs/table1v2/x10_y1/models/diffusion_10k/`.

## `reference/`

| file | run |
|---|---|
| `table1_detach.md`, `table1_v3_ncond_ramp.md`, `table1_swd_5x1.md` | kernel-bandwidth variants |
| `table1_long.md`, `table1_wide.md`, `checks/{long,wide}_*.json` | retrained networks (50,000 batches / 256 units) |
| `table1_10x1_r100.md` | (10, 1) with 100 restarts |
| `table1_gbfeso_d256.{json,md,tex}` | MLGD\* against GB-FESO (KL, MMD) |
| `checks/table1_*.json` | network accuracy of the main-table networks (the four large cells) |
| `smc/fig_smc_hard.{json,npz}`, `smc/sweep/` | twisted-SMC populations of the toy example, and the parameter sweep |
