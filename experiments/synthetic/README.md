# `experiments/synthetic/` — Gaussian-mixture simulations

The synthetic experiments of the paper are one setup. A Gaussian mixture `P(X, Y)` with
known parameters defines a target `G(Y) = P(Y | X = x*)`. Each algorithm searches for an
`x_hat` whose conditional `P(Y | X = x_hat)` matches `G`, and is scored against the exact
mixture. The algorithms are MLGD (conditional-diffusion sampler), MLGD-F (consistency
sampler), their particle versions PMLGD and PMLGD-F, and the oracles `exact` (the analytic
conditional as sampler) and `exactloss` (closed-form MMD).

| path | contents |
|---|---|
| `pipeline/` | the Python entry points (below); every script has `--help` |
| `slurm/` | SLURM job scripts for the HUJI cluster, and `cluster_setup.txt` |
| `configs/mog/` | fixed mixture parameter files (`generate_data.py --params-file`) |
| `reference/` | outputs of the runs behind the paper, imported from `orineo1/GlobalConditional` (this repository before it was renamed `orineo1/CDM`) |
| `results/` | a full rerun of the paper's synthetic experiments with the current code (detached bandwidth) |

Run everything from the repository root after `pip install -e .` (or with
`PYTHONPATH=$PWD`). Outputs go to `runs/`, `results/` and `checkpoints/` at the repository root, which are
git-ignored.

## The search (shared by all experiments)

Every experiment runs the same algorithm, `cdm.guidance.search.search`: R independent guided
trajectories (`--n-restarts`), each optionally refined after guidance (`--refine-steps`,
`--refine-n-cond`, `--refine-keep-best`), then selection of one design: the smallest final loss, or
with `--select-top K` the K smallest re-scored with `--select-n` fresh conditional samples
(`--select-repeats`, and `--select-samplers` to average over several trained samplers where they
exist). The gradient in the guided steps is backpropagation through the conditional sampler
(`--guide-grad pathwise`, default) or zeroth-order central differences of the loss
(`--guide-grad zo`; 2 dim(x) + 1 loss values per step, so for low-dimensional designs). The
driver of each experiment (`search_<dataset>.py`) only loads that dataset's models and target and
builds its per-trajectory configuration; the search flags are the same in every driver.

## `pipeline/`

| script | role |
|---|---|
| `generate_data.py` | draw `P(X, Y)` (random mixture `--random-mog --standardize`, or `--params-file`), define `G`, save `data.pt` |
| `train_models.py` | train the prior `P(X)` (unconditional diffusion) and a conditional sampler `P(Y \| X)` (`--architecture diffusion` or `cm`) |
| `search_mog.py` | the method: prior + conditional sampler + target → `x_hat` (restarts or particles, guidance, refinement, oracles) |
| `run_optimization.py` | older step 3 of the pipeline; also holds `load_model`, the checkpoint loader the other scripts import |
| `check_models.py` | trained networks against the analytic mixture (score error, prior SWD, sampler error at `x*` and over `x`) |
| `make_table1.py` | the grid driver: per scenario runs `generate_data`, `train_models`, `check_models`, `search_mog`, then collects the tables |
| `fig1_toy_example.py` | Figure 1: the 1-D toy (`--setting hard`), its networks, the four methods and the β-sweep |
| `fig_smc_correction.py` | twisted-SMC correction of the β-sweep (Figure 1(d)) |
| `refine_endpoints.py`, `diag_trajectories.py`, `diag_xT_vs_noise.py`, `diag_sampler_at_xstar.py` | diagnostics of the (10, 1) cell (research, not in the paper) |

## Smoke test (CPU, about 15 minutes)

```bash
python experiments/synthetic/pipeline/make_table1.py --settings 1x1 \
    --n-samples 500 --epochs 30 --batch-size 256 --nunits 16 --nblocks 1 \
    --budget 2 --attempts 1 --n-cond 16 --refine-steps 2 --refine-n-cond 32 \
    --guidance-optim adam --step-size 0.05 --skip-check --device cpu \
    --runs-dir runs/smoke --out results/smoke_table1
python experiments/synthetic/pipeline/fig1_toy_example.py --quick --setting hard --device cpu \
    --ckpt-dir checkpoints/smoke --out results/smoke_fig1_hard.png
```

This checks that the code runs. It does not reproduce any number.

## Reproducing the paper

The grid commands share the Table 1 search flags and the runs directory. A flag given again
after `$T1` overrides it (argparse keeps the last value).

```bash
MT="python experiments/synthetic/pipeline/make_table1.py"
T1="--budget 10 --attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device cuda --runs-dir runs/table1v2"
```

The data and training settings are the `make_table1.py` defaults: 4 components, 20,000
samples, networks MLP 3 × 128 trained for 10,000 fresh batches of 1,024, and resampling
`exp(-5 · loss)` when ESS < N/2. Every step whose output exists is skipped, so the variants
below reuse the Table 1 networks and differ only in `--tag`. On the cluster,
`slurm/slurm_table1.sh` runs one scenario per array task with the same flags and takes
`TAG`, `RUNS`, `ARGS` and `OUT` from the environment (`TAG=_detach ARGS="" sbatch ...`).

Table 1 ran before the RBF bandwidth was detached from the gradient, which is now the
library default. `--extra=--bandwidth-grad` restores the old behavior. Write it with `=`,
because argparse rejects a separate value that starts with `-`.

| paper result | command(s) | reference |
|---|---|---|
| **Table 1** `tab:sim_main`: TV per scenario × method | `$MT $T1 --tag _adam_refine --extra=--bandwidth-grad --out results/table1_adam_refine` → `_compact.tex` | — |
| **`tab:sim_full`**, refined columns (‖x̂ − x*‖, relative L²) | same run → `results/table1_adam_refine.tex`; per-run values, basin hits and resamplings in `.json` / `_appendix.md` | — |
| `tab:sim_full`, ablation *plain* (no refinement) | `$MT $T1 --refine-steps 0 --tag _adam --extra=--bandwidth-grad --out results/table1_adam` | — |
| `tab:sim_full`, ablation *annealed* | `$MT $T1 --methods PMLGD PMLGD-F --resample-schedule linear --refine-steps 0 --tag _adam_anneal --extra=--bandwidth-grad --out results/table1_adam_anneal` | — |
| `tab:sim_full`, ablation *oracle f_φ*; the (10, 1) "exact conditional" value | `$MT $T1 --methods MLGD-X PMLGD-X --refine-steps 0 --tag _adam_oracle --extra=--bandwidth-grad --out results/table1_adam_oracle` | — |
| **`tab:model_quality`**: network accuracy | the Table 1 run (it writes `runs/table1v2/<scenario>/check.json`) → `results/table1_adam_refine_models.tex`. Stand-alone: `python experiments/synthetic/pipeline/check_models.py --prior <d>/models/uncond/model_unconditional.pt --sampler <d>/models/diffusion/model_conditional.pt <d>/models/cm/model_conditional.pt --data <d>/data.pt --out <d>/check` | `reference/checks/table1_*.json` (4 of 9 cells) |
| App. "Kernel bandwidth": detached rerun | `$MT $T1 --tag _detach --out results/table1_detach` | `reference/table1_detach.md` |
| App. "Kernel bandwidth": `n_cond` ramp 50→250 + cosine step decay | `$MT $T1 --methods MLGD MLGD-F --tag _v3 --extra "--n-cond-schedule linear --n-cond-min 50 --step-decay 0.1" --out results/table1_v3` | `reference/table1_v3_ncond_ramp.md` |
| App. "Kernel bandwidth": sliced Wasserstein at (5, 1); App. "(10, 1) cell": SWD there | `$MT $T1 --settings 5x1 10x1 --methods MLGD MLGD-F --tag _adam_swd_refine --extra "--loss swd" --out results/table1_swd` | `reference/table1_swd_5x1.md` |
| App. "Sampler training": 50,000 batches | `$MT $T1 --runs-dir runs/table1v2_long --epochs 50000 --settings 5x5 5x10 10x5 10x10 --methods MLGD MLGD-F --tag _long --out results/table1_long` | `reference/table1_long.md`, `reference/checks/long_*.json` |
| App. "Sampler training": 256 hidden units | `$MT $T1 --runs-dir runs/table1v2_wide --nunits 256 --settings 5x5 5x10 10x5 10x10 --methods MLGD MLGD-F --tag _wide --out results/table1_wide` | `reference/table1_wide.md`, `reference/checks/wide_*.json` |
| App. "(10, 1) cell": 100 restarts per run | `$MT $T1 --settings 10x1 --methods MLGD MLGD-F MLGD-X --budget 100 --attempts 3 --tag _r100 --out results/table1_10x1_r100` | `reference/table1_10x1_r100.md` |
| App. "(10, 1) cell": conditional sampler retrained for 50,000 batches | see [below](#10-1-with-a-retrained-conditional-sampler) | — |
| App. "Guidance step rule": constant ζ′ vs Adam push | see [below](#guidance-step-rule-sweep) | — |
| **Figure 1** `fig:sim_2d` (toy, panels a–d) | see [below](#figure-1) | `reference/smc/` |
| App. toy "Models": network check of the toy | `python experiments/synthetic/pipeline/check_models.py --setting hard` (after Figure 1 has trained the networks) | — |
| **`tab:nonstar_results`**, `fig:nonstar` (no-star target) | see [below](#no-star-target) | — |

### Figure 1

```bash
P=experiments/synthetic/pipeline
python $P/fig1_toy_example.py --setting hard --device cuda     # trains the three toy networks (checkpoints/toy/hard/),
                                                               # runs MLGD, MLGD-F, PMLGD, PMLGD-F (10 trajectories each)
                                                               # and the beta-sweep; writes results/fig1_hard{.png,_sim.json,_sweep.npz}
python $P/fig_smc_correction.py --setting hard --betas 0 2 8 16 --n-particles 2000 --n-repeats 10 \
    --max-shift 0.25 --ess-frac 1.0 --bins 80 --device cuda   # results/fig_smc_hard.{png,json,npz}
python $P/fig1_toy_example.py --setting hard --smc-npz results/fig_smc_hard.npz   # redraw from the cache: panel (d) plain vs corrected
```

The last command redraws `results/fig1_hard.png` (the paper's `fig1_toy_hard_smc.png`).
The paper's main-text Figure 1 is the compact version (panels a, b and the beta sweep, larger fonts),
redrawn from the same caches; the four-panel version above is the appendix figure:

```bash
python $P/fig1_toy_example.py --setting hard --smc-npz experiments/synthetic/reference/smc/fig_smc_hard.npz \
    --panels abd --figsize 11 8.5 --font-scale 1.9 --compact --out results/fig1_hard_compact.png
```

(the caches must be named after `--out`: copy `results/fig1_hard_sim.json` / `_sweep.npz` to `results/fig1_hard_compact_*`). To
draw it from the published populations, use
`--smc-npz experiments/synthetic/reference/smc/fig_smc_hard.npz`. `reference/smc/sweep/`
holds the `--max-shift` / `--ess-frac` sweep that chose 0.25 / 1.0. On the cluster:
`sbatch experiments/synthetic/slurm/slurm_fig1.sh hard`.

### (10, 1) with a retrained conditional sampler

```bash
D=runs/table1v2/x10_y1; P=experiments/synthetic/pipeline
python $P/train_models.py --data $D/data.pt --model conditional --architecture diffusion --fresh-samples \
    --epochs 50000 --batch-size 1024 --nunits 128 --nblocks 3 --seed 42 --device cuda --output-dir $D/models/diffusion_50k
python $P/search_mog.py --prior $D/models/uncond/model_unconditional.pt --sampler $D/models/diffusion_50k/model_conditional.pt \
    --target $D/data.pt --algo lgd --n-restarts 10 --n-attempts 5 --n-cond 250 --guidance-optim adam --step-size 0.05 \
    --refine-steps 100 --refine-lr 0.02 --refine-n-cond 1000 --seed 42 --device cuda --out $D/MLGD_50k.json
```

In the cluster copy of `runs/table1v2/x10_y1/`, `models/diffusion/` holds the 50k network and
`models/diffusion_10k/` the Table 1 network. The diagnostic job scripts use the second one.

### Guidance step rule sweep

MLGD-F, three runs per cell, constant ζ′ (`sgd`) against the Adam-form push:

```bash
for cell in 1x1 1x10 5x5 10x10; do
  for z in 0.03 0.1 0.3 1.0; do $MT $T1 --settings $cell --methods MLGD-F --attempts 3 --refine-steps 0 \
      --guidance-optim sgd  --step-size $z --tag _sgd$z  --out results/steprule_sgd$z;  done
  for z in 0.02 0.05 0.1 0.2; do $MT $T1 --settings $cell --methods MLGD-F --attempts 3 --refine-steps 0 \
      --guidance-optim adam --step-size $z --tag _adam$z --out results/steprule_adam$z; done
done
```

The refinement setting of this sweep was not recorded. It ran before refinement was adopted,
hence `--refine-steps 0`.

### No-star target

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
```

Each JSON holds the 25 `x_hat` and the time per attempt. The "% at x_bi" column is the
fraction of `x_hat` closer to `x_bi = -2` than to `x_uni = 2`. No script computes it.
The published table and figure came from the notebook
`simulations/notebooks/Exp_nonstar_mmd_target_ablation.ipynb` of
`orineo1/conditional-matching-paper` (`main`, `fc4389d`), whose constants
`configs/mog/nonstar_2d.txt` transcribes.

### Paper numbers without a script

- App. "How to read TV": the TV of `x* + δu` for random unit directions `u` and
  δ ∈ {0.01, 0.03, 0.1, 0.3}.
- The same paragraph: the TV of a random draw from the prior (≈ 0.7 at (1, 1), ≈ 1.0 at
  (10, 10)).
- The combined `tab:sim_full` body (`aistats2027/tables/sim_appendix_rows.tex` in the paper
  repository) was assembled by hand. Its sources are `table1_adam_refine.tex` and the TV
  columns of `_adam`, `_adam_anneal` and `_adam_oracle`.

### Trained networks

The Table 1 networks (`runs/table1v2/<scenario>/models/`) are on Hugging Face under
`simulations/checkpoints`. The paper links them as `\hfsim`.

## Improved search at (10, 1) (research, after the AISTATS submission)

Table 1 reaches TV 0.49 at (10, 1). Three changes bring it to a mean of 0.08 over five seeds (0.11 at
worst, about 25 min per run on an L40S):

1. A more accurate conditional sampler: diffusion, 256 units, 4 blocks, 100k batches of fresh draws.
   Its TV to `G` at `x*` is 0.07, against 0.46 for the Table 1 network (`check_sampler_accuracy.py`).
2. Zeroth-order guidance (`--guide-grad zo`): central differences of the sample loss with common
   random numbers, pulled back to `x_t` through the Tweedie estimate. It recovers the mixture-weight
   part of the gradient that backpropagation through the sampler misses.
3. Refinement with 3,000 conditional samples that returns its best iterate (`--refine-keep-best`),
   after 30 guided trajectories, then reselection of the 5 lowest-loss ones.

```bash
S=experiments/synthetic/slurm
ARCH=diffusion BATCHES=100000 UNITS=256 BLOCKS=4 OUTD=runs/night/x10_y1/models/d256 \
  sbatch --job-name=tr_d256 --export=ALL $S/slurm_train_sampler.sh
NAME=D3_ref100_n3000 SAMPLER=runs/night/x10_y1/models/d256/model_conditional.pt \
  EXTRA="--n-restarts 30 --guide-grad zo --refine-keep-best --refine-steps 100 --refine-n-cond 3000 \
         --select-top 5 --select-n 2000 --select-repeats 5" \
  sbatch --array=0-4 --job-name=D3 --export=ALL $S/slurm_search.sh
python experiments/synthetic/pipeline/summarize_runs.py --runs-dir runs/night/x10_y1 --data runs/table1v2/x10_y1/data.pt
```

The analysis of why the Table 1 configuration fails here is in Section 13 of `optimization_notes.pdf`
(paper repository).

## GB-FESO competitor (Table 1 budget)

GB-FESO (Zick & Mendels, arXiv:2607.06421) optimizes x directly by Adam through the differentiated
conditional sampler, with no prior over X, using a KDE estimate of the KL.
`pipeline/search_gbfeso.py` ports their optimizer loop (`run_feso.py`, `run_phase`) onto our samplers,
and `cdm/losses/kde_kl.py` holds their loss, copied unchanged (MIT). It runs on the Table 1 networks
and data, so the comparison with MLGD differs only in the search and the loss:

| method | search | loss |
|---|---|---|
| MLGD (Table 1) | guided reverse diffusion under the prior, then refinement | MMD |
| `GBFESO-MMD` | GB-FESO: direct Adam on x from a random training design | MMD (same kernel as Table 1) |
| `GBFESO-KL` | GB-FESO | their KDE-KL (h = 0.5, symmetric, Euclidean coordinates) |

GB-FESO's optimizer is theirs (command-line defaults): 201 Adam steps per phase at lr 0.05, phase-0
noise 0.05 decaying linearly to 0, best iterate kept, x clamped to the range of X in the data. Phase 1
(201 more steps, no noise) runs in every start for both losses (`--phase1 always`; their code runs it
only when the phase-0 KL exceeds 0.2, a threshold with no MMD counterpart). Each run has 10 starts,
each a random row of X from the training data. Each start's answer, the better of the two phases, is
chosen and scored on fresh samples, and the lowest score wins, as MLGD's 10 restarts; 5 runs per
scenario. The sample budget is MLGD's in Table 1: 250 conditional samples per step in phase 0 (MLGD:
per guided step), 1,000 in phase 1 (MLGD: per refinement step), scored on 1,000; every loss compares
the generated samples with as many target samples, drawn fresh at each step.
`--select-top K --select-n N --select-repeats R` replaces the lowest-score rule by MLGD's selection
(`cdm.guidance.search.select`, the same function): the K lowest-scoring starts are re-scored R times on
N fresh samples and the lowest average is returned, as MLGD\* does with K = 5, N = 5,000, R = 5.

On the cluster, reusing the Table 1 networks (`RUNS`), from the repository root, with the environment active:

```bash
RUNS=runs/table1v2 VENV=$VIRTUAL_ENV TAG=_main OUT=results/table1_gbfeso ARGS="--methods GBFESO-KL GBFESO-MMD --extra='--num-samples 250 --num-samples-phase1 1000 --final-n 1000'" sbatch --partition=salmon --gres=gpu:1 --time=12:00:00 experiments/synthetic/slurm/slurm_table1.sh
```

The last array task writes the table; rebuild it once every task is done (`--collect` reads the JSONs only):
`python experiments/synthetic/pipeline/make_table1.py --collect --runs-dir runs/table1v2 --tag _main --methods GBFESO-KL GBFESO-MMD --out results/table1_gbfeso`.
The committed results are `results/table1_gbfeso.{json,tex}` (the `--collect` output) and `results/table1_gbfeso.md`, written from it and `results/table1.json` by `python experiments/synthetic/pipeline/make_table1_gbfeso.py`.

### GB-FESO on the retrained samplers (against MLGD\*)

MLGD\* runs on retrained conditional samplers (diffusion, 256 units, 4 blocks; on the cluster
`runs/night/grid/<cell>/d256/`, and `runs/night/x10_y1/models/d256/` for (10, 1)) with the Table 1 data.
GB-FESO is rerun on the same samplers and data, at MLGD\*'s budget, so the two differ only in the
search: 30 starts per run, 5 runs (seeds 42–46, one result file each), 250 samples per phase-0 step,
3,000 per phase-1 step (MLGD\*: per refinement step), each start scored on 1,000, and MLGD\*'s selection
(the 5 lowest re-scored 5 × 5,000). GB-FESO uses no prior over X. One array task per cell, loss and seed
(9 × 2 × 5 = 90, about 50 min each on an L40S), from the repository root:

```bash
OR=/sci/labs/orzuk/orzuk/github/CDM     # samplers and data
sbatch --array=0-89%20 --job-name=gbfeso_d256 --partition=salmon --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=03:00:00 \
  --output=logs/gbfeso_d256_%A_%a.out --wrap='i=$SLURM_ARRAY_TASK_ID; OR='$OR'; \
cell=$(echo x1_y1 x1_y5 x1_y10 x5_y1 x5_y5 x5_y10 x10_y1 x10_y5 x10_y10 | cut -d" " -f$((i/10+1))); \
loss=$(echo kl mmd | cut -d" " -f$((i/5%2+1))); seed=$((42+i%5)); L=$(echo $loss | tr a-z A-Z); \
if [ $cell = x10_y1 ]; then S=$OR/runs/night/x10_y1/models/d256/model_conditional.pt; else S=$OR/runs/night/grid/$cell/d256/model_conditional.pt; fi; \
D=runs/gbfeso_d256/$cell; mkdir -p $D; PYTHONPATH=$PWD python experiments/synthetic/pipeline/search_gbfeso.py \
  --sampler $S --target $OR/runs/table1v2/$cell/data.pt --loss $loss --n-restarts 30 --n-attempts 1 --seed $seed \
  --num-samples 250 --num-samples-phase1 3000 --final-n 1000 --select-top 5 --select-n 5000 --select-repeats 5 \
  --device cuda --out $D/GBFESO-${L}_s$seed.json'
```

Then link MLGD\*'s runs next to them (`MLGDstar_s<seed>.json` -> `runs/night/grid/<cell>/G_zo_s<seed>.json`;
at (10, 1) `runs/night/x10_y1/E6_sel5000_n3000_s<seed>.json`, the same recipe) and write the table:

```bash
python experiments/synthetic/pipeline/make_table1_d256.py --runs-dir runs/gbfeso_d256 \
    --data-root $OR/runs/table1v2 --out experiments/synthetic/results/table1_gbfeso_d256
```

`results/table1_gbfeso_d256.{json,md,tex}`, in the format of `table1_gbfeso.*`: the `.json` and `.tex` as
`make_table1.py --collect` writes them (exact TV, ‖x̂ − x*‖, relative L2, time per cell and method), the `.md` the
TV table and every column. The earlier `table1_gbfeso.*` ran GB-FESO on the
Table 1 networks with 10 starts.

## Research diagnostics (not in the paper)

All of these use the Table 1 networks and flags. The usage line is at the top of each script.

| job | what it runs |
|---|---|
| `slurm/slurm_screen.sh` | `search_mog --init screen` (screened start) on the grid, tags `_screen_refine` / `_screen_guided` |
| `slurm/slurm_zoguide.sh` | zeroth-order guidance `--guide-grad zo`: the (10, 1) exact-sampler test and the grid, tag `_zoguide` |
| `slurm/slurm_diag_10x1.sh` | (10, 1) with the `exact` and `exactloss` oracles, 10 and 100 trajectories |
| `slurm/slurm_refine_select.sh` | `refine_endpoints.py`: pathwise or zeroth-order refinement and reselection of the Table 1 end points at (10, 1) |
| `slurm/slurm_xt_vs_noise.sh` | `diag_xT_vs_noise.py`: the start `x_T` against the guidance noise at (10, 1) |
| `pipeline/diag_trajectories.py` | per-trajectory TV and best-of-k statistics from result JSONs |
| `pipeline/diag_sampler_at_xstar.py` | a trained sampler at `x*`: TV to `G`, and the loss at `x*` against the loss at the end points |
| `pipeline/summarize_runs.py` | per-variant summary of `<variant>_s<seed>.json` result files |
| `pipeline/check_sampler_accuracy.py` | TV of trained samplers to the exact conditional at `x*` and at random `x` (dim Y = 1) |
| `slurm/slurm_search.sh`, `slurm/slurm_train_sampler.sh` | one search variant over seeds; train one network on fresh mixture draws |

## `reference/`

Imported from `orineo1/GlobalConditional` `master` `9ddf570` (`results/`), except
`smc/fig_smc_hard.npz` (`1a5edcd`). Tables use the `make_table1.py` Markdown format.

| file | run |
|---|---|
| `table1_detach.md` | detached-bandwidth rerun of the grid |
| `table1_v3_ncond_ramp.md` | `n_cond` ramp + cosine step decay |
| `table1_swd_5x1.md` | sliced-Wasserstein loss at (5, 1) |
| `table1_long.md`, `table1_wide.md` | retrained networks, 50,000 batches / 256 units |
| `table1_10x1_r100.md` | (10, 1) with 100 restarts |
| `checks/table1_*.json`, `checks/{long,wide}_*.json` | `check_models.py` output of the Table 1 / long / wide networks (the four large cells) |
| `smc/fig_smc_hard.{json,npz}` | twisted-SMC populations and TVs of Figure 1(d), published configuration |
| `smc/fig_smc_hard_cpu.json` | an earlier CPU run of 2,000 particles, from before the script recorded `--n-repeats` and `--max-shift` |
| `smc/sweep/ms*_ess*.json` | the `--max-shift` / `--ess-frac` sweep |

## `results/`

A rerun of the synthetic experiments on NVIDIA L40S GPUs with the code of this repository
(`search_mog.py` and the losses as on `master` at `602c09a`; later commits add options that
are off by default). Every run uses the library default, a **detached** RBF bandwidth (no
`--bandwidth-grad`), so the grid corresponds to the paper's detached rerun
(`reference/table1_detach.md`), not to the published Table 1 run.

`results/run.sh` reruns all of it with the exact flags, step by step or at once
(`bash experiments/synthetic/results/run.sh all`, one GPU, about two days; the steps are
`grid`, `checks`, `ablations`, `nonstar`, `fig1`, `smc`). It writes to the git-ignored
`runs/reproduction/` and `results/reproduction/`.

| file | command (from the repository root) |
|---|---|
| `table1.{md,json,tex}` | `$MT $T1 --runs-dir runs/table1v2` (no `--tag`, no `--extra`), then `--collect`: Table 1 and the refined columns of `tab:sim_full` |
| `table1_gbfeso.{md,json,tex}` | the GB-FESO competitor at the Table 1 budget (section "GB-FESO competitor"): `.md` next to the Table 1 methods; `.json` and `.tex` from `make_table1.py --collect --tag _main --methods GBFESO-KL GBFESO-MMD` |
| `checks/table1_*.json` | `check_models.py` on the nine scenarios' networks, `--n-random-x 100`: `tab:model_quality` |
| `nonstar/arm_*.json` | the [no-star target](#no-star-target) commands above: `tab:nonstar_results` |
| `fig1/fig1_hard{.png,_sim.json,_sweep.npz}` | `fig1_toy_example.py --setting hard --n-restarts 25 --n-cond 250 --n-mc 3 --methods MLGD MLGD-F PMLGD PMLGD-F --betas 0 2 8 16 --n-beta-samples 2000 --rerun-sim --device cuda` (25 trajectories per method, not 10) |
| `smc/fig_smc_hard.{json,png}` | `fig_smc_correction.py` with the Figure 1 flags above, on the networks of `fig1/` |

Seeds are fixed (42), but trained weights differ across GPUs and library versions, so the
numbers agree with the paper within run-to-run noise rather than digit for digit. The
`tab:sim_full` ablation columns (plain, annealed, oracle) are not in this folder yet.

