# toy_example

**Produces:** Figure 1 `fig:sim_2d` — four panels.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig1_toy_example.py` |
| Full configuration | four beta values x 2,000 trajectories, plus the four methods |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min (the script's own `--quick`) |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Or on a cluster: `sbatch experiments/synthetic/slurm/slurm_fig1.sh`. Toy networks are cached under `outputs/ckpt/<setting>/`.

`run.sh full` passes `--rerun-sim` together with `--rerun-sweep`. `--rerun-sweep` keeps the
method results of a previous run and redoes only the beta sweep, so it needs that run's cache
(`outputs/fig1_<setting>_sim.json`). **In case there is no previous run** (a fresh `outputs/`),
run the full configuration directly with the same flags minus `--rerun-sweep`: `--rerun-sim`
alone computes the methods and the beta sweep. Only the `hard` setting is the published figure;
`bimodal` and `paper` are earlier variants.

### Submitting on a SLURM cluster (from the repository root)

```bash
sbatch --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=06:00:00 \
  --job-name=cdm-toy --output=logs/cdm-toy-%j.out \
  --wrap="python experiments/synthetic/pipeline/fig1_toy_example.py --setting hard --n-restarts 25 --n-cond 250 --n-mc 3 --methods MLGD MLGD-F PMLGD PMLGD-F --betas 0 2 8 16 --n-beta-samples 2000 --rerun-sim --device cuda --ckpt-dir experiments/synthetic/toy_example/outputs/ckpt --out experiments/synthetic/toy_example/outputs/fig1_hard"
```

On a partition whose GPUs are registered with a type, name it (e.g. `--gres=gpu:l40s:1`). `mkdir -p logs` first. Once a run
exists, `bash run.sh full` redoes everything.

The paper reports ten trajectories per method; the command above runs 25 (`--n-restarts 25`, as
`run.sh full`), so compare the fraction of trajectories in the `x*` basin, not the count. The
results are `outputs/fig1_hard.png`, `outputs/fig1_hard_sim.json` and the per-method summary lines
in the job log (`best of 25 ... k/25 restarts in the x* basin`). The twisted-SMC correction
(`../smc_correction`) reads the networks this run trains under `outputs/ckpt/hard/`.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
