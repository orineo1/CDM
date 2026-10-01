# smc_correction

**Produces:** Figure 3 `fig:smc_correction`.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/fig_smc_correction.py` (smoke first trains the toy networks via `fig1_toy_example.py --quick`) |
| Full configuration | 4 betas x 10 pooled populations x 2,000 particles x 2 arms |
| GPU | optional — a CPU variant of the reference output exists |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Both configurations pass four betas; the script requires more than one `--betas` value.

The full configuration **loads** the toy networks and does not train them: run the Figure 1
simulation first (`../toy_example/README.md`). `run.sh full` looks for the networks under this
directory's `outputs/ckpt/`; in case they are not there, point it at the ones Figure 1 trained
with a symlink before submitting.

### Submitting on a SLURM cluster (from the repository root)

```bash
mkdir -p experiments/synthetic/smc_correction/outputs logs
ln -sfn ../../toy_example/outputs/ckpt experiments/synthetic/smc_correction/outputs/ckpt
sbatch --partition=<gpu partition> --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=06:00:00 \
  --job-name=cdm-smc --output=logs/cdm-smc-%j.out --dependency=afterok:<toy job id> \
  --wrap="bash experiments/synthetic/smc_correction/run.sh full"
```

Drop `--dependency` when the toy run has already finished. On a partition whose GPUs are
registered with a type, name it (e.g. `--gres=gpu:l40s:1`). The result is
`outputs/fig_smc_hard.json` and the `[smc] beta=...: TV(hist, Q_beta) plain ... -> corrected ...`
lines in the job log.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
