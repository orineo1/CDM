# main_grid

**Produces:** Table 1 (`tab:sim_main`) and the full-grid appendix table (`tab:sim_full`):
the 9 scenarios `(dim X, dim Y) in {1, 5, 10}^2` x 4 algorithms (MLGD, MLGD-F, PMLGD, PMLGD-F),
scored by the exact total variation between `P(Y | X = x_hat)` and the target.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py`, which drives `generate_data.py`, `train_models.py`, `check_models.py` and `find_xstar.py` per scenario |
| Full configuration | 9 scenarios x 4 methods x 5 runs; 10 trajectories per run, `n_cond = 250`, Adam guidance step 0.05, 100 refinement steps (lr 0.02, 1,000 samples); `config/paper_config.txt` records it |
| GPU | yes. About 8 min per MLGD run and 50 s per MLGD-F run on an L40S, plus training three networks per scenario (10,000 batches each) |
| Reduced configuration (`smoke`) | CPU, a few minutes: one scenario, tiny networks, 2 trajectories, 1 run |

## Run

```bash
bash experiments/synthetic/main_grid/run.sh smoke        # reduced configuration, checks the code path
bash experiments/synthetic/main_grid/run.sh full         # the paper's configuration, all 9 scenarios on one GPU (~1 day)
SETTINGS="5x5" bash experiments/synthetic/main_grid/run.sh full   # one scenario
```

On a SLURM cluster, one array task per scenario (edit the `#SBATCH` partition / venv lines first):

```bash
sbatch experiments/synthetic/slurm/slurm_table1.sh
```

Every step whose output already exists is skipped, so a killed run can be resubmitted.
Outputs go to `outputs/` (gitignored): `outputs/runs/<scenario>/` holds data, networks, the
model check and one JSON per method; `outputs/table1.{json,md,tex}` is the table.

## Reference

`results/PUBLISHED.md` has the paper's numbers for this experiment; `results/table1.md` is
the table as produced by the run (same file `run.sh full` writes to `outputs/`).
