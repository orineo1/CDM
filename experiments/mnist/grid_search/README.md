# grid_search

Hyperparameter search for MLGD-F on the **uniform** and **bimodal** MNIST targets,
logged to Weights & Biases and ranked by the classifier on each configuration's best
seeds.

Each configuration (one combination of the lists in `grids/<target>.args`) runs 15
seeds exactly like `../table_mlgdf/run_mlgdf.py`: same optimiser, seeding, no clamp,
the paper's classifier (`../checkpoints/robust_classifier.pth`) and the paper's
threshold 0.75. Then:

- every recovered image gets the classifier's label and confidence;
- the 5 seeds with the lowest final SWD are the **top-5**;
- the configuration is scored by the classifier on its top-5:
  `top5_clf_score = (# of the top-5 the classifier labels) + (their mean confidence)`,
  so 5 labelled images beat 4 and confidence breaks ties (range 0 to 6).

Per configuration, in `outputs/<target>/<configuration>/`:

| file | contents |
|---|---|
| `<configuration>_all_seeds.png` | all 15 seeds, each titled `Seed \| Loss \| label (confidence)` |
| `<configuration>_top5.png` | the top-5 by SWD, same titles |
| `metrics.json` | the score, top-5 labels and confidences, SWD all / top-5, time |
| `record.pkl` | images, losses, labels, confidences (same format as Table 2's records) |

In W&B (project `cdm-mnist-grid`, one run per configuration, grouped by target): the
config, the metrics as summary fields, both figures, and a table with every seed's
image, SWD, label, confidence and whether it is in the top-5. In the W&B runs table,
sort by `top5_clf_score`.

## Run on the cluster

```bash
wandb login                                            # once, on the login node
bash experiments/mnist/grid_search/submit.sh uniform   # 36 configurations
bash experiments/mnist/grid_search/submit.sh bimodal   # 48 configurations
```

`submit.sh` counts the grid, submits one SLURM array task per configuration
(`slurm.sh`), and queues a ranking job that runs when the array has ended. Finished
configurations are skipped on resubmission. Compute nodes without internet:
`WANDB_MODE=offline bash submit.sh ...`, then `wandb sync wandb/offline-run-*`.

## The ranking, locally

```bash
python experiments/mnist/grid_search/summarize.py --target uniform --top 10
```

prints the configurations sorted by `top5_clf_score` (ties: lower top-5 SWD) and
writes `outputs/<target>/ranking.md`; each row names the folder with its figures.

## The grids

`grids/uniform.args`: n_cond {600, 1000} x T {290, 300, 400} x n_MC {3, 10} x
step rule {original, half, double}. `grids/bimodal.args`: n_cond {1000, 1500} x
T {100, 125, 150} x n_MC {3, 10} x step rule {original, double} x variance {252, 300}.
The paper's configurations (uniform 600 / 290 / 3 / original; bimodal 1500 / 125 / 10
/ original / 252) are included. Edit the files to change the grid.
