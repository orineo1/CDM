# Provenance -- table2_mlgdf

## Specification
Paper Section 5.2 (`sec:exp:mnist`), Table 2 (`tab:mnist`), Table 13
(`tab:mnist_lgd_hparams`), Appendix C.

## Code

Ported from `orineo1/conditional-matching-paper`:

| what | path in the source repo |
|---|---|
| runner | `MNIST/run_mlgdf.py` |
| conditional iCT angle model | `MNIST/src/cond_model.py` |
| UNet prior wrapper | `MNIST/src/uncond_model.py` |
| rotation-robust classifier | `MNIST/src/classifier.py` |
| dataset construction | `MNIST/src/dataset.py` |

Branch and commit: **`main`, MNIST tree at `7c856e2`** ("fix path to new dir").
The repository tip at the time of the port was `841417c`.

Why this branch:
- `change-dir-HF-mnist_v2` @ `a298e21` has a **byte-identical** `MNIST/` tree
  (`git diff upstream/main upstream/change-dir-HF-mnist_v2 -- MNIST` is empty);
  it is the branch that was merged, so `main` is the same code.
- `MNIST-CLEAN` @ `7b12001` is older: it predates the HuggingFace `subfolder="MNIST"`
  fix and the `REPO_ROOT` / `PYTHONPATH` handling (3 files, 6 lines).
- `TFG_in_MNIST` @ `7613b2e` is `main` plus a TFG guidance extension
  (+90 lines in `run_mlgdf.py`, a later experiment, not the paper's MLGD-F).
- `gread-search-mnist` @ `6b5af8b` is the hyperparameter-search branch; it
  carries grid-search drivers and committed artifacts (below), not the runner.
- `MNIST-EXP` does not exist as a branch; the paper's `\repomnist` points at
  `MNIST/notebooks/MNIST_experiments.ipynb`, which exists only on
  `gread-search-mnist`.

## Committed artifacts in the source repository

Run records committed on branch `gread-search-mnist` @ `6b5af8b`, under
`MNIST/checkpoints_and_results/` (each a pickled payload of 15 seeds: images,
per-seed final SWD, per-seed time):

| file | blob |
|---|---|
| `Bimodal.pkl` | `ba7b7d3194e6e806feeed8df4d47f5c49d20f14b` |
| `Uniform.pkl` | `b15158c6df6278dc8458f51d95bcb8c92464841e` |
| `Unimodal.pkl` | `26450cdb651aa851379aa9fd4fd3a418421ae716` |
| `uni_var515_st130_ssdouble_xt3.pkl` | `6e50ee21cf580482619d4125c27181ee98f44a1c` |
| `unif_ns1500_st300_sshalf_xt10.pkl` | `ed87cbe05c8a6f163f4c1d8bb22f3b6b351f281d` |
| `unif_ns600_st290_ssoriginal_xt3.pkl` | `a8f1cdda79d6e75ddd82bc8eb46e85b59ea6c5b9` |
| `robust_classifier.pth` | `e0689b9e9ccddbe4d7b6031e719c261794ce5b14` |

They are not copied into `results/`; `results/PUBLISHED.md` carries the paper's
numbers themselves.

## Absent
- The unconditional prior checkpoint `MnistUncond100Epoch.pth` is not in either
  repository; it is fetched from HuggingFace.
- No run record is committed on `main`; `MNIST/results/` does not exist there.

## What was produced locally
`outputs/` only, by the smoke tier, and it was deleted afterwards. The directory
is empty. Nothing in `outputs/` is a published number.
