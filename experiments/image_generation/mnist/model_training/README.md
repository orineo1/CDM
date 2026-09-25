# model_training

**Produces:** the two frozen MNIST networks (inventory MN-05, MN-07 to MN-09):
the unconditional DDPM prior `P(X)` over rotated MNIST (architecture of
Table 11) and the conditional iCT angle sampler `P(Y|X)`, together with the
dataset-construction classifier and the rotation-augmented search set the
conditional model trains on. Hyperparameters are Table 12 `tab:mnist_hparams`.

| | |
|---|---|
| Scripts | `train_uncond.py`, `train_conditional.py` (this directory) |
| Full tier | prior: 100 epochs, batch 256, AdamW 1e-3 (~178 min on a T4). conditional: 500 epochs, batch 256, AdamW 1e-4 (~162 min on a T4), after 15 epochs of the construction classifier |
| GPU | yes (T4 in the paper) |
| Smoke tier | CPU, ~2 min: 1 epoch over 64 images, then 2 epochs over 32 images |
| **Tier actually run here** | **smoke only** (2026-09-25, CPU, PASS) |

Neither training has to be run to reproduce Table 2: the published checkpoints
are downloaded (see the area README).

## Run

```
bash run.sh smoke
bash run.sh full
```

MNIST is downloaded to `outputs/data/` unless `MNIST_DATA_ROOT` is set.

## Reference numbers

`results/PUBLISHED.md` -- the published training configuration and the reported
training times.
