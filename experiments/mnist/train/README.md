# train

**Produces:** the two frozen MNIST networks:
the unconditional DDPM prior `P(X)` over rotated MNIST (architecture of
Table 11) and the conditional iCT angle sampler `P(Y|X)`, together with the
dataset-construction classifier and the rotation-augmented search set the
conditional model trains on. Hyperparameters are Table 12 `tab:mnist_hparams`.

| | |
|---|---|
| Scripts | `train_uncond.py`, `train_conditional.py` (this directory) |
| Full configuration | prior: 100 epochs, batch 256, AdamW 1e-3 (~178 min on a T4). conditional: 500 epochs, batch 256, AdamW 1e-4 (~162 min on a T4), after 15 epochs of the construction classifier |
| GPU | yes (T4 in the paper) |
| Reduced configuration (`smoke`) | CPU, ~2 min: 1 epoch over 64 images, then 2 epochs over 32 images |

Neither training has to be run to reproduce Table 2: the published checkpoints
are downloaded instead (see the area README).

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```

MNIST is downloaded to `outputs/data/` unless `MNIST_DATA_ROOT` is set.

## Reference numbers

`results/PUBLISHED.md` -- the published training configuration and the reported
training times.
