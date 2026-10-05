# train

Trains the two networks. Not needed to reproduce the results: `../table_mlgdf` downloads the published checkpoints.

- `train_uncond.py`: the unconditional DDPM prior `P(X)` over rotated MNIST (UNet; 100 epochs, batch 256,
  AdamW 1e-3, `squaredcos_cap_v2` schedule, 1000 steps).
- `train_conditional.py`: the conditional iCT angle sampler `P(Y | X)` (500 epochs, batch 256, AdamW 1e-4), after
  a 15-epoch classifier that builds its training set.

```bash
bash run.sh smoke   # CPU, ~2 min, throwaway weights
bash run.sh full    # GPU, a few hours per network; writes outputs/full/
```

MNIST is downloaded to `outputs/data/` unless `MNIST_DATA_ROOT` is set. To use your own networks, copy the
checkpoints into `../checkpoints/` under the names given there.
