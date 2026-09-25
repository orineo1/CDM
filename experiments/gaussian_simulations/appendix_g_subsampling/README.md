# appendix_g_subsampling

**Produces:** The Appendix G selection-rule control on the 10D setting. No Gaussian float in the paper backs it.

| | |
|---|---|
| Script | `scripts/control_subsample.py` |
| Full tier | 6 configurations x 25 restarts |
| GPU | yes |
| Smoke tier | CPU, ~1.5 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
Needs `CMP_SIM_DIR` pointing at `conditional-matching-paper/simulations`; downloads two checkpoints from HuggingFace on first use. Witness configurations are not run in either tier.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
