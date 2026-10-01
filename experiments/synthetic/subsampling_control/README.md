# subsampling_control

**Produces:** The Appendix G selection-rule control on the 10D setting. No Gaussian float in the paper backs it.

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/control_subsample.py` |
| Full configuration | 6 configurations x 25 restarts |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1.5 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Needs `CMP_SIM_DIR` pointing at `conditional-matching-paper/simulations`; downloads two checkpoints from HuggingFace on first use. Witness configurations are not run in either tier.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Results

`results/` holds the outputs of `run.sh full` behind the paper, copied from `outputs/`.
