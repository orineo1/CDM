# detached_bandwidth_grid

**Produces:** The App. B.6 bandwidth comparison (paper prose, no float).

| | |
|---|---|
| Script | `experiments/synthetic/pipeline/make_table1.py` |
| Full configuration | one grid per arm |
| GPU | yes |
| Reduced configuration (`smoke`) | CPU, ~1 min |

## Run

```
bash run.sh smoke   # reduced configuration, for a quick run
bash run.sh full    # the full configuration
```
Detached is the library default; `--extra=--bandwidth-grad` selects the non-detached arm. Note the `=`: argparse refuses a lone `-flag` as a value.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
Other files in `results/` are the outputs of the run reported in the paper's repository.
