# detached_bandwidth_grid

**Produces:** The App. B.6 bandwidth comparison (paper prose, no float).

| | |
|---|---|
| Script | `scripts/make_table1.py` |
| Full tier | one grid per arm |
| GPU | yes |
| Smoke tier | CPU, ~1 min |
| Tier run here | smoke |

## Run

```
bash run.sh smoke
bash run.sh full
```
Detached is the library default; `--extra=--bandwidth-grad` selects the non-detached arm. Note the `=`: argparse refuses a lone `-flag` as a value.

Outputs go to `outputs/`, which is gitignored and ships empty.

## Reference

`results/PUBLISHED.md` — the paper's numbers for this experiment, transcribed.
See `PROVENANCE.md` for anything else in `results/`.
