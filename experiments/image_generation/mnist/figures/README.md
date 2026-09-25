# figures

**Produces:** Figure 5 `fig:mnist` (inventory MN-02) -- the top-5 recovered
digits per target with polar and cartesian histograms of P(Y | X = x_hat)
against the target density -- and Figures 6-8
`fig:app_{bimodal,uniform,unimodal}_full` (inventory MN-03), the same panels for
all 15 seeds of each target.

| | |
|---|---|
| Script | `make_figures.py` (this directory) |
| Full tier | reads `../table2_mlgdf/outputs/full/*.pkl`; 500 angle draws per displayed digit |
| GPU | optional -- only the conditional sampler runs, which is small |
| Smoke tier | CPU, ~1 min, over the smoke run records |
| **Tier actually run here** | **smoke only** (2026-09-25, CPU, PASS) |

## Run

```
bash run.sh smoke            # needs ../table2_mlgdf/run.sh smoke first
bash run.sh full             # needs ../table2_mlgdf/run.sh full first
bash run.sh full <dir>       # or point it at any directory of .pkl records
```

`--no-titles` (set by the full tier) also writes the title-free copies, which
are the versions that appear in the paper.

## Reference figures

`results/PUBLISHED.md` lists the figure files as they appear in the paper.
Nothing in `outputs/` is a published figure.
