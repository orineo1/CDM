# figures

**Produces:** Figure 5 `fig:mnist` -- the top-5 recovered
digits per target, with polar and cartesian histograms of P(Y | X = x_hat)
against the target density -- and Figures 6-8
`fig:app_{bimodal,uniform,unimodal}_full`, the same panels for
all 15 seeds of each target.

| | |
|---|---|
| Script | `make_figures.py` (this directory) |
| Full configuration | reads `../table_mlgdf/outputs/full/*.pkl`; 500 angle draws per displayed digit |
| GPU | optional -- only the conditional sampler runs, which is small |
| Reduced configuration (`smoke`) | CPU, ~1 min, over the records of a reduced `table_mlgdf` run |

## Run

```
bash run.sh smoke            # reduced; needs ../table_mlgdf/run.sh smoke first
bash run.sh full             # full; needs ../table_mlgdf/run.sh full first
bash run.sh full <dir>       # or point it at any directory of .pkl records
```

`--no-titles` (set by the full tier) also writes the title-free copies, which
are the versions that appear in the paper.

## Reference figures

`results/PUBLISHED.md` lists the figure files as they appear in the paper.
`outputs/` holds local runs and is never a published figure.

`<run>_all_seeds.png`: all 15 seeds of one target in one grid, each titled
`Seed i | Loss x | label` (label = the digit the run's classifier assigned, `None` below
the threshold). One file per target, i.e. per row of Table 2.

