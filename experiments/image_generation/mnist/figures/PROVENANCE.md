# Provenance -- figures

## Specification
Paper Figure 5 (`fig:mnist`) and Figures 6-8
(`fig:app_{bimodal,uniform,unimodal}_full`), Appendix C.

## Code
Source: `MNIST/visualize.py` of `orineo1/conditional-matching-paper`, branch
**`main`**, MNIST tree at **`7c856e2`**. The branch is discussed in
`../table2_mlgdf/PROVENANCE.md`.

The plotting functions live in `../mnist_cdm/plotting.py`; `make_figures.py` is
the entry point.

## Absent
- The paper's appendix figure filenames (`appendix_*_{toprow,midrow,buttrow}.png`)
  are not emitted by `MNIST/visualize.py`, which writes
  `<run>_all_row<N>.png`, `<run>_top<k>.png`, `<run>_dist_cart.png` and
  `<run>_dist_polar.png`. The renaming step is not in the source repository.
- No rendered figure file is committed in either code repository.

## Outputs
`outputs/` is gitignored and holds only what a local run writes.
