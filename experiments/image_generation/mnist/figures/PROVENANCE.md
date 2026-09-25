# Provenance -- figures

## Specification
Paper Figure 5 (`fig:mnist`) and Figures 6-8
(`fig:app_{bimodal,uniform,unimodal}_full`), Appendix C.

## Code
Ported from `MNIST/visualize.py` of `orineo1/conditional-matching-paper`,
branch **`main`**, MNIST tree at **`7c856e2`** (repository tip `841417c` at the
time of the port). Branch choice is argued in `../table2_mlgdf/PROVENANCE.md`.

The plotting functions now live in `../mnist_cdm/plotting.py`; `make_figures.py`
is the entry point. In the source repo `visualize.py` imported `mog_pdf`,
`classify_generated_images`, `sliced_wasserstein_distance` and `set_seed` from
`run_mlgdf.py`; that dependency is removed.

## Absent
- The paper's appendix figure filenames (`appendix_*_{toprow,midrow,buttrow}.png`)
  are not emitted by `MNIST/visualize.py`, which writes
  `<run>_all_row<N>.png`, `<run>_top<k>.png`, `<run>_dist_cart.png` and
  `<run>_dist_polar.png`. The renaming step is not in the source repository.
- No rendered figure file is committed in either code repository.

## What was produced locally
`outputs/` only, by the smoke tier, and it was deleted afterwards. The directory
is empty.
