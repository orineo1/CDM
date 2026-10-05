# interpolation

The gender and age interpolation figures, for the MLGD-F scribbles of `../four_targets/results/`
(`full_GenderInterpolation_seed8`, `full_AgeInterpolation_seed2`):

1. **Sorted grid**: 16 of the scribble's 100 portraits, ordered along the target axis.
2. **Projection**: the target groups and the scribble's portraits in CLIP space.

Open `interpolation.ipynb` and run it top to bottom on a GPU (L40S, L4 or A100; a 16 GB T4 runs out of memory). The
code is `cdm/models/sd/interpolation.py`. Output: `outputs/<run>/` with `sorted_grid.png`, `pca_scatter.png`,
`ranked_all.png`, `picks.json` and `embeddings.npz`. On Colab, upload the repository to `/content/CDM` first.

The target axis is PC1 of a PCA on Woman vs Man (gender), or the direction from the mean age-40 to the mean age-79
embedding (age). CLIP order is only a proxy for perceived gender or age, so the 16 portraits are picked near even
spacing (about every 6–7 ranks) as the ones that best show their position. The ranks are fixed in the notebook
(`SCENARIOS[...]["picks"]`) and `ranked_all.png` shows all 100 with their ranks; `picks=None` gives exact even
spacing. The ranks hold for the same seed, GPU type and library versions.
