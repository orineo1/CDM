# interpolation

**Produces:** the paper's two interpolation figures, for the MLGD-F scribble of each
continuum scenario (gender interpolation, seed 8; age interpolation, seed 2):

1. **Sorted grid:** 16 of the scribble's 100 portraits, ordered along the target axis.
2. **PCA scatter:** the target groups and the scribble's portraits in CLIP space.

Nothing here is a metric; the numbers are `four_targets/run.sh eval`.

| | |
|---|---|
| Entry point | `interpolation.ipynb` -> `cdm/models/sd/interpolation.py` |
| Input | the committed scribbles in `../four_targets/results/full_GenderInterpolation_seed8/` and `full_AgeInterpolation_seed2/` (set in `SCENARIOS`) |
| GPU | yes (L40S, L4 or A100; a 16 GB T4 runs out of memory while loading); runs on the cluster or on Colab |
| Output | `outputs/<run>/`: `sorted_grid.png`, `pca_scatter.png`, `ranked_all.png`, `picks.json`, `embeddings.npz` |

For each scenario a target is generated on the scribble (the four gender groups, 75 each, or
3 portraits per age 40–79), 100 portraits of the scribble are generated under the neutral
prompt, and each portrait gets a score along the target axis: gender, PC1 of a PCA fitted on
Woman vs Man; age, the axis from the mean age-40 to the mean age-79 embedding.

## How the grid's portraits are picked

The CLIP order is a proxy for what a person sees: two portraits a few ranks apart can look
the other way round, and lighting, hair or expression move a portrait's CLIP score without
changing its perceived gender or age. So the 16 portraits are picked near even spacing
(about every 6–7 ranks of 100) and, within a few ranks of each position, the one whose
perceived gender or age best shows that position is taken. The ranks are fixed in the
notebook (`SCENARIOS[...]["picks"]`) and saved to `picks.json`; `ranked_all.png` shows all
100 portraits with their ranks, so the choice can be checked. `picks=None` gives the purely
evenly spaced grid.

The ranks refer to these exact portraits: they hold for the same seed and `N_EVAL` on the
same GPU type and library versions (the run is deterministic).

## Run

Open `interpolation.ipynb` and run top to bottom. On Colab the first cell clones the
repository (a read token in a Colab secret named `GITHUB`, removed from the git config after
the clone) and installs the requirements; the last cell downloads `outputs/` as a zip.
