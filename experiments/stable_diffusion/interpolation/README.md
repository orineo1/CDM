# interpolation

**Produces:** the gender- and age-interpolation figures: a scribble's portraits sorted along
the target axis, and the PCA scatter of the target groups with the scribble's portraits on
top. No metric is computed here; the numbers are `four_targets/run.sh eval`.

| | |
|---|---|
| Entry point | `interpolation.ipynb`, a thin notebook over `cdm/models/sd/interpolation.py` |
| Input | a scribble: by default the MLGD-F scribble of `../four_targets/outputs/full_<GenderInterpolation\|AgeInterpolation>/`, or any PNG (`SCRIBBLE = ...`) |
| GPU | yes (generates the target and the portraits); a few minutes |

Ported from the paper's `eval_scribbl_interpolation.ipynb` and
`eval_scribbl_interpolation_age.ipynb`, minus the MMD, the classifier and the sample rows.
As in those notebooks the target is generated **on the scribble itself**.

## Run

Set `SCENARIO = "gender"` or `"age"` in the first cell and run top to bottom, in Jupyter on
a GPU node, or without opening it:

```
jupyter nbconvert --to notebook --execute --inplace experiments/stable_diffusion/interpolation/interpolation.ipynb
```

What happens:

1. **Target**: gender, the four groups Woman / WomanMasc / ManFem / Man (`N_TARGET` images
   in total); age, one group per age from `AGE_MIN` to `AGE_MAX` (`N_PER_AGE` each).
2. **Portraits**: `N_EVAL` portraits of the scribble under the neutral prompt.
3. **Axis**: gender, PC1 of a PCA fitted on Woman + Man, oriented feminine -> masculine;
   age, the unit vector from the mean age-40 to the mean age-79 embedding, with PC1 of the
   residuals as the second dimension. Every portrait gets a score along it.
4. **`ranked_all.png`**: every portrait in score order, labelled with its rank, to pick from
   by eye.
5. **`sorted_grid.png`**: `N_PICKS` portraits along the axis. `PICKS = None` spaces them
   evenly; `PICKS = [1, 4, 7, ...]` (ranks into the sorted order) is the by-hand choice.
6. **`pca_scatter.png`**: the target groups (coloured by axis score, or by age) and the
   portraits (gold).
7. **Save**: `picks.json` (ranks, photo indices, scores, the full ranking) and
   `embeddings.npz`, next to the PNGs, in `outputs/<Scenario>/`.

`python experiments/stable_diffusion/export_results.py` copies the PNGs and `picks.json`
into `results/<Scenario>/`. `outputs/` is gitignored.
