> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.8, "Models" (paper.txt:1030-1034)

> "Checked against the analytic mixture (script check_models.py), the prior's score
> network has a relative Fisher error below 0.02 at every noise level but the last,
> unguided samples match P(X) (normalised SWD 0.02), and the conditional-diffusion sampler
> reproduces P(Y | X = x*) to Wasserstein-1 distance 0.06; the consistency sampler is much
> less accurate at x* (Wasserstein-1 0.56)."

| quantity | value |
|---|---|
| prior score net, relative Fisher error | < 0.02 at every noise level but the last |
| prior unguided samples, normalised SWD to P(X) | 0.02 |
| conditional DDPM at x*, W1 | 0.06 |
| consistency sampler (iCT) at x*, W1 | 0.56 |

This is the toy setting, not the grid of Table 5.

Transcribed from `paper.txt`. The text does not name the `--setting`; `run.sh full` runs
both `hard` and `paper`.
