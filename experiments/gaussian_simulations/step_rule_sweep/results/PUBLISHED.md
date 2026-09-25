> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.6, "Guidance step rule" (paper.txt:841-846)

> "With the consistency sampler and three runs per cell, the best constant is
> zeta' ~ 0.03-0.1 at (dim X, dim Y) = (1, 1) (TV 0.09-0.11) but 0.3 at (5, 5) (0.50), and
> every constant fails at (10, 10) (>= 0.90); zeta' = 1 sends trajectories off the support
> in most scenarios (TV ~ 1). We therefore apply the push in Adam form."

| cell | zeta' | TV |
|---|---|---|
| (1, 1)   | 0.03-0.1 | 0.09-0.11 |
| (5, 5)   | 0.3      | 0.50 |
| (10, 10) | any      | >= 0.90 |
| most     | 1.0      | ~ 1 |

Adopted setting: Adam form, step 0.05, betas 0.9/0.999.

Transcribed from `paper.txt`. The text gives a range rather than the individual constants
swept; `config/zeta_grid.txt` lists the endpoints it names.
