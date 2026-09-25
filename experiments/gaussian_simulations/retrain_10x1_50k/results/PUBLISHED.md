> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.7, "The (10, 1) cell" (paper.txt:908-911)

> "the conditional sampler trained for 50,000 instead of 10,000 batches (its error at x*
> drops from 0.99 to 0.21) gives TV 0.58 +- 0.30; the exact conditional as sampler gives
> 0.89 +- 0.10 (no refinement); the sliced-Wasserstein loss [...] gives 0.81 +- 0.27; the
> table value with the standard sampler is 0.49 +- 0.08."

| arm at (10, 1) | sampler error at x* | TV |
|---|---|---|
| standard sampler | 0.99 | 0.49 +- 0.08 |
| conditional retrained 50,000 batches | 0.21 | 0.58 +- 0.30 |
| exact conditional as sampler (no refinement) | -- | 0.89 +- 0.10 |
| sliced-Wasserstein loss | -- | 0.81 +- 0.27 |

Transcribed from `paper.txt`. The text describes retraining the conditional sampler;
`make_table1.py --epochs 50000 --force` retrains all three networks.
