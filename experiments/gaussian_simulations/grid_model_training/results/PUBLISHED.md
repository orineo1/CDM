> The declared reference for this experiment: the numbers as published in the paper,
> transcribed with their location in the source. `../outputs/` is gitignored and holds
> only whatever a local run emitted.

# App. B.4 -- the 27 grid network trainings

Three networks (unconditional diffusion prior, conditional DDPM, conditional iCT) x nine
scenarios. Declared configuration: MLP 3 blocks x 128 units, fresh draws, 10,000 batches
of 1,024, T = 100 diffusion steps, AdamW lr 1e-4 with weight decay 1e-4 and a cosine
schedule, classifier-free dropout p = 0.2 for the conditional DDPM.

There is no result value here beyond the checkpoints; the quality numbers these networks
produce are Table 5 (`../../table5_model_checks/`). The checkpoints are published on the
anonymous HuggingFace repo the paper cites as `\hfsim`.

Transcribed from `paper.txt`.
