# CDM: Inverse Design for Conditional Distribution Matching

Code for the paper *Inverse Design for Conditional Distribution Matching*. Given generative models of a joint
`P(X, Y)` and a target distribution `G(Y)`, find an input `x` whose induced conditional `P(Y | X = x)` matches
`G`. MLGD solves this by loss-guided reverse diffusion over `X`; its loss (MMD or sliced Wasserstein) is estimated
from samples of a differentiable conditional sampler. MLGD-F uses a few-step sampler.

## Install

From the root of this repository:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# GPU:                    pip install torch --index-url https://download.pytorch.org/whl/cu121
# Stable Diffusion only:  pip install -r experiments/stable_diffusion/requirements.txt
```

## Quick start (CPU, about 2 minutes)

A random Gaussian mixture `P(X, Y)` with `dim X = 3`, `dim Y = 1`, and the target `G = P(Y | X = x*)` for a known
`x*`. The search prints the recovered `x_hat`, its distance to `x*` and the exact L2 distance between
`P(Y | X = x_hat)` and `G`. The networks are trained briefly here; `--epochs 2000` gives a closer `x_hat`.

```bash
P=experiments/synthetic/pipeline
python $P/generate_data.py --random-mog --dim 4 --n-components 3 --condition-on 3 --n-samples 10000 \
    --output runs/quick/data.pt
python $P/train_models.py --data runs/quick/data.pt --model both --architecture cm --epochs 200 --batch-size 1024 \
    --output-dir runs/quick/models
python $P/search_mog.py --prior runs/quick/models/model_unconditional.pt \
    --sampler runs/quick/models/model_conditional.pt --target runs/quick/data.pt --n-restarts 3 --out runs/quick/xstar.json
```

## Experiments

Each directory's README has the commands, the settings and the committed results.

| directory | results in the paper |
|---|---|
| [`experiments/synthetic/`](experiments/synthetic/README.md) | toy example and β-sweep; Gaussian-mixture grid (MLGD, MLGD\*, MLGD-F, particle variants, ablations, network accuracy); comparison with GB-FESO; no-star target |
| [`experiments/mnist/`](experiments/mnist/README.md) | rotated digits: SWD, recovered digits, angle histograms; network and search settings |
| [`experiments/stable_diffusion/`](experiments/stable_diffusion/README.md) | sketch-to-portrait with four distributional targets against baselines; scribble edits and saliency; interpolation grids; per-step cost; Jacobian error of the few-step sampler |
| [`experiments/peptide/`](experiments/peptide/README.md) | molecular design: comparison with GB-FESO on the four-bead peptide |

## Code

```
cdm/guidance/   the search: guided reverse diffusion (core.py), its settings (config.py), restarts and selection (search.py)
cdm/losses/     sample distances: MMD, sliced Wasserstein, GB-FESO's KDE-KL, closed-form MMD for mixtures
cdm/models/     frozen networks: diffusion prior, conditional samplers, the MNIST and Stable Diffusion models
cdm/utils/      Gaussian mixtures and exact distances between them, network blocks
tests/          CPU tests, also run on every push: bash tests/run_all_tests.sh
```

## License

MIT, see [`LICENSE`](LICENSE).
