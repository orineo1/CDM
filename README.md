# Conditional Distribution Optimization: Method Comparison

Master's thesis implementation comparing optimization methods for finding optimal conditioning values in generative models. Given joint distribution P(X,Y), find x* that minimizes the distance between P(Y|X=x*) and target distribution G(Y).

## Methods Compared

**LGD (Loss-Guided Diffusion)**: Unconditional and conditional diffusion models with Monte Carlo gradient estimation through DDIM sampling  
**LGD-CM**: Loss-guided optimization using unconditional diffusion with conditional consistency models  
**D-FLOW**: Flow matching-based optimization using ODE solvers
**Superposition**: Score-based isosurface sampling with linear interpolation between conditional models

## Quick Start

### LGD Method
```python
from Diffusion import DiffusionModel
from ConsistencyModels import ConsistencyModeliCT
from Optimization import optimize_LGD
from LossFunctions import MMDLoss, RBF
import dist_utils

# Generate MoG parameters
mu_list, Sigma_list, alpha, mog_means, mog_variances, weights, x_star = \
    dist_utils.get_param_mog_with_target(dim_data=10, num_components=5, device='cuda')

# Data generators
data_gen = lambda n, **kw: dist_utils.generate_mog_samples_not_differentiable(
    n, mu_list, Sigma_list, alpha)

# Train unconditional diffusion model
model_uncond = DiffusionModel(nfeatures=9, condition=False, device='cuda')
model_uncond.train_model(data_generator=data_gen, nepochs=1000, batch_size=2048, device='cuda')

# Train conditional diffusion model
model_cond = DiffusionModel(nfeatures=10, condition=True, condition_on=9, device='cuda')
model_cond.train_model(data_generator=data_gen, nepochs=1000, batch_size=2048, 
                       condition_on=9, device='cuda')

# Optimize to find best x*
best_x, _, final_loss = optimize_LGD(
    model_uncond, model_cond, mog_means, mog_variances, weights,
    mu_list, Sigma_list, alpha, nsamples=500, num_x_t=3, device='cuda'
)

print(f"Optimized x*: {best_x}")
print(f"Final MMD loss: {final_loss:.6f}")
```

### D-FLOW Method
```python
from FlowMatching import FMModel
from Optimization import optimize_DFLOW

# Train flow matching models
vf_X = FMModel(nfeatures=9, condition_on=0, nunits=128, device='cuda')
vf_X.train_FM(data_generator=data_gen, nepochs=5000, batch_size=1024)

vf_y_cond_x = FMModel(nfeatures=1, condition_on=9, nunits=128, device='cuda')
vf_y_cond_x.train_FM(data_generator=data_gen, nepochs=5000, batch_size=1024)

# Optimize using differentiable flows
best_x1, final_loss = optimize_DFLOW(
    vf_y_cond_x, vf_X, device='cuda',
    mog_means=mog_means, mog_variances=mog_variances, weights=weights,
    max_iter=100, n_sample=500, loss_method="MMD"
)

print(f"Optimized x*: {best_x1}")
print(f"Final loss: {final_loss:.6f}")
```

## Evaluation Metrics

**SWD (Sliced Wasserstein Distance)**: Primary metric for distribution matching quality  
**MMD (Maximum Mean Discrepancy)**: Kernel-based distance metric for distribution comparison  
**L1 Distance**: Direct integration-based distance for low-dimensional conditionals  
**Optimization Time**: Computational efficiency and wall-clock time comparison

## Repository Structure

**Core Implementation:**
- `CompareMethods-Final.ipynb`: Main comparison notebook with all methods and experiments
- `Optimization.py`: LGD, LGD-CM, and D-FLOW optimization algorithms

**Generative Models:**
- `Diffusion.py`: DDIM diffusion model for LGD
- `ConsistencyModels.py`: Consistency models (CT/iCT) for LGD-CM
- `FlowMatching.py`: Conditional flow matching for D-FLOW
- `DiffusionGeneral.py`: Score-based diffusion for Superposition method

**Utilities:**
- `LossFunctions.py`: MMD, SWD distance metrics
- `dist_utils.py`: Mixture of Gaussians utilities and conditional computation
- `evalModels.py`: Comprehensive evaluation framework with statistical analysis
- `superposition_utils.py`: Isosurface sampling and interpolation utilities
- `NN_utils.py`: Neural network building blocks
- `EncoderDecoder.py`: VAE for latent space optimization


**Diffusion Models:**
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020) - https://arxiv.org/abs/2006.11239
- Song et al., "Denoising Diffusion Implicit Models" (2021) - https://arxiv.org/abs/2010.02502

**Consistency Models:**
- Song et al., "Consistency Models" (2023) - https://arxiv.org/abs/2303.01469
- Song & Dhariwal, "Improved Consistency Training" (2023) - https://arxiv.org/abs/2310.14189

**Flow Matching:**
- Lipman et al., "Flow Matching for Generative Modeling" (2023) - https://arxiv.org/abs/2210.02747

**Optimization Methods:**
- Song et al., "Loss-Guided Diffusion Models for Plug-and-Play Controllable Generation", ICML 2023 - https://proceedings.mlr.press/v202/song23k.html
- Ben-Hamu et al., "D-Flow: Differentiating through Flows for Controlled Generation" (2024) - https://arxiv.org/abs/2402.14017
- Skreta et al., "The Superposition of Diffusion Models Using the Itô Density Estimator" (2025) - https://arxiv.org/abs/2412.17762


