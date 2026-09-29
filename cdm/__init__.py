"""Conditional Distribution Matching (CDM) by loss-guided diffusion.

    cdm.guidance   Algorithm 1: the guided reverse-diffusion outer loop (MLGD / MLGD-F)
    cdm.losses     Algorithm 2: the inner estimator -- MMD, sliced Wasserstein, selection
    cdm.models     the frozen networks: diffusion prior, conditional samplers,
                   and the MNIST / Stable Diffusion instantiations
    cdm.utils      Gaussian mixtures, exact distances, network building blocks
"""
