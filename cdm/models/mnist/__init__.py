"""MNIST rotated-digits experiment: the MNIST-specific half of the code.

Everything here is specific to the MNIST task -- the rotated-MNIST dataset, the
two classifiers, the conditional iCT angle model, the UNet prior, the three
angle targets and the figure code. The method itself is NOT here: the guided
optimiser comes from `cdm/guidance/`, the sliced-Wasserstein loss and the circular
helpers from `cdm/losses/`, the mixture sampling from `cdm/utils/`.

Import it by putting `experiments/mnist/` on `sys.path`; the
entry-point scripts in the sibling experiment directories do that themselves.
"""

__all__ = [
    "checkpoints", "classifier", "cond_model", "dataset",
    "guidance", "plotting", "targets", "uncond_model",
]
