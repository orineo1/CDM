"""Frozen networks, split by the role they play in CDM.

`models/` holds the networks the method treats as given; `cdm/guidance/` and
`cdm/losses/` hold the method itself.

    models/diffusion.py   DiffusionModel -- discrete-time (timestep-indexed
                          DDPM/DDIM).  "Discrete-time", NOT discrete-state:
                          there is no discrete-state diffusion in this repo.
    models/priors/        P(X):    the unconditional prior over X.
    models/samplers/      P(Y|X):  consistency, exact.

The same `DiffusionModel` class serves both roles: `DiffusionModel(condition=True)`
is P(Y|X) on the diffusion-sampler path, which is why the class lives here
rather than inside `priors/`.
"""

from . import diffusion  # noqa: F401

from .diffusion import DiffusionModel  # noqa: F401

from . import priors  # noqa: F401,E402
from . import samplers  # noqa: F401,E402
