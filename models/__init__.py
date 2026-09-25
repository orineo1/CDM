"""Frozen networks, split by the role they play in CDM.

`models/` holds the networks the method treats as given; `guidance/` and
`losses/` hold the method itself.

    models/diffusion.py          DiffusionModel   -- discrete-time (timestep-indexed
                                 DDPM/DDIM).  "Discrete-time", NOT discrete-state:
                                 there is no discrete-state diffusion in this repo.
    models/diffusion_general.py  DiffusionModel   -- continuous-time SDE formulation.
    models/priors/               P(X):    the unconditional prior over X.
    models/samplers/             P(Y|X):  consistency, exact.

Both modules define a class named `DiffusionModel`, and the same class serves
both roles: `DiffusionModel(condition=True)` is P(Y|X) on the diffusion-sampler
path, which is why the class lives here rather than inside `priors/`.  A package
can export only one name, so:

    models.DiffusionModel     -> models.diffusion.DiffusionModel      (discrete-time)
    models.DiffusionModelCT   -> models.diffusion_general.DiffusionModel (continuous-time)

Both remain reachable by full module path.  `models.diffusion_general` also carries
its own pre-existing in-module alias `DiffusionModelSDE` for the same class.  Every
call site in this repo uses the discrete-time class; `diffusion_general` has no
importers.
"""

from . import diffusion  # noqa: F401
from . import diffusion_general  # noqa: F401

from .diffusion import DiffusionModel  # noqa: F401
from .diffusion_general import DiffusionModel as DiffusionModelCT  # noqa: F401

from . import priors  # noqa: F401,E402
from . import samplers  # noqa: F401,E402
