"""Locating and loading the two frozen MNIST networks.

Both checkpoints live in the HuggingFace repo named in the paper's appendix,
under the `MNIST/` subfolder. A local path always wins, so a run can be pointed
at `experiments/mnist/checkpoints/` (see its README) or at a
scratch directory without touching the network.
"""

import os

import torch

HF_REPO_ID = "anon-submission-cdm/cdm-inverse-design"
HF_SUBFOLDER = "MNIST"
COND_FILENAME = "MnistConditional500Epoch.pt"
UNCOND_FILENAME = "MnistUncond100Epoch.pth"


def resolve_checkpoint(filename, local_dir=None):
    """Return a path to `filename`: local_dir first, then the HF download."""
    if local_dir:
        candidate = os.path.join(local_dir, filename)
        if os.path.exists(candidate):
            return candidate
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename, subfolder=HF_SUBFOLDER,
        token=os.environ.get("HF_TOKEN") or None,
    )


def load_cond_model(device="cpu", local_dir=None, path=None):
    """The conditional iCT angle sampler f_phi, in eval mode."""
    from .cond_model import CircularAngleConsistencyModel
    path = path or resolve_checkpoint(COND_FILENAME, local_dir)
    model = CircularAngleConsistencyModel(
        nfeatures=2, img_features=784, eps=0.002,
        nunits=128, depth=5, device=str(device))
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_uncond_model(device="cpu", local_dir=None, path=None, allow_untrained=False):
    """The unconditional DDPM prior, in eval mode.

    `allow_untrained=True` returns a freshly initialised UNet when no checkpoint
    is reachable. That is for the smoke tier only: it exercises the code path
    and nothing else.
    """
    from .uncond_model import UnconditionalUnet
    model = UnconditionalUnet().to(device)
    try:
        path = path or resolve_checkpoint(UNCOND_FILENAME, local_dir)
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        if not allow_untrained:
            raise
        print("[checkpoints] no unconditional checkpoint reachable — "
              "using an UNTRAINED UNet (smoke tier only)")
    model.eval()
    return model


def make_noise_scheduler():
    """The DDPM training schedule the prior was trained with (Table 12)."""
    from diffusers import DDPMScheduler
    return DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
