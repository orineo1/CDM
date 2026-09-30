"""
cdm/models/sd/metrics.py -- SD-specific metrics, plus the seam to the shared MMD.

MMD is NOT implemented here. `compute_mmd`, `mmd_report` and `compute_witness_scores`
are re-exported from `cdm.models.sd.mmd`, which aliases the SHARED `cdm.losses.functions.compute_mmd`
-- the port of the paper repo's own MMD, SD kernel included. This module keeps the
upstream names so the ported pipeline code reads as it did.

SWD below IS SD-specific and stays. The shared `cdm.losses.functions.sliced_wasserstein_distance`
is a different estimator (per-projection 1-D Wasserstein via `ot.wasserstein_1d`) with a
different signature and return type; SD's is a mean absolute difference of sorted
projections with an adaptive projection count and interpolation for unequal sample sizes.
They are not interchangeable, so SD's is carried here rather than aliased.

Upstream: SD_cond_SD_controlnet/src/metrics.py @ cb8bc91.
"""

import numpy as np
import torch
import torchvision.transforms.functional as TF

from cdm.models.sd.mmd import (  # the shared MMD, under the upstream names
    sd_mmd as compute_mmd,
    mmd_report,
    compute_witness_scores,
    estimate_bandwidth,
)

__all__ = [
    "compute_mmd", "mmd_report", "compute_witness_scores", "estimate_bandwidth",
    "compute_swd", "evaluate_distribution_mmd",
]

def compute_swd(
    x,
    y,
    n_projections=None,
    tol=1e-3,
    min_projections=10,
    step=10,
    max_projections=500,
    seed=None,
):
    """
    Sliced Wasserstein Distance between two sets of embeddings.

    Args:
        x:             [n, d] generated embeddings (grad flows through).
        y:             [m, d] target embeddings (detached).
        n_projections: if given, uses exactly that many projections.
                       Otherwise grows adaptively until convergence.
        tol:           convergence threshold |SWD(n+step) - SWD(n)| < tol.
        min_projections: starting number for adaptive mode.
        step:          projections added per iteration in adaptive mode.
        max_projections: hard cap for adaptive mode.
        seed:          if given, the random projection directions are drawn from a
                       dedicated Generator seeded with this value, so the SAME
                       projections (hence the exact same SWD value) are reproduced
                       across runs regardless of what else has consumed the global
                       torch RNG beforehand. None (default) uses the global RNG
                       unseeded, as before -- not reproducible run to run.

    Returns:
        Scalar SWD estimate.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    dev = x.device
    x = x.float().to(dev)
    y = y.float().to(dev).detach()

    if x.dim() > 2:
        x = x.reshape(x.shape[0], -1)
    if y.dim() > 2:
        y = y.reshape(y.shape[0], -1)

    d = x.shape[1]
    generator = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None

    def _swd_fixed(n_proj):
        projections = torch.randn(n_proj, d, device=dev, generator=generator)
        projections = projections / projections.norm(dim=1, keepdim=True)

        x_proj = projections @ x.T  # [n_proj, n]
        y_proj = projections @ y.T  # [n_proj, m]

        x_sorted = x_proj.sort(dim=1).values
        y_sorted = y_proj.sort(dim=1).values

        if x_sorted.shape[1] != y_sorted.shape[1]:
            y_sorted = torch.nn.functional.interpolate(
                y_sorted.unsqueeze(0),
                size=x_sorted.shape[1],
                mode="linear",
                align_corners=False,
            ).squeeze(0)

        return (x_sorted - y_sorted).abs().mean()

    if n_projections is not None:
        return _swd_fixed(n_projections)

    # Adaptive: grow until converged
    n = min_projections
    prev_swd = _swd_fixed(n)
    while n + step <= max_projections:
        n += step
        curr_swd = _swd_fixed(n)
        delta = (curr_swd - prev_swd).abs()
        print(
            f"      [SWD] n_proj={n}  swd={curr_swd.item():.6f}  "
            f"delta={delta.item():.6f}",
            flush=True,
        )
        if delta.item() < tol:
            print(f"      [SWD] Converged at n_projections={n}", flush=True)
            return curr_swd
        prev_swd = curr_swd

    print(
        f"      [SWD] Reached max_projections={max_projections} without convergence",
        flush=True,
    )
    return prev_swd


def evaluate_distribution_mmd(
    latent,
    architect_vae,
    architect_image_processor,
    sprinter,
    clip_model,
    clip_processor,
    all_clip_embeddings,
    eval_prompt,
    n_eval=10,
    device="cuda",
    controlnet_scale=0.8,
    generator=None,
):
    """
    Full evaluation: latent -> scribble PIL -> sprinter photos -> CLIP -> MMD.

    Args:
        latent:                  Architect latent tensor [1, C, H, W].
        architect_vae:           VAE from the Architect pipeline.
        architect_image_processor: image processor from the Architect pipeline.
        sprinter:                Sprinter pipeline.
        clip_model:              Frozen CLIP model.
        clip_processor:          CLIP processor.
        all_clip_embeddings:     [N, 768] target CLIP embeddings (detached).
        eval_prompt:             Text prompt for Sprinter generation.
        n_eval:                  Number of Sprinter photos to generate.
        device:                  torch device string.
        controlnet_scale:        ControlNet conditioning scale -- must match whatever
                                 scale built the target distribution and whatever scale
                                 the guidance loop (run_dps_step_clip) uses, so eval is
                                 conditioned consistently with the rest of the pipeline
                                 (run_mlgd_f.py passes args.controlnet_scale).
        generator:               torch.Generator for the Sprinter noise. With one, this
                                 evaluation draws nothing from the global RNG, so calling
                                 it (or not) leaves the guided trajectory unchanged.

    Returns:
        (mmd_scalar, eval_photos_list, clip_embs) -- mmd_scalar is
        the value upstream reports, sqrt(|MMD^2| + 1e-8), for display/comparison.
    """
    from cdm.models.sd.clip_utils import encode_images_clip
    from cdm.models.sd.image_utils import latent_to_pil

    with torch.no_grad():
        scribble_pil = latent_to_pil(latent, architect_vae, architect_image_processor)

    original_vae_dtype = sprinter.vae.dtype
    sprinter.vae.to(dtype=torch.float16)
    eval_photos = []
    with torch.no_grad():
        for start in range(0, n_eval, 2):
            bs = min(2, n_eval - start)
            result = sprinter(
                prompt=[eval_prompt] * bs,
                image=[scribble_pil] * bs,
                num_inference_steps=2,
                guidance_scale=0.0,
                controlnet_conditioning_scale=controlnet_scale,
                output_type="pil",
                return_dict=True,
                generator=generator,
            )
            eval_photos.extend(result.images)
    sprinter.vae.to(dtype=original_vae_dtype)

    tensors = [TF.to_tensor(img).unsqueeze(0) for img in eval_photos]
    photo_tensor = torch.cat(tensors, dim=0).to(device)
    clip_model.to(device)
    with torch.no_grad():
        clip_embs = encode_images_clip(photo_tensor, clip_model, clip_processor)
    clip_model.to("cpu")

    # Report-only value: this function exists purely to log/compare an MMD number
    # (called under torch.no_grad(), never feeds a loss). `mmd_report` is a pass-through
    # now that `compute_mmd` returns upstream's square-rooted value directly.
    mmd = mmd_report(compute_mmd(clip_embs, all_clip_embeddings)).item()

    return mmd, eval_photos, clip_embs
