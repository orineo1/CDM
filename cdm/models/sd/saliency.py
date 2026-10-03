# Port of orineo1/conditional-matching-paper SD_cond_SD_controlnet/notebooks/gender_saliency_eval.ipynb,
# reduced to what the paper reports. Run as:
#     bash experiments/stable_diffusion/four_targets/run.sh saliency <scenario> --run_dir <run>
#     python -m cdm.models.sd.saliency --help

"""
saliency.py -- where MLGD-F edits the scribble, and why there (the paper's scribble-edit and saliency figures).

For one finished run (source scribble -> MLGD-F scribble):

1. Scribble change. The signed per-pixel difference (MLGD-F - source, channel mean),
   Gaussian-smoothed (--sigma); a pixel is changed if |smoothed difference| >= --tau, added if
   positive, removed if negative. Reported: % unchanged / added / removed.
2. Gender saliency. |ds/dx| on the source scribble x, with
   s(x) = cos(CLIP(I(x)), t_man) - cos(CLIP(I(x)), t_woman), backpropagated from the image
   I(x) through scribble -> ControlNet + Sprinter -> VAE decoder -> CLIP; |gradient| summed
   over channels, averaged over --n_seeds Sprinter draws, Gaussian-smoothed.
3. Saliency vs change. On the source's stroke pixels: the smoothed L2 change
   against the saliency, binned by saliency percentile; Spearman rho and Pearson r.
4. Latent geometry. On the 64x64 VAE latent grid: the source latent norm against the norm of
   the latent edit (MLGD-F - source); Pearson r and Spearman rho.

Writes to --output_dir (run.sh: <run>/saliency/): saliency_results.json (every number above,
the settings, model revisions, library versions, W&B link), scribble_change.png (source | MLGD-F |
signed change) and saliency_correlation.png (binned change vs saliency | saliency map).
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

MALE_PROMPT      = "a photo of a man"
FEMALE_PROMPT    = "a photo of a woman"
VARIATION_PROMPT = "a superrealistic professional photograph of"
PAPER_FIG_DPI    = 600


# ---------------------------------------------------------------------------
# Scribble change and the saliency-change correlation (numpy only)
# ---------------------------------------------------------------------------

def load_scribble(path, size=512):
    """(float32 HWC array in [0, 1], PIL image), resized to size x size."""
    pil = Image.open(path).convert("RGB").resize((size, size))
    return np.asarray(pil).astype(np.float32) / 255.0, pil


def scribble_change(src, opt, sigma):
    """(change, signed): the smoothed per-pixel L2 change, and the smoothed signed change
    (> 0 where a stroke was added, < 0 where one was removed)."""
    change = gaussian_filter(np.linalg.norm(opt - src, axis=2), sigma=sigma)
    signed = gaussian_filter((opt - src).mean(axis=2), sigma=sigma)
    return change, signed


def change_masks(signed, tau):
    changed = np.abs(signed) >= tau
    return changed, changed & (signed > 0), changed & (signed < 0)


def change_counts(signed, tau):
    """% of pixels unchanged / added / removed (reported with the scribble-change figure)."""
    changed, added, removed = change_masks(signed, tau)
    n = changed.size
    return {"unchanged_pct": 100 * float(n - changed.sum()) / n,
            "added_pct": 100 * float(added.sum()) / n,
            "removed_pct": 100 * float(removed.sum()) / n}


def line_mask(scribble, threshold=0.05):
    """The stroke pixels of a scribble (mean intensity above threshold)."""
    return scribble.mean(axis=2) > threshold


def saliency_change_correlation(saliency, change, scribble, n_bins=20, line_threshold=0.05):
    """On the stroke pixels: saliency vs change, the correlations and the per-bin means
    (bins = saliency percentiles)."""
    m = line_mask(scribble, line_threshold).ravel()
    sal, chg = saliency.ravel()[m], change.ravel()[m]
    rho, p_rho = spearmanr(sal, chg)
    r, p_r = pearsonr(sal, chg)
    edges = np.percentile(sal, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.digitize(sal, edges) - 1, 0, n_bins - 1)
    return {"n_line_pixels": int(m.sum()), "spearman_rho": float(rho), "spearman_p": float(p_rho),
            "pearson_r": float(r), "pearson_p": float(p_r),
            "bin_saliency": [float(sal[idx == k].mean()) for k in range(n_bins)],
            "bin_change": [float(chg[idx == k].mean()) for k in range(n_bins)]}


def latent_correlation(src_norm, edit_norm):
    """Source latent norm vs latent edit magnitude, over the latent grid."""
    if src_norm.std() < 1e-6 or edit_norm.std() < 1e-6:
        return {"pearson_r": None, "spearman_rho": None, "note": "constant map (identical scribbles?)"}
    r, p = pearsonr(src_norm.ravel(), edit_norm.ravel())
    rho, _ = spearmanr(src_norm.ravel(), edit_norm.ravel())
    return {"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho)}


# ---------------------------------------------------------------------------
# The two paper figures
# ---------------------------------------------------------------------------

def plot_scribble_change(src, opt, signed, tau, path):
    """Source | MLGD-F | smoothed signed change (green added, red removed)."""
    changed, _, _ = change_masks(signed, tau)
    vmax = np.abs(signed[changed]).max() if changed.any() else 1.0
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.2), gridspec_kw={"wspace": 0.04})
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(src); axes[1].imshow(opt)
    im = axes[2].imshow(signed, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
    pos = axes[2].get_position()
    plt.colorbar(im, cax=fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])).ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=PAPER_FIG_DPI, bbox_inches="tight"); plt.close(fig)


def plot_saliency_correlation(saliency, corr, scribble, path, line_threshold=0.05):
    """Binned change vs saliency (left), saliency on the stroke pixels (right)."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    masked = np.where(line_mask(scribble, line_threshold), saliency, 0.0)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.2), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.15})
    axes[0].plot(corr["bin_saliency"], corr["bin_change"], "o-", lw=1.5, ms=4)
    axes[0].set_xlabel("Gender Saliency (mean per bin)", fontsize=8)
    axes[0].set_ylabel("Scribble Change (mean)", fontsize=8)
    axes[0].tick_params(labelsize=7); axes[0].grid(True, alpha=0.3, linewidth=0.5)
    axes[0].text(0.97, 0.05, f"Spearman $\\rho$ = {corr['spearman_rho']:.2f}", transform=axes[0].transAxes,
                 ha="right", va="bottom", fontsize=8,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray", linewidth=0.5))
    axes[1].imshow(np.zeros((*saliency.shape, 3), dtype=np.uint8))
    vmax = np.percentile(masked[masked > 0], 90) if (masked > 0).any() else 1.0
    im = axes[1].imshow(masked, cmap="hot", vmin=0, vmax=vmax); axes[1].axis("off")
    cb = plt.colorbar(im, cax=make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05))
    cb.ax.tick_params(labelsize=7)
    fig.savefig(path, dpi=PAPER_FIG_DPI, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
# GPU part
# ---------------------------------------------------------------------------

def latent_norms(src_pil, opt_pil, vae, device):
    """(source latent norm, latent edit norm), each [64, 64]: L2 over the 4 channels."""
    import torch
    import torchvision.transforms.functional as TF

    def encode(pil):
        t = TF.to_tensor(pil).unsqueeze(0).float().to(device) * 2.0 - 1.0   # the pipeline's [-1, 1]
        with torch.no_grad():
            return vae.encode(t.to(vae.dtype)).latent_dist.mean.float() * vae.config.scaling_factor
    src_lat, opt_lat = encode(src_pil), encode(opt_pil)
    return src_lat[0].norm(dim=0).cpu().numpy(), (opt_lat - src_lat)[0].norm(dim=0).cpu().numpy()


def gender_direction(clip_model, clip_processor, device, male=MALE_PROMPT, female=FEMALE_PROMPT):
    """[2, 768] unit text embeddings t_man, t_woman."""
    import torch
    inputs = clip_processor(text=[male, female], return_tensors="pt", padding=True).to(device)
    clip_model.to(device)
    with torch.no_grad():
        out = clip_model.get_text_features(**inputs)
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output   # transformers 5.x: output object
    return emb / emb.norm(dim=-1, keepdim=True)


def compute_saliency(scribble, sprinter, clip_model, clip_processor, g_embs, device,
                     n_seeds=20, sigma=2.5, prompt=VARIATION_PROMPT, controlnet_scale=0.5, seed=0):
    """|ds/dx| summed over channels, averaged over n_seeds Sprinter draws (torch seeds
    seed .. seed + n_seeds - 1), smoothed. CLIP embeddings are unit-norm, so the dot
    products are the cosines of s(x)."""
    import torch
    import torch.utils.checkpoint
    from cdm.models.sd.clip_utils import encode_images_clip

    vae = sprinter.vae
    sf = vae.config.scaling_factor
    accum = np.zeros(scribble.shape[:2], dtype=np.float64)

    def forward(ctrl):
        lat = sprinter(prompt=[prompt], image=ctrl, num_inference_steps=2, guidance_scale=0.0,
                       controlnet_conditioning_scale=controlnet_scale, output_type="latent",
                       return_dict=True).images
        px = vae.decode((lat.float() / sf).to(vae.dtype)).sample
        px = torch.clamp((px.float() + 1.0) / 2.0, 0.0, 1.0)
        with torch.amp.autocast("cuda", enabled=False):
            return encode_images_clip(px.float(), clip_model, clip_processor)

    for k in range(n_seeds):
        torch.manual_seed(seed + k)
        x = torch.tensor(scribble, device=device, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        x.requires_grad_(True)
        emb = torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=False)
        score = (emb @ g_embs[0]) - (emb @ g_embs[1])
        score.sum().backward()
        g = x.grad.detach().abs().sum(dim=1)[0].cpu().numpy()
        print(f"  seed {seed + k}: s={score.item():+.4f}  |grad| sum={g.sum():.6f}", flush=True)
        accum += g.astype(np.float64)
        del x, emb, score
        torch.cuda.empty_cache()
    return gaussian_filter(accum / n_seeds, sigma=sigma)


def _first(d, *names):
    for n in names:
        p = os.path.join(d, n)
        if os.path.exists(p):
            return p
    return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, required=True,
                   help="a finished run (scribble.png + final_scribble_mlgd_f.png)")
    p.add_argument("--output_dir", type=str, default=None, help="default: <run_dir>/saliency")
    p.add_argument("--n_seeds", type=int, default=20, help="Sprinter draws averaged into the saliency map")
    p.add_argument("--sigma", type=float, default=2.5, help="Gaussian smoothing (pixels)")
    p.add_argument("--tau", type=float, default=0.1, help="changed pixel: |smoothed signed change| >= tau")
    p.add_argument("--line_threshold", type=float, default=0.05, help="stroke pixel: mean intensity above this")
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--male_prompt", type=str, default=MALE_PROMPT)
    p.add_argument("--female_prompt", type=str, default=FEMALE_PROMPT)
    p.add_argument("--variation_prompt", type=str, default=VARIATION_PROMPT)
    p.add_argument("--controlnet_scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0, help="first torch seed of the saliency draws (notebook: 0)")
    p.add_argument("--no_deterministic", action="store_true")
    for m in ("architect", "sprinter", "controlnet", "hed", "clip"):
        p.add_argument(f"--{m}_revision", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="sdxl-saliency")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from cdm.models.sd.clip_utils import CLIP_MODEL_ID, load_clip_model
    from cdm.models.sd.models import (DEFAULT_CONTROLNET_MODEL_ID, DEFAULT_SPRINTER_MODEL_ID,
                                      freeze_module, load_models, make_deterministic, resolved_revision)
    from cdm.models.sd.run_mlgd_f import library_versions

    src_path = _first(args.run_dir, "scribble.png", "scribble_source.png")
    opt_path = _first(args.run_dir, "final_scribble_mlgd_f.png", "scribble_mlgdd.png")
    if not (src_path and opt_path):
        sys.exit(f"{args.run_dir}: needs scribble.png and final_scribble_mlgd_f.png")
    out_dir = args.output_dir or os.path.join(args.run_dir, "saliency")
    os.makedirs(out_dir, exist_ok=True)
    out = lambda name: os.path.join(out_dir, name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.no_deterministic:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
    else:
        make_deterministic(args.seed)
    t0 = time.time()

    # ── 1. scribble change (CPU) ─────────────────────────────────────────────
    src, src_pil = load_scribble(src_path)
    opt, opt_pil = load_scribble(opt_path)
    change, signed = scribble_change(src, opt, args.sigma)
    counts = change_counts(signed, args.tau)
    print("Scribble change: " + ", ".join(f"{k} {v:.1f}" for k, v in counts.items()), flush=True)
    plot_scribble_change(src, opt, signed, args.tau, out("scribble_change.png"))

    # ── models ───────────────────────────────────────────────────────────────
    architect, sprinter = load_models(device, sprinter_revision=args.sprinter_revision,
                                      controlnet_revision=args.controlnet_revision,
                                      architect_revision=args.architect_revision)
    del architect
    torch.cuda.empty_cache()
    clip_model, clip_processor = load_clip_model(device, revision=args.clip_revision)
    sprinter.unet.enable_gradient_checkpointing()
    sprinter.controlnet.enable_gradient_checkpointing()
    for m in (sprinter.unet, sprinter.controlnet, sprinter.vae):
        freeze_module(m)
    sprinter.vae.to(dtype=torch.float32)   # gradients flow through the decode
    model_revisions = {
        DEFAULT_SPRINTER_MODEL_ID: resolved_revision(DEFAULT_SPRINTER_MODEL_ID, args.sprinter_revision),
        DEFAULT_CONTROLNET_MODEL_ID: resolved_revision(DEFAULT_CONTROLNET_MODEL_ID, args.controlnet_revision),
        CLIP_MODEL_ID: resolved_revision(CLIP_MODEL_ID, args.clip_revision),
    }
    run = None
    if not args.no_wandb:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity or None,
                         config={**vars(args), "source": src_path, "mlgd_f": opt_path,
                                 "model_revisions": model_revisions, "versions": library_versions()})

    # ── 4. latent geometry ───────────────────────────────────────────────────
    latent = latent_correlation(*latent_norms(src_pil, opt_pil, sprinter.vae, device))
    print(f"Latent norm vs edit: {latent}", flush=True)

    # ── 2-3. gender saliency of the source scribble, and its correlation ─────
    g = gender_direction(clip_model, clip_processor, device, args.male_prompt, args.female_prompt)
    print(f"Gender saliency of the source scribble ({args.n_seeds} draws)...", flush=True)
    saliency = compute_saliency(src, sprinter, clip_model, clip_processor, g, device,
                                n_seeds=args.n_seeds, sigma=args.sigma, prompt=args.variation_prompt,
                                controlnet_scale=args.controlnet_scale, seed=args.seed)
    corr = saliency_change_correlation(saliency, change, src, args.n_bins, args.line_threshold)
    plot_saliency_correlation(saliency, corr, src, out("saliency_correlation.png"), args.line_threshold)
    print(f"Saliency vs change on {corr['n_line_pixels']} stroke pixels: Spearman rho={corr['spearman_rho']:.3f}, "
          f"Pearson r={corr['pearson_r']:.3f}", flush=True)

    # ── save ─────────────────────────────────────────────────────────────────
    record = {"run_dir": os.path.abspath(args.run_dir), "source": src_path, "mlgd_f": opt_path,
              "scribble_change": counts, "saliency_vs_change": corr, "latent_norm_vs_edit": latent,
              "settings": {k: v for k, v in vars(args).items() if not k.endswith("_revision")},
              "deterministic": not args.no_deterministic, "minutes": (time.time() - t0) / 60,
              "model_revisions": model_revisions, "versions": library_versions(),
              "wandb": ({"project": run.project, "name": run.name, "id": run.id, "url": run.url} if run else None)}
    with open(out("saliency_results.json"), "w") as f:
        json.dump(record, f, indent=2)
    if run:
        import wandb
        wandb.log({f: wandb.Image(out(f + ".png")) for f in ("scribble_change", "saliency_correlation")})
        wandb.summary.update({**{f"scribble_change/{k}": v for k, v in counts.items()},
                              "saliency_vs_change/spearman_rho": corr["spearman_rho"],
                              "saliency_vs_change/pearson_r": corr["pearson_r"],
                              "latent_norm_vs_edit/pearson_r": latent["pearson_r"]})
        wandb.finish()
    print(f"Saved to {out_dir} ({record['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
