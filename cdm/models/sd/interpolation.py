# Port of orineo1/conditional-matching-paper SD_cond_SD_controlnet/notebooks/
# eval_scribbl_interpolation.ipynb and eval_scribbl_interpolation_age.ipynb, minus the MMD,
# the classifier and the sample rows: the functions behind the interpolation figures, to call
# from experiments/stable_diffusion/interpolation/interpolation.ipynb.

"""
interpolation.py -- where do the portraits of one scribble fall along the target axis?

Gender: the axis is PC1 of a PCA fitted on the two extreme groups (first and last of the
target prompts, Woman and Man), oriented feminine -> masculine.
Age: the axis is the unit vector from the mean age-40 embedding to the mean age-79
embedding; the second dimension is PC1 of the residuals, as in the paper's figures.

    target = build_target(...)          # {group: CLIP embeddings}, generated on the scribble
    photos, embs = eval_photos(...)     # N portraits of the scribble + embeddings
    axis = gender_axis(target) | age_axis(target, young, old)
    scores, coords = axis.project(embs)
    sorted_grid(photos, scores, picks=None or [ranks], ...)
    scatter(axis, target, embs, ...)
"""

import json
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

from cdm.models.sd.clip_utils import encode_images_clip
from cdm.models.sd.generation import generate_and_store_cs

EVAL_PROMPT = "a superrealistic professional photograph of"

GENDER_GROUPS = [   # the paper's 4-class gender-interpolation target, feminine -> masculine
    ("Woman",     "superrealistic portrait photograph of a woman, extremely feminine features, studio lighting"),
    ("WomanMasc", "a superrealistic portrait photograph of a woman with masculine features, heavy brow ridge, studio lighting"),
    ("ManFem",    "a superrealistic portrait photograph of a man with extremely feminine features, soft delicate face, high cheekbones, studio lighting"),
    ("Man",       "a superrealistic portrait photograph of a man, extremely masculine features, studio lighting"),
]


def age_prompt(age, gender="man"):
    return (f"a superrealistic portrait photograph of a {age}-year-old {gender}, "
            "studio lighting, sharp focus, photographic")


# ---------------------------------------------------------------------------
# Generation + encoding
# ---------------------------------------------------------------------------

def _embed(images, clip_model, clip_processor, device, batch=32):
    clip_model.to(device)
    out = []
    with torch.no_grad():
        for s in range(0, len(images), batch):
            t = torch.cat([TF.to_tensor(im).unsqueeze(0) for im in images[s:s + batch]]).to(device)
            out.append(encode_images_clip(t, clip_model, clip_processor).cpu())
    clip_model.to("cpu")
    return torch.cat(out, dim=0).numpy()


def generate(sprinter, scribble, prompt, n, seed, cn_scale=0.5):
    """n Sprinter portraits of `scribble` from its own seeded generator."""
    gen = torch.Generator(device=sprinter.device).manual_seed(seed)
    imgs, _ = generate_and_store_cs(sprinter, prompt, scribble, n, batch_size=2,
                                    cn_scale=cn_scale, generator=gen)
    return imgs


def build_target(sprinter, clip_model, clip_processor, scribble, groups, n_per_group, seed,
                 device, cn_scale=0.5, seed_step=1000):
    """{group name: [n, 768] embeddings}, one prompt per group, generated on `scribble`.
    `groups`: [(name, prompt)]; `n_per_group`: int or {name: n}. Seeds seed + i*seed_step."""
    target = {}
    for i, (name, prompt) in enumerate(groups):
        n = n_per_group[name] if isinstance(n_per_group, dict) else n_per_group
        print(f"  target [{name}] n={n}", flush=True)
        imgs = generate(sprinter, scribble, prompt, n, seed + i * seed_step, cn_scale)
        target[name] = _embed(imgs, clip_model, clip_processor, device)
    return target


def eval_photos(sprinter, clip_model, clip_processor, scribble, n, seed, device,
                prompt=EVAL_PROMPT, cn_scale=0.5):
    """(photos, [n, 768] embeddings) of `scribble` under the neutral prompt."""
    imgs = generate(sprinter, scribble, prompt, n, seed, cn_scale)
    return imgs, _embed(imgs, clip_model, clip_processor, device)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------

class Axis:
    """A 1-D score (position along the target axis) and 2-D coordinates for plotting."""

    def __init__(self, score_fn, coords_fn, left, right, name):
        self._score, self._coords = score_fn, coords_fn
        self.left, self.right, self.name = left, right, name

    def project(self, embs):
        embs = np.asarray(embs, dtype=np.float64)
        return self._score(embs), self._coords(embs)


def gender_axis(target, low=None, high=None):
    """PCA(2) fitted on the two extreme groups (default: first and last of `target`),
    PC1 oriented so `high` (masculine) is positive."""
    names = list(target)
    low, high = low or names[0], high or names[-1]
    pca = PCA(n_components=2, svd_solver="full")
    pca.fit(np.vstack([target[low], target[high]]))
    flip = 1.0 if target[high].mean(0) @ pca.components_[0] > target[low].mean(0) @ pca.components_[0] else -1.0
    coords = lambda e: pca.transform(e) * np.array([flip, 1.0])
    return Axis(lambda e: coords(e)[:, 0], coords, left=low, right=high, name="PC1 (Woman -> Man)")


def age_axis(target, young, old):
    """Unit vector from the mean `young` embedding to the mean `old` one; y = PC1 of the
    residuals of all target embeddings after removing that axis."""
    young, old = str(young), str(old)
    axis = target[old].mean(0) - target[young].mean(0)
    axis = axis / np.linalg.norm(axis)
    all_t = np.vstack(list(target.values()))
    resid = all_t - np.outer(all_t @ axis, axis)
    pca2 = PCA(n_components=1, svd_solver="full").fit(resid)

    def coords(e):
        x = e @ axis
        y = pca2.transform(e - np.outer(x, axis))[:, 0]
        return np.stack([x, y], axis=1)
    return Axis(lambda e: e @ axis, coords, left=young, right=old, name=f"age axis ({young} -> {old})")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def even_picks(n, n_picks):
    """Ranks spaced evenly from first to last of a sorted list of n."""
    return [int(round(i * (n - 1) / (n_picks - 1))) for i in range(n_picks)] if n_picks > 1 else [0]


def ranked_strip(photos, scores, save_path=None, thumb=1.1, per_row=20):
    """Every photo, ordered by score, labelled with its rank: the sheet to pick from by eye."""
    order = np.argsort(scores)
    n = len(order)
    rows = int(np.ceil(n / per_row))
    fig, axes = plt.subplots(rows, per_row, figsize=(per_row * thumb, rows * (thumb + 0.25)))
    axes = np.atleast_2d(axes)
    for k, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if k < n:
            ax.imshow(photos[order[k]])
            ax.set_title(f"{k}  ({scores[order[k]]:.2f})", fontsize=6)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=90)
    return fig


def sorted_grid(photos, scores, picks=None, n_picks=16, left="", right="", title=None,
                show_scores=True, save_path=None):
    """
    n_picks photos along the axis, in two rows. `picks=None`: ranks evenly spaced from the
    lowest to the highest score. `picks=[ranks]`: your own choice, as ranks into the
    sorted order (0 = lowest score), e.g. after looking at ranked_strip().
    Returns (fig, picks, indices): indices are the photo indices actually shown.
    """
    order = np.argsort(scores)
    picks = list(picks) if picks is not None else even_picks(len(order), n_picks)
    idx = [int(order[r]) for r in picks]
    cols = int(np.ceil(len(idx) / 2))
    fig, axes = plt.subplots(2, cols, figsize=(cols * 1.6, 3.6))
    axes = np.atleast_2d(axes)
    for ax in axes.flatten():
        ax.axis("off")
    for ax, r, i in zip(axes.flatten(), picks, idx):
        ax.imshow(photos[i])
        if show_scores:
            ax.set_title(f"#{r}  {scores[i]:.2f}", fontsize=7)
    if left:
        axes[0, 0].text(0, -0.08, f"<- {left}", transform=axes[0, 0].transAxes, fontsize=9, va="top")
    if right:
        axes[0, -1].text(1, -0.08, f"{right} ->", transform=axes[0, -1].transAxes, fontsize=9, va="top", ha="right")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig, picks, idx


def scatter(axis, target, embs, color_by=None, cmap=None, colorbar_label=None, save_path=None):
    """
    The target groups (circles) and the scribble's portraits (gold) in the axis' 2-D
    coordinates, the paper's PCA figure. `color_by`: None colours the target circles by
    their own axis score (gender); a {group: value} colours them by that value (age).
    """
    cmap = cmap or (cm.plasma if color_by else cm.RdBu_r)
    t_coords = {g: axis.project(e)[1] for g, e in target.items()}
    all_t = np.vstack(list(t_coords.values()))
    e_scores, e_coords = axis.project(embs)
    if color_by is None:
        c = np.concatenate([axis.project(e)[0] for e in target.values()])
        vmin, vmax = min(c.min(), e_scores.min()), max(c.max(), e_scores.max())
    else:
        c = np.concatenate([[color_by[g]] * len(e) for g, e in target.items()])
        vmin, vmax = min(color_by.values()), max(color_by.values())
    fig, ax = plt.subplots(figsize=(14, 5))
    sc = ax.scatter(all_t[:, 0], all_t[:, 1], c=c, cmap=cmap, vmin=vmin, vmax=vmax,
                    s=100, alpha=0.6, edgecolors="black", linewidths=0.5, label="target")
    ax.scatter(e_coords[:, 0], e_coords[:, 1], c="gold", s=150, alpha=1.0,
               edgecolors="black", linewidths=0.8, zorder=10, label="scribble's portraits")
    if colorbar_label:
        fig.colorbar(sc, ax=ax, pad=0.01).set_label(colorbar_label, fontsize=12)
    ax.set_xlabel("PC1", fontsize=18, labelpad=10); ax.set_ylabel("PC2", fontsize=18, labelpad=10)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.tick_params(axis="both", length=0)
    ax.grid(True, linestyle="--", alpha=0.3); ax.set_axisbelow(True); ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130)
    return fig


def save_run(out_dir, scenario, scribble_path, picks, indices, scores, embs, target, extra=None):
    """picks.json (what the figure shows) and embeddings.npz (to redraw without a GPU)."""
    os.makedirs(out_dir, exist_ok=True)
    order = np.argsort(scores)
    with open(os.path.join(out_dir, "picks.json"), "w") as f:
        json.dump({"scenario": scenario, "scribble": scribble_path, "picks_ranks": [int(r) for r in picks],
                   "photo_indices": [int(i) for i in indices],
                   "scores": [float(scores[i]) for i in indices],
                   "ranking": [int(i) for i in order], **(extra or {})}, f, indent=2)
    np.savez_compressed(os.path.join(out_dir, "embeddings.npz"), eval=embs, scores=scores,
                        **{f"target_{g}": e for g, e in target.items()})
