"""Figure code for the MNIST rotated-digits experiment.

Ported from `MNIST/visualize.py` of `orineo1/conditional-matching-paper`
(branch `main`, MNIST tree at 7c856e2): the per-seed image rows, the top-k
image panel, and the cartesian + polar histograms of P(Y | X = x_hat) against
the target density.

The source module imported `mog_pdf`, `classify_generated_images`,
`sliced_wasserstein_distance` and `set_seed` from `run_mlgdf.py`, making the
figure code depend on the runner. That import is gone: the classifier call
comes from `cdm.models.mnist.classifier`, the angle helpers from `cdm.losses.circular`,
and nothing here computes a distance.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from cdm.losses.circular import circular_to_angles

from .classifier import classify_generated_images

ICT_TS = [150.0, 50.0, 20.0, 10.0, 5.0, 1.0]


def clamp_normalize(payload):
    """Map each stored image from [-1, 1] to [0, 1] for display."""
    fixed = []
    for img in payload["results"]:
        t = torch.as_tensor(np.asarray(img), dtype=torch.float32).clamp(-1.0, 1.0)
        fixed.append(((t + 1.0) / 2.0).numpy())
    payload = dict(payload)
    payload["results"] = fixed
    return payload


def plot_all_images(payload, plots_dir, ncols=5, dpi=100,
                    save_no_title=False, classifier=None, device=None):
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    n     = len(results)
    nrows = math.ceil(n / ncols)
    preds, _ = (classify_generated_images(results, classifier, device, threshold=0.75)
                if classifier is not None else ([None]*n, [None]*n))

    for row in range(nrows):
        start, end  = row * ncols, min(row * ncols + ncols, n)
        row_results = results[start:end]
        row_losses  = loss_log[start:end]
        row_seeds   = seed_log[start:end]
        row_preds   = preds[start:end]
        n_in_row    = len(row_results)

        def _make(show_titles):
            fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3, 3),
                                     gridspec_kw=dict(wspace=0.02))
            axes = np.array(axes).reshape(ncols)
            for c, (img, loss, seed, pred) in enumerate(
                    zip(row_results, row_losses, row_seeds, row_preds)):
                axes[c].imshow(img, cmap='gray')
                if show_titles:
                    title = f'Seed {seed} | Loss {loss:.4f}'
                    if classifier is not None:
                        title += f' | {str(pred) if pred is not None else "None"}'
                    axes[c].set_title(title, fontsize=11, pad=2)
                axes[c].axis('off')
            for c in range(n_in_row, ncols):
                axes[c].axis('off')
            plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
            return fig

        base = os.path.join(plots_dir, f'{experiment_name}_all_row{row+1}.png')
        fig  = _make(show_titles=True)
        fig.savefig(base, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {base}')

        if save_no_title:
            base_nt = os.path.join(plots_dir, f'{experiment_name}_all_row{row+1}_notitle.png')
            fig_c   = _make(show_titles=False)
            fig_c.savefig(base_nt, dpi=dpi, bbox_inches='tight')
            plt.close(fig_c)


def plot_all_seeds_grid(payload, plots_dir, ncols=5, dpi=100, indices=None,
                        suffix="all_seeds", show_conf=False, title=None):
    """Seeds of one target in a single grid, titled 'Seed i | Loss x | label'.

    The label is the digit the run's classifier assigned (`predicted_labels` in the
    run record, the same labels Table 2 counts); "None" when it was below the
    threshold. Records without labels get titles without the label field.

    indices   draw only these seeds, in this order (e.g. the top-5 by SWD);
              default all seeds in seed order
    suffix    file name suffix: <experiment_name>_<suffix>.png
    show_conf append the classifier's top-class confidence, 'label (0.93)'
    title     optional figure title
    """
    results, loss_log, seed_log = payload["results"], payload["loss_log"], payload["seed_log"]
    preds = payload.get("predicted_labels")
    confs = payload.get("predicted_confidences")
    order = list(range(len(results))) if indices is None else list(indices)
    n = len(order)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.3 + (0.3 if title else 0)),
                             squeeze=False)
    for slot, ax in enumerate(axes.flat):
        ax.axis("off")
        if slot >= n:
            continue
        k = order[slot]
        ax.imshow(results[k], cmap="gray", vmin=0.0, vmax=1.0)
        text = f"Seed {seed_log[k]} | Loss {loss_log[k]:.4f}"
        if preds is not None:
            text += f" | {preds[k] if preds[k] is not None else 'None'}"
            if show_conf and confs is not None:
                text += f" ({confs[k]:.2f})"
        ax.set_title(text, fontsize=8, pad=2)
    fig.tight_layout(pad=0.3, h_pad=1.2, w_pad=0.3,
                     rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()) if title else None)
    if title:
        fig.suptitle(title, fontsize=9)
    path = os.path.join(plots_dir, f"{payload['experiment_name']}_{suffix}.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def plot_top_k_images(payload, plots_dir, top_k=5, dpi=100, save_no_title=False):
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    k      = min(top_k, len(results))
    top_ix = np.argsort(loss_log)[:k]
    ncols  = min(5, k)
    nrows  = math.ceil(k / ncols)

    def _make(show_titles):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 3, nrows * 3),
                                 gridspec_kw=dict(wspace=0.02, hspace=0.1))
        axes = np.array(axes).reshape(nrows, ncols)
        for rank, idx in enumerate(top_ix):
            r, c = divmod(rank, ncols)
            axes[r, c].imshow(results[idx], cmap='gray')
            if show_titles:
                axes[r, c].set_title(
                    f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                    fontsize=7, pad=2)
            axes[r, c].axis('off')
        for rank in range(k, nrows * ncols):
            r, c = divmod(rank, ncols)
            axes[r, c].axis('off')
        if show_titles:
            plt.suptitle(f'[{experiment_name}] Top {k} Images',
                         fontsize=12, fontweight='bold', y=1.002)
        plt.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)
        return fig

    base = os.path.join(plots_dir, f'{experiment_name}_top{k}.png')
    fig  = _make(show_titles=True)
    fig.savefig(base, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {base}')

    if save_no_title:
        base_nt = os.path.join(plots_dir, f'{experiment_name}_top{k}_notitle.png')
        fig_c   = _make(show_titles=False)
        fig_c.savefig(base_nt, dpi=dpi, bbox_inches='tight')
        plt.close(fig_c)

    return top_ix


def plot_top_k_distributions(payload, model_cond, top_ix, plots_dir,
                              top_k_dist=5, dpi=150, save_no_title=False,
                              device='cpu', n_cond_plot=500):
    results, loss_log, seed_log = (
        payload['results'], payload['loss_log'], payload['seed_log']
    )
    experiment_name = payload['experiment_name']
    x_range_np      = payload['x_range']
    target_pdf_np   = payload['target_pdf']

    dist_k  = min(top_k_dist, len(top_ix))
    dist_ix = top_ix[:dist_k]

    all_max_y = target_pdf_np.max()
    temp_angs = []
    for idx in dist_ix:
        cond_t = (torch.tensor(results[idx], dtype=torch.float32)
                  .flatten().unsqueeze(0))
        ang = circular_to_angles(
            model_cond.sample(nsamples=n_cond_plot, condition_x=cond_t,
                              ts=ICT_TS)[0]
        )
        temp_angs.append(ang)
        h, _ = np.histogram(ang.detach().cpu().numpy(),
                             bins=30, range=(0, 360), density=True)
        all_max_y = max(all_max_y, h.max() if len(h) else 0)
    y_lim = all_max_y * 1.1

    target_rad_x        = np.deg2rad(x_range_np)
    target_rad_y        = target_pdf_np * (360.0 / (2 * np.pi))
    target_rad_x_closed = np.append(target_rad_x, target_rad_x[0])
    target_rad_y_closed = np.append(target_rad_y, target_rad_y[0])

    local_polar_max = target_rad_y.max()
    for ang in temp_angs:
        counts, _ = np.histogram(
            np.deg2rad(ang.detach().cpu().numpy()),
            bins=np.linspace(0, 2 * np.pi, 37), density=True)
        local_polar_max = max(local_polar_max, counts.max() if len(counts) else 0)
    polar_ylim = local_polar_max * 1.1

    def _make_cart(show_titles, show_legend):
        fig, cart_axes = plt.subplots(1, dist_k, figsize=(dist_k * 4, 4), sharey=True)
        cart_axes = np.array(cart_axes).reshape(dist_k)
        for rank, (idx, ang) in enumerate(zip(dist_ix, temp_angs)):
            ang_np = ang.detach().cpu().numpy()
            ax     = cart_axes[rank]
            ax.hist(ang_np, bins=30, alpha=0.6, color='skyblue', edgecolor='black',
                    range=(0, 360), density=True, label='Sampled')
            ax.plot(x_range_np, target_pdf_np, color='orange', linewidth=2, label='Target')
            ax.fill_between(x_range_np, target_pdf_np, alpha=0.25, color='orange')
            ax.set_xlim(0, 360); ax.set_ylim(0, y_lim)
            ax.set_xlabel('Angle (°)', fontsize=13)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(axis='x', labelsize=12); ax.tick_params(axis='y', labelsize=11)
            ax.grid(True, alpha=0.3)
            if show_titles:
                ax.set_title(f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                             fontsize=9)
            if rank == 0:
                ax.set_ylabel('Density', fontsize=12)
                if show_legend:
                    ax.legend(fontsize=8)
        if show_titles:
            fig.suptitle(f'[{experiment_name}] Top {dist_k} Distributions — Cartesian',
                         fontsize=13, fontweight='bold')
        fig.tight_layout()
        return fig

    def _make_polar(show_titles, show_legend):
        fig, polar_axes = plt.subplots(1, dist_k, figsize=(dist_k * 4, 4),
                                       subplot_kw={'projection': 'polar'})
        polar_axes = np.array(polar_axes).reshape(dist_k)
        for rank, (idx, ang) in enumerate(zip(dist_ix, temp_angs)):
            ang_rad   = np.deg2rad(ang.detach().cpu().numpy())
            bin_edges = np.linspace(0, 2 * np.pi, 37)
            counts, _ = np.histogram(ang_rad, bins=bin_edges, density=True)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            width       = bin_edges[1] - bin_edges[0]
            ax = polar_axes[rank]
            ax.bar(bin_centers, counts, width=width, alpha=0.55,
                   color='skyblue', edgecolor='black', label='Sampled', zorder=2)
            ax.fill(target_rad_x_closed, target_rad_y_closed,
                    color='orange', alpha=0.35, zorder=3)
            ax.plot(target_rad_x_closed, target_rad_y_closed,
                    color='orange', linewidth=2.5, zorder=4, label='Target')
            ax.set_theta_zero_location('N'); ax.set_theta_direction(-1)
            ax.set_yticklabels([]); ax.set_yticks([])
            ax.tick_params(axis='x', labelsize=12)
            ax.set_ylim(0, polar_ylim)
            if show_titles:
                ax.set_title(f'Rank {rank+1} | Loss {loss_log[idx]:.4f} | Seed {seed_log[idx]}',
                             fontsize=9, pad=14)
            if rank == 0 and show_legend:
                ax.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.35, 1.15))
        if show_titles:
            fig.suptitle(f'[{experiment_name}] Top {dist_k} Distributions — Polar',
                         fontsize=13, fontweight='bold')
        fig.tight_layout()
        return fig

    base_cart  = os.path.join(plots_dir, f'{experiment_name}_dist_cart.png')
    base_polar = os.path.join(plots_dir, f'{experiment_name}_dist_polar.png')
    _make_cart(True, True).savefig(base_cart,  dpi=dpi, bbox_inches='tight'); plt.close()
    _make_polar(True, True).savefig(base_polar, dpi=dpi, bbox_inches='tight'); plt.close()
    print(f'  Saved: {base_cart}')
    print(f'  Saved: {base_polar}')

    if save_no_title:
        _make_cart(False, False).savefig(
            base_cart.replace('.png', '_notitle.png'),  dpi=dpi, bbox_inches='tight'); plt.close()
        _make_polar(False, False).savefig(
            base_polar.replace('.png', '_notitle.png'), dpi=dpi, bbox_inches='tight'); plt.close()

