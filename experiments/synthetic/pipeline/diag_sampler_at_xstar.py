"""How far is a trained conditional sampler from the truth at x*, and where is the minimum of its loss?

For each sampler: (1) TV between the sampler's draws at x* and G (1-D Y: histogram against the exact
target density; the same estimate on exact draws gives the noise floor); (2) the Table 1 sample loss
(MMD, mean of 5 x 2000 draws) at x* and at the saved endpoints of a search_mog run, with the endpoints'
exact TV; (3) refinement from x* on the sampler's own loss (zeroth-order and pathwise gradients):
exact TV and sampler loss before and after, i.e. where the sampler's loss pulls a design that
starts at the true answer.

  python experiments/synthetic/pipeline/diag_sampler_at_xstar.py --runs-dir runs/table1v2/x10_y1 \
      --samplers models/diffusion_10k/model_conditional.pt:MLGD_r100.json models/cm/model_conditional.pt:MLGD-F_r100.json \
      --out results/diag_sampler_at_xstar_x10_y1.json
"""
import argparse, dataclasses, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import search_mog as fx                                                # noqa: E402
from make_table1 import _load_mixture, scale_free_distances            # noqa: E402
from run_optimization import load_model                                # noqa: E402
from cdm.guidance.core import refine as refine_x, _final_loss          # noqa: E402
from cdm.models.samplers.exact import ExactConditionalSampler          # noqa: E402


def hist_tv(y, m, v, w, bins=400):
    y = y.reshape(-1).cpu().double().numpy()
    mu = np.array([float(a) for a in m]); sd = np.sqrt(np.array([float(torch.as_tensor(a).reshape(-1)[0]) for a in v]))
    wt = np.array([float(a) for a in torch.as_tensor(w).reshape(-1)])
    lo, hi = (mu - 6 * sd).min(), (mu + 6 * sd).max()
    edges = np.linspace(lo, hi, bins + 1); c = 0.5 * (edges[1:] + edges[:-1]); dx = edges[1] - edges[0]
    q = (wt[:, None] * np.exp(-0.5 * ((c[None] - mu[:, None]) / sd[:, None]) ** 2) / (np.sqrt(2 * np.pi) * sd[:, None])).sum(0)
    p, _ = np.histogram(y, bins=edges); p = p / len(y) / dx
    out = ((y < lo) | (y > hi)).mean()                                  # mass outside the window counts fully
    return float(0.5 * np.abs(p - q).sum() * dx + 0.5 * out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", required=True)
    p.add_argument("--samplers", nargs="+", required=True, help="model_path:run_json pairs, relative to --runs-dir")
    p.add_argument("--n-draw", type=int, default=20000)
    p.add_argument("--max-endpoints", type=int, default=100)
    p.add_argument("--refine-steps", type=int, default=300)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    D, dev = a.runs_dir, a.device
    sys.argv = ["search_mog", "--prior", f"{D}/models/uncond/model_unconditional.pt", "--sampler", "x", "--target",
                f"{D}/data.pt", "--device", dev, "--n-cond", "250", "--refine-lr", "0.02", "--refine-n-cond", "1000",
                "--guidance-optim", "adam", "--step-size", "0.05"]
    args = fx.parse_args()
    prior, _ = load_model(args.prior, dev)
    gen, info = fx.load_target(args.target, dev)
    m, v, w = info["target"]
    data = _load_mixture(f"{D}/data.pt")
    xs = info["x_star"].float().view(1, -1).to(dev)
    loss_fn = fx.make_loss(args)
    c0 = torch.load(args.target, map_location="cpu", weights_only=False)["condition_on"]
    res = {"x_star_exact_tv": scale_free_distances(data, xs.view(-1).tolist(), n_mc=20000)["tv"]}
    ex = ExactConditionalSampler(info["mu_list"], info["Sigma_list"], info["alpha"], c0)
    y = ex.sample(a.n_draw, condition_x=xs.repeat(a.n_draw, 1), device=dev)[0]
    res["hist_tv_floor_exact_draws"] = hist_tv(y[:, c0:] if y.shape[1] > 1 else y, m, v, w)
    for spec in a.samplers:
        mp, rj = spec.split(":")
        sampler, arch = load_model(os.path.join(D, mp), dev)
        cfg = fx.make_cfg(args, prior, arch, gen, 0)
        sel = dataclasses.replace(cfg, final_n_cond=2000)
        L = lambda x: float(sum(_final_loss(sampler, loss_fn, sel, x) for _ in range(5)) / 5)   # noqa: E731
        with torch.no_grad():
            out = sampler.sample(nsamples=a.n_draw, condition_x=xs.repeat(a.n_draw, 1), device=dev)
        yy = out[0] if isinstance(out, tuple) else out
        if cfg.condition_on_slice:
            yy = yy[:, sampler.condition_on:]
        r = {"model": mp, "hist_tv_at_xstar": hist_tv(yy, m, v, w),
             "draws_mean_std_at_xstar": [float(yy.mean()), float(yy.std())], "loss_at_xstar": L(xs)}
        ends = []
        for att in json.load(open(os.path.join(D, rj)))["attempts"]:
            for q in att.get("restarts", [att]):
                ends.append(q["x_hat"])
        ends = ends[:a.max_endpoints]
        el = [(L(torch.tensor(x, device=dev).view(1, -1).float()),
               scale_free_distances(data, np.array(x).reshape(-1).tolist(), n_mc=5000)["tv"]) for x in ends]
        el.sort()
        r["endpoints"] = {"n": len(el), "n_loss_below_xstar": sum(l < r["loss_at_xstar"] for l, _ in el),
                          "lowest_loss_5": [[round(l, 5), round(t, 3)] for l, t in el[:5]],
                          "exact_tv_of_lowest_loss": el[0][1]}
        for g in ["zo", "pathwise"]:
            c = dataclasses.replace(cfg, refine_steps=a.refine_steps, refine_grad=g)
            torch.manual_seed(1)
            xr, tr = refine_x(prior, sampler, loss_fn, c, xs.clone())
            xr = xr.detach()
            r[f"refine_from_xstar_{g}"] = {"loss_after": L(xr), "exact_tv_after": scale_free_distances(data, xr.view(-1).tolist(), n_mc=20000)["tv"],
                                           "dist_moved": float((xr - xs).norm())}
        res[mp] = r
        print(json.dumps(r), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == "__main__":
    main()
