"""TV of trained conditional samplers to the exact conditional P(Y | X = x), at x* and at random x from the joint.

Only for dim Y = 1 (histogram TV against the exact density, diag_sampler_at_xstar.hist_tv). Prints, per
model and DDIM eta, the TV at x* and the mean and maximum over --n-random draws of x.

  python experiments/synthetic/pipeline/check_sampler_accuracy.py --data runs/table1v2/x10_y1/data.pt \
      --models runs/night/x10_y1/models/d256/model_conditional.pt runs/table1v2/x10_y1/models/cm/model_conditional.pt \
      --etas 0 0.5 1 --out results/sampler_accuracy_x10_y1.json
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from run_optimization import load_model                     # noqa: E402
from diag_sampler_at_xstar import hist_tv                   # noqa: E402
from cdm.utils.dist import compute_conditionals, compute_alpha, generate_mog_samples_not_differentiable  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--etas", nargs="+", type=float, default=[0.0], help="DDIM eta values (diffusion samplers only)")
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--n-draw", type=int, default=20000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    d = torch.load(a.data, map_location="cpu", weights_only=False)
    mu, Sig, al, c0 = d["mu_list"], d["Sigma_list"], d["alpha"], d["condition_on"]
    torch.manual_seed(0)
    J = generate_mog_samples_not_differentiable(a.n_random, mu, Sig, al).float()
    X = torch.cat([torch.as_tensor(d["x_star"]).float().view(1, -1), J[:, :c0]], 0)

    def exact(x):
        m, S = compute_conditionals(mu, Sig, x.reshape(-1, 1))
        w = compute_alpha(mu, Sig, al, x.flatten())
        return ([float(torch.as_tensor(t).reshape(-1)[0]) for t in m],
                [float(torch.as_tensor(t).reshape(-1)[0]) for t in S], w)

    out = {}
    for path in a.models:
        model, arch = load_model(path, a.device)
        for eta in (a.etas if arch == "diffusion" else [None]):
            tvs = []
            for x in X:
                kw = {} if eta is None else {"eta": eta}
                with torch.no_grad():
                    o = model.sample(nsamples=a.n_draw, condition_x=x.view(1, -1).repeat(a.n_draw, 1).to(a.device),
                                     device=a.device, **kw)
                y = o[0] if isinstance(o, tuple) else o
                if y.shape[1] > 1:
                    y = y[:, c0:]
                tvs.append(hist_tv(y, *exact(x)))
            key = f"{path} eta={eta}"
            out[key] = {"x_star": tvs[0], "random_mean": float(np.mean(tvs[1:])), "random_max": float(np.max(tvs[1:]))}
            print(key, {k: round(v, 3) for k, v in out[key].items()}, flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
