"""EXP-012 equivalence checks A2, A3, A4. Any failure halts the confirmatory run."""
import json, math, sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tfg._compat import ensure_legacy_importable   # noqa: E402
ensure_legacy_importable()

import dist_utils                                   # noqa: E402
from LossFunctions import MMDLoss, RBF              # noqa: E402
from tfg import oracle                              # noqa: E402
from tfg.curriculum_experiment import FIXED_BW, optimize   # noqa: E402
from tfg.equivalence_e2e import legacy_replica      # noqa: E402
from tfg.run_estimator_v2 import make_fixed_target_set     # noqa: E402
from tfg.target_hierarchy import TargetHierarchy, pointwise_loss, weighted_mmd2  # noqa: E402
from tfg.train_cm import PAPER_TS, SEEDS, load_one  # noqa: E402
from tfg.train_uncond import load as load_uncond    # noqa: E402


def check_A2(mc, mu, p, H, ns=25, nx=2, seed=101):
    """New wrapper in legacy-compat mode == legacy_replica, full trajectory."""
    tr = []
    torch.manual_seed(seed)
    x_ref, _ = legacy_replica(mu, mc, p, nsamples=ns, num_x_t=nx, trace=tr)
    torch.manual_seed(seed)
    x_new, info = optimize(mc, mu, H, method="full", restart=0,
                           legacy_compat=True, params=p, num_x_t=nx,
                           ts=None, bandwidth=None, n_override=ns)
    steps = min(len(tr), len(info["trajectory"]))
    dx = max(abs(tr[i]["x"] - info["trajectory"][i]["x"]) for i in range(steps))
    dg = max(abs(tr[i]["grad"] - info["trajectory"][i]["grad"]) for i in range(steps))
    dl = max(abs(tr[i]["loss"] - info["trajectory"][i]["loss"]) for i in range(steps))
    return {"check": "A2", "steps_compared": steps,
            "max_abs_dx": dx, "max_abs_dgrad": dg, "max_abs_dloss": dl,
            "x_legacy": x_ref, "x_new": x_new,
            "final_abs_dx": abs(x_ref - x_new),
            "pass": dx < 1e-9 and dg < 1e-9 and dl < 1e-9}


def check_A3(mc, p, H, n=64, seed=7):
    """Fixed-bandwidth `full` loss/grad recomputed independently."""
    torch.manual_seed(seed)
    x = torch.tensor([-4.5], dtype=torch.float32, requires_grad=True)
    cond = x.reshape(1, -1).repeat(n, 1)
    y, _, _ = mc.sample(nsamples=n, condition_x=cond, device="cpu", ts=PAPER_TS)
    c, w, _ = H.level(H.N)
    L_ours = weighted_mmd2(y.double(), c, w, FIXED_BW)
    g_ours, = torch.autograd.grad(L_ours, x, retain_graph=True)

    # independent: repository MMDLoss at the same fixed bandwidth vs raw S_G
    L_repo = MMDLoss(kernel=RBF(bandwidth=FIXED_BW, device="cpu"),
                     device="cpu")(y.double(), H.S_G)
    g_repo, = torch.autograd.grad(L_repo, x)

    # finite differences of the loss w.r.t. x, same conditional noise
    def loss_at(v):
        torch.manual_seed(seed)
        xx = torch.tensor([v], dtype=torch.float32)
        cc = xx.reshape(1, -1).repeat(n, 1)
        yy, _, _ = mc.sample(nsamples=n, condition_x=cc, device="cpu", ts=PAPER_TS)
        return float(weighted_mmd2(yy.double(), c, w, FIXED_BW))
    h = 1e-3
    g_fd = (loss_at(-4.5 + h) - loss_at(-4.5 - h)) / (2 * h)

    dl = abs(float(L_ours) - float(L_repo))
    dg = abs(float(g_ours) - float(g_repo))
    rel_fd = abs(float(g_ours) - g_fd) / max(1e-9, abs(g_fd))
    return {"check": "A3", "loss_ours": float(L_ours), "loss_repo": float(L_repo),
            "abs_dloss": dl, "grad_ours": float(g_ours), "grad_repo": float(g_repo),
            "abs_dgrad": dg, "grad_fd": g_fd, "rel_err_vs_fd": rel_fd,
            "pass": dl < 1e-9 and dg < 1e-9 and rel_fd < 0.05}


def check_A4(mc, H, seed=13):
    """Constant K=n=1 tier == ordinary point-target TFG predictor."""
    torch.manual_seed(seed)
    x = torch.tensor([-4.5], dtype=torch.float32, requires_grad=True)
    y, _, _ = mc.sample(nsamples=1, condition_x=x.reshape(1, -1), device="cpu",
                        ts=PAPER_TS)
    L_ours = pointwise_loss(y.double(), H.mean())
    g_ours, = torch.autograd.grad(L_ours, x, retain_graph=True)

    # independent point-target TFG loss: squared distance to E_G[Y]
    mean_target = float(H.S_G.mean())
    L_indep = (y.double().reshape(-1)[0] - mean_target) ** 2
    g_indep, = torch.autograd.grad(L_indep, x)

    return {"check": "A4", "target_mean": mean_target,
            "loss_ours": float(L_ours), "loss_indep": float(L_indep),
            "abs_dloss": abs(float(L_ours) - float(L_indep)),
            "grad_ours": float(g_ours), "grad_indep": float(g_indep),
            "abs_dgrad": abs(float(g_ours) - float(g_indep)),
            "pass": abs(float(L_ours) - float(L_indep)) < 1e-12
                    and abs(float(g_ours) - float(g_indep)) < 1e-12}


if __name__ == "__main__":
    p = oracle.load_params()
    S_G = make_fixed_target_set(p).double()
    H = TargetHierarchy(S_G)
    mc, _ = load_one(SEEDS[0], "tfg_results/cm_checkpoints")
    mu, _ = load_uncond(out_dir="tfg_results/cm_checkpoints")

    res = [check_A2(mc, mu, p, H), check_A3(mc, p, H), check_A4(mc, H)]
    for r in res:
        print(f"{r['check']}: {'PASS' if r['pass'] else 'FAIL'}  " +
              "  ".join(f"{k}={v:.3e}" for k, v in r.items()
                        if isinstance(v, float) and k.startswith(("max_", "abs_", "rel_", "final_"))))
    out = _ROOT / "tfg_results" / "equivalence_A2_A4.json"
    out.write_text(json.dumps(res, indent=2))
    print("written", out)
    sys.exit(0 if all(r["pass"] for r in res) else 1)
