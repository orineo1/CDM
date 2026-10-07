"""Stage 3b: reweighted (from the x0 reference walkers) vs direct (walkers simulated at x_val with the same
frozen bias, unbiased by V only) state populations and phi marginals at the validation designs.
Errors: standard error over walkers (each walker is an independent trajectory)."""
import argparse, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import system as S
from rw import Frames

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="/mnt/c/Code/Github/CDM/runs/trialanine")
ap.add_argument("--vals", default="val1,val2,val3")
ap.add_argument("--direct_pattern", default="prod_{v}_w*.npz")
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
a = ap.parse_args()
R = Frames(glob.glob(f"{a.runs}/prod_x0_w*.npz"))
res = dict(N_ref=R.N, vals={})
vals = a.vals.split(",")
fig, axes = plt.subplots(len(vals), 2, figsize=(11, 3.4 * len(vals)), squeeze=False)
edges = np.linspace(-np.pi, np.pi, 37)
for vi, v in enumerate(vals):
    Dv = Frames(glob.glob(f"{a.runs}/" + a.direct_pattern.format(v=v)))
    x = Dv.xs[0]
    pr = R.pops(x); br = R.pops_blocks(x); er = br.std(0, ddof=1) / np.sqrt(len(br))
    pd = Dv.pops(x); bd = Dv.pops_blocks(x); ed = bd.std(0, ddof=1) / np.sqrt(len(bd))
    z = np.where(np.maximum(pr, pd) > 1e-3, (pr - pd) / np.sqrt(er ** 2 + ed ** 2 + 1e-12), 0.0)  # states > 0.1%
    ess_r, per_r = R.ess(x); ess_d, per_d = Dv.ess(x)
    rec = dict(x=x.tolist(), dist_kT=float(np.linalg.norm(x) / R.kT), pops_reweighted=pr.tolist(), se_reweighted=er.tolist(),
               pops_direct=pd.tolist(), se_direct=ed.tolist(), z=z.tolist(), max_abs_z=float(np.abs(z).max()),
               tv=float(0.5 * np.abs(pr - pd).sum()), ess_reweighted=ess_r, ess_reweighted_per_state=per_r.tolist(),
               ess_direct=ess_d, N_direct=Dv.N, pops_x0=R.pops(np.zeros(S.D)).tolist())
    res["vals"][v] = rec
    print(v, f"|x|={rec['dist_kT']:.2f} kT TV={rec['tv']:.4f} max|z|={rec['max_abs_z']:.2f} ESS_rw={ess_r:.0f}")
    print("  rw ", np.round(pr, 4), "\n  dir", np.round(pd, 4))
    ax = axes[vi, 0]; i8 = np.arange(8)
    ax.errorbar(i8 - 0.1, pr, 2 * er, fmt="o", label="reweighted from x0")
    ax.errorbar(i8 + 0.1, pd, 2 * ed, fmt="s", label="direct at x")
    ax.plot(i8, rec["pops_x0"], "k_", ms=12, label="x0")
    ax.set_yscale("log"); ax.set_xticks(i8, S.STATE_LABELS); ax.set_title(f"{v}: |x|={rec['dist_kT']:.1f} kT, TV={rec['tv']:.3f}")
    ax.legend(fontsize=7)
    ax = axes[vi, 1]
    wr, wd = R.weights(x), Dv.weights(x)
    for i in range(3):
        hr, _ = np.histogram(R.phi[:, i], edges, weights=wr); hd, _ = np.histogram(Dv.phi[:, i], edges, weights=wd)
        c = np.degrees(0.5 * (edges[1:] + edges[:-1]))
        ax.plot(c, hr + 1e-6, f"C{i}-", label=f"phi{i+1} rw"); ax.plot(c, hd + 1e-6, f"C{i}--", label=f"phi{i+1} direct")
    ax.set_yscale("log"); ax.set_xlabel("phi (deg)"); ax.legend(fontsize=6, ncol=3)
fig.tight_layout(); fig.savefig(os.path.join(a.out, "validate.png"), dpi=130)
json.dump(res, open(os.path.join(a.out, "validate.json"), "w"), indent=1)
