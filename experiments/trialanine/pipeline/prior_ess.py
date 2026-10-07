"""Stage 3a: reweighting ESS vs distance from x0 for x ~ N(0, sigma^2 I) (sigma in units of kT),
plus population statistics of the prior draws. Writes results/prior_ess.json and a plot."""
import argparse, glob, json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import system as S
from rw import Frames

ap = argparse.ArgumentParser()
ap.add_argument("--frames", default="/mnt/c/Code/Github/CDM/runs/trialanine/prod_x0_w*.npz")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "results"))
a = ap.parse_args()
t0 = time.time()
F = Frames(glob.glob(a.frames))
rng = np.random.default_rng(0)
ess0, per0 = F.ess(np.zeros(S.D))
res = dict(N=F.N, ess_x0=ess0, ess_x0_per_state=per0.tolist(), pops_x0=F.pops(np.zeros(S.D)).tolist(),
           kT_kJ=F.kT, sigmas_kT=[], rows=[])
sig_list = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
for sig in sig_list:
    X = rng.normal(0, sig * F.kT, size=(a.n, S.D))
    for x in X:
        e, per = F.ess(x)
        p = F.pops(x)
        res["rows"].append(dict(sigma_kT=sig, dist_kT=float(np.linalg.norm(x) / F.kT), ess=e, ess_frac=e / ess0,
                                min_state_ess=float(per.min()), pops=p.tolist(), max_pop=float(p.max()),
                                entropy=float(-(p * np.log(p + 1e-300)).sum())))
summary = {}
for sig in sig_list:
    r = [q for q in res["rows"] if q["sigma_kT"] == sig]
    f = lambda k, fn: float(fn([q[k] for q in r]))
    summary[str(sig)] = dict(ess_frac_median=f("ess_frac", np.median), ess_frac_p10=f("ess_frac", lambda v: np.percentile(v, 10)),
                             min_state_ess_median=f("min_state_ess", np.median),
                             min_state_ess_p10=f("min_state_ess", lambda v: np.percentile(v, 10)),
                             frac_min_state_ess_ge_100=float(np.mean([q["min_state_ess"] >= 100 for q in r])),
                             max_pop_median=f("max_pop", np.median), entropy_median=f("entropy", np.median))
res["summary"] = summary
res["wall_s"] = time.time() - t0
json.dump(res, open(os.path.join(a.out, "prior_ess.json"), "w"), indent=1)
print(json.dumps(dict(N=F.N, ess_x0=ess0, per0=per0.round(0).tolist(), pops_x0=np.round(res["pops_x0"], 4).tolist()), indent=0))
print(json.dumps(summary, indent=1))
d = np.array([q["dist_kT"] for q in res["rows"]]); e = np.array([q["ess_frac"] for q in res["rows"]])
m = np.array([q["min_state_ess"] for q in res["rows"]]); sg = np.array([q["sigma_kT"] for q in res["rows"]])
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sc = ax[0].scatter(d, e, c=sg, s=6, cmap="viridis"); ax[0].set_yscale("log")
ax[0].set_xlabel("|x - x0| / kT"); ax[0].set_ylabel("ESS(x) / ESS(x0)"); plt.colorbar(sc, ax=ax[0], label="prior sigma / kT")
ax[1].scatter(d, m, c=sg, s=6, cmap="viridis"); ax[1].set_yscale("log"); ax[1].axhline(100, color="k", lw=0.5)
ax[1].set_xlabel("|x - x0| / kT"); ax[1].set_ylabel("min over 8 states of per-state ESS")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "prior_ess.png"), dpi=130)
