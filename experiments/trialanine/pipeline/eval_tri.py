"""Exact evaluation of the trialanine CDM benchmark: every returned design is scored by reweighting the x0
reference frames (no sampler involved).

Per run: label-free count populations P(n '+' residues | x_hat) and their TV / KL to the target G_count,
the labeled 8-state TV (for information), the per-residue '+' marginals and the mode (residue with the most
'+' population; the target's mode is that of x*), |x_hat - x*| in kT, the reweighting ESS at x_hat, and the
exact label-free ensemble MMD (Gaussian kernel, bandwidths 0.5 / 1 / 2 on the standardized residue-sorted
features, 600 random Fourier features; target = reweighted frames at x*). Solved = count TV <= --tv-solved.
Compute: wall time on the run's GPU and the number of conditional samples drawn.
"""
import argparse, glob, json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S
from rw import Frames

RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]


def samples_mlgd(args, n_restarts):
    """Conditional samples drawn by one search_mog run (pathwise guidance), from its flags."""
    T = 100   # prior diffusion steps (train_models default)
    guided = n_restarts * (T * args["n_cond"] + (args.get("final_n_cond") or 4 * args["n_cond"]))
    refine = n_restarts * (args["refine_steps"] * (args["refine_n_cond"] or args["n_cond"]))
    select = args["select_top"] * args["select_repeats"] * args["select_n"]
    return guided + refine + select


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/trialanine/cdm")
    p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
    p.add_argument("--thin", type=int, default=2)
    p.add_argument("--tv-solved", type=float, default=0.05)
    p.add_argument("--out", default="experiments/trialanine/results/cdm")
    a = p.parse_args()
    meta = json.load(open(os.path.join(a.root, "meta.json")))
    F = Frames(glob.glob(a.frames), thin=a.thin); kT = F.kT; scale = meta["x_scale_kJ"]
    C = (S.canonical_features(F.phi, F.psi) - np.array(meta["y_mean"])) / np.array(meta["y_std"])
    rng = np.random.default_rng(0)
    W = np.concatenate([rng.normal(0, 1 / h, (200, 12)) for h in (0.5, 1.0, 2.0)])
    Z = np.concatenate([np.cos(C @ W.T), np.sin(C @ W.T)], 1) / np.sqrt(len(W))
    tg = {t["id"]: t for t in meta["targets"] if t["id"] is not None}
    gfeat = {i: F.weights(np.array(t["x_star_kJ"])) @ Z for i, t in tg.items()}

    def score(x_kJ, t):
        w = F.weights(x_kJ); P = w @ F.onehot; Pc = P @ NPLUS; G = np.array(t["G_count"])
        plus = P @ RES; r = w @ Z - gfeat[t["id"]]
        return dict(count_tv=float(0.5 * np.abs(Pc - G).sum()),
                    count_kl=float((G * (np.log(np.maximum(G, 1e-12)) - np.log(np.maximum(Pc, 1e-12)))).sum()),
                    state_tv=float(0.5 * np.abs(P - np.array(t["G_states"])).sum()), mmd_exact=float(r @ r),
                    plus=plus.round(4).tolist(), mode=int(np.argmax(plus)), mode_star=t["mode_star"],
                    mode_match=bool(int(np.argmax(plus)) == t["mode_star"]),
                    dist_kT=float(np.linalg.norm(x_kJ - np.array(t["x_star_kJ"])) / kT), ess=float(1 / (w ** 2).sum()),
                    P_count=Pc.round(4).tolist())

    rows = []
    for i, t in tg.items():
        rows.append(dict(method="oracle (x*)", target=i, kind=t["kind"], time_s=0.0, samples=0,
                         **score(np.array(t["x_star_kJ"]), t)))
    times = {}
    for f in sorted(glob.glob(os.path.join(a.root, "results", "*_t*.json"))):
        m = re.match(r"(.+)_t(\d+)\.json", os.path.basename(f))
        name, i = m.group(1), int(m.group(2))
        J = json.load(open(f))
        if name.startswith("MLGD") and "+polish" not in name:
            att = J["attempts"][0]
            x_std, t_s, ns = np.array(att["x_hat"]), att["time_s"], samples_mlgd(J["args"], J["budget"])
        else:
            x_std, t_s, ns = np.array(J["best"]["x_hat"]), J["time_s"], J["n_conditional_samples"]
        times[(name, i)] = (t_s, ns)
        rows.append(dict(method=name, target=i, kind=tg[i]["kind"], time_s=t_s, samples=ns, x_hat_std=x_std.tolist(),
                         **score(x_std * scale, tg[i])))
    for r in rows:   # the hybrid's compute includes the MLGD run it starts from
        if r["method"].endswith("+polish"):
            t0, n0 = times.get((r["method"][:-7], r["target"]), (np.nan, 0))
            r["time_s_polish"], r["time_s"], r["samples"] = r["time_s"], r["time_s"] + t0, r["samples"] + n0
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(meta={k: v for k, v in meta.items() if k != "targets"}, tv_solved=a.tv_solved, rows=rows),
              open(os.path.join(a.out, "per_run.json"), "w"), indent=1)

    order = ["oracle (x*)", "GBFESO-x0", "GBFESO-rand4", "MLGD-KL", "MLGD-MMD", "MLGD-KL+polish", "MLGD-MMD+polish"]
    names = [n for n in order if any(r["method"] == n for r in rows)] + sorted({r["method"] for r in rows} - set(order))
    lines = ["| method | targets | solved (count TV <= %.2f) | solved, random / engineered | median count TV | mean count TV | "
             "median count KL | median exact MMD | median labeled-state TV | mode = x*'s mode | median abs(x - x*) (kT) | "
             "median time (s) | median samples (M) |" % a.tv_solved, "|" + "---|" * 13]
    for n in names:
        R = [r for r in rows if r["method"] == n]
        tv = np.array([r["count_tv"] for r in R]); sol = tv <= a.tv_solved
        rnd = [r["count_tv"] <= a.tv_solved for r in R if r["kind"] == "random"]
        eng = [r["count_tv"] <= a.tv_solved for r in R if r["kind"] != "random"]
        med = lambda k: float(np.median([r[k] for r in R]))   # noqa: E731
        lines.append(f"| {n} | {len(R)} | {sol.mean():.0%} | {np.mean(rnd) if rnd else np.nan:.0%} / {np.mean(eng) if eng else np.nan:.0%} | "
                     f"{np.median(tv):.3f} | {tv.mean():.3f} | {med('count_kl'):.4f} | {med('mmd_exact'):.2e} | {med('state_tv'):.3f} | "
                     f"{np.mean([r['mode_match'] for r in R]):.0%} | {med('dist_kT'):.2f} | {med('time_s'):.0f} | {med('samples') / 1e6:.2f} |")
    # per-mode: success by the target's mode, and which mode the solved runs found
    lines += ["", "Per target mode (residue carrying the most '+' at x*): solved share, and among solved runs the share that "
              "found x*'s mode / another residue's mode", "",
              "| method | " + " | ".join(f"x* mode res{m + 1} (n)" for m in range(3)) + " |", "|---|---|---|---|"]
    for n in names:
        cells = []
        for m in range(3):
            R = [r for r in rows if r["method"] == n and r["mode_star"] == m]
            if not R:
                cells.append("-"); continue
            S_ = [r for r in R if r["count_tv"] <= a.tv_solved]
            same = np.mean([r["mode_match"] for r in S_]) if S_ else np.nan
            cells.append(f"{len(S_) / len(R):.0%} solved, {same:.0%} same mode / {1 - same:.0%} other (n={len(R)})")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    open(os.path.join(a.out, "results_table.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
