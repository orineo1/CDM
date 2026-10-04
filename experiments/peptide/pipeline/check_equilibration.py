"""
Equilibration check of the simulation protocol (replicas started from the straight chain): are the basin
populations stable in time and reproducible between independent replicas?

    python experiments/peptide/pipeline/sim_ensembles.py --n-beads 6 --eps-file eps.npy --n-rep 16 --no-subsample \
        --seed 11 --out runs/peptide_N6/equil_ens.pt
    python experiments/peptide/pipeline/check_equilibration.py runs/peptide_N6/equil_ens.pt [--out table.md]

For every eps, with the frames of replica r in time order (sim_ensembles.py --no-subsample):
  early / late : basin populations (tuple basins of common.py; for N = 4 also their utils.py boxes) of the
                 first vs the second half of every replica's frames, pooled over replicas;
  seeds        : the first vs the second half of the replicas (independent chains), whole runs;
  first 10%    : the first 10% of the frames after burn-in vs the rest (residual memory of the start).
TV = total variation between the two population vectors; dF = the largest |dF| difference (k_BT, floor 1e-3)
over basins with > 5% population in either half. Drift shows as early/late (or first-10%) TV well above the
seeds TV, which measures the replica-to-replica noise at this number of replicas.
tau = integrated autocorrelation time (in MD steps) of the indicator "in the most populated tuple basin",
averaged over replicas (summed up to where the autocorrelation drops below 0.05; '>' = it never does within
a run); share = that basin's share of each replica's frames, min / median / max over replicas (a spread from
0 to 1 means replicas stay in one basin for the whole run).
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402
from simulate import Design   # noqa: E402


def probs(Y, which):
    if which == "tuple":
        return common.basin_probs(Y)
    from utils import basin_probabilities     # GB-FESO, N = 4
    return basin_probabilities(Y)


def compare(p, q):
    tv = float(0.5 * (p - q).abs().sum())
    act = [i for i in range(len(p)) if max(float(p[i]), float(q[i])) > 0.05]
    if len(act) < 2:
        return tv, 0.0
    ref = max(act, key=lambda i: float(p[i] + q[i]))
    fp = -torch.log(p.clamp(min=1e-3) / p.clamp(min=1e-3)[ref])
    fq = -torch.log(q.clamp(min=1e-3) / q.clamp(min=1e-3)[ref])
    return tv, float(max(abs(float(fp[i] - fq[i])) for i in act))


def tuple_class(Y):
    st = common.angle_states(Y)
    na = st.shape[1]
    code = (st * (4 ** torch.arange(na - 1, -1, -1))).sum(1)
    return common.basin_classes(na + 2)[0][code]


def autocorr_time(c, stride):
    """c [R, F] basin ids in time order -> (tau in steps, never-decorrelated flag, per-replica top shares)."""
    top = torch.mode(c.flatten()).values
    x = (c == top).float().numpy()
    xc = x - x.mean()
    var = max(float((xc ** 2).mean()), 1e-12)
    F = x.shape[1]
    f = np.fft.rfft(xc, n=2 * F, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :F].mean(0) / np.arange(F, 0, -1) / var
    below = ac < 0.05
    k = int(np.argmax(below)) if below.any() else F
    return (1 + 2 * ac[1:k].sum()) * stride, not below.any(), x.mean(1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ens")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    d = torch.load(a.ens, weights_only=False)
    design = Design.from_dict(d.get("design"))
    pr = d["protocol"]
    assert pr.get("no_subsample"), "needs an ensemble simulated with --no-subsample"
    R = pr["n_rep"]
    Y = d["Y"]
    F = Y.shape[1] // R
    kinds = ["tuple"] + (["their"] if design.n_beads == 4 else [])
    lines = [f"Equilibration check: {design}, {R} replicas x {pr['n_steps']:,} steps, burn-in {pr['burn']:,}, "
             f"{F} frames per replica", "",
             "| eps | basins | n active (>5%) | top basins (pop.) | early/late TV | early/late max dF | seeds TV | seeds max dF | first-10% TV | tau (steps) | share min/med/max |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    names = common.basin_classes(design.n_beads)[1]
    summary = {k: [] for k in ("el", "sd", "f10", "tau")}
    for e in range(Y.shape[0]):
        Z = Y[e].view(R, F, -1)
        for kind in kinds:
            pa = probs(Z.reshape(-1, Z.shape[-1]), kind)
            early, late = probs(Z[:, :F // 2].reshape(-1, Z.shape[-1]), kind), probs(Z[:, F // 2:].reshape(-1, Z.shape[-1]), kind)
            s1, s2 = probs(Z[:R // 2].reshape(-1, Z.shape[-1]), kind), probs(Z[R // 2:].reshape(-1, Z.shape[-1]), kind)
            f1, f2 = probs(Z[:, :F // 10].reshape(-1, Z.shape[-1]), kind), probs(Z[:, F // 10:].reshape(-1, Z.shape[-1]), kind)
            el, sd, f10 = compare(early, late), compare(s1, s2), compare(f1, f2)
            if kind == "tuple":
                summary["el"].append(el[0]); summary["sd"].append(sd[0]); summary["f10"].append(f10[0])
            nm = names if kind == "tuple" else ["A", "B", "C", "out"]
            top = torch.argsort(pa, descending=True)[:3].tolist()
            if kind == "tuple":
                tau, never, share = autocorr_time(tuple_class(Z.reshape(-1, Z.shape[-1])).view(R, F), pr["stride"])
                summary["tau"].append(tau)
                tcol = f" {'>' if never else ''}{tau:,.0f} | {share.min():.2f}/{np.median(share):.2f}/{share.max():.2f} |"
            else:
                tcol = " | |"
            lines.append(f"| {[round(float(v), 2) for v in d['eps'][e]]} | {kind} | {int((pa > 0.05).sum())} | "
                         + ", ".join(f"{nm[i]} {float(pa[i]):.2f}" for i in top)
                         + f" | {el[0]:.3f} | {el[1]:.2f} | {sd[0]:.3f} | {sd[1]:.2f} | {f10[0]:.3f} |" + tcol)
    lines += ["", f"tuple basins, median over eps: early/late TV {np.median(summary['el']):.3f}, seeds TV "
              f"{np.median(summary['sd']):.3f}, first-10% TV {np.median(summary['f10']):.3f}; max: "
              f"{max(summary['el']):.3f}, {max(summary['sd']):.3f}, {max(summary['f10']):.3f}; tau median "
              f"{np.median(summary['tau']):,.0f}, max {max(summary['tau']):,.0f} steps (run {pr['n_steps']:,})"]
    print("\n".join(lines))
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
