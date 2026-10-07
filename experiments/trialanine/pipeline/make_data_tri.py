"""Trialanine CDM benchmark data, all by reweighting the x0 reference frames (no new MD):

  <out>/data.pt         rows [x_std (12), y (12)] for train_models.py --model conditional (condition_on = 12)
  <out>/prior_data.pt   x_std draws of the design prior, for train_models.py --model unconditional
  <out>/targets/target_XXX.npy   standardized y samples of the target ensemble at a hidden x*
  <out>/meta.json       scales, x* (kJ/mol and standardized), label-free count targets, ESS, flags

Design prior (training, MLGD prior, GB-FESO random starts): x = sigma_train kT * x_std, x_std ~ N(0, I)
truncated to the box [-B, B]^12 (default sigma_train = 2 kT, B = 2, i.e. |x_j| <= 4 kT).
y = label-free dihedral features: the residues' (phi, psi) sorted by phi, as cos / sin (system.canonical_features),
standardized per feature. Each training design gets --frames-per-x frames resampled (multinomial) from the
reference frames with the importance weights at x; the per-design Kish ESS is recorded.
Targets: --n-random hidden designs from the narrower target prior N(0, (sigma_target kT)^2) truncated to the box,
plus engineered ones (one or two residues pushed to '+'). A target is flagged 'low_ess' (and dropped unless
--keep-low-ess) when its total ESS < --min-ess or a count class with G > 0.05 has ESS < --min-class-ess.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system as S
from rw import Frames


def trunc_normal(rng, n, d, box):
    out = rng.normal(size=(n, d))
    bad = np.abs(out) > box
    while bad.any():
        out[bad] = rng.normal(size=bad.sum())
        bad = np.abs(out) > box
    return out


def class_ess(w, labels, k):
    out = []
    for c in range(k):
        ws = w[labels == c]
        out.append(float(ws.sum() ** 2 / (ws ** 2).sum()) if ws.sum() > 0 else 0.0)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
    p.add_argument("--out", default="runs/trialanine/cdm")
    p.add_argument("--n-designs", type=int, default=1000)
    p.add_argument("--frames-per-x", type=int, default=1500)
    p.add_argument("--sigma-train", type=float, default=2.0, help="kT per coefficient")
    p.add_argument("--box", type=float, default=2.0, help="in units of sigma_train")
    p.add_argument("--n-prior", type=int, default=200_000)
    p.add_argument("--n-random", type=int, default=24)
    p.add_argument("--sigma-target", type=float, default=1.25)
    p.add_argument("--eng-amp", type=float, default=3.5, help="kT, engineered sin(phi_i) coefficient (negative = '+')")
    p.add_argument("--n-target-frames", type=int, default=20000)
    p.add_argument("--min-ess", type=float, default=1000)
    p.add_argument("--min-class-ess", type=float, default=200)
    p.add_argument("--keep-low-ess", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    F = Frames(glob.glob(a.frames))
    kT = F.kT; scale = a.sigma_train * kT
    Yc = S.canonical_features(F.phi, F.psi).astype(np.float32)
    ymean, ystd = Yc.mean(0), Yc.std(0)
    Ys = (Yc - ymean) / ystd
    count = S.count_plus(F.phi)
    RES = np.array([[(k >> (2 - i)) & 1 for i in range(3)] for k in range(8)], float)
    NPLUS = np.eye(4)[np.array([bin(k).count("1") for k in range(8)])]

    # training pairs
    Xs = trunc_normal(rng, a.n_designs, S.D, a.box)
    rows, ess = [], []
    for xs in Xs:
        w = F.weights(xs * scale)
        ess.append(float(1 / (w ** 2).sum()))
        idx = rng.choice(F.N, a.frames_per_x, p=w)
        rows.append(np.concatenate([np.repeat(xs[None].astype(np.float32), a.frames_per_x, 0), Ys[idx]], 1))
    samples = torch.from_numpy(np.concatenate(rows))
    os.makedirs(os.path.join(a.out, "targets"), exist_ok=True)
    torch.save({"samples": samples, "condition_on": S.D, "y_mean": torch.from_numpy(ymean), "y_std": torch.from_numpy(ystd),
                "x_scale_kJ": scale}, os.path.join(a.out, "data.pt"))
    torch.save({"samples": torch.from_numpy(trunc_normal(rng, a.n_prior, S.D, a.box).astype(np.float32)), "condition_on": S.D},
               os.path.join(a.out, "prior_data.pt"))

    # targets
    cands = []
    rt = np.random.default_rng(a.seed + 1)
    for k in range(a.n_random):
        cands.append(("random", trunc_normal(rt, 1, S.D, a.box * a.sigma_train / a.sigma_target)[0] * a.sigma_target * kT))
    for res_set in ([0], [1], [2], [0, 1], [1, 2], [0, 2]):
        x = np.zeros(S.D)
        for i in res_set:
            x[4 * i + 1] = -a.eng_amp * kT
        cands.append(("eng_plus" + "".join(str(i + 1) for i in res_set), x))
    targets, tid = [], 0
    for kind, x in cands:
        w = F.weights(x)
        P = w @ F.onehot
        Gc = P @ NPLUS
        e_tot = float(1 / (w ** 2).sum())
        e_cls = class_ess(w, count, 4)
        low = e_tot < a.min_ess or any(g > 0.05 and e < a.min_class_ess for g, e in zip(Gc, e_cls))
        rec = dict(kind=kind, x_star_kJ=x.tolist(), x_star_std=(x / scale).tolist(), G_count=Gc.tolist(), G_states=P.tolist(),
                   plus_marginals=(P @ RES).tolist(), mode_star=int(np.argmax(P @ RES)), ess=e_tot, ess_count_classes=e_cls,
                   low_ess=bool(low))
        if low and not a.keep_low_ess:
            rec["id"] = None
        else:
            rec["id"] = tid
            idx = rng.choice(F.N, a.n_target_frames, p=w)
            np.save(os.path.join(a.out, "targets", f"target_{tid:03d}.npy"), Ys[idx])
            tid += 1
        targets.append(rec)
        print(f"[target] {kind:12s} id={rec['id']} ESS={e_tot:7.0f} G_count={np.round(Gc, 3)} plus={np.round(P @ RES, 3)}"
              + ("  LOW ESS" if low else ""), flush=True)
    meta = dict(frames=a.frames, N_frames=F.N, kT_kJ=kT, x_scale_kJ=scale, sigma_train_kT=a.sigma_train, box_std=a.box,
                y_mean=ymean.tolist(), y_std=ystd.tolist(), n_designs=a.n_designs, frames_per_x=a.frames_per_x,
                train_ess=dict(min=float(np.min(ess)), p10=float(np.percentile(ess, 10)), median=float(np.median(ess))),
                frac_train_ess_lt_500=float(np.mean(np.array(ess) < 500)), sigma_target_kT=a.sigma_target,
                eng_amp_kT=a.eng_amp, targets=targets, wall_s=time.time() - t0, args=vars(a))
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"[make_data_tri] data {tuple(samples.shape)}, {tid} targets kept of {len(targets)}, "
          f"train ESS median {meta['train_ess']['median']:.0f} min {meta['train_ess']['min']:.0f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
