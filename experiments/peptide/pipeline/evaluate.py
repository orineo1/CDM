"""
Ground-truth evaluation: every returned eps_hat is re-simulated (sim_ensembles.py, the same protocol as
the targets) and compared with the target ensemble using GB-FESO's own metrics (their utils.py basins
and Delta F, their loss.py KDE-KL with bandwidth 0.5, kl_lambda 1, raw 5D coordinates).

  1. python experiments/peptide/pipeline/evaluate.py collect --results runs/peptide/results/*.json \
         --targets-ens runs/peptide/targets_ens.pt --out runs/peptide/eval_eps.npy
     (adds an 'oracle' row eps_hat = eps*: an independent re-simulation at the true eps, the noise floor)
  2. python experiments/peptide/pipeline/sim_ensembles.py --eps-file runs/peptide/eval_eps.npy --keep 20000 \
         --seed 3 --out runs/peptide/eval_ens.pt
  3. python experiments/peptide/pipeline/evaluate.py metrics --eval-eps runs/peptide/eval_eps.npy \
         --eval-ens runs/peptide/eval_ens.pt --targets-ens runs/peptide/targets_ens.pt \
         --out experiments/peptide/results/

Success = every target basin with mass > 5% has mass >= 0.5% at eps_hat (their compare_basin_metrics:
`missing_active_basin` is False). Delta F error = their per-run median |dF_hat - dF| over the target's
non-reference active basins (k_BT, capped at 6), multi-basin targets only. eps_hat is clamped to the box
[0.21, 1.55]^K before simulation (GB-FESO clamps during the search; ours is clamped only here).

N beads (the design is read from the target ensemble). Their basins are boxes in (theta123, theta234), so
for N > 4 (or with --basins tuple) the basins are the tuple basins of common.py: every angle is +folded,
-folded, +open or -open (|theta| < 1.55 = folded), a conformation is the tuple of the N - 2 angle states,
and a basin is a class of tuples under the mirror and chain-reversal symmetries (N = 5: 20 basins,
N = 6: 72). Success and Delta F follow their rules on these basins (active > 5%, missing < 0.5%,
floor 1e-3, cap 6). KDE-KL: their loss with the N - 2 angle columns wrapped. Distribution score that
does not depend on basins: the energy distance between 2,048 target and 2,048 re-simulated frames in our
standardized feature space (bond lengths, cos / sin of the angles; `ED` in the table; oracle = noise floor).

    python experiments/peptide/pipeline/evaluate.py sampler --prior ... --sampler runs/peptide_N5/models/diff/model_conditional.pt \
        --targets-ens runs/peptide_N5/targets_ens.pt --meta runs/peptide_N5/meta.json [--gbfeso-ckpt ...] --target-ids 0-9
    (accuracy of the trained samplers at held-out eps*: sampler draws vs the simulated target ensemble)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402,F401
from simulate import EPS_BOX, Design   # noqa: E402
from utils import compare_basin_metrics   # noqa: E402  (GB-FESO)
from loss import kde_kl_target_to_generated   # noqa: E402  (GB-FESO)


def load_runs(paths, design=None):
    runs = []
    for p in paths:
        for f in sorted(glob.glob(p)):
            d = json.load(open(f))
            if "runs" in d:
                for r in d["runs"]:
                    method = d["method"] + (f" [{r['start']}]" if "start" in r else "")   # GB-FESO: one row per start
                    runs.append({"method": method, "family": d["method"], **{k: r[k] for k in r if k not in ("loss_trace",)}})
            else:    # search_peptide output: one target per file, named <method><tag>_t<id>.json
                stem, tid = os.path.basename(f)[:-len(".json")].rsplit("_t", 1)
                tid, b = int(tid), d["best"]
                if stem.endswith("_theirs"):        # MLGD run on GB-FESO's surrogate (gbfeso_sampler.py)
                    stem = stem[:-len("_theirs")] + ", their surrogate"
                runs.append({"method": stem, "family": stem, "target_id": tid,
                             "x_hat_std": b["x_hat"], "eps_hat": common.x_from_std(np.array(b["x_hat"], dtype=np.float64), design).tolist()
                             if design is not None and not design.is_legacy else
                             (np.array(b["x_hat"]) * common.X_STD + common.X_MEAN).tolist(),
                             "budget": d.get("budget"),
                             "surrogate_loss": b.get("select_loss", b["final_loss"]),
                             "time_s": d["summary"]["time_s"]["mean"]})
    return runs


def cmd_collect(a):
    te = torch.load(a.targets_ens, weights_only=False)
    design = Design.from_dict(te.get("design"))
    runs = load_runs(a.results, design)
    for tid in ([] if a.no_oracle else range(te["eps"].shape[0])):
        runs.append({"method": "oracle (eps_hat = eps*)", "family": "oracle", "target_id": tid,
                     "eps_hat": te["eps"][tid].tolist(), "time_s": 0.0})
    for r in runs:
        r["eps_star"] = te["eps"][r["target_id"]].tolist()
        r["eps_eval"] = (np.clip(r["eps_hat"], *EPS_BOX).tolist() if design.is_legacy
                         else design.clip(r["eps_hat"]).tolist())
    np.save(a.out, np.array([r["eps_eval"] for r in runs], dtype=np.float32))
    json.dump(runs, open(os.path.splitext(a.out)[0] + "_runs.json", "w"))
    print(f"[evaluate] {len(runs)} runs -> {a.out}")


def score_pair(T, G, design, basins, kl_n, scaler, g, dev):
    """All metrics of one (target ensemble T, re-simulated ensemble G), raw [n, 2N - 3] frames."""
    if basins == "their":
        m = compare_basin_metrics(T, G)
    else:
        m = common.compare_tuple_basins(T, G)
    it = torch.randperm(T.shape[0], generator=g)[:kl_n]
    ig = torch.randperm(G.shape[0], generator=g)[:kl_n]
    angles = tuple(range(design.n_beads - 1, design.dim_y))
    kl = float(kde_kl_target_to_generated(T[it].to(dev), G[ig].to(dev), bandwidth=0.5, kl_lambda=1.0, angle_indices=angles))
    ie, je = it[:2048], ig[:2048]
    ed = common.energy_distance(scaler.fwd(T[ie]).to(dev), scaler.fwd(G[je]).to(dev))
    return {"success": not m["missing_active_basin"], "multi_basin": m["target_type"] == "multi_basin",
            "df_median_abs_error": m["df_median_abs_error"], "df_mae": m["df_mae"], "tv_basins": m["total_variation"],
            "target_probs": m["target_probs"], "generated_probs": m["generated_probs"], "kl": kl, "ed": ed}


def get_scaler(a, te):
    """our standardization of the features (meta.json of the run), for the energy distance"""
    meta = os.path.join(os.path.dirname(os.path.abspath(a.targets_ens)), "meta.json")
    if os.path.exists(meta):
        mj = json.load(open(meta))
        return common.YScaler(mj["y_mean"], mj["y_std"])
    f = common.raw_to_feat(te["Y"].reshape(-1, te["Y"].shape[-1])[:200000])
    return common.YScaler(f.mean(0), f.std(0))


def cmd_metrics(a):
    runs = json.load(open(os.path.splitext(a.eval_eps)[0] + "_runs.json"))
    ev = torch.load(a.eval_ens, weights_only=False)
    te = torch.load(a.targets_ens, weights_only=False)
    design = Design.from_dict(te.get("design"))
    common.set_design(design)
    basins = a.basins or ("their" if design.n_beads == 4 else "tuple")
    assert basins == "tuple" or design.n_beads == 4, "their basin boxes exist for N = 4 only"
    scaler = get_scaler(a, te)
    assert ev["Y"].shape[0] == len(runs)
    for k in ("n_steps", "burn", "stride", "n_rep"):
        assert ev["protocol"][k] == te["protocol"][k], f"evaluation and target protocols differ in {k}"
    g = torch.Generator().manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for i, r in enumerate(runs):
        T, G = te["Y"][r["target_id"]], ev["Y"][i]
        r.update(score_pair(T, G, design, basins, a.kl_n, scaler, g, dev))
        r["eps_err"] = float(np.linalg.norm(np.array(r["eps_eval"]) - np.array(r["eps_star"])))
        r["basins"] = basins
    os.makedirs(a.out, exist_ok=True)
    json.dump(runs, open(os.path.join(a.out, f"per_run{a.suffix}.json"), "w"), indent=0)
    write_table(runs, os.path.join(a.out, f"results_table{a.suffix}.md"))


def cmd_sampler(a):
    """Accuracy of the trained samplers at held-out eps*: draws of our conditional sampler(s) (and of the
    GB-FESO surrogate) at eps* against the simulated target ensemble; 'split' = two disjoint halves of the
    target ensemble (a lower bound of the noise floor; the oracle row of the evaluation is the real one)."""
    import dataclasses
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "synthetic", "pipeline"))
    import search_mog as fx
    from cdm.guidance.core import _draw_conditional, _EvalView
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = torch.load(a.targets_ens, weights_only=False)
    design = Design.from_dict(te.get("design"))
    common.set_design(design)
    basins = a.basins or ("their" if design.n_beads == 4 else "tuple")
    mj = json.load(open(a.meta))
    sc = common.YScaler(mj["y_mean"], mj["y_std"])
    from gbfeso import parse_ids
    ids = parse_ids(a.target_ids, te["eps"].shape[0])
    g = torch.Generator().manual_seed(0)
    rows = {}
    samplers = []
    for path in a.sampler:
        sys.argv = ["search_mog", "--prior", a.prior, "--sampler", path, "--target", "x.npy", "--device", dev]
        args = fx.parse_args()
        prior, _ = fx.load_model(a.prior, dev)
        smp, stype = fx.load_model(path, dev)
        cfg = fx.make_cfg(args, prior, stype, lambda n, d: None, 0)
        samplers.append((os.path.basename(os.path.dirname(path)), smp, cfg))
    gb = None
    if a.gbfeso_ckpt:
        import gbfeso
        Yt = te["Y"]
        gb = gbfeso.load_model(a.gbfeso_ckpt, gbfeso.traj_list(Yt[:, :10], te["eps"]), dev)
    for tid in ids:
        T = te["Y"][tid]
        perm = torch.randperm(T.shape[0], generator=g)
        h = T.shape[0] // 2
        rows.setdefault("split halves of the target", []).append(
            score_pair(T[perm[:h]], T[perm[h:]], design, basins, a.n, sc, g, dev))
        x = common.x_to_std(te["eps"][tid], design).view(1, -1).to(dev)
        for name, smp, cfg in samplers:
            with torch.no_grad():
                Z = _draw_conditional(smp, x, _EvalView(dataclasses.replace(cfg, final_n_cond=a.n), a.n)).float().cpu()
            rows.setdefault(f"ours: {name}", []).append(score_pair(T, sc.inv(Z.view(a.n, -1)), design, basins, a.n, sc, g, dev))
        if gb is not None:
            model, frame_ds, gcfg, _ = gb
            with torch.no_grad():
                xs = model.ddim_sample(te["eps"][tid].to(dev).view(1, -1).expand(a.n, -1), shape=(a.n, frame_ds.target_dim),
                                       sample_steps=40, eta=0.0, clamp_x0=False)
            Gs = frame_ds.denormalize_target(xs, clamp=False).float().cpu()
            rows.setdefault("GB-FESO surrogate (40 DDIM steps)", []).append(score_pair(T, Gs, design, basins, a.n, sc, g, dev))
    lines = [f"Surrogate accuracy at {len(ids)} held-out eps* ({design}, basins: {basins}): {a.n} draws of each sampler at eps* "
             "vs {a.n} frames of the simulated target ensemble. Ours: conditional diffusion (MLGD's sampler) and consistency model "
             "(MLGD-F's); theirs: GB-FESO's surrogate with 40 DDIM steps, as their optimizer draws. dF errors over the target's "
             "active basins (> 5%), multi-basin targets only; ED = energy distance in our standardized features. "
             "'split halves' = two disjoint random halves of the target ensemble (a lower bound of the noise).".replace("{a.n}", str(a.n)), "",
             "| sampler | basin success | median basin TV | mean basin TV | median dF err | mean dF err (multi-basin targets) | median KDE-KL | median ED | mean ED |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, rs in rows.items():
        df = [r["df_median_abs_error"] for r in rs if r["df_median_abs_error"] is not None]
        lines.append(f"| {name} | {100 * np.mean([r['success'] for r in rs]):.0f}% | {np.median([r['tv_basins'] for r in rs]):.3f} | "
                     f"{np.mean([r['tv_basins'] for r in rs]):.3f} | {np.median(df) if df else float('nan'):.3f} | "
                     f"{np.mean(df) if df else float('nan'):.3f} (n={len(df)}) | {np.median([r['kl'] for r in rs]):.3f} | "
                     f"{np.median([r['ed'] for r in rs]):.4f} | {np.mean([r['ed'] for r in rs]):.4f} |")
    print("\n".join(lines))
    if a.out:
        open(a.out, "w").write("\n".join(lines) + "\n")


def cmd_table(a):
    path = os.path.join(a.out, f"per_run{a.suffix}.json")
    runs = json.load(open(path))
    if a.add:       # merge evaluated runs of new methods (same protocol) into per_run<suffix>.json
        new = [r for f in a.add for r in json.load(open(f))]
        keys = {(r["method"], r["target_id"]) for r in new}
        runs = [r for r in runs if (r["method"], r["target_id"]) not in keys] + new
        json.dump(runs, open(path, "w"), indent=0)
    write_table(runs, os.path.join(a.out, f"results_table{a.suffix}.md"))


def best_of_starts(runs):
    """Per target, the start whose returned eps has the lowest surrogate loss (what a user running all
    starts would pick); its time is the sum over the starts."""
    out = []
    for fam in sorted({r["family"] for r in runs if "start" in r}):
        by_t = {}
        for r in runs:
            if r["family"] == fam and "start" in r:
                by_t.setdefault(r["target_id"], []).append(r)
        for tid, rs_all in sorted(by_t.items()):
            corners = [q for q in rs_all if not q["start"].startswith("rand")]
            # with random starts present, report both the four corners and all starts
            fixed = "corner starts" if all("-" in q["start"] and not q["start"].startswith("all") for q in corners) else "fixed starts"
            for rs, what in ([(corners, fixed), (rs_all, "starts")] if len(corners) < len(rs_all)
                             else [(rs_all, "starts")]):
                b = dict(min(rs, key=lambda q: q["surrogate_loss"]))
                name = f"{fam}, best of {len(rs)} {what}"
                b.update({"method": name, "family": name, "time_s": sum(q["time_s"] for q in rs)})
                del b["start"]
                out.append(b)
    return out


def strict(r):
    """success and, for multi-basin targets, mean |dF error| <= 1 k_BT (catches basins that keep a few
    tenths of a percent of their mass, which the 0.5% rule still counts as populated)."""
    return r["success"] and (not r["multi_basin"] or r["df_mae"] is None or r["df_mae"] <= 1.0)


def summarize(rs):
    df = [r["df_median_abs_error"] for r in rs if r["multi_basin"] and r["df_median_abs_error"] is not None]
    return {"n": len(rs), "success": np.mean([r["success"] for r in rs]),
            "n_fail": int(sum(not r["success"] for r in rs)),
            "strict": np.mean([strict(r) for r in rs]),
            "df_med": np.median(df) if df else float("nan"), "n_multi": len(df),
            "df_mean": np.mean(df) if df else float("nan"),
            "df_le03": np.mean([d <= 0.3 for d in df]) if df else float("nan"),
            "kl_med": np.median([r["kl"] for r in rs]), "kl_mean": np.mean([r["kl"] for r in rs]),
            "tv_med": np.median([r["tv_basins"] for r in rs]),
            "ed_med": np.median([r["ed"] for r in rs]) if all("ed" in r for r in rs) else float("nan"),
            "ed_mean": np.mean([r["ed"] for r in rs]) if all("ed" in r for r in rs) else float("nan"),
            "eps_err_med": np.median([r["eps_err"] for r in rs]),
            "time_med": np.median([r["time_s"] for r in rs]), "time_mean": np.mean([r["time_s"] for r in rs])}


def write_table(runs, path):
    """One row per method family (sub-rows per GB-FESO start), families sorted by their median time per run
    and grouped by compute in units of one GB-FESO start (the median time of the GB-FESO family's runs)."""
    runs = runs + best_of_starts(runs)
    order = []
    for r in runs:
        if r["method"] not in order:
            order.append(r["method"])
    fams = []
    for r in runs:
        if r["family"] not in fams:
            fams.append(r["family"])
    gb = [r["time_s"] for r in runs if r["family"] == "GB-FESO"]
    unit = float(np.median(gb)) if gb else None
    tmed = {f: float(np.median([r["time_s"] for r in runs if r["family"] == f])) for f in fams}
    fams.sort(key=lambda f: (f.startswith("oracle"), tmed[f]))
    has_ed = all("ed" in r for r in runs)
    head = "| method | runs | success | success, mean dF err <= 1 k_BT | median dF err (k_BT) | mean dF err | dF err <= 0.3 | median KDE-KL | mean KDE-KL | median basin TV |"
    head += " median ED | mean ED |" if has_ed else ""
    head += " median abs(eps_hat - eps*) | time / run (s), median (mean) |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]

    def row(name, rs):
        s = summarize(rs)
        ed = f" {s['ed_med']:.4f} | {s['ed_mean']:.4f} |" if has_ed else ""
        lines.append(f"| {name} | {s['n']} | {100 * s['success']:.1f}% | {100 * s['strict']:.1f}% | {s['df_med']:.3f} (n={s['n_multi']}) | {s['df_mean']:.3f} | {100 * s['df_le03']:.0f}% | "
                     f"{s['kl_med']:.3f} | {s['kl_mean']:.3f} | {s['tv_med']:.3f} |{ed} {s['eps_err_med']:.3f} | {s['time_med']:.1f} ({s['time_mean']:.1f}) |")
    group = None
    for f in fams:
        if unit and not f.startswith("oracle"):
            g = max(1, int(round(tmed[f] / unit)))
            g = min((1, 2, 4, 6, 10, 20), key=lambda c: abs(np.log(c / g)))
            if g != group:
                group = g
                lines.append(f"| *compute ~{g} GB-FESO start{'s' if g > 1 else ''} ({g * unit:.0f} s)* |" + " |" * (head.count("|") - 2))
        elif f.startswith("oracle"):
            lines.append("| *noise floor* |" + " |" * (head.count("|") - 2))
        rs = [r for r in runs if r["family"] == f]
        row(f"**{f}** (all)", rs)
        subs = [m for m in order if m != f and any(r["method"] == m and r["family"] == f for r in runs)]
        if len(subs) == 1:      # a single variant is the family row itself
            subs = []
        for m in subs:
            row(f"&nbsp;&nbsp;{m}", [r for r in runs if r["method"] == m])
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--results", nargs="+", required=True)
    c.add_argument("--targets-ens", required=True)
    c.add_argument("--no-oracle", action="store_true", help="no oracle rows (evaluating extra methods only)")
    c.add_argument("--out", required=True)
    m = sub.add_parser("metrics")
    m.add_argument("--eval-eps", required=True)
    m.add_argument("--eval-ens", required=True)
    m.add_argument("--targets-ens", required=True)
    m.add_argument("--kl-n", type=int, default=4096)
    m.add_argument("--basins", choices=["their", "tuple"], default=None,
                   help="their utils.py boxes (N = 4 only; the default for N = 4) or the tuple basins (default N > 4)")
    m.add_argument("--suffix", default="", help="results_table<suffix>.md / per_run<suffix>.json, e.g. _N5")
    m.add_argument("--out", required=True)
    t = sub.add_parser("table", help="rewrite results_table<suffix>.md from <out>/per_run<suffix>.json")
    t.add_argument("--suffix", default="")
    t.add_argument("--add", nargs="*", default=None, help="per_run json files of extra methods to merge in first")
    t.add_argument("--out", required=True)
    sp = sub.add_parser("sampler", help="sampler accuracy at held-out eps*")
    sp.add_argument("--prior", required=True)
    sp.add_argument("--sampler", nargs="+", required=True)
    sp.add_argument("--gbfeso-ckpt", default=None)
    sp.add_argument("--targets-ens", required=True)
    sp.add_argument("--meta", required=True)
    sp.add_argument("--target-ids", default="0-9")
    sp.add_argument("--n", type=int, default=4096)
    sp.add_argument("--basins", choices=["their", "tuple"], default=None)
    sp.add_argument("--out", default=None)
    a = p.parse_args()
    {"collect": cmd_collect, "metrics": cmd_metrics, "table": cmd_table, "sampler": cmd_sampler}[a.cmd](a)


if __name__ == "__main__":
    main()
