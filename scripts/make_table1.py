"""
Table 1 end to end: for every (dim X, dim Y) scenario, generate the mixture and its data,
train the prior and the two conditional samplers, check them against the analytic mixture,
run the x*-finding algorithms, and collect the table.

    python scripts/make_table1.py                                   # all nine scenarios below
    python scripts/make_table1.py --settings 5x1 10x10 --attempts 3
    python scripts/make_table1.py --settings 1x1 --dry-run          # print the commands only
    python scripts/make_table1.py --collect                         # only rebuild the table from existing JSONs

A scenario is "<dim X>x<dim Y>": a random Gaussian mixture with 4 components in dim X + dim Y
dimensions in which two components share the x-part of their mean, so x* is that shared
vector and the target G = P(Y | X = x*) is bimodal (scripts/generate_data.py --random-mog).
The paper's parameter files can be used instead with "file:<path>".

Per scenario, under <runs-dir>/<scenario>/ (every step is skipped when its output exists,
--force redoes it):
  data.pt                          scripts/generate_data.py
  models/uncond/model_unconditional.pt      scripts/train_models.py --model unconditional
  models/diffusion/model_conditional.pt     scripts/train_models.py --model conditional --architecture diffusion
  models/cm/model_conditional.pt            scripts/train_models.py --model conditional --architecture cm
  check.json / check.png           scripts/check_models.py (networks vs. the analytic mixture)
  <METHOD>.json                    scripts/find_xstar.py, METHOD in MLGD, MLGD-F, PMLGD, PMLGD-F:
                                   --n-attempts independent runs, each returning the best of
                                   --budget trajectories (restarts or particles)

Output: <out>.json (everything), <out>.md and <out>.tex (the table: exact L2 between
P(Y | X = x_hat) and G, mean +- std over the attempts, and the time per attempt), plus
<out>_appendix.md (per-scenario model-check numbers, basin hit rates, resamplings).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(REPO, "scripts")
METHODS = ["MLGD", "MLGD-F", "PMLGD", "PMLGD-F"]
DEFAULT_SETTINGS = ["1x1", "1x5", "1x10", "5x1", "5x5", "5x10", "10x1", "10x5", "10x10"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", nargs="+", default=DEFAULT_SETTINGS, help="'<dimX>x<dimY>' or 'file:<params.txt>'")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--runs-dir", default="runs/table1")
    p.add_argument("--out", default="results/table1")
    g = p.add_argument_group("data and training")
    g.add_argument("--n-components", type=int, default=4)
    g.add_argument("--n-samples", type=int, default=20000)
    g.add_argument("--epochs", type=int, default=3000, help="training epochs for each network")
    g.add_argument("--batch-size", type=int, default=1024)
    g.add_argument("--nunits", type=int, default=128)
    g.add_argument("--nblocks", type=int, default=3)
    g = p.add_argument_group("x* search")
    g.add_argument("--budget", type=int, default=10, help="trajectories per attempt: restarts (MLGD) or particles (PMLGD)")
    g.add_argument("--attempts", type=int, default=5, help="independent runs of each algorithm; mean/std over them")
    g.add_argument("--n-cond", type=int, default=250)
    g.add_argument("--resample-beta", type=float, default=5.0)
    g.add_argument("--refine-steps", type=int, default=0)
    g.add_argument("--extra", default="", help="extra flags passed verbatim to find_xstar.py")
    g = p.add_argument_group("control")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", default=None)
    g.add_argument("--force", action="store_true", help="redo steps whose outputs exist")
    g.add_argument("--skip-check", action="store_true")
    g.add_argument("--collect", action="store_true", help="only rebuild the table from existing JSONs")
    g.add_argument("--no-collect", action="store_true", help="run the scenarios but do not write the table (array jobs)")
    g.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def run(cmd, args, log=None):
    print("  $ " + " ".join(cmd), flush=True)
    if args.dry_run:
        return 0
    t0 = time.time()
    with open(log, "a") if log else open(os.devnull, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    print(f"    -> exit {r.returncode} in {time.time() - t0:.0f}s", flush=True)
    if r.returncode != 0:
        tail = open(log).read()[-3000:] if log else ""
        raise SystemExit(f"step failed (see {log}):\n{tail}")
    return r.returncode


def parse_setting(tok):
    if tok.startswith("file:"):
        return {"name": os.path.splitext(os.path.basename(tok[5:]))[0], "params_file": tok[5:]}
    dx, dy = tok.lower().split("x")
    return {"name": f"x{dx}_y{dy}", "dim_x": int(dx), "dim_y": int(dy)}


def prepare(setting, args):
    """data, three networks, model check. Returns the paths."""
    py = sys.executable
    d = os.path.join(args.runs_dir, setting["name"])
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "log.txt")
    paths = {"data": os.path.join(d, "data.pt"),
             "prior": os.path.join(d, "models", "uncond", "model_unconditional.pt"),
             "diffusion": os.path.join(d, "models", "diffusion", "model_conditional.pt"),
             "cm": os.path.join(d, "models", "cm", "model_conditional.pt"),
             "check": os.path.join(d, "check")}
    dev = ["--device", args.device] if args.device else []

    if args.force or not os.path.exists(paths["data"]):
        cmd = [py, os.path.join(SCRIPTS, "generate_data.py"), "--n-samples", str(args.n_samples), "--output", paths["data"],
               "--seed", str(args.seed)]
        if "params_file" in setting:
            cmd += ["--params-file", setting["params_file"]]
        else:
            cmd += ["--random-mog", "--dim", str(setting["dim_x"] + setting["dim_y"]), "--condition-on", str(setting["dim_x"]),
                    "--n-components", str(args.n_components)]
        run(cmd, args, log)
    common = ["--data", paths["data"], "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
              "--nunits", str(args.nunits), "--nblocks", str(args.nblocks), "--seed", str(args.seed)] + dev
    if args.force or not os.path.exists(paths["prior"]):
        run([py, os.path.join(SCRIPTS, "train_models.py"), "--model", "unconditional", "--architecture", "diffusion",
             "--output-dir", os.path.dirname(paths["prior"])] + common, args, log)
    for arch in ("diffusion", "cm"):
        if args.force or not os.path.exists(paths[arch]):
            run([py, os.path.join(SCRIPTS, "train_models.py"), "--model", "conditional", "--architecture", arch,
                 "--output-dir", os.path.dirname(paths[arch])] + common, args, log)
    if not args.skip_check and (args.force or not os.path.exists(paths["check"] + ".json")):
        run([py, os.path.join(SCRIPTS, "check_models.py"), "--prior", paths["prior"], "--sampler", paths["diffusion"], paths["cm"],
             "--data", paths["data"], "--out", paths["check"]] + dev, args, log)
    return paths, log


def search(setting, paths, log, method, args):
    py = sys.executable
    out = os.path.join(args.runs_dir, setting["name"], f"{method}.json")
    if not args.force and os.path.exists(out):
        return out
    sampler = paths["cm"] if method.endswith("-F") else paths["diffusion"]
    algo = "particles" if method.startswith("P") else "lgd"
    cmd = [py, os.path.join(SCRIPTS, "find_xstar.py"), "--prior", paths["prior"], "--sampler", sampler, "--target", paths["data"],
           "--algo", algo, "--n-restarts", str(args.budget), "--n-particles", str(args.budget),
           "--n-attempts", str(args.attempts), "--n-cond", str(args.n_cond), "--resample-beta", str(args.resample_beta),
           "--refine-steps", str(args.refine_steps), "--seed", str(args.seed), "--out", out, "--no-store"]
    if args.device:
        cmd += ["--device", args.device]
    if args.extra:
        cmd += args.extra.split()
    run(cmd, args, log)
    return out


# ----------------------------------------------------------------------------
# collect
# ----------------------------------------------------------------------------
def collect(settings, args):
    rows, appendix, everything = [], [], {}
    for st in settings:
        d = os.path.join(args.runs_dir, st["name"])
        entry = {"setting": st}
        cp = os.path.join(d, "check.json")
        if os.path.exists(cp):
            entry["check"] = json.load(open(cp))
        for m in args.methods:
            fp = os.path.join(d, f"{m}.json")
            if not os.path.exists(fp):
                continue
            r = json.load(open(fp))
            entry[m] = {"summary": r["summary"], "budget": r["budget"], "n_attempts": len(r["attempts"]),
                        "attempts": [{k: a.get(k) for k in ("x_hat", "final_loss", "l2_gmm", "l2_to_xstar", "time_s", "n_resamples")}
                                     for a in r["attempts"]]}
            # basin hit rate over all trajectories of all attempts, when x* is known: a trajectory
            # counts as a hit if its x_hat is closer to x* than half the distance to the nearest other cluster centre
            hits, total = None, 0
            if r.get("x_star") is not None:
                data = _load_mixture(os.path.join(d, "data.pt"))
                if data is not None:
                    xs = np.array(r["x_star"]); c = len(xs)
                    centres = np.unique(np.round(np.array([m_[:c] for m_ in data["mu_list"]]), 6), axis=0)
                    others = [o for o in centres if not np.allclose(o, xs)]
                    half = 0.5 * min(np.linalg.norm(o - xs) for o in others) if others else np.inf
                    pts = []
                    for a in r["attempts"]:
                        pts += [q["x_hat"] for q in a.get("restarts", a.get("particles", [a]))]
                    hits = int(sum(np.linalg.norm(np.array(p_) - xs) < half for p_ in pts)); total = len(pts)
            entry[m]["hits"] = None if hits is None else f"{hits}/{total}"
            s = r["summary"]
            rows.append({"setting": st["name"], "dim_x": st.get("dim_x"), "dim_y": st.get("dim_y"), "method": m,
                         "l2_mean": s["l2_gmm"]["mean"], "l2_std": s["l2_gmm"]["std"], "n": s["l2_gmm"]["n"],
                         "time_mean": s["time_s"]["mean"], "hits": entry[m]["hits"],
                         "resamplings": (np.mean([a["n_resamples"] for a in r["attempts"]]) if m.startswith("P") else None)})
        everything[st["name"]] = entry
    return rows, everything


def _load_mixture(path):
    try:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        return {"mu_list": [m.numpy() for m in d["mu_list"]]}
    except Exception:
        return None


def write_tables(rows, everything, args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"rows": rows, "scenarios": everything, "args": vars(args)}, f, indent=2)
    # markdown
    md = ["| dim X | dim Y | method | L2 GMM of returned x_hat (mean +- std) | time / attempt (s) | in x* basin | resamplings |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        rs = "" if r["resamplings"] is None else f"{r['resamplings']:.1f}"
        md.append(f"| {r['dim_x']} | {r['dim_y']} | {r['method']} | {r['l2_mean']:.4f} +- {r['l2_std']:.4f} (n={r['n']}) | "
                  f"{r['time_mean']:.0f} | {r['hits'] or ''} | {rs} |")
    open(args.out + ".md", "w").write("\n".join(md) + "\n")
    # latex rows (booktabs body, same column order as the paper's table)
    tex = []
    last = None
    for r in rows:
        key = (r["dim_x"], r["dim_y"])
        if key != last:
            if last is not None:
                tex.append("\\midrule")
            tex.append(f"\\multirow{{{sum(1 for q in rows if (q['dim_x'], q['dim_y']) == key)}}}{{*}}{{$\\dim X={r['dim_x']}$, $\\dim Y={r['dim_y']}$}}")
            last = key
        name = {"MLGD": "\\mlgd{}", "MLGD-F": "\\mlgdf{}", "PMLGD": "P\\mlgd{}", "PMLGD-F": "P\\mlgdf{}"}[r["method"]]
        tex.append(f" & {name} & ${r['l2_mean']:.3f} \\pm {r['l2_std']:.3f}$ & ${r['time_mean']:.0f}$ \\\\")
    open(args.out + ".tex", "w").write("\n".join(tex) + "\n")
    # appendix: model checks, hits, resamplings
    ap = ["# Table 1 appendix: networks and search diagnostics", ""]
    for name, e in everything.items():
        ap.append(f"## {name}")
        if "check" in e:
            c = e["check"]
            sc = c.get("score", [])
            worst = max((o["rel_fisher"] for o in sc if o["t"] > 1), default=float("nan"))
            pr = c.get("prior", {})
            ap.append(f"- score network: max relative Fisher error over t > 1: {worst:.3f}; prior samples SWD (norm) {pr.get('swd_norm', float('nan')):.3f}")
            for k in ("cond_diffusion", "cond_cm"):
                if k in c:
                    pts = c[k]["points"]
                    at = pts.get("x*", {}).get("mean", float("nan"))
                    ap.append(f"- {k}: {c[k]['metric_at_points']} at x* {at:.3f}; SWD over random x {c[k]['swd_norm_random_x_mean']:.3f}")
        for m in args.methods:
            if m in e:
                ap.append(f"- {m}: {e[m]['n_attempts']} attempts x budget {e[m]['budget']}; in x* basin {e[m]['hits']}; "
                          f"per-attempt L2 {[round(a['l2_gmm'], 4) for a in e[m]['attempts']]}")
        ap.append("")
    open(args.out + "_appendix.md", "w").write("\n".join(ap))
    print("\n".join(md))
    print(f"[table] wrote {args.out}.json / .md / .tex / _appendix.md")


def main():
    args = parse_args()
    settings = [parse_setting(t) for t in args.settings]
    if not args.collect:
        for st in settings:
            print(f"=== scenario {st['name']} ===", flush=True)
            paths, log = prepare(st, args)
            for m in args.methods:
                search(st, paths, log, m, args)
    if args.dry_run or args.no_collect:
        return
    rows, everything = collect(settings, args)
    write_tables(rows, everything, args)


if __name__ == "__main__":
    main()
