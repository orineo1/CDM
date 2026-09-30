"""Redo only the post-refinement and the selection of a finished find_xstar run.

Reads a result JSON whose restarts saved their pre-refinement end points (refine.x_before), refines
every end point again with the chosen refinement (pathwise or zeroth-order gradient) and selects the
returned x_hat per attempt (lowest final loss, optionally re-estimated for the top K). The guided
trajectories are reused unchanged, so the comparison isolates refinement and selection. Output has the
find_xstar format (make_table1 --collect and diag_trajectories.py read it).

  python experiments/synthetic/pipeline/refine_endpoints.py --result runs/table1v2/x10_y1/MLGD_adam_refine.json \
      --refine-grad zo --select-top 3 --out runs/table1v2/x10_y1/MLGD_zo_sel3.json
"""
import argparse, json, os, sys, time, dataclasses
import torch

sys.path.insert(0, os.path.dirname(__file__))
import find_xstar as fx                                                       # noqa: E402
from cdm.guidance.core import refine, _final_loss                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--refine-grad", choices=["pathwise", "zo"], default="pathwise")
    p.add_argument("--refine-steps", type=int, default=None)
    p.add_argument("--refine-lr", type=float, default=None)
    p.add_argument("--zo-eps", type=float, default=0.05)
    p.add_argument("--zo-dirs", type=int, default=0)
    p.add_argument("--select-top", type=int, default=0)
    p.add_argument("--select-n", type=int, default=2000)
    p.add_argument("--select-repeats", type=int, default=5)
    p.add_argument("--sampler", default=None, help="override the conditional sampler path stored in the result")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    r = json.load(open(a.result))
    # current defaults first (results written before a flag existed lack it), then the saved settings
    argv0 = sys.argv
    sys.argv = ["find_xstar.py", "--prior", "_", "--sampler", "_", "--target", "_"]
    args = fx.parse_args()
    sys.argv = argv0
    for k, v in r["args"].items():
        setattr(args, k, v)
    for k in ("refine_grad", "zo_eps", "zo_dirs", "select_top", "select_n", "device"):
        setattr(args, k, getattr(a, k))
    if a.sampler is not None:
        args.sampler = a.sampler
    if a.refine_steps is not None:
        args.refine_steps = a.refine_steps
    if a.refine_lr is not None:
        args.refine_lr = a.refine_lr
    dev = args.device
    prior, _ = fx.load_model(args.prior, dev)
    if args.sampler == "exact":
        from cdm.models.samplers.exact import ExactConditionalSampler
        _, i0 = fx.load_target(args.target, "cpu")
        c0 = torch.load(args.target, map_location="cpu", weights_only=False)["condition_on"]
        sampler, arch = ExactConditionalSampler(i0["mu_list"], i0["Sigma_list"], i0["alpha"], c0), "exact"
    else:
        sampler, arch = fx.load_model(args.sampler, dev)
    stype = arch if args.sampler_type == "auto" else args.sampler_type
    gen, info = fx.load_target(args.target, dev)
    loss_fn = fx.make_loss(args)
    attempts = []
    for att in r["attempts"]:
        t0 = time.time()
        runs = []
        for q in att["restarts"]:
            cfg = fx.make_cfg(args, prior, stype, gen, q["seed"])
            torch.manual_seed(q["seed"])
            x0 = torch.tensor(q["refine"]["x_before"], device=dev).view(1, -1)
            x, _ = refine(prior, sampler, loss_fn, cfg, x0) if args.refine_steps > 0 else (x0, None)
            fl = float(_final_loss(sampler, loss_fn, cfg, x))
            xh = x.detach().float().view(-1)
            runs.append({"seed": q["seed"], "x_hat": xh.cpu().tolist(), "final_loss": fl,
                         "refine": {"x_before": q["refine"]["x_before"]}, **fx.evaluate(xh, info)})
        best = min(runs, key=lambda q: q["final_loss"])
        if a.select_top and len(runs) > 1:
            cfg_sel = dataclasses.replace(fx.make_cfg(args, prior, stype, gen, att["attempt"]), final_n_cond=a.select_n)
            cand = sorted(runs, key=lambda q: q["final_loss"])[:a.select_top]
            for q in cand:
                q["select_loss"] = float(sum(_final_loss(sampler, loss_fn, cfg_sel, torch.tensor(q["x_hat"], device=dev).view(1, -1)) for _ in range(a.select_repeats)) / a.select_repeats)
            best = min(cand, key=lambda q: q["select_loss"])
        attempts.append({"attempt": att["attempt"], **{k: v for k, v in best.items()}, "restarts": runs, "time_s": time.time() - t0})
        print(f"[{att['attempt']+1}/{len(r['attempts'])}] loss={best['final_loss']:.5f} ({time.time()-t0:.0f}s)", flush=True)
    summ = {}
    for k in ("final_loss", "l2_gmm", "l2_to_xstar", "time_s"):
        vals = [x[k] for x in attempts if k in x]
        if vals:
            import numpy as np
            summ[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    out = {**{k: r[k] for k in ("method", "algo", "guidance", "budget", "x_star") if k in r},
           "args": {**r["args"], "refine_grad": a.refine_grad, "zo_eps": a.zo_eps, "zo_dirs": a.zo_dirs,
                    "select_top": a.select_top, "select_n": a.select_n, "select_repeats": a.select_repeats, "refine_steps": args.refine_steps,
                    "refine_lr": args.refine_lr, "source": a.result, "sampler": args.sampler},
           "summary": summ, "attempts": attempts}
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
