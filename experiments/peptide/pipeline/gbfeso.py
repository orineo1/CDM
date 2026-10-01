"""
GB-FESO baseline (Zick & Mendels 2026) run with THEIR code on our simulated ensembles.

  train : their train.train() (FiLM residual MLP DDPM, TrainConfig defaults: width 256, depth 7,
          1000 diffusion steps, AdamW lr 1.5e-4, batch 256, best-epoch checkpoint), with their
          GeometryFrameDataset normalization. Only the data plumbing is replaced: their loader parses
          LAMMPS dumps one frame at a time, ours hands the same normalized (x0, eps, tid) triples to
          the same loop from a tensor (identical content, shuffled mini-batches, drop_last).
  opt   : their run_feso.optimize_epsilon_condition() unchanged (Adam on raw eps, DDIM through the
          frozen model, KDE-KL with bandwidth 0.5 and kl_lambda 1, phase 0 with decaying random
          perturbations, phase 1 at lr_refine if the phase-0 best KL > 0.2, eps clamped to the
          training range, best-loss iterate returned), defaults of their CLI (201 steps, lr 5e-2,
          lr_refine 5e-2, 3500 samples), DDIM steps 40 (their paper; their code default is 100).
          Plotting is switched off (no-op plotters), which only removes figure I/O from the timing.

    python experiments/peptide/pipeline/gbfeso.py train --train-ens runs/peptide/train_ens.pt \
        --frames-per-eps 4000 --epochs 80 --out runs/peptide/models/gbfeso.pt
    python experiments/peptide/pipeline/gbfeso.py opt --ckpt runs/peptide/models/gbfeso.pt \
        --targets-ens runs/peptide/targets_ens.pt --target-ids 0-39 --out runs/peptide/results/gbfeso_0.json
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402,F401  (puts GB-FESO on the path, aliases their renamed classes)
import dataset as gb_dataset   # noqa: E402
import train as gb_train   # noqa: E402

STARTS = {"low-low": (0.25, 0.25), "low-high": (0.25, 1.4), "high-low": (1.4, 0.25), "high-high": (1.4, 1.4)}


def traj_list(Y, eps):
    return [{"target": Y[i].float(), "eps": eps[i].float()} for i in range(Y.shape[0])]


def parse_ids(s, n):
    if s in (None, "all"):
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def cmd_train(a):
    ens = torch.load(a.train_ens, weights_only=False)
    Y = ens["Y"][:, :a.frames_per_eps]
    frame_ds = gb_dataset.GeometryFrameDataset(traj_list(Y, ens["eps"]))

    def fast_loader(root_dir, batch_size):
        x0 = frame_ds.normalize_target(Y.reshape(-1, 5))
        e = ens["eps"].float().repeat_interleave(Y.shape[1], 0)
        tid = torch.arange(Y.shape[0]).repeat_interleave(Y.shape[1])
        return DataLoader(TensorDataset(x0, e, tid), batch_size=batch_size, shuffle=True, drop_last=True), frame_ds
    gb_train.make_dataloader = fast_loader

    cfg = gb_train.TrainConfig(root_dir="simulated", epochs=a.epochs, seed=a.seed)
    t0 = time.time()
    model, frame_ds_out = gb_train.train(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    gb_train.save_checkpoint(a.out, model, frame_ds_out, cfg)
    print(f"[gbfeso train] {Y.shape[0]} eps x {Y.shape[1]} frames, {a.epochs} epochs, {time.time() - t0:.0f}s -> {a.out}")


def load_model(path, frame_traj, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = gb_train.TrainConfig(**{k: v for k, v in ck["train_config"].items() if k in gb_train.TrainConfig.__dataclass_fields__})
    cfg.device = device
    model = gb_train.build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    frame_ds = gb_dataset.GeometryFrameDataset(frame_traj, eps_range=(ck["eps_min"], ck["eps_max"]),
                                               distance_range=(ck["distance_min"], ck["distance_max"]))
    return model, frame_ds, cfg


def cmd_opt(a):
    import plotters   # their plotting: switched off
    for name in dir(plotters):
        if name.startswith("plot_"):
            setattr(plotters, name, lambda *args, **kw: None)
    import run_feso
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te = torch.load(a.targets_ens, weights_only=False)
    g = torch.Generator().manual_seed(0)
    # their target is a trajectory; ours an ensemble of replica frames: shuffle so that their
    # linspace subsampling to max_target_points takes an unbiased subset
    Yt = torch.stack([y[torch.randperm(y.shape[0], generator=g)] for y in te["Y"]])
    model, frame_ds, cfg = load_model(a.ckpt, traj_list(Yt, te["eps"]), dev)
    ids = parse_ids(a.target_ids, Yt.shape[0])
    starts = [s.strip() for s in a.starts.split(",")]
    records = []
    if os.path.exists(a.out):
        records = json.load(open(a.out))["runs"]
    done = {(r["target_id"], r["start"]) for r in records}
    for tid in ids:
        for sname in starts:
            if (tid, sname) in done:
                continue
            e1, e2 = STARTS[sname]
            torch.manual_seed(a.seed * 100003 + tid * 7 + starts.index(sname))
            oc = run_feso.EpsilonOptConfig(
                traj_id=tid, optimize_eps="both", kde_bandwidth=a.kde_bandwidth, sample_steps=a.sample_steps,
                init_optimized_raw_1=e1, init_optimized_raw_2=e2, fixed_eps_raw=0.5, seed=a.seed,
                steps=a.steps, lr=a.lr, lr_refine=a.lr_refine, plot_steps=None, plot_dir=a.plot_dir,
                plot_loss_surface=False, num_samples=a.num_samples, clamp_eps=True, kl_lambda=a.kl_lambda,
                print_every=a.print_every)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            res = run_feso.optimize_epsilon_condition(model, frame_ds, oc)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"target_id": tid, "start": sname, "start_eps": [e1, e2], "eps_star": te["eps"][tid].tolist(),
                   "eps_hat": list(res["eval_eps"]), "surrogate_loss": res["eval_loss"], "eval_phase": res["eval_phase"],
                   "phase1_ran": res["best_loss_1"] is not None, "time_s": dt,
                   "n_iters": len(res["history"]["loss"]) + (0 if res["history_1"] is None else len(res["history_1"]["loss"])),
                   "loss_trace": res["history"]["loss"] + ([] if res["history_1"] is None else res["history_1"]["loss"]),
                   "surrogate_metrics": {k: res["eval_metrics"].get(k) for k in ("missing_active_basin", "df_median_abs_error")}
                   if res["eval_metrics"] else None}
            records.append(rec)
            print(f"[gbfeso] target {tid} start {sname}: eps*={[round(v, 3) for v in rec['eps_star']]} "
                  f"eps_hat={[round(v, 3) for v in rec['eps_hat']]} KL={rec['surrogate_loss']:.4f} ({dt:.1f}s)", flush=True)
            json.dump({"method": "GB-FESO", "args": vars(a), "runs": records}, open(a.out, "w"))
    print(f"[gbfeso] wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train-ens", required=True)
    t.add_argument("--frames-per-eps", type=int, default=4000)
    t.add_argument("--epochs", type=int, default=80)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    o = sub.add_parser("opt")
    o.add_argument("--ckpt", required=True)
    o.add_argument("--targets-ens", required=True)
    o.add_argument("--target-ids", default="all")
    o.add_argument("--starts", default="low-low,low-high,high-low,high-high")
    o.add_argument("--steps", type=int, default=201)
    o.add_argument("--lr", type=float, default=5e-2)
    o.add_argument("--lr-refine", type=float, default=5e-2)
    o.add_argument("--num-samples", type=int, default=3500)
    o.add_argument("--sample-steps", type=int, default=40)
    o.add_argument("--kde-bandwidth", type=float, default=0.5)
    o.add_argument("--kl-lambda", type=float, default=1.0)
    o.add_argument("--print-every", type=int, default=50)
    o.add_argument("--plot-dir", default="/tmp/gbfeso_plots_unused")
    o.add_argument("--seed", type=int, default=0)
    o.add_argument("--out", required=True)
    a = p.parse_args()
    {"train": cmd_train, "opt": cmd_opt}[a.cmd](a)


if __name__ == "__main__":
    main()
