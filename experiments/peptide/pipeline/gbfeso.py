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

N beads (`Design` read from the ensemble file; N = 4 with eps1, eps2 free runs their code untouched).
Their code is written for Y = 5 numbers (3 distances, 2 angles) and a 2-dimensional eps. For N > 4
(dim eps = K = number of free eps_d, dim Y = 2N - 3) `install_n_bead_patches` makes these changes,
all in this file (the GB-FESO checkout is not edited), and nothing else:
  dataset.py  GeometryFrameDatasetN: their GeometryFrameDataset with N - 1 distance columns (min-max to
              [-1, 1]) and N - 2 angle columns (/ pi) instead of 3 and 2, and K raw eps per trajectory.
  model.py    prepare_eps_cond: [B, K] instead of [B, 2]. epsilon_features: their features of (eps1, eps2)
              (eps1, eps2, eps1 + eps2, eps1 - eps2, eps1 eps2, eps1^2, eps2^2) generalized to K: all eps_i,
              then eps_i + eps_j, eps_i - eps_j, eps_i eps_j over pairs i < j, then eps_i^2 (identical
              for K = 2); FiLMResMLP.eps_mlp's first layer takes K + 3 K (K - 1) / 2 + K inputs instead of 7.
  loss.py     kde_kl_target_to_generated wraps the last N - 2 columns as angles (their default wraps
              the last 2 when dim != 2), via angle_indices.
  run_feso.py optimize_epsilon_condition, re-compiled from their source with four textual replacements
              (each asserted to match exactly once): eps_pair_raw.view(1, 2) -> view(1, -1);
              grad_pair = [None, None] -> [None] * K; best_pair / initial_pair / final_pair keep all K
              coordinates instead of (eps1, eps2). _parse_optimize_eps('all') -> all K coordinates;
              _eps_pair_tensor accepts K numbers. Their basin metrics (2-angle boxes) are switched off
              during the search (eval_basin_metrics = False: they are only printed and stored, they do
              not affect the optimization).
Everything else is theirs: Adam on raw eps, DDIM through the frozen model, KDE-KL (bandwidth 0.5,
kl_lambda 1) on the full raw geometry, decaying perturbations in phase 0 (std 0.005 when every start
coordinate is >= 0.9, their high-high rule), phase 1 when the best KL > 0.2, clamping, best iterate.
Starts for K > 2 (the four corners do not generalize): all-low (0.25, ..., 0.25), all-high (1.4, ..., 1.4)
(their low / high values) and rand<k>, uniform in the box, fixed per (target, k).
"""
import argparse
import functools
import inspect
import json
import math
import os
import sys
import textwrap
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common   # noqa: E402,F401  (puts GB-FESO on the path, aliases their renamed classes)
import dataset as gb_dataset   # noqa: E402
import train as gb_train   # noqa: E402
from simulate import EPS_BOX, Design   # noqa: E402

STARTS = {"low-low": (0.25, 0.25), "low-high": (0.25, 1.4), "high-low": (1.4, 0.25), "high-high": (1.4, 1.4)}
LOW, HIGH = 0.25, 1.4          # their low / high start values


def is_original(design):
    """N = 4 with (eps1, eps2) free: their code runs unpatched."""
    return design.n_beads == 4 and design.dim_x == 2


def default_starts(design):
    return "low-low,low-high,high-low,high-high" if design.dim_x == 2 else "all-low,all-high,rand1,rand2"


def start_eps(name, tid, K):
    if name in STARTS and K == 2:
        return list(STARTS[name])
    if name == "all-low":
        return [LOW] * K
    if name == "all-high":
        return [HIGH] * K
    assert name.startswith("rand"), f"unknown start {name}"
    gr = torch.Generator().manual_seed(1000 * tid + int(name[4:]))     # uniform in the box, fixed per (target, k)
    return (EPS_BOX[0] + (EPS_BOX[1] - EPS_BOX[0]) * torch.rand(K, generator=gr)).tolist()


# ---------------------------------------------------------------------------------------------------
# N-bead generalization of their code (see the module docstring); installed only when not is_original
# ---------------------------------------------------------------------------------------------------
class GeometryFrameDatasetN(gb_dataset.GeometryFrameDataset):
    """Their GeometryFrameDataset for N beads: x0 = [N - 1 distances min-max scaled to [-1, 1], N - 2 angles
    / pi], eps = K raw values. Same rules as theirs (eps range = data min / max per coordinate unless
    given; distance range from the data unless given)."""

    def __init__(self, traj_ds, design, eps_range=None, distance_range=None):
        self.traj_ds = traj_ds
        self.n_dist, self.n_ang, K = design.n_beads - 1, design.n_beads - 2, design.dim_x
        self.target_dim = design.dim_y
        self.distance_indices = tuple(range(self.n_dist))
        self.angle_indices = tuple(range(self.n_dist, self.target_dim))
        self.target_columns = tuple([f"r{i}{i + 1}" for i in range(1, design.n_beads)]
                                    + [f"theta{i - 1}{i}{i + 1}" for i in range(2, design.n_beads)])
        self.offsets, self.eps_raw, all_targets = [0], [], []
        for tid in range(len(traj_ds)):
            s_ = traj_ds[tid]
            t = s_["target"].detach().cpu().numpy().astype(np.float32)
            assert t.ndim == 2 and t.shape[1] == self.target_dim, t.shape
            self.offsets.append(self.offsets[-1] + t.shape[0])
            all_targets.append(t)
            e = s_["eps"].detach().cpu().numpy().astype(np.float32)
            assert e.shape == (K,), e.shape
            self.eps_raw.append(e)
        self.total = self.offsets[-1]
        ev = np.stack(self.eps_raw, 0)
        if eps_range is None:
            self.eps_min, self.eps_max = ev.min(0).astype(np.float32), ev.max(0).astype(np.float32)
        else:
            r = np.asarray(eps_range, dtype=np.float32)
            self.eps_min, self.eps_max = (np.full(K, r[0], np.float32), np.full(K, r[1], np.float32)) if r.shape == (2,) \
                else (r[0].astype(np.float32), r[1].astype(np.float32))
        targets = np.concatenate(all_targets, 0)
        nd = self.n_dist
        if distance_range is None:
            self.distance_min = targets[:, :nd].min(0).astype(np.float32)
            self.distance_max = targets[:, :nd].max(0).astype(np.float32)
        else:
            dr = np.asarray(distance_range, dtype=np.float32)
            assert dr.shape == (2, nd), dr.shape
            self.distance_min, self.distance_max = dr[0].copy(), dr[1].copy()
        bad = (self.distance_max - self.distance_min) <= 0
        self.distance_min[bad] -= 0.5
        self.distance_max[bad] += 0.5
        self.target_min = np.concatenate([self.distance_min, np.full(self.n_ang, -math.pi, np.float32)])
        self.target_max = np.concatenate([self.distance_max, np.full(self.n_ang, math.pi, np.float32)])

    def normalize_target(self, target):
        t = torch.as_tensor(target, dtype=torch.float32)
        nd = self.n_dist
        dmin = torch.as_tensor(self.distance_min, device=t.device)
        dmax = torch.as_tensor(self.distance_max, device=t.device)
        out = torch.empty_like(t)
        out[..., :nd] = 2.0 * (t[..., :nd] - dmin) / (dmax - dmin) - 1.0
        out[..., nd:] = t[..., nd:] / math.pi
        return out

    def denormalize_target(self, x_norm, clamp=True):
        x = torch.as_tensor(x_norm, dtype=torch.float32)
        if clamp:
            x = x.clamp(-1.0, 1.0)
        nd = self.n_dist
        dmin = torch.as_tensor(self.distance_min, device=x.device)
        dmax = torch.as_tensor(self.distance_max, device=x.device)
        out = torch.empty_like(x)
        out[..., :nd] = dmin + 0.5 * (x[..., :nd] + 1.0) * (dmax - dmin)
        a = math.pi * x[..., nd:]
        out[..., nd:] = torch.atan2(torch.sin(a), torch.cos(a))
        return out

    def target_angles(self, target):
        return target[..., self.n_dist:]

    def get_traj_eps_raw(self, tid):
        return tuple(float(v) for v in self.eps_raw[tid])


def n_eps_features(K):
    return K + 3 * K * (K - 1) // 2 + K


def _epsilon_features_K(eps_cond):
    """Their epsilon_features generalized to K eps (identical order and values for K = 2)."""
    K = eps_cond.shape[1]
    e = [eps_cond[:, i:i + 1] for i in range(K)]
    prs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    return torch.cat(e + [e[i] + e[j] for i, j in prs] + [e[i] - e[j] for i, j in prs]
                     + [e[i] * e[j] for i, j in prs] + [x.square() for x in e], dim=-1)


_PATCHED = {}


def install_n_bead_patches(design):
    """Generalize their code path to dim eps = K, dim Y = 2N - 3 (module docstring). Idempotent."""
    if is_original(design) or _PATCHED.get("design") == design.to_dict():
        return
    import model as gb_model
    import loss as gb_loss
    import run_feso
    K, N = design.dim_x, design.n_beads

    def prepare_eps_cond(eps_cond, batch_size, device, dtype=torch.float32):
        e = torch.as_tensor(eps_cond, device=device, dtype=dtype)
        e = e.view(1, K) if e.ndim == 1 else e
        assert e.ndim == 2 and e.shape[1] == K, f"eps_cond must have shape [B, {K}], got {tuple(e.shape)}"
        return e.expand(batch_size, K) if e.shape[0] == 1 and batch_size != 1 else e
    gb_model.prepare_eps_cond = prepare_eps_cond
    gb_model.epsilon_features = _epsilon_features_K

    orig_build = _PATCHED.setdefault("build_model", gb_train.build_model)

    def build_model(cfg):
        m = orig_build(cfg)
        old = m.model.eps_mlp[0]
        m.model.eps_mlp[0] = torch.nn.Linear(n_eps_features(K), old.out_features)
        return m
    gb_train.build_model = build_model

    angles = tuple(range(N - 1, 2 * N - 3))
    orig_kl = _PATCHED.setdefault("kde_kl", gb_loss.kde_kl_target_to_generated)
    run_feso.kde_kl_target_to_generated = functools.partial(orig_kl, angle_indices=angles)
    run_feso._eps_pair_tensor = lambda e, device=None, dtype=torch.float32: torch.as_tensor(e, device=device, dtype=dtype).reshape(-1)
    orig_parse = _PATCHED.setdefault("parse", run_feso._parse_optimize_eps)
    run_feso._parse_optimize_eps = lambda name: tuple(range(K)) if str(name).lower() in ("all", "both") else orig_parse(name)

    if "optimize" not in _PATCHED:
        src = textwrap.dedent(inspect.getsource(run_feso.optimize_epsilon_condition))
        for a, b in [
            ("eps_cond = eps_pair_raw.view(1, 2).expand(cfg.num_samples, 2)",
             "eps_cond = eps_pair_raw.view(1, -1).expand(cfg.num_samples, -1)"),
            ("grad_pair = [None, None]", "grad_pair = [None] * eps_pair_raw.numel()"),
            ("initial_pair_out = (eps1_value, eps2_value)",
             "initial_pair_out = tuple(float(v) for v in eps_pair_raw.detach().tolist())"),
            ("best_pair = (eps1_value, eps2_value)",
             "best_pair = tuple(float(v) for v in eps_pair_raw.detach().tolist())"),
            ('"final_pair": (history["eps1"][-1], history["eps2"][-1]),',
             '"final_pair": tuple(float(v) for v in eps_pair_raw.detach().tolist()),'),
        ]:
            assert src.count(a) == 1, f"their run_feso changed: {a!r} found {src.count(a)} times"
            src = src.replace(a, b)
        ns = {}
        exec(compile(src, run_feso.__file__ + " [N-bead patch]", "exec"), run_feso.__dict__, ns)
        _PATCHED["optimize"] = ns["optimize_epsilon_condition"]
    run_feso.optimize_epsilon_condition = _PATCHED["optimize"]
    _PATCHED["design"] = design.to_dict()
    print(f"[gbfeso] N-bead patches installed: {design}, dim eps {K}, dim Y {design.dim_y}, "
          f"{n_eps_features(K)} eps features, angle columns {angles}", flush=True)


def frame_dataset(traj, design, **kw):
    if is_original(design):
        return gb_dataset.GeometryFrameDataset(traj, **kw)
    return GeometryFrameDatasetN(traj, design, **kw)


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
    design = Design.from_dict(ens.get("design"))
    install_n_bead_patches(design)
    Y = ens["Y"][:, :a.frames_per_eps]
    frame_ds = frame_dataset(traj_list(Y, ens["eps"]), design)

    def fast_loader(root_dir, batch_size):
        x0 = frame_ds.normalize_target(Y.reshape(-1, Y.shape[-1]))
        e = ens["eps"].float().repeat_interleave(Y.shape[1], 0)
        tid = torch.arange(Y.shape[0]).repeat_interleave(Y.shape[1])
        return DataLoader(TensorDataset(x0, e, tid), batch_size=batch_size, shuffle=True, drop_last=True), frame_ds
    gb_train.make_dataloader = fast_loader

    cfg = gb_train.TrainConfig(root_dir="simulated", epochs=a.epochs, seed=a.seed, x_dim=design.dim_y)
    t0 = time.time()
    model, frame_ds_out = gb_train.train(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    gb_train.save_checkpoint(a.out, model, frame_ds_out, cfg)
    if not is_original(design):     # record the design in the checkpoint (their format has no field for it)
        ck = torch.load(a.out, weights_only=False)
        ck["peptide_design"] = design.to_dict()
        torch.save(ck, a.out)
    print(f"[gbfeso train] {Y.shape[0]} eps x {Y.shape[1]} frames, {a.epochs} epochs, {time.time() - t0:.0f}s -> {a.out}")


def load_model(path, frame_traj, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    design = Design.from_dict(ck.get("peptide_design"))
    install_n_bead_patches(design)
    cfg = gb_train.TrainConfig(**{k: v for k, v in ck["train_config"].items() if k in gb_train.TrainConfig.__dataclass_fields__})
    cfg.device = device
    model = gb_train.build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    frame_ds = frame_dataset(frame_traj, design, eps_range=(ck["eps_min"], ck["eps_max"]),
                             distance_range=(ck["distance_min"], ck["distance_max"]))
    return model, frame_ds, cfg, design


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
    model, frame_ds, cfg, design = load_model(a.ckpt, traj_list(Yt, te["eps"]), dev)
    assert Design.from_dict(te.get("design")).to_dict() == design.to_dict(), "targets and checkpoint designs differ"
    ids = parse_ids(a.target_ids, Yt.shape[0])
    starts = [s.strip() for s in (a.starts or default_starts(design)).split(",")]
    records = []
    if os.path.exists(a.out):
        records = json.load(open(a.out))["runs"]
    done = {(r["target_id"], r["start"]) for r in records}
    for tid in ids:
        for sname in starts:
            if (tid, sname) in done:
                continue
            e0 = start_eps(sname, tid, design.dim_x)
            torch.manual_seed(a.seed * 100003 + tid * 7 + starts.index(sname))
            init = dict(init_optimized_raw_1=e0[0], init_optimized_raw_2=e0[1]) if is_original(design) \
                else dict(init_optimized_raw_1=e0)
            oc = run_feso.EpsilonOptConfig(
                traj_id=tid, optimize_eps="both", kde_bandwidth=a.kde_bandwidth, sample_steps=a.sample_steps,
                **init, fixed_eps_raw=0.5, seed=a.seed,
                steps=a.steps, lr=a.lr, lr_refine=a.lr_refine, plot_steps=None, plot_dir=a.plot_dir,
                plot_loss_surface=False, num_samples=a.num_samples, clamp_eps=True, kl_lambda=a.kl_lambda,
                print_every=a.print_every, eval_basin_metrics=is_original(design))
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            res = run_feso.optimize_epsilon_condition(model, frame_ds, oc)
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            rec = {"target_id": tid, "start": sname, "start_eps": e0, "eps_star": te["eps"][tid].tolist(),
                   "eps_hat": list(res["eval_eps"]), "surrogate_loss": res["eval_loss"], "eval_phase": res["eval_phase"],
                   "phase1_ran": res["best_loss_1"] is not None, "time_s": dt,
                   "n_iters": len(res["history"]["loss"]) + (0 if res["history_1"] is None else len(res["history_1"]["loss"])),
                   "loss_trace": res["history"]["loss"] + ([] if res["history_1"] is None else res["history_1"]["loss"]),
                   "surrogate_metrics": {k: res["eval_metrics"].get(k) for k in ("missing_active_basin", "df_median_abs_error")}
                   if res["eval_metrics"] else None}
            records.append(rec)
            print(f"[gbfeso] target {tid} start {sname}: eps*={[round(v, 3) for v in rec['eps_star']]} "
                  f"eps_hat={[round(v, 3) for v in rec['eps_hat']]} KL={rec['surrogate_loss']:.4f} ({dt:.1f}s)", flush=True)
            json.dump({"method": a.method_name, "args": vars(a), "runs": records}, open(a.out, "w"))
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
    o.add_argument("--starts", default=None,
                   help="comma-separated: low-low, low-high, high-low, high-high (2 free eps), all-low, all-high, or "
                        "rand<k> (uniform in the box); default: the four corners for 2 free eps, else "
                        "all-low,all-high,rand1,rand2")
    o.add_argument("--method-name", default="GB-FESO", help="name of this variant in the results table")
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
