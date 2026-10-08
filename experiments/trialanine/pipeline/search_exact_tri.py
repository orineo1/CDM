"""Trialanine count targets with the EXACT conditional (no trained sampler): the search question in isolation.

Exact loss: for a standardized design x_std (x = x_scale * x_std, kJ/mol), the reference frames are reweighted,
w_i(x) ∝ exp(-(x . B_i)/kT) * w_i(x0)  (rw.py: log w = base - B x / kT), and
    L(x) = KL(G_count || P_count(x)),   P_count(x)_n = sum_{i: n_i = n} w_i(x)
the label-free count distribution (n = number of residues in the phi ~ +60 basin). It is a weighted expectation
over a fixed frame set, differentiable in x through the weights (torch autograd), with no sampling noise.
Every method below sees this same function; the number of loss(+gradient) evaluations is counted.

  mlgd    search_mog.py (shared driver, all its flags) with its loss-oracle mode: the 'sampler' returns its
          condition (cdm.losses.exact_mmd.IdentitySampler) and the loss is L at the Tweedie estimate, i.e. the
          weighted-expectation loss is used directly inside the guidance steps, the refinement and the
          re-selection (no resampling of frames). Prior: the trained unconditional diffusion over x_std.
  gbfeso  their optimizer loop (search_gbfeso.run_phase settings: Adam lr 0.05, 201 steps, perturbation noise
          0.05 decaying to 0, clamping to the box, best iterate kept; phase 1 at lr 0.05 without noise when the
          best phase-0 loss > 0.2) on L; starts: --init zero (ff14SB), randn (prior draws, truncated to the box),
          json (the --init-top best restarts of an MLGD result: the hybrid), or --n-starts K random starts.
          --match-evals FILE sets K so that the evaluations match the MLGD run in FILE (equal compute).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "..", "synthetic", "pipeline"))
from rw import Frames   # noqa: E402
import system as S      # noqa: E402


class ExactCountLoss:
    def __init__(self, frames_glob, G_count, x_scale, device="cpu", thin=1, k=None, p_star=None, wP=0.0):
        F = Frames(glob.glob(frames_glob), thin=thin)
        dt = torch.float64
        self.B = torch.tensor(F.B, dtype=dt, device=device)
        self.base = torch.tensor(F.base, dtype=dt, device=device)
        self.cls = torch.tensor(S.count_plus(F.phi), device=device)
        self.G = torch.tensor(G_count, dtype=dt, device=device)
        self.kT, self.scale, self.N, self.device = F.kT, x_scale, F.N, device
        # optional residue-specific restraint (deceptive-target benchmark): + wP * (P(phi_k '+' | x) - p_star)^2
        self.k, self.p_star, self.wP = k, p_star, wP
        if k is not None:
            d = np.degrees(F.phi[:, k])
            self.plus_k = torch.tensor(((d > 0) & (d < 120)).astype(float), dtype=dt, device=device)
        self.n_evals = 0

    def counts(self, x_std):
        x = x_std.reshape(-1).to(self.device, torch.float64) * self.scale
        lw = self.base - self.B @ x / self.kT
        w = torch.softmax(lw, 0)
        return torch.zeros(4, dtype=w.dtype, device=w.device).index_add(0, self.cls, w), w

    def __call__(self, x_std):
        self.n_evals += 1
        P, w = self.counts(x_std)
        G = self.G
        m = G > 0
        L = (G[m] * (torch.log(G[m]) - torch.log(P[m].clamp_min(1e-300)))).sum()
        if self.k is not None and self.wP > 0:
            L = L + self.wP * (w @ self.plus_k - self.p_star) ** 2
        return L.float()

    def ess(self, x_std):
        with torch.no_grad():
            _, w = self.counts(torch.as_tensor(x_std))
            return float(1.0 / (w ** 2).sum())


def target_info(meta, target_npy, target_id=None):
    tid = target_id if target_id is not None else int(os.path.basename(target_npy).split("_")[1].split(".")[0])
    t = [q for q in meta["targets"] if q["id"] == tid][0]
    return tid, t


def make_loss(a, meta, t):
    if a.wP > 0:   # restraint on the '+' population of the x*-mode residue
        k = t["mode_star"]
        return ExactCountLoss(a.frames, t["G_count"], meta["x_scale_kJ"], a.device, k=k, p_star=t["plus_marginals"][k], wP=a.wP)
    return ExactCountLoss(a.frames, t["G_count"], meta["x_scale_kJ"], a.device)


def run_mlgd(a, rest):
    import search_mog
    meta = json.load(open(a.meta)); tid, t = target_info(meta, a.target, a.target_id)
    L = make_loss(a, meta, t)
    _load = search_mog.load_target

    def load_target(path, device):
        gen, info = _load(path, device)
        info.update(mu_list="exact-count", Sigma_list=None, alpha=None, target=None)   # oracle mode, no GMM metrics
        return gen, info
    search_mog.load_target = load_target
    search_mog.ExactConditionalMMD = lambda *args: L
    sys.argv = [sys.argv[0], "--sampler", "exactloss", "--target", a.target, "--out", a.out] + rest
    t0 = time.time()
    search_mog.main()
    J = json.load(open(a.out))
    J.update(method=a.method_name or "MLGD-exact", n_loss_evals=L.n_evals, wall_s=time.time() - t0, target_id=tid,
             ess_best=L.ess(torch.tensor(J["attempts"][0]["x_hat"])), device=a.device)
    json.dump(J, open(a.out, "w"), indent=1)
    print(f"[exact mlgd] {L.n_evals} loss evaluations, {time.time() - t0:.0f}s, loss {J['attempts'][0]['final_loss']:.5f}")


def adam_phase(L, x0, steps, lr, noise, box):
    x = x0.clone().clamp(-box, box).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    best, bx, trace = float("inf"), x.detach().clone(), []
    for k in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = L(x)
        v = float(loss.detach()); trace.append(v)
        if v < best:
            best, bx = v, x.detach().clone()
        if k == steps:
            break
        loss.backward(); opt.step()
        with torch.no_grad():
            if noise > 0:
                x.add_(noise * (1.0 - k / max(steps, 1)) * torch.randn_like(x))
            x.clamp_(-box, box)
    return bx, best, trace


def run_gbfeso(a):
    meta = json.load(open(a.meta)); tid, t = target_info(meta, a.target, a.target_id)
    L = make_loss(a, meta, t)
    torch.manual_seed(a.seed)
    c = S.D
    n = a.n_starts
    if a.match_evals:
        J = json.load(open(a.match_evals))
        per = (a.steps + 1) * (2 if a.phase1 == "always" else 1)
        n = max(1, int(round(J["n_loss_evals"] / per)))
    if a.init == "zero":
        starts = [torch.zeros(c)]
    elif a.init == "json":
        J = json.load(open(a.init_json)); rs = J["attempts"][0]["restarts"]
        key = lambda q: (0, q["select_loss"]) if q.get("select_loss") is not None else (1, q["final_loss"])  # noqa: E731
        starts = [torch.tensor(q["x_hat"]).float() for q in sorted(rs, key=key)[:a.init_top]]
    else:
        starts = []
        for _ in range(n):
            z = torch.randn(c)
            while (z.abs() > a.box).any():
                z = torch.where(z.abs() > a.box, torch.randn_like(z), z)
            starts.append(z)
    t0, runs = time.time(), []
    for r, x0 in enumerate(starts):
        torch.manual_seed(a.seed * 1000 + r)
        b0, l0, tr0 = adam_phase(L, x0.to(a.device), a.steps, a.lr, a.noise_std, a.box)
        do1 = a.phase1 == "always" or (a.phase1 == "threshold" and l0 > a.phase1_threshold)
        b1, l1 = (adam_phase(L, b0, a.steps, a.lr, 0.0, a.box)[:2]) if do1 else (b0, float("inf"))
        xh, lh = (b1, l1) if l1 < l0 else (b0, l0)
        runs.append(dict(x0=x0.tolist(), x_hat=xh.view(-1).tolist(), final_loss=lh, phase1=bool(do1), trace0=tr0[::10]))
    best = min(runs, key=lambda q: q["final_loss"])
    name = a.method_name or f"GBFESO-exact-{a.init}{len(starts)}"
    json.dump(dict(method=name, args=vars(a), target_id=tid, best=dict(x_hat=best["x_hat"], final_loss=best["final_loss"]),
                   restarts=runs, n_starts=len(starts), time_s=time.time() - t0, n_loss_evals=L.n_evals,
                   n_conditional_samples=0, ess_best=L.ess(torch.tensor(best["x_hat"])), device=a.device),
              open(a.out, "w"), indent=1)
    print(f"[exact gbfeso] {name}: {len(starts)} starts, loss {best['final_loss']:.6f}, {L.n_evals} evals, {time.time() - t0:.0f}s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["mlgd", "gbfeso"])
    p.add_argument("--meta", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--frames", default="runs/trialanine/prod_x0_w*.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--method-name", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--box", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=201)
    p.add_argument("--lr", type=float, default=5e-2)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--phase1", choices=["always", "threshold", "never"], default="threshold")
    p.add_argument("--phase1-threshold", type=float, default=0.2)
    p.add_argument("--init", choices=["zero", "randn", "json"], default="zero")
    p.add_argument("--n-starts", type=int, default=1)
    p.add_argument("--match-evals", default=None)
    p.add_argument("--init-json", default=None)
    p.add_argument("--init-top", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-id", type=int, default=None, help="target id in --meta (else parsed from --target's name)")
    p.add_argument("--wP", type=float, default=0.0, help="weight of the residue-specific '+' restraint (0: count only)")
    a, rest = p.parse_known_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    if a.mode == "mlgd":
        run_mlgd(a, rest + ["--device", a.device, "--seed", str(a.seed)])
    else:
        run_gbfeso(a)


if __name__ == "__main__":
    main()
