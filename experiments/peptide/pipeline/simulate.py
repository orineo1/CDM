"""
Batched GPU Langevin simulator of the N-bead 2D Lennard-Jones chain of Zick & Mendels (2026), GB-FESO
(arXiv:2607.06421), a PyTorch port of their LAMMPS input (`GB-FESO/lammps_scripts/gener.py`), generalized
from their 4 beads to N beads. N = 4 with the default design is their benchmark, unchanged.

Physics (LJ units, 2D, N beads of mass 1):
  bonds (i, i+1): E = K (r - r0)^2, K = 0.15, r0 = 1.5          (LAMMPS bond_style harmonic, no 1/2)
  LJ on all N(N-1)/2 pairs (special_bonds lj 1 1 1), sigma = 1, plain lj/cut 4.0 (no shift); pair (i, j)
     has strength eps_d, d = |i - j| the sequence distance, d = 1..N-1.
  Design (`Design`): which eps_d are free (the search variable X) and the values of the fixed ones.
     Default N = 4: eps1, eps2 free, eps3 = 0.5 fixed (their benchmark); N >= 5: all N - 1 free.
  fix langevin T = 0.09, damp = 1 (gamma = 1) + fix nve (velocity Verlet), dt = 0.005,
  start: straight chain with spacing 1.0, velocities drawn at T = 0.09 with zero net momentum.
The step follows LAMMPS: half kick with the stored total force, drift, new force = conservative +
friction on the half-step velocity + random force (Gaussian, same variance 2 gamma kT / dt as the
LAMMPS uniform noise), second half kick.

Output frames: Y = (r_12, ..., r_{N-1,N}, theta_123, ..., theta_{N-2,N-1,N}), 2N - 3 numbers: the N - 1 bond
lengths, then the N - 2 signed xy-plane bond angles via atan2, exactly as their dataset.py
(`_signed_angle_xy`): theta_{i-1,i,i+1} is the angle from (i -> i-1) to (i -> i+1).

All chains (n_eps x replicas) advance in one tensor. The inner loop of `stride` steps is captured once
as a CUDA graph (one replay per saved frame), so the per-step Python overhead disappears.

    from simulate import simulate, Design
    Y = simulate(eps, n_rep=8, n_steps=4_000_000, stride=200, burn=50_000)   # N = 4: [n_eps, n_rep * n_frames, 5]
    Y = simulate(eps, design=Design(6))                                      # eps [n_eps, 5] -> Y [..., 9]
"""
import math
import time

import torch

KBOND, R0, EPS3, SIGMA, RCUT = 0.15, 1.5, 0.5, 1.0, 4.0
TEMP, GAMMA, DT = 0.09, 1.0, 0.005
EPS_BOX = (0.21, 1.55)          # their gener.py range for eps1, eps2 (used for every free eps_d)


class Design:
    """The design space of the N-bead chain: eps_d for sequence distance d = 1..N-1; `free` lists the d
    that the search varies (in this order, X = (eps_d for d in free)), `fixed` maps the others to values.
    Default: N = 4 -> free (1, 2), fixed {3: 0.5} (their benchmark); N >= 5 -> all free."""

    def __init__(self, n_beads=4, free=None, fixed=None):
        n = int(n_beads)
        assert n >= 3
        if free is None:
            free = (1, 2) if n == 4 else tuple(range(1, n))
        fixed = dict(fixed) if fixed is not None else ({3: EPS3} if n == 4 and tuple(free) == (1, 2) else {})
        fixed = {int(d): float(v) for d, v in fixed.items()}
        free = tuple(int(d) for d in free)
        missing = [d for d in range(1, n) if d not in free and d not in fixed]
        assert not missing, f"eps_d for d in {missing} is neither free nor fixed"
        assert all(1 <= d < n for d in free) and not set(free) & set(fixed), (free, fixed)
        self.n_beads, self.free, self.fixed = n, free, {d: fixed[d] for d in sorted(fixed) if d < n}

    # sizes
    @property
    def dim_x(self):
        return len(self.free)

    @property
    def dim_y(self):
        return 2 * self.n_beads - 3

    @property
    def n_bonds(self):
        return self.n_beads - 1

    def to_dict(self):
        return {"n_beads": self.n_beads, "free": list(self.free), "fixed": {str(d): v for d, v in self.fixed.items()}}

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls(4)
        return cls(d["n_beads"], d["free"], {int(k): v for k, v in d["fixed"].items()})

    @classmethod
    def from_args(cls, n_beads=4, free=None, fixed=None):
        """CLI form: free '1,2,3' (None = default), fixed '3=0.5,4=0.2' (None = default)."""
        fr = None if free in (None, "") else tuple(int(s) for s in str(free).split(","))
        fx = None if fixed is None else ({} if fixed == "" else
                                         {int(k): float(v) for k, v in (s.split("=") for s in str(fixed).split(","))})
        return cls(n_beads, fr, fx)

    def __repr__(self):
        return f"Design(N={self.n_beads}, free={self.free}, fixed={self.fixed})"

    def pairs(self):
        """(i, j, d) for every pair, ordered by sequence distance d then i: the bonds come first. For N = 4
        this is (0,1),(1,2),(2,3),(0,2),(1,3),(0,3), the order of the original 4-bead code."""
        return [(i, i + d, d) for d in range(1, self.n_beads) for i in range(self.n_beads - d)]

    def pair_eps(self, eps):
        """[B, dim_x] free eps -> [B, n_pairs] LJ epsilon of each pair."""
        col = {d: eps[:, k:k + 1] for k, d in enumerate(self.free)}
        cols = []
        for d in range(1, self.n_beads):
            c = col[d] if d in col else torch.full_like(eps[:, :1], self.fixed[d])
            cols += [c] * (self.n_beads - d)
        return torch.cat(cols, dim=1)


# the 4-bead constants of the original code (kept for imports)
PI = (0, 1, 2, 0, 1, 0)
PJ = (1, 2, 3, 2, 3, 3)
BOND_MASK = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _incidence(device, design=None):
    design = design or Design(4)
    P = design.pairs()
    A = torch.zeros(design.n_beads, len(P), device=device)
    for k, (i, j, _) in enumerate(P):
        A[j, k] += 1.0
        A[i, k] -= 1.0
    return A


def _bond_mask(design):
    return torch.tensor([1.0 if d == 1 else 0.0 for (_, _, d) in design.pairs()])


def pair_eps(eps, design=None):
    """[B, dim_x] free eps -> [B, n_pairs]; N = 4 default: (eps1, eps2) -> (e1, e1, e1, e2, e2, 0.5)."""
    return (design or Design(4)).pair_eps(eps)


def conservative_force(x, peps, A, bmask):
    """x [B, N, 2], peps [B, P] -> force [B, N, 2]. A = _incidence(device): bead j of pair k gets +f_k."""
    d = torch.einsum("pk,bpd->bkd", A, x)               # [B, P, 2], vector i -> j  (x_j - x_i)
    r2 = (d * d).sum(-1)
    inv2 = SIGMA ** 2 / r2
    inv6 = inv2 ** 3
    inside = (r2 < RCUT ** 2).to(x.dtype)
    # -dU/dr / r for LJ: 24 eps (2 s^12/r^14 - s^6/r^8)
    f_lj = 24.0 * peps * (2.0 * inv6 * inv6 - inv6) / r2 * inside
    r = torch.sqrt(r2)
    f_b = -2.0 * KBOND * (r - R0) / r * bmask
    fpair = (f_lj + f_b)[..., None] * d                 # force on bead j; bead i gets minus
    return torch.einsum("pk,bkd->bpd", A, fpair)


def internal_coords(x):
    """x [..., N, 2] -> [..., 2N - 3] = (r_12, ..., r_{N-1,N}, theta_123, ..., theta_{N-2,N-1,N})."""
    n = x.shape[-2]
    back = [x[..., i - 1, :] - x[..., i, :] for i in range(n)]          # back[i] = i -> i-1  (i >= 1)
    fwd = [x[..., i + 1, :] - x[..., i, :] for i in range(n - 1)]       # fwd[i]  = i -> i+1

    def ang(u, v):
        return torch.atan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))
    # the 4-bead code: r12 = |v21|, r23 = |v23|, r34 = |v34| with v21 = back[1], v23 = fwd[1], v34 = fwd[2]
    r = [back[1].norm(dim=-1)] + [fwd[i].norm(dim=-1) for i in range(1, n - 1)]
    th = [ang(back[i], fwd[i]) for i in range(1, n - 1)]
    return torch.stack(r + th, -1)


def initial_state(B, device, gen=None, n_beads=4):
    x = torch.zeros(B, n_beads, 2, device=device)
    x[:, :, 0] = torch.arange(n_beads, device=device, dtype=torch.float32)
    v = torch.randn(B, n_beads, 2, device=device, generator=gen) * math.sqrt(TEMP)
    v = v - v.mean(1, keepdim=True)
    return x, v


@torch.no_grad()
def simulate(eps, n_rep=8, n_steps=4_000_000, stride=200, burn=50_000, device=None, seed=0,
             use_graph=True, verbose=True, return_temperature=False, design=None):
    """Simulate every eps (rows of [n_eps, design.dim_x] free eps) with n_rep independent replicas; returns Y
    frames [n_eps, n_rep * n_frames, 2N - 3] (float32, CPU; replica-major), frames every `stride` steps after
    `burn` steps. design = Design(4) by default (their 4-bead benchmark)."""
    design = design or Design(4)
    N, dy = design.n_beads, design.dim_y
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    eps = torch.as_tensor(eps, dtype=torch.float32).view(-1, design.dim_x)
    E = eps.shape[0]
    B = E * n_rep
    peps = design.pair_eps(eps.to(device)).repeat_interleave(n_rep, dim=0)       # chain b = e * n_rep + r
    A, bmask = _incidence(device, design), _bond_mask(design).to(device)
    x, v = initial_state(B, device, n_beads=N)
    F = conservative_force(x, peps, A, bmask)
    noise_sd = math.sqrt(2.0 * GAMMA * TEMP / DT)
    state = {"x": x, "v": v, "F": F, "ke": torch.zeros(B, device=device)}

    def one_step():
        s = state
        v_half = s["v"] + 0.5 * DT * s["F"]
        x_new = s["x"] + DT * v_half
        Fc = conservative_force(x_new, peps, A, bmask)
        Fn = Fc - GAMMA * v_half + noise_sd * torch.randn_like(v_half)
        v_new = v_half + 0.5 * DT * Fn
        s["x"].copy_(x_new)
        s["v"].copy_(v_new)
        s["F"].copy_(Fn)

    def block():
        for _ in range(stride):
            one_step()
        state["ke"].copy_(0.5 * (state["v"] ** 2).sum((1, 2)))

    run_block = block
    if use_graph and device != "cpu" and str(device).startswith("cuda"):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            block()                               # warm-up (also advances the chains by one block)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            block()
        run_block = g.replay

    n_blocks = n_steps // stride
    n_burn = burn // stride
    frames, ke = [], []
    t0 = time.time()
    for k in range(n_blocks):
        run_block()
        if k >= n_burn:
            frames.append(internal_coords(state["x"]))
            ke.append(state["ke"].mean())
        if verbose and (k + 1) % max(1, n_blocks // 10) == 0:
            torch.cuda.synchronize() if str(device).startswith("cuda") else None
            print(f"  [simulate] {(k + 1) * stride:,}/{n_steps:,} steps, {B} chains, {time.time() - t0:.0f}s", flush=True)
    nf = len(frames)
    Y = torch.stack(frames, 0)                    # [n_frames, B, 2N - 3]
    del frames
    Y = Y.cpu()                                   # reorder on the host: the GPU holds at most two copies
    Y = Y.view(nf, E, n_rep, dy).permute(1, 2, 0, 3).reshape(E, n_rep * nf, dy)
    if return_temperature:
        # 2D, N beads: kinetic temperature 2 KE / (2N) per chain (Langevin breaks momentum conservation)
        T = float(torch.stack(ke).mean() * 2.0 / (2 * N))
        return Y, T
    return Y
