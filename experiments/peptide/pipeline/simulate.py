"""
Batched GPU Langevin simulator of the four-bead 2D Lennard-Jones chain of Zick & Mendels (2026),
GB-FESO (arXiv:2607.06421), a PyTorch port of their LAMMPS input (`GB-FESO/lammps_scripts/gener.py`).

Physics (LJ units, 2D, 4 beads of mass 1):
  bonds (1,2),(2,3),(3,4): E = K (r - r0)^2, K = 0.15, r0 = 1.5     (LAMMPS bond_style harmonic, no 1/2)
  LJ on all six pairs (special_bonds lj 1 1 1), sigma = 1, plain lj/cut 4.0 (no shift):
     (1,2),(2,3),(3,4) -> eps1;  (1,3),(2,4) -> eps2;  (1,4) -> eps3 = 0.5
  fix langevin T = 0.09, damp = 1 (gamma = 1) + fix nve (velocity Verlet), dt = 0.005,
  start: straight chain with spacing 1.0, velocities drawn at T = 0.09 with zero net momentum.
The step follows LAMMPS: half kick with the stored total force, drift, new force = conservative +
friction on the half-step velocity + random force (Gaussian, same variance 2 gamma kT / dt as the
LAMMPS uniform noise), second half kick.

Output frames: Y = (r12, r23, r34, theta123, theta234), signed xy-plane angles via atan2, exactly as
their dataset.py (`_signed_angle_xy`): theta123 from (2->1) to (2->3), theta234 from (3->2) to (3->4).

All chains (n_eps x replicas) advance in one tensor. The inner loop of `stride` steps is captured once
as a CUDA graph (one replay per saved frame), so the per-step Python overhead disappears.

    from simulate import simulate
    Y = simulate(eps, n_rep=8, n_steps=4_000_000, stride=200, burn=50_000)   # [n_eps, n_rep * n_frames, 5]
"""
import math
import time

import torch

KBOND, R0, EPS3, SIGMA, RCUT = 0.15, 1.5, 0.5, 1.0, 4.0
TEMP, GAMMA, DT = 0.09, 1.0, 0.005
EPS_BOX = (0.21, 1.55)          # their gener.py range for eps1, eps2

# pair k joins beads I[k] -> J[k]; 0..2 are the bonds
PI = (0, 1, 2, 0, 1, 0)
PJ = (1, 2, 3, 2, 3, 3)
BOND_MASK = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])


def _incidence(device):
    A = torch.zeros(4, 6, device=device)
    for k, (i, j) in enumerate(zip(PI, PJ)):
        A[j, k] += 1.0
        A[i, k] -= 1.0
    return A


def pair_eps(eps):
    """[B, 2] (eps1, eps2) -> [B, 6] LJ epsilon of each pair."""
    e1, e2 = eps[:, 0:1], eps[:, 1:2]
    return torch.cat([e1, e1, e1, e2, e2, torch.full_like(e1, EPS3)], dim=1)


def conservative_force(x, peps, A, bmask):
    """x [B, 4, 2], peps [B, 6] -> force [B, 4, 2]. A = _incidence(device): bead j of pair k gets +f_k."""
    d = torch.einsum("pk,bpd->bkd", A, x)               # [B, 6, 2], vector i -> j  (x_j - x_i)
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
    """x [..., 4, 2] -> [..., 5] = (r12, r23, r34, theta123, theta234)."""
    v21 = x[..., 0, :] - x[..., 1, :]
    v23 = x[..., 2, :] - x[..., 1, :]
    v32 = -v23
    v34 = x[..., 3, :] - x[..., 2, :]

    def ang(u, v):
        return torch.atan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))
    return torch.stack([v21.norm(dim=-1), v23.norm(dim=-1), v34.norm(dim=-1), ang(v21, v23), ang(v32, v34)], -1)


def initial_state(B, device, gen=None):
    x = torch.zeros(B, 4, 2, device=device)
    x[:, :, 0] = torch.arange(4, device=device, dtype=torch.float32)
    v = torch.randn(B, 4, 2, device=device, generator=gen) * math.sqrt(TEMP)
    v = v - v.mean(1, keepdim=True)
    return x, v


@torch.no_grad()
def simulate(eps, n_rep=8, n_steps=4_000_000, stride=200, burn=50_000, device=None, seed=0,
             use_graph=True, verbose=True, return_temperature=False):
    """Simulate every eps pair with n_rep independent replicas; returns Y frames
    [n_eps, n_rep * n_frames, 5] (float32, CPU), frames every `stride` steps after `burn` steps."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    eps = torch.as_tensor(eps, dtype=torch.float32).view(-1, 2)
    E = eps.shape[0]
    B = E * n_rep
    peps = pair_eps(eps.to(device)).repeat_interleave(n_rep, dim=0)       # chain b = e * n_rep + r
    A, bmask = _incidence(device), BOND_MASK.to(device)
    x, v = initial_state(B, device)
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
    Y = torch.stack(frames, 0)                    # [n_frames, B, 5]
    Y = Y.view(len(frames), E, n_rep, 5).permute(1, 2, 0, 3).reshape(E, n_rep * len(frames), 5).cpu()
    if return_temperature:
        # 2D, 4 beads: kinetic temperature 2 KE / (2N) per chain (Langevin breaks momentum conservation)
        T = float(torch.stack(ke).mean() * 2.0 / 8.0)
        return Y, T
    return Y
