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


SIGMA_BOX = (0.7, 1.4)          # residue size sigma_i (Design with free sigma)
ETA_BOX = (0.3, 2.0)            # residue stickiness eta_i (Design with free eta)
CHARGE_BOX = (-1.0, 1.0)        # residue charge q_i (Design with free charge)
BOXES = {"eps": EPS_BOX, "sigma": SIGMA_BOX, "eta": ETA_BOX, "charge": CHARGE_BOX}
DEFAULT_VALUE = {"sigma": 1.0, "eta": 1.0, "charge": 0.0}     # per-bead parameters that are neither free nor fixed
# screened electrostatics between charged beads (Design with charges): U_ij = A q_i q_j exp(-r / debye) / r for
# r < rcut (plain cutoff, like the LJ; LAMMPS coul/debye), on every pair, bonded ones included (special_bonds 1 1 1)
ELEC = {"A": 1.0, "debye": 2.0, "rcut": 6.0}


def _group(name):
    return name.rstrip("0123456789")


class Design:
    """The design space of the N-bead chain. Pair (i, j) at sequence distance d = |i - j| interacts by LJ with

        eps_ij   = eps_d * sqrt(eta_i * eta_j)      chain-distance strength x residue stickiness
        sigma_ij = (sigma_i + sigma_j) / 2          residue sizes (Lorentz-Berthelot), cutoff 4 sigma_ij

    with charges also U_ij = A q_i q_j exp(-r / debye) / r (r < rcut; `elec`, defaults ELEC): like charges repel,
    unlike ones attract, so the best charge of one bead depends on the others (frustration).

    Parameters: eps1..eps{N-1}, sigma1..sigmaN, eta1..etaN, charge1..chargeN. `free` lists those the search varies (X, in this
    order), `fixed` gives values to others; sigma_i, eta_i that are neither are 1 (identical residues), every
    eps_d must be free or fixed, charges that are neither are 0 (no electrostatics). `free` / `fixed` accept parameter names, group names ("eps", "sigma", "eta":
    every parameter of the group not given otherwise) and, as before, integers d for eps_d.

    Zick & Mendels' benchmark is the default Design(4): free eps1, eps2, fixed eps3 = 0.5, identical residues;
    Design(N >= 5) frees every eps_d. A residue (heteropolymer) design on top of fixed chain strengths:
    Design(8, free="sigma,eta", fixed="eps=0.5").

    `box` maps a group or a parameter to its range (defaults BOXES); `prior` is how `sample` draws designs:
    "uniform" (in the box), or "band:w" (each bead with free sigma_i and eta_i: eta_i follows sigma_i along the
    diagonal of their box, with Gaussian spread w in box units: larger residues are stickier, as in amino
    acids; the rest uniform). `temp` (None: TEMP) is the Langevin temperature of every simulation of this design;
    kept with the design so that data, targets and evaluation share it."""

    def __init__(self, n_beads=4, free=None, fixed=None, box=None, prior="uniform", temp=None, elec=None):
        n = int(n_beads)
        assert n >= 3
        self.n_beads = n
        params = [f"eps{d}" for d in range(1, n)] + [f"sigma{i}" for i in range(1, n + 1)] + [f"eta{i}" for i in range(1, n + 1)] \
            + [f"charge{i}" for i in range(1, n + 1)]
        if free is None:
            free = (1, 2) if n == 4 else tuple(range(1, n))
        if fixed is None:
            fixed = {3: EPS3} if n == 4 and tuple(free) in ((1, 2), ("eps1", "eps2")) else {}

        def tokens(spec):
            return [str(t).strip() for t in (spec.split(",") if isinstance(spec, str) else spec) if str(t).strip()]

        def expand(t):       # (parameters, explicit): a group name expands to all its parameters
            if t.isdigit():
                return [f"eps{t}"], True
            if t in BOXES:
                return [p for p in params if _group(p) == t], False
            assert t in params, f"unknown parameter {t} (N = {n})"
            return [t], True
        if isinstance(fixed, str):
            fixed = dict(kv.split("=") for kv in fixed.split(",") if kv)
        fixed = {str(k): float(v) for k, v in fixed.items()}
        # a parameter named explicitly (free or fixed) wins over a group name on the other side
        explicit_free = {p for t in tokens(free) for p in expand(t)[0] if expand(t)[1]}
        explicit_fixed = {p for k in fixed for p in expand(k)[0] if expand(k)[1]}
        fr = []
        for t in tokens(free):
            ps, ex = expand(t)
            fr += [p for p in ps if ex or p not in explicit_fixed]
        fr = list(dict.fromkeys(fr))
        fx = {}
        for k, v in fixed.items():
            ps, ex = expand(k)
            for p in ps:
                if ex or (p not in explicit_free and p not in fr):
                    fx[p] = v
        fx = {p: v for p, v in fx.items() if p not in fr}
        missing = [p for p in params if _group(p) == "eps" and p not in fr and p not in fx]
        assert not missing, f"{missing} are neither free nor fixed"
        self.free = tuple(fr)
        self.fixed = {p: fx[p] for p in params if p in fx}
        bx = {g: tuple(map(float, r)) for g, r in BOXES.items()}
        bx.update({k: tuple(map(float, v)) for k, v in (box or {}).items()})
        self.box_spec = {k: v for k, v in bx.items() if k not in BOXES or v != BOXES[k]}
        self._box = bx
        self.prior = prior
        self.temp = None if temp is None else float(temp)
        self.elec = dict(ELEC, **{k: float(v) for k, v in (elec or {}).items()})

    # ---- sizes and identity
    @property
    def dim_x(self):
        return len(self.free)

    @property
    def dim_y(self):
        return 2 * self.n_beads - 3

    @property
    def n_bonds(self):
        return self.n_beads - 1

    @property
    def is_legacy(self):
        """only chain strengths eps_d, default box and prior: the design of the earlier runs (same files, same numbers)"""
        return (all(_group(p) == "eps" for p in list(self.free) + list(self.fixed))
                and not self.box_spec and self.prior == "uniform" and self.temp is None)

    @property
    def reversal_symmetric(self):
        """reading the chain backwards leaves the energy unchanged (identical or palindromic residues, no free ones)"""
        if any(_group(p) != "eps" for p in self.free):
            return False
        n = self.n_beads
        return all(self.value(f"{g}{i}") == self.value(f"{g}{n + 1 - i}") for g in ("sigma", "eta", "charge") for i in range(1, n + 1))

    def value(self, p):
        """the fixed or default value of a parameter that is not free"""
        return self.fixed.get(p, DEFAULT_VALUE.get(_group(p)))

    def to_dict(self):
        if self.is_legacy:
            return {"n_beads": self.n_beads, "free": [int(p[3:]) for p in self.free],
                    "fixed": {p[3:]: v for p, v in self.fixed.items()}}
        d = {"n_beads": self.n_beads, "free": list(self.free), "fixed": dict(self.fixed),
             "box": {k: list(v) for k, v in self.box_spec.items()}, "prior": self.prior}
        if self.temp is not None:
            d["temp"] = self.temp
        if self._uses("charge"):
            d["elec"] = dict(self.elec)
        return d

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return cls(4)
        if d.get("kind") == "residue":            # the short-lived residue-only format (sigma_i, eps_i per bead)
            raise ValueError("residue-only design files are not supported; use Design(free='sigma,eta', fixed='eps=...')")
        fixed = {(f"eps{k}" if str(k).isdigit() else k): v for k, v in d["fixed"].items()}
        return cls(d["n_beads"], d["free"], fixed, d.get("box"), d.get("prior", "uniform"), d.get("temp"), d.get("elec"))

    @classmethod
    def from_args(cls, n_beads=4, free=None, fixed=None, box=None, prior="uniform", temp=None, elec=None):
        """CLI form: free '1,2,3' or 'sigma,eta' or 'eps1,sigma3' or 'charge' (None = default); fixed '3=0.5' or
        'eps=0.5,eta4=2'; box 'sigma=0.7:1.4,eta=0.3:2'; elec 'A=1,debye=2,rcut=6'."""
        fr = None if free in (None, "") else free
        fx = None if fixed is None else ({} if fixed == "" else fixed)
        bx = None if not box else {k: tuple(float(t) for t in v.split(":")) for k, v in (s.split("=") for s in box.split(","))}
        el = None if not elec else {k: float(v) for k, v in (s.split("=") for s in elec.split(","))}
        return cls(n_beads, fr, fx, bx, prior or "uniform", temp, el)

    def __repr__(self):
        extra = "" if self.prior == "uniform" else f", prior={self.prior}"
        extra += "" if self.temp is None else f", T={self.temp:g}"
        extra += f", elec={self.elec}" if self._uses("charge") else ""
        return f"Design(N={self.n_beads}, free={','.join(self.free)}, fixed={self.fixed}{extra})"

    # ---- the box, standardization and the prior
    def bounds(self):
        """(lo, hi) of every free parameter, as Python floats"""
        lo, hi = [], []
        for p in self.free:
            r = self._box.get(p, self._box[_group(p)])
            lo.append(r[0]); hi.append(r[1])
        return lo, hi

    def to_std(self, v):
        """free parameters -> standardized X (each the uniform box mapped to mean 0, std 1)"""
        lo, hi = self.bounds()
        lo, hi = torch.tensor(lo, dtype=torch.float64), torch.tensor(hi, dtype=torch.float64)
        mean, std = ((lo + hi) / 2).float(), ((hi - lo) / math.sqrt(12.0)).float()
        return (torch.as_tensor(v, dtype=torch.float32) - mean) / std

    def from_std(self, x):
        lo, hi = self.bounds()
        lo, hi = torch.tensor(lo, dtype=torch.float64), torch.tensor(hi, dtype=torch.float64)
        mean, std = ((lo + hi) / 2).float(), ((hi - lo) / math.sqrt(12.0)).float()
        return torch.as_tensor(x, dtype=torch.float32) * std + mean

    def clip(self, v):
        lo, hi = self.bounds()
        return torch.max(torch.min(torch.as_tensor(v, dtype=torch.float32), torch.tensor(hi)), torch.tensor(lo))

    def sample(self, n, generator=None):
        """[n, dim_x] designs drawn from the prior"""
        lo, hi = self.bounds()
        if self.is_legacy:          # bit-identical to the earlier eps draws
            return lo[0] + (hi[0] - lo[0]) * torch.rand(n, self.dim_x, generator=generator)
        lo_t, span = torch.tensor(lo), torch.tensor(hi) - torch.tensor(lo)
        u = torch.rand(n, self.dim_x, generator=generator)
        if self.prior.startswith("band"):
            w = float(self.prior.split(":")[1]) if ":" in self.prior else 0.1
            col = {p: k for k, p in enumerate(self.free)}
            for i in range(1, self.n_beads + 1):
                s, e = col.get(f"sigma{i}"), col.get(f"eta{i}")
                if s is not None and e is not None:
                    u[:, e] = (u[:, s] + w * torch.randn(n, generator=generator)).clamp(0.0, 1.0)
        else:
            assert self.prior == "uniform", f"unknown prior {self.prior}"
        return lo_t + span * u

    # ---- pairs
    def pairs(self):
        """(i, j, d) for every pair, ordered by sequence distance d then i: the bonds come first. For N = 4
        this is (0,1),(1,2),(2,3),(0,2),(1,3),(0,3), the order of the original 4-bead code."""
        return [(i, i + d, d) for d in range(1, self.n_beads) for i in range(self.n_beads - d)]

    def _col(self, x, p):
        if p in self.free:
            k = self.free.index(p)
            return x[:, k:k + 1]
        return torch.full_like(x[:, :1], self.value(p))

    def _uses(self, g):
        n = self.n_beads
        return any(_group(p) == g for p in self.free) or any(self.value(f"{g}{i}") != DEFAULT_VALUE[g] for i in range(1, n + 1))

    def pair_eps(self, x):
        """[B, dim_x] free parameters -> [B, n_pairs] LJ epsilon of each pair."""
        cols = []
        for d in range(1, self.n_beads):
            c = self._col(x, f"eps{d}")
            cols += [c] * (self.n_beads - d)
        e = torch.cat(cols, dim=1)
        if self._uses("eta"):
            eta = torch.cat([self._col(x, f"eta{i}") for i in range(1, self.n_beads + 1)], 1)
            P = self.pairs()
            e = e * torch.sqrt(eta[:, [i for i, _, _ in P]] * eta[:, [j for _, j, _ in P]])
        return e

    def pair_sigma(self, x):
        """[B, dim_x] -> [B, n_pairs] LJ sigma of each pair, or None when every residue has sigma = 1."""
        if not self._uses("sigma"):
            return None
        sig = torch.cat([self._col(x, f"sigma{i}") for i in range(1, self.n_beads + 1)], 1)
        P = self.pairs()
        return 0.5 * (sig[:, [i for i, _, _ in P]] + sig[:, [j for _, j, _ in P]])

    def pair_charge(self, x):
        """[B, dim_x] -> [B, n_pairs] charge products q_i q_j, or None when no bead is charged."""
        if not self._uses("charge"):
            return None
        q = torch.cat([self._col(x, f"charge{i}") for i in range(1, self.n_beads + 1)], 1)
        P = self.pairs()
        return q[:, [i for i, _, _ in P]] * q[:, [j for _, j, _ in P]]


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


def conservative_force(x, peps, A, bmask, psig=None, pqq=None, elec=None):
    """x [B, N, 2], peps [B, P] -> force [B, N, 2]. A = _incidence(device): bead j of pair k gets +f_k.
    psig [B, P]: LJ sigma of each pair (Design with residue sizes), cutoff RCUT * sigma; None: sigma = 1 for all pairs.
    pqq [B, P]: charge products q_i q_j (Design with charges), screened Coulomb with elec = {A, debye, rcut}."""
    d = torch.einsum("pk,bpd->bkd", A, x)               # [B, P, 2], vector i -> j  (x_j - x_i)
    r2 = (d * d).sum(-1)
    if psig is None:
        inv2 = SIGMA ** 2 / r2
        inside = (r2 < RCUT ** 2).to(x.dtype)
    else:
        inv2 = psig * psig / r2
        inside = (inv2 > 1.0 / RCUT ** 2).to(x.dtype)  # r < RCUT * sigma
    inv6 = inv2 ** 3
    # -dU/dr / r for LJ: 24 eps (2 s^12/r^14 - s^6/r^8)
    f_lj = 24.0 * peps * (2.0 * inv6 * inv6 - inv6) / r2 * inside
    r = torch.sqrt(r2)
    f_b = -2.0 * KBOND * (r - R0) / r * bmask
    if pqq is not None:
        # -dU/dr / r for U = A qq exp(-r / l) / r: A qq exp(-r / l) (1 / r + 1 / l) / r^2
        el = elec or ELEC
        f_b = f_b + el["A"] * pqq * torch.exp(-r / el["debye"]) * (1.0 / r + 1.0 / el["debye"]) / r2 \
            * (r < el["rcut"]).to(x.dtype)
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


def initial_state(B, device, gen=None, n_beads=4, temp=TEMP):
    x = torch.zeros(B, n_beads, 2, device=device)
    x[:, :, 0] = torch.arange(n_beads, device=device, dtype=torch.float32)
    v = torch.randn(B, n_beads, 2, device=device, generator=gen) * math.sqrt(temp)
    v = v - v.mean(1, keepdim=True)
    return x, v


@torch.no_grad()
def simulate(eps, n_rep=8, n_steps=4_000_000, stride=200, burn=50_000, device=None, seed=0,
             use_graph=True, verbose=True, return_temperature=False, design=None, temp=None):
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
    psig = design.pair_sigma(eps.to(device))
    psig = None if psig is None else psig.repeat_interleave(n_rep, dim=0)
    pqq = design.pair_charge(eps.to(device))
    pqq = None if pqq is None else pqq.repeat_interleave(n_rep, dim=0)
    fkw = {} if pqq is None else {"pqq": pqq, "elec": design.elec}
    A, bmask = _incidence(device, design), _bond_mask(design).to(device)
    temp = float(temp) if temp is not None else (design.temp if design.temp is not None else TEMP)
    x, v = initial_state(B, device, n_beads=N, temp=temp)
    F = conservative_force(x, peps, A, bmask, psig, **fkw)
    noise_sd = math.sqrt(2.0 * GAMMA * temp / DT)
    state = {"x": x, "v": v, "F": F, "ke": torch.zeros(B, device=device)}

    def one_step():
        s = state
        v_half = s["v"] + 0.5 * DT * s["F"]
        x_new = s["x"] + DT * v_half
        Fc = conservative_force(x_new, peps, A, bmask, psig, **fkw)
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
