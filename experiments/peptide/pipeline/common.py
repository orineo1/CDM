"""
Shared helpers of the peptide experiment.

* Imports Zick & Mendels' GB-FESO code (github.com/.../GB-FESO, MIT), cloned next to this repository
  (`<parent of CDM>/GB-FESO`, or $GBFESO_DIR). Their modules import two names that their dataset.py no
  longer defines (`LammpsTrajectoryAngleDataset`, `AngleFrameDataset`, renamed to *Geometry*); we alias them
  so that train.py, run_feso.py and plotters.py import unchanged.
* The Y representation of OUR networks: for N beads, 3N - 5 features (the N - 1 bond lengths, then
  cos th, sin th of each of the N - 2 angles; N = 4: r12, r23, r34, cos th123, sin th123, cos th234,
  sin th234), standardized, so that Euclidean MMD treats the angles periodically. N is read off the
  last dimension (raw 2N - 3, features 3N - 5). X = the free eps_d, each standardized with the box
  [0.21, 1.55]. GB-FESO keeps its own representation (raw 2N - 3 numbers, distances min-max scaled,
  angles / pi).
"""
import math
import os
import sys

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
GBFESO_DIR = os.environ.get("GBFESO_DIR", os.path.join(os.path.dirname(REPO), "GB-FESO", "GB-FESO"))
if GBFESO_DIR not in sys.path:
    sys.path.insert(0, GBFESO_DIR)
import dataset as _gb_dataset   # noqa: E402
_gb_dataset.AngleFrameDataset = _gb_dataset.GeometryFrameDataset
_gb_dataset.LammpsTrajectoryAngleDataset = _gb_dataset.LammpsTrajectoryGeometryDataset

# X standardization: uniform box [0.21, 1.55]
X_MEAN, X_STD = 0.88, 1.34 / math.sqrt(12.0)


def n_beads_raw(Y):
    """N from raw Y [..., 2N - 3]."""
    return (Y.shape[-1] + 3) // 2


def raw_to_feat(Y):
    """[..., 2N - 3] raw -> [..., 3N - 5] unstandardized features (N = 4: 5 -> 7)."""
    nb = n_beads_raw(Y) - 1
    th = [Y[..., nb + k] for k in range(Y.shape[-1] - nb)]
    return torch.cat([Y[..., :nb], torch.stack([f for t in th for f in (t.cos(), t.sin())], -1)], -1)


def feat_to_raw(F):
    """[..., 3N - 5] unstandardized features -> [..., 2N - 3] raw (angles by atan2)."""
    nb = (F.shape[-1] + 5) // 3 - 1
    na = (F.shape[-1] - nb) // 2
    return torch.cat([F[..., :nb], torch.stack([torch.atan2(F[..., nb + 2 * k + 1], F[..., nb + 2 * k])
                                                for k in range(na)], -1)], -1)


def x_to_std(eps, design=None):
    """free design parameters -> standardized X; the legacy constants unless a non-legacy design is given"""
    if design is not None and not design.is_legacy:
        return design.to_std(eps)
    return (torch.as_tensor(eps, dtype=torch.float32) - X_MEAN) / X_STD


def x_from_std(x, design=None):
    if design is not None and not design.is_legacy:
        return design.from_std(x)
    return torch.as_tensor(x, dtype=torch.float32) * X_STD + X_MEAN


# Basin classes merge tuples related by the mirror image and, when the energy is symmetric under reading the chain
# backwards (identical or palindromic residues), also by the reversal. Scripts that load a design call
# set_design(design) so the classes follow it; the default (True) is the chain-distance benchmark.
REVERSAL = True


def set_design(design):
    global REVERSAL
    REVERSAL = bool(design.reversal_symmetric) if design is not None else True


class YScaler:
    def __init__(self, mean, std):
        self.mean, self.std = torch.as_tensor(mean).float(), torch.as_tensor(std).float()

    def fwd(self, Y_raw):          # raw 2N - 3 -> standardized 3N - 5
        return (raw_to_feat(Y_raw) - self.mean) / self.std

    def inv(self, Z):              # standardized 3N - 5 -> raw 2N - 3
        return feat_to_raw(Z * self.std.to(Z.device) + self.mean.to(Z.device))


# ---------------------------------------------------------------------------------------------------
# Conformational basins for any N (the evaluation of N > 4; GB-FESO's utils.py boxes cover N = 4 only)
# ---------------------------------------------------------------------------------------------------
# Per-angle state: sign of theta and whether |theta| < FOLD_MAX ("folded": beads i-1 and i+1 in LJ contact)
# or not ("open"). State codes 0 = +folded, 1 = -folded, 2 = +open, 3 = -open. A conformation is the tuple of
# the N - 2 angle states. The potential is symmetric under the mirror image (every theta -> -theta) and under
# reading the chain backwards ((theta_1..theta_n) -> (-theta_n..-theta_1)), so tuples related by these maps
# have equal equilibrium populations; a basin is one class of tuples under the two maps (their four N = 4
# basins are classes of the same two symmetries: A anti-diagonal = (+f, -f) and (-f, +f), B same-sign =
# (+f, +f) and (-f, -f), C outer = one angle open). Basin populations are the frequencies of the classes.
# FOLD_MAX = 1.55 rad is the minimum of the |theta| marginal pooled over eps (N = 4, 5, 6 alike) between the
# contact peak (|theta| ~ 1.0, r_{i-1,i+1} ~ 2^(1/6)) and the open peak (|theta| ~ 2.05).
FOLD_MAX = 1.55


def angle_states(Y, fold_max=FOLD_MAX):
    """raw Y [n, 2N - 3] -> [n, N - 2] state codes (0 +f, 1 -f, 2 +o, 3 -o)."""
    nb = n_beads_raw(Y) - 1
    th = torch.as_tensor(Y)[:, nb:]
    return (th < 0).long() + 2 * (th.abs() >= fold_max).long()


def _flip(s):
    return s ^ 1          # sign flip: 0 <-> 1, 2 <-> 3


def basin_classes(n_beads, reversal=None):
    """(class_of [4 ** (N - 2)] tuple index -> class id, names [n_classes]) under mirror + reversal."""
    import itertools
    na = n_beads - 2
    sym = "fFoO"          # f/o folded/open, lower case + sign, upper case - sign
    tuples = list(itertools.product(range(4), repeat=na))
    index = {t: i for i, t in enumerate(tuples)}
    class_of, names, canon = [0] * len(tuples), [], {}
    for t in tuples:
        rev = REVERSAL if reversal is None else reversal
        orbit = {t, tuple(_flip(s) for s in t)}
        if rev:
            orbit |= {tuple(_flip(s) for s in reversed(t)), tuple(reversed(t))}
        c = min(orbit)
        if c not in canon:
            canon[c] = len(names)
            names.append("".join(sym[s] for s in c))
        class_of[index[t]] = canon[c]
    return torch.tensor(class_of), names


def basin_probs(Y, fold_max=FOLD_MAX):
    """raw Y [n, 2N - 3] -> class probabilities [n_classes] (names: basin_classes(N)[1])."""
    st = angle_states(Y, fold_max)
    na = st.shape[1]
    code = (st * (4 ** torch.arange(na - 1, -1, -1))).sum(1)
    class_of, names = basin_classes(na + 2)
    counts = torch.bincount(class_of[code], minlength=len(names)).float()
    return counts / counts.sum()


def compare_tuple_basins(T, G, active_target_prob=0.05, min_generated_active_prob=0.005, prob_floor=1e-3, cap=6.0,
                         fold_max=FOLD_MAX):
    """Their compare_basin_metrics rules on the tuple basins: active = target population > 5%; missing = an
    active basin below 0.5% at eps_hat; dF_i = -log(P_i / P_ref) with the most populated active basin as
    reference, floor 1e-3, |error| capped at 6, over the non-reference active basins."""
    pt, pg = basin_probs(T, fold_max), basin_probs(G, fold_max)
    active = [i for i in range(len(pt)) if float(pt[i]) > active_target_prob]
    missing = [i for i in active if float(pg[i]) < min_generated_active_prob]
    out = {"target_probs": pt.tolist(), "generated_probs": pg.tolist(), "active_indices": active,
           "missing_active_basin": len(missing) > 0, "target_type": "multi_basin" if len(active) >= 2 else "single_basin",
           "total_variation": float(0.5 * (pt - pg).abs().sum()), "df_mae": None, "df_median_abs_error": None}
    if len(active) >= 2:
        ref = max(active, key=lambda i: float(pt[i]))
        idx = torch.tensor([i for i in active if i != ref])
        ft, fg = -torch.log(pt.clamp(min=prob_floor) / pt.clamp(min=prob_floor)[ref]), \
            -torch.log(pg.clamp(min=prob_floor) / pg.clamp(min=prob_floor)[ref])
        err = (fg[idx] - ft[idx]).abs().clamp(max=cap)
        out.update({"df_mae": float(err.mean()), "df_median_abs_error": float(err.median())})
    return out


def energy_distance(A, B):
    """Energy distance 2 E|a - b| - E|a - a'| - E|b - b'| between two samples [n, d] (V-statistics)."""
    A, B = A.float(), B.float()
    return float(2 * torch.cdist(A, B).mean() - torch.cdist(A, A).mean() - torch.cdist(B, B).mean())
