"""The N-bead generalization of the peptide experiment (experiments/peptide/pipeline): N = 4 keeps the
four-bead benchmark exactly, and the GB-FESO generalization reduces to their code at two free eps.
CPU, seconds; the GB-FESO checks are skipped when their checkout (../GB-FESO or $GBFESO_DIR) is absent."""
import os
import sys

import pytest
import torch

PIPE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "peptide", "pipeline")
sys.path.insert(0, PIPE)
from simulate import Design, simulate, internal_coords   # noqa: E402

REPO = os.path.abspath(os.path.join(PIPE, "..", "..", ".."))
GBFESO = os.environ.get("GBFESO_DIR", os.path.join(os.path.dirname(REPO), "GB-FESO", "GB-FESO"))
needs_gbfeso = pytest.mark.skipif(not os.path.isfile(os.path.join(GBFESO, "run_feso.py")), reason="GB-FESO checkout absent")


def test_design_defaults():
    d4 = Design(4)
    assert d4.free == (1, 2) and d4.fixed == {3: 0.5} and d4.dim_x == 2 and d4.dim_y == 5
    assert [(i, j) for i, j, _ in d4.pairs()] == [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]
    assert torch.equal(d4.pair_eps(torch.tensor([[0.3, 0.7]])), torch.tensor([[0.3, 0.3, 0.3, 0.7, 0.7, 0.5]]))
    d6 = Design(6)
    assert d6.free == (1, 2, 3, 4, 5) and d6.fixed == {} and d6.dim_y == 9 and len(d6.pairs()) == 15
    assert Design.from_dict(d6.to_dict()).to_dict() == d6.to_dict()
    assert Design.from_args(5, "1,2", "3=0.5,4=0.2").fixed == {3: 0.5, 4: 0.2}


def test_internal_coords_n4_matches_four_bead_formula():
    x = torch.randn(7, 4, 2)
    v21, v23, v34 = x[:, 0] - x[:, 1], x[:, 2] - x[:, 1], x[:, 3] - x[:, 2]

    def ang(u, v):
        return torch.atan2(u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0], (u * v).sum(-1))
    ref = torch.stack([v21.norm(dim=-1), v23.norm(dim=-1), v34.norm(dim=-1), ang(v21, v23), ang(-v23, v34)], -1)
    assert torch.equal(internal_coords(x), ref)


def test_simulate_n6_shapes():
    Y, T = simulate(torch.full((2, 5), 0.8), n_rep=2, n_steps=400, stride=100, burn=100, device="cpu",
                    verbose=False, return_temperature=True, design=Design(6))
    assert Y.shape == (2, 2 * 3, 9) and torch.isfinite(Y).all() and 0.0 < T < 1.0


@needs_gbfeso
def test_features_and_basins():
    import common
    Y = torch.rand(50, 9) * 3
    assert torch.allclose(common.feat_to_raw(common.raw_to_feat(Y)), Y, atol=1e-5)     # angles in [0, 3] < pi
    assert common.raw_to_feat(torch.zeros(3, 5)).shape == (3, 7)
    assert [len(common.basin_classes(n)[1]) for n in (4, 5, 6)] == [6, 20, 72]
    p = common.basin_probs(Y)
    assert p.shape == (72,) and abs(float(p.sum()) - 1) < 1e-6


@needs_gbfeso
def test_gbfeso_generalization_reduces_to_theirs_at_two_eps():
    import gbfeso
    import model as gb_model
    import dataset as gb_dataset
    e = torch.rand(5, 2)
    assert torch.equal(gbfeso._epsilon_features_K(e), gb_model.epsilon_features(e))
    assert gbfeso.n_eps_features(2) == 7
    Y = torch.rand(2, 30, 5) + torch.tensor([1.0, 1.0, 1.0, -0.5, -0.5])
    traj = gbfeso.traj_list(Y, torch.rand(2, 2))
    theirs, ours = gb_dataset.GeometryFrameDataset(traj), gbfeso.GeometryFrameDatasetN(traj, Design(4))
    x = Y.reshape(-1, 5)
    assert torch.allclose(theirs.normalize_target(x), ours.normalize_target(x))
    assert torch.allclose(theirs.denormalize_target(theirs.normalize_target(x)), ours.denormalize_target(ours.normalize_target(x)))


@needs_gbfeso
def test_gbfeso_patch_installs_for_n5():
    import gbfeso
    import run_feso
    gbfeso.install_n_bead_patches(Design(5))
    assert run_feso._parse_optimize_eps("both") == (0, 1, 2, 3)
    assert run_feso.optimize_epsilon_condition.__code__.co_filename.endswith("[N-bead patch]")
