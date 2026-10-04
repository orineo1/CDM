"""Peptide experiment: the shared search on the four-bead chain (X = (eps1, eps2), Y = internal coordinates).

The networks are checkpoints from `train_models.py` and the targets are `.npy` arrays of target samples, the
same kind of vector data as the Gaussian-mixture experiments, so this entry point runs that driver
(`experiments/synthetic/pipeline/search_mog.py`) with all its flags; nothing in it is mixture-specific
when the target is a sample array. The search flags (--n-restarts, --guide-grad, --refine-*, --select-*)
are the shared ones of `cdm.guidance.search`. --sampler may also be GB-FESO's surrogate checkpoint
(`models/gbfeso.pt`): MLGD on their surrogate (gbfeso_sampler.py).

    python experiments/peptide/pipeline/search_peptide.py --prior <prior.pt> --sampler <sampler.pt> \\
        --target <target_XXX.npy> --guide-grad zo --n-restarts 10 --out <result.json>
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "synthetic", "pipeline"))
import search_mog  # noqa: E402

_load_model = search_mog.load_model


def load_model(path, device):
    """--sampler may also be a GB-FESO checkpoint (gbfeso.py train): their surrogate as our conditional
    sampler (gbfeso_sampler.py), for MLGD on their surrogate."""
    sys.path.insert(0, HERE)
    from gbfeso_sampler import is_gbfeso_checkpoint, GBFESOSampler
    if is_gbfeso_checkpoint(path):
        return GBFESOSampler(path, device=device), "diffusion"
    return _load_model(path, device)


search_mog.load_model = load_model

if __name__ == "__main__":
    search_mog.main()
