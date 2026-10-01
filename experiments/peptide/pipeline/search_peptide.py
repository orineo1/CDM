"""Peptide experiment: the shared search on the four-bead chain (X = (eps1, eps2), Y = internal coordinates).

The networks are checkpoints from `train_models.py` and the targets are `.npy` arrays of target samples, the
same kind of vector data as the Gaussian-mixture experiments, so this entry point runs that driver
(`experiments/synthetic/pipeline/search_mog.py`) with all its flags; nothing in it is mixture-specific
when the target is a sample array. The search flags (--n-restarts, --guide-grad, --refine-*, --select-*)
are the shared ones of `cdm.guidance.search`.

    python experiments/peptide/pipeline/search_peptide.py --prior <prior.pt> --sampler <sampler.pt> \\
        --target <target_XXX.npy> --guide-grad zo --n-restarts 10 --out <result.json>
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "synthetic", "pipeline"))
import search_mog  # noqa: E402

if __name__ == "__main__":
    search_mog.main()
