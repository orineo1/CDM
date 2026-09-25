"""Retrain the conditional consistency model from the recorded configuration.

PROVENANCE: this is a RE-TRAINING, not a reproduction.  The weights used for
the paper's reported synthetic results were trained in a Colab session and
discarded; no GMM checkpoint exists in the repository or on the cluster.  The
configuration below is recovered verbatim from ``CompareMethods-Final.ipynb``
cells 9 and 18:

    nfeatures    = X.shape[1] - condition_on = 1
    condition_on = 1
    nunits       = 128           (depth left at its default of 6)
    nepochs_CM   = 7_500
    batch_size   = 1_024
    data_gen     = generate_mog_samples_not_differentiable(mu_list, Sigma_list, alpha)
    use_improved_training = True

Three models are trained under fixed, documented seeds.  The FIRST seed is the
main run and the other two are robustness checks; the best model is never
selected, because selecting on validation fidelity would bias every downstream
estimator measurement.
"""

import argparse
import json
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import dist_utils                                     # noqa: E402
from ConsistencyModels import ConsistencyModeliCT     # noqa: E402
from tfg import oracle                                # noqa: E402

SEEDS = (20240401, 20240402, 20240403)      # main, robustness-1, robustness-2

# ARCHITECTURE AUDIT (see checkpoint report). The paper's Table 3 ("Model
# architectures and training hyperparameters for the 2D setting") and the
# notebook DISAGREE:
#
#            paper Table 3     notebook cell 9/18        used here
#   Blocks   3                 6  (depth not passed ->   3   (paper)
#                                  class default 6)
#   Units    128               128                       128
#   Epochs   20,000            7,500 (nepochs_CM)        20,000 (paper)
#   Batch    1,024             1,024                     1,024
#
# The paper is the artefact we are reproducing, so its values win. The
# notebook's omission of `depth` is almost certainly an oversight, since it
# passes `nunits` explicitly and the surrounding diffusion models are given
# nblocks=3.
CONFIG = dict(nfeatures=1, condition_on=1, nunits=128, depth=3,
              nepochs=20_000, batch_size=1_024)

# The paper (Appendix A.5) states sampling uses noise levels
# [150, 50, 20, 10, 5, 1]. That is the default `ts` of ConsistencyModel.sample
# (ConsistencyModels.py:144), NOT of ConsistencyModeliCT.sample
# (ConsistencyModels.py:506), whose default is a 15-entry ladder. Both are
# evaluated and reported; this is a genuine code/paper discrepancy.
PAPER_TS = [150.0, 50.0, 20.0, 10.0, 5.0, 1.0]


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_ROOT),
                              capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return None


def train_one(seed, params, nepochs=None, device="cpu", out_dir=None):
    """Train a single conditional consistency model under ``seed``."""
    nepochs = CONFIG["nepochs"] if nepochs is None else nepochs
    torch.manual_seed(seed)

    # The model is float32 (the repository's default, as in the notebook),
    # while tfg.oracle loads the mixture in float64 for the analytic work.
    # Cast the generator's parameters so training matches the original run.
    data_generator = partial(dist_utils.generate_mog_samples_not_differentiable,
                             means=[m.float() for m in params["mu_list"]],
                             variances=[s.float() for s in params["Sigma_list"]],
                             weights=params["alpha"].float())

    model = ConsistencyModeliCT(nfeatures=CONFIG["nfeatures"],
                                condition_on=CONFIG["condition_on"],
                                nunits=CONFIG["nunits"],
                                depth=CONFIG["depth"],
                                device=torch.device(device))
    t0 = time.perf_counter()
    model.train_model(X=None, nepochs=nepochs, batch_size=CONFIG["batch_size"],
                      device=device, condition=CONFIG["condition_on"],
                      data_generator=data_generator,
                      use_improved_training=True)
    seconds = time.perf_counter() - t0
    model.eval()

    record = {
        "seed": seed,
        "config": {**CONFIG, "nepochs": nepochs},
        "seconds": seconds,
        "device": device,
        "dtype": "float32 (model default)",
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "provenance": "RE-TRAINED from recorded config; not the paper's weights",
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"cm_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "record": record}, path)
        record["checkpoint"] = str(path)

    return model, record


def load_one(seed, out_dir, device="cpu"):
    blob = torch.load(Path(out_dir) / f"cm_seed{seed}.pt",
                      map_location=device, weights_only=False)
    model = ConsistencyModeliCT(nfeatures=CONFIG["nfeatures"],
                                condition_on=CONFIG["condition_on"],
                                nunits=CONFIG["nunits"],
                                depth=CONFIG["depth"],
                                device=torch.device(device))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob["record"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="tfg_results/cm_checkpoints")
    ap.add_argument("--nepochs", type=int, default=CONFIG["nepochs"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(SEEDS))
    args = ap.parse_args()

    params = oracle.load_params()
    records = []
    for i, seed in enumerate(args.seeds):
        role = "MAIN" if i == 0 else f"robustness-{i}"
        print(f"\n=== training seed {seed} ({role}) ===", flush=True)
        _, rec = train_one(seed, params, nepochs=args.nepochs,
                           device=args.device, out_dir=args.out_dir)
        rec["role"] = role
        records.append(rec)
        print(f"    done in {rec['seconds']:.1f}s -> {rec.get('checkpoint')}",
              flush=True)

    out = Path(args.out_dir) / "training_records.json"
    out.write_text(json.dumps(records, indent=2))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
