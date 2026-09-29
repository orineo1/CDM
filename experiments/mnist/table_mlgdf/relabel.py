"""Relabel finished MNIST runs with a chosen classifier and threshold, then rebuild Table 2.

The classifier only labels the recovered images; it does not affect the
optimisation or the SWDs, so this never re-runs anything. It reads every `.pkl`
in --results-dir, recomputes `predicted_labels` / `predicted_confidences`, writes
the relabelled records to --out, prints each seed's top guess and confidence
(so the effect of a threshold is visible), and builds <out>/table2.md.

An image is labelled with its top class when that class's probability is above
--threshold, and `None` otherwise. The paper's runs used 0.75.

    python experiments/mnist/table_mlgdf/relabel.py \
        --results-dir experiments/mnist/table_mlgdf/outputs/full \
        --out experiments/mnist/table_mlgdf/outputs/full_thr0.5 --threshold 0.5
"""

import argparse
import glob
import os
import pickle
import subprocess
import sys

import torch

from cdm.models.mnist.classifier import classify_generated_images, load_or_train_classifier

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_CLASSIFIER = os.path.join(os.path.dirname(HERE), "checkpoints", "robust_classifier.pth")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results-dir", required=True, help="directory with the run .pkl records")
    p.add_argument("--out", required=True, help="where to write the relabelled records and table2.md")
    p.add_argument("--threshold", type=float, default=0.75,
                   help="label an image only if its top class probability is above this")
    p.add_argument("--classifier", default=PAPER_CLASSIFIER,
                   help="classifier weights (.pth); trained by the code and saved here if absent")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.results_dir, "*.pkl")))
    if not paths:
        raise SystemExit(f"no run records (.pkl) in {args.results_dir}")
    if os.path.abspath(args.out) == os.path.abspath(args.results_dir):
        raise SystemExit("--out must differ from --results-dir (the originals are kept)")
    os.makedirs(args.out, exist_ok=True)

    clf = load_or_train_classifier(args.classifier, device=args.device)
    print(f"threshold = {args.threshold}")
    for path in paths:
        with open(path, "rb") as fh:
            rec = pickle.load(fh)
        # threshold 0: every image gets its top class, so confs/top are complete
        top, confs = classify_generated_images(rec["results"], clf, args.device, threshold=0.0)
        preds = [t if c > args.threshold else None for t, c in zip(top, confs)]
        rec["predicted_labels"], rec["predicted_confidences"] = preds, confs
        with open(os.path.join(args.out, os.path.basename(path)), "wb") as fh:
            pickle.dump(rec, fh)
        print(f"\n== {rec.get('target', os.path.basename(path))}")
        for seed, t, c, lab in zip(rec["seed_log"], top, confs, preds):
            print(f"  seed {seed:2d}  top {t}  conf {c:.3f}  -> {lab if lab is not None else 'None'}")

    subprocess.run([sys.executable, os.path.join(HERE, "make_table.py"), args.out], check=True)


if __name__ == "__main__":
    main()
