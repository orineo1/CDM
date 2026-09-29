#!/usr/bin/env python
"""Re-classify finished grid-search configurations and summarise them in a CSV.

Reads outputs/<target>/*/record.pkl (the saved images; nothing is re-run and the
records are not modified), clamps every image to [-1, 1] (as the figures show it;
--no-clamp classifies the raw image, as the paper did), labels it with the
classifier, and writes one row per configuration to
outputs/<target>/classification_<tag>.csv, best first:

    top5_n_labelled   of the 5 lowest-SWD seeds, how many the classifier labels
    top5_n_0          of those 5, how many it labels as 0
    n_labelled        of all seeds, how many it labels (not None)
    n_none            of all seeds, how many it leaves unlabelled
    digit_0..digit_9  of all seeds, how many it labels as each digit
    ...               plus the configuration, top-5 labels / confidence and SWDs

Rows are sorted by top5_n_labelled, then top5_n_0, then n_labelled (all highest
first), then the top-5 SWD (lowest first). A second file, classification_<tag>_seeds.csv,
has one row per seed (configuration, seed, SWD, in top-5, label, confidence).

    python experiments/mnist/grid_search/reclassify.py --target uniform \\
        --classifier-path experiments/mnist/checkpoints/robust_classifier_code.pth
"""

import argparse
import csv
import glob
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MNIST_DIR = os.path.dirname(HERE)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target", choices=["uniform", "bimodal"], required=True)
    p.add_argument("--out", default=os.path.join(HERE, "outputs"))
    p.add_argument("--classifier-path",
                   default=os.path.join(MNIST_DIR, "checkpoints", "robust_classifier.pth"))
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--no-clamp", action="store_true",
                   help="classify the raw image (the paper's way) instead of clamping to [-1, 1] first")
    p.add_argument("--tag", default=None,
                   help="suffix of the CSV names (default: clamp or noclamp, plus the threshold)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torch
    from cdm.models.mnist.classifier import classify_generated_images, load_or_train_classifier

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(args.out, args.target)
    paths = sorted(glob.glob(os.path.join(root, "*", "record.pkl")))
    if not paths:
        raise SystemExit(f"no finished configurations (record.pkl) under {root}")
    if not os.path.exists(args.classifier_path):
        raise SystemExit(f"classifier not found: {args.classifier_path}")
    clf = load_or_train_classifier(args.classifier_path, device=device)
    tag = args.tag or f"{'noclamp' if args.no_clamp else 'clamp'}_thr{args.threshold:g}"

    rows, seed_rows = [], []
    for path in paths:
        with open(path, "rb") as fh:
            rec = pickle.load(fh)
        images = [np.asarray(x, dtype=np.float32) for x in rec["results"]]
        if not args.no_clamp:
            images = [np.clip(x, -1.0, 1.0) for x in images]
        preds, confs = classify_generated_images(images, clf, device, threshold=args.threshold)
        losses = np.asarray(rec["loss_log"], dtype=float)
        top = [int(i) for i in np.argsort(losses)[:5]]
        top_preds = [preds[i] for i in top]
        labelled = [x for x in preds if x is not None]
        cfg = rec.get("config", {})
        row = {
            "configuration": rec.get("experiment_name", os.path.basename(os.path.dirname(path))),
            "n_cond": cfg.get("n_cond"), "steps": cfg.get("steps"), "n_mc": cfg.get("n_mc"),
            "step_size_mode": cfg.get("step_size_mode"), "bimodal_var": cfg.get("bimodal_var"),
            "top5_n_labelled": sum(x is not None for x in top_preds),
            "top5_n_0": sum(x == 0 for x in top_preds),
            "n_labelled": len(labelled),
            "n_none": len(preds) - len(labelled),
            **{f"digit_{d}": sum(x == d for x in labelled) for d in range(10)},
            "top5_labels": " ".join("None" if x is None else str(x) for x in top_preds),
            "top5_conf_mean": round(float(np.mean([confs[i] for i in top])), 4),
            "swd_top5_mean": round(float(losses[top].mean()), 5),
            "swd_all_mean": round(float(losses.mean()), 5),
            "n_seeds": len(preds),
        }
        rows.append(row)
        for k in range(len(preds)):
            seed_rows.append({"configuration": row["configuration"], "seed": int(rec["seed_log"][k]),
                              "swd": round(float(losses[k]), 5), "in_top5": k in top,
                              "label": "None" if preds[k] is None else preds[k],
                              "confidence": round(float(confs[k]), 4)})

    rows.sort(key=lambda r: (-r["top5_n_labelled"], -r["top5_n_0"], -r["n_labelled"], r["swd_top5_mean"]))
    out_csv = os.path.join(root, f"classification_{tag}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    seeds_csv = os.path.join(root, f"classification_{tag}_seeds.csv")
    with open(seeds_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(seed_rows[0]))
        w.writeheader()
        w.writerows(seed_rows)

    print(f"{len(rows)} configurations, classifier {args.classifier_path}, threshold {args.threshold}, "
          f"{'raw images' if args.no_clamp else 'clamped to [-1, 1]'}\n")
    show = ["configuration", "top5_n_labelled", "top5_n_0", "n_labelled", "n_none"] + \
           [f"digit_{d}" for d in range(10)] + ["swd_top5_mean"]
    print(",".join(show))
    for r in rows:
        print(",".join(str(r[c]) for c in show))
    print(f"\nwrote {out_csv}\nwrote {seeds_csv}")


if __name__ == "__main__":
    main()
