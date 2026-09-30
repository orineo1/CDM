#!/usr/bin/env python
"""Compare two Stable Diffusion runs and say whether they are the same run.

Two runs with the same seed, on the same GPU model and library versions, must give
identical per-step numbers, the same final MMD and the same images. This compares:

    steps.csv                       every numeric column, exact and max |difference|
    metrics.json                    final MMD (and the unguided MMD when present),
                                    seed, model revisions, library versions, GPU
    scribble.png, source_portrait.png, final_scribble_mlgd_f.png
                                    pixel-exact, else the fraction of differing pixels

Exit status 0 when everything matches, 1 otherwise, so it can end a job:

    python experiments/stable_diffusion/compare_runs.py outputs/det_a outputs/det_b
    python experiments/stable_diffusion/compare_runs.py a b --atol 1e-6   # tolerate fp noise
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

IMAGES = ["scribble.png", "source_portrait.png", "final_scribble_mlgd_f.png",
          "final_scribble_regular.png"]


def load_steps(d):
    with open(os.path.join(d, "steps.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        if all(v == "" for v in vals):
            continue
        try:
            cols[k] = np.array([float(v) if v != "" else np.nan for v in vals])
        except ValueError:
            pass
    return cols


def compare_steps(a, b, atol):
    ok = True
    sa, sb = load_steps(a), load_steps(b)
    if len(next(iter(sa.values()))) != len(next(iter(sb.values()))):
        print(f"  steps.csv: {len(next(iter(sa.values())))} vs {len(next(iter(sb.values())))} rows")
        return False
    print(f"  steps.csv: {len(next(iter(sa.values())))} rows")
    for k in sa:
        if k not in sb or k == "elapsed_s":       # wall-clock is never equal
            continue
        x, y = sa[k], sb[k]
        m = ~(np.isnan(x) & np.isnan(y))
        diff = np.abs(x[m] - y[m])
        worst = float(diff.max()) if diff.size else 0.0
        first = int(np.argmax(diff > atol)) + 1 if (diff > atol).any() else None
        same = worst <= atol
        ok &= same
        print(f"    {k:<20} {'same' if same else 'DIFFERS':8} max|diff|={worst:.3g}"
              + (f"  first at step {first}" if first else ""))
    return ok


def compare_metrics(a, b, atol):
    ok = True
    with open(os.path.join(a, "metrics.json")) as fh:
        ma = json.load(fh)
    with open(os.path.join(b, "metrics.json")) as fh:
        mb = json.load(fh)
    for key in ("final_mlgd_f_mmd", "final_regular_mmd"):
        x, y = ma.get(key), mb.get(key)
        if x is None and y is None:
            continue
        same = x is not None and y is not None and abs(x - y) <= atol
        ok &= same
        print(f"    {key:<20} {'same' if same else 'DIFFERS':8} {x} vs {y}")
    for key in ("seed", "deterministic", "model_revisions"):
        same = ma.get(key) == mb.get(key)
        ok &= same
        print(f"    {key:<20} {'same' if same else 'DIFFERS':8} {ma.get(key) if same else (ma.get(key), mb.get(key))}")
    va, vb = ma.get("versions") or {}, mb.get("versions") or {}
    diff_v = {k: (va.get(k), vb.get(k)) for k in set(va) | set(vb) if va.get(k) != vb.get(k)}
    print(f"    {'versions/gpu':<20} {'same' if not diff_v else 'DIFFERS':8} {diff_v or va.get('gpu')}")
    if diff_v:
        print("    (different GPU or library versions: identical numbers are not expected)")
    return ok


def compare_images(a, b):
    from PIL import Image
    ok = True
    for name in IMAGES:
        pa, pb = os.path.join(a, name), os.path.join(b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            continue
        x = np.asarray(Image.open(pa).convert("RGB"), dtype=np.int16)
        y = np.asarray(Image.open(pb).convert("RGB"), dtype=np.int16)
        if x.shape != y.shape:
            print(f"    {name:<28} DIFFERS  shape {x.shape} vs {y.shape}")
            ok = False
            continue
        frac = float((np.abs(x - y).max(axis=2) > 0).mean())
        same = frac == 0.0
        ok &= same
        print(f"    {name:<28} {'same' if same else 'DIFFERS':8}"
              + ("" if same else f" {100 * frac:.2f}% of pixels differ, max |diff| {int(np.abs(x - y).max())}"))
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("--atol", type=float, default=0.0,
                   help="tolerance on every number (default 0: bit-exact)")
    args = p.parse_args()
    for d in (args.run_a, args.run_b):
        if not os.path.exists(os.path.join(d, "metrics.json")):
            sys.exit(f"{d}: no metrics.json (run not finished?)")

    print(f"A: {args.run_a}\nB: {args.run_b}")
    print("steps")
    ok = compare_steps(args.run_a, args.run_b, args.atol)
    print("metrics")
    ok &= compare_metrics(args.run_a, args.run_b, args.atol)
    print("images")
    ok &= compare_images(args.run_a, args.run_b)
    print("\nRESULT:", "IDENTICAL" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
