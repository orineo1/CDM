"""
End-to-end convenience wrapper that calls generate_data.py, train_models.py,
and run_optimization.py in sequence.

For fine-grained control, run each script separately:

  Step 1:  python generate_data.py --n-samples 10000 --output data/mog2d.pt
  Step 2:  python train_models.py --data data/mog2d.pt --model both --epochs 5 --output-dir models/run1
  Step 3:  python run_optimization.py --data data/mog2d.pt --model-dir models/run1 --method LGD_CM

This wrapper runs all three with a single command:

  python run_experiment.py --n-samples 10000 --epochs 5 --method LGD_CM
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full pipeline: generate data -> train models -> optimize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data generation
    p.add_argument("--n-samples", type=int, default=10000,
                   help="Samples to draw from P(X,Y).")
    p.add_argument("--params-file", type=str, default=None,
                   help="SimulationParameters/*.txt file (optional).")
    p.add_argument("--random-mog", action="store_true")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--condition-on", type=int, default=1)
    p.add_argument("--uncond-dim", type=int, default=1)
    p.add_argument("--min-separation", type=float, default=5.0)

    # Training
    p.add_argument("--architecture", type=str, default="cm",
                   choices=["diffusion", "cm", "fm"])
    p.add_argument("--epochs", type=int, default=5,
                   help="True epochs over the fixed dataset.")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--nunits", type=int, default=128)
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--diffusion-steps", type=int, default=100)

    # Optimization
    p.add_argument("--method", type=str, default="LGD_CM",
                   choices=["LGD", "LGD_CM", "DFLOW"])
    p.add_argument("--n-attempts", type=int, default=5)
    p.add_argument("--optim-samples", type=int, default=250)
    p.add_argument("--mc-samples", type=int, default=3)

    # General
    p.add_argument("--work-dir", type=str, default="experiment_output",
                   help="Directory for all intermediate and output files.")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)

    return p.parse_args()


def run(cmd):
    """Run a command, printing it first."""
    print(f"\n{'='*60}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nCommand failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    args = parse_args()
    py = sys.executable
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_path = os.path.join(args.work_dir, "data.pt")
    model_dir = os.path.join(args.work_dir, "models")
    results_path = os.path.join(args.work_dir, "results.pt")

    # ---- Step 1: Generate data ----
    cmd = [py, os.path.join(script_dir, "generate_data.py"),
           "--n-samples", str(args.n_samples),
           "--condition-on", str(args.condition_on),
           "--output", data_path]
    if args.params_file:
        cmd += ["--params-file", args.params_file]
    if args.random_mog:
        cmd += ["--random-mog", "--dim", str(args.dim),
                "--n-components", str(args.n_components),
                "--uncond-dim", str(args.uncond_dim),
                "--min-separation", str(args.min_separation)]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    run(cmd)

    # ---- Step 2: Train models ----
    cmd = [py, os.path.join(script_dir, "train_models.py"),
           "--data", data_path,
           "--model", "both",
           "--architecture", args.architecture,
           "--epochs", str(args.epochs),
           "--batch-size", str(args.batch_size),
           "--nunits", str(args.nunits),
           "--nblocks", str(args.nblocks),
           "--diffusion-steps", str(args.diffusion_steps),
           "--output-dir", model_dir]
    if args.device:
        cmd += ["--device", args.device]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    run(cmd)

    # ---- Step 3: Optimize ----
    cmd = [py, os.path.join(script_dir, "run_optimization.py"),
           "--data", data_path,
           "--model-dir", model_dir,
           "--method", args.method,
           "--n-attempts", str(args.n_attempts),
           "--optim-samples", str(args.optim_samples),
           "--mc-samples", str(args.mc_samples),
           "--output", results_path]
    if args.device:
        cmd += ["--device", args.device]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]
    run(cmd)

    print(f"\nAll outputs in: {args.work_dir}/")
    print(f"  data.pt         - dataset + distribution parameters")
    print(f"  models/          - trained model checkpoints")
    print(f"  results.pt       - optimization results")


if __name__ == "__main__":
    main()
