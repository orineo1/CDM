"""Speed and sanity check of the simulator: kinetic T, bond lengths, basin populations (their utils.py boxes)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulate import simulate
import common
from utils import basin_probabilities, basin_names   # GB-FESO
eps = torch.tensor([[0.25, 0.25], [0.25, 1.4], [1.4, 0.25], [1.4, 1.4], [0.8, 0.8], [0.5, 1.0]])
for B_rep, steps in [(8, 200_000), (512, 200_000)]:
    t0 = time.time()
    Y, T = simulate(eps, n_rep=B_rep, n_steps=steps, burn=20_000, return_temperature=True, verbose=False)
    print(f"n_rep={B_rep} chains={B_rep*len(eps)} steps={steps}: {time.time()-t0:.1f}s  T={T:.4f}")
Y, T = simulate(eps, n_rep=8, n_steps=int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000, burn=50_000, return_temperature=True, verbose=True)
print("T", T, basin_names())
for e, y in zip(eps, Y):
    print(e.tolist(), "bond mean/sd", y[:, :3].mean().item(), y[:, :3].std().item(),
          "basins", [round(v, 3) for v in basin_probabilities(y).tolist()])
