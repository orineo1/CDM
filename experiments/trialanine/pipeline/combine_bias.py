"""Sum the per-walker self-bias grids of a multiple-walker metadynamics run into one frozen bias npz.
usage: python combine_bias.py 'runs/trialanine/metad_TAG_w*_bias.npz' runs/trialanine/metad_TAG_bias.npz"""
import glob, sys
import numpy as np
fs = sorted(glob.glob(sys.argv[1]))
tot = sum(np.load(f)["self_grid"] for f in fs)
d = np.load(fs[0])
np.savez(sys.argv[2], grid=tot, T=d["T"], bias_factor=d["bias_factor"], walkers=len(fs))
print(len(fs), "walkers, max bias", tot.max())
