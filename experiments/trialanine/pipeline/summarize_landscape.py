"""Print a compact table of landscape_*.json results."""
import glob, json, os, sys
import numpy as np
d0 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
for f in sorted(glob.glob(os.path.join(d0, "landscape_*.json"))):
    d = json.load(open(f))
    for ti, t in enumerate(d["targets"]):
        for k, r in t["losses"].items():
            bands = [b["band_barrier"] for b in r["band_barriers"]]
            print(f"{os.path.basename(f):24s} t{ti} {k:8s} |x*|={np.linalg.norm(t['xstar'])/d['kT']:5.2f}kT "
                  f"Lrand_med={r['L_random_median']:.3g} Lmin={r['L_min']:.1e} global={r['n_global']}/{r['n_starts']} "
                  f"comps={r['n_components']} sizes={[c['size'] for c in r['components']]} "
                  f"line_max={r['max_line_barrier']:.3g} band_barriers={np.round(bands, 4).tolist()} "
                  f"spread={r['global_rms_spread_kT']:.2f}kT minESS={min(r['ess_all_min']):.0f} at_box_max={max(r['at_box'])}")
