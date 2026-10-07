"""Sampling for capped trialanine in vacuum (OpenMM, Reference platform: fastest for 42 atoms).

  metad : well-tempered metadynamics on (phi1, phi2, phi3) at design x; writes the final bias grid (npz).
  prod  : Langevin MD at design x with an optional FROZEN bias grid V(phi) (static Hamiltonian
          U0 + x.B + V); saves phi, psi, V per frame. Reweighting to any design x' uses
          log w = -(x' - x).B/kT + V/kT.
Every run writes a diagnostics JSON (wall time, frames, state counts, phi transitions).
"""
import argparse, json, os, time
import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as u
from openmm.app.metadynamics import Metadynamics, BiasVariable
import system as S

HERE = os.path.dirname(os.path.abspath(__file__))
PDB = os.path.join(HERE, "..", "trialanine.pdb")


def phi_forces(phis):
    out = []
    for q in phis:
        f = mm.CustomTorsionForce("theta")
        f.addTorsion(*q, [])
        out.append(f)
    return out


def bias_force(phis, grid, lim=np.pi):
    """Static bias V(phi1,phi2,phi3) from a metadynamics grid (OpenMM ordering), force group 2."""
    cv = mm.CustomCVForce("table(phi1, phi2, phi3)")
    for i, f in enumerate(phi_forces(phis)):
        cv.addCollectiveVariable(f"phi{i+1}", f)
    w = grid.shape[::-1]
    cv.addTabulatedFunction("table", mm.Continuous3DFunction(*w, grid.flatten(), -lim, lim, -lim, lim, -lim, lim, True))
    cv.setForceGroup(2)
    return cv


def parse_x(s):
    if s is None:
        return np.zeros(S.D)
    if s.endswith(".npy"):
        return np.load(s)
    return np.array([float(v) for v in s.split(",")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["metad", "prod"])
    ap.add_argument("--x", default=None, help="design, comma list (kJ/mol) or .npy; default x0 = 0")
    ap.add_argument("--bias", default=None, help="frozen bias npz (prod)")
    ap.add_argument("--ns", type=float, default=10.0)
    ap.add_argument("--save_ps", type=float, default=1.0)
    ap.add_argument("--T", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--equil_ps", type=float, default=100.0)
    ap.add_argument("--bias_factor", type=float, default=8.0)
    ap.add_argument("--height", type=float, default=0.5, help="kJ/mol")
    ap.add_argument("--width", type=float, default=0.3, help="rad")
    ap.add_argument("--pace", type=int, default=250)
    ap.add_argument("--bias_dir", default=None, help="shared dir for multiple-walker metadynamics")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    x = parse_x(a.x)
    pdb, system, phis, psis = S.make_system(PDB, x)
    dt = 0.002
    integ = mm.LangevinMiddleIntegrator(a.T * u.kelvin, 1 / u.picosecond, dt * u.picoseconds)
    integ.setRandomNumberSeed(a.seed + 1)
    if a.mode == "prod" and a.bias:
        grid = np.load(a.bias)["grid"]
        system.addForce(bias_force(phis, grid))
    meta = None
    if a.mode == "metad":
        vars_ = [BiasVariable(f, -np.pi, np.pi, a.width, True, 36) for f in phi_forces(phis)]
        kw = dict(saveFrequency=a.pace * 20, biasDir=a.bias_dir) if a.bias_dir else {}
        if a.bias_dir:
            os.makedirs(a.bias_dir, exist_ok=True)
        meta = Metadynamics(system, vars_, a.T * u.kelvin, a.bias_factor, a.height * u.kilojoules_per_mole, a.pace, **kw)
    sim = app.Simulation(pdb.topology, system, integ, mm.Platform.getPlatformByName("Reference"))
    sim.context.setPositions(pdb.positions)
    sim.minimizeEnergy()
    sim.context.setVelocitiesToTemperature(a.T * u.kelvin, a.seed + 1)
    t0 = time.time()
    run = (lambda n: meta.step(sim, n)) if meta else sim.step
    run(int(a.equil_ps / dt))
    nsave = int(round(a.save_ps / dt))
    nframes = int(a.ns * 1000 / a.save_ps)
    phi = np.empty((nframes, 3)); psi = np.empty((nframes, 3)); V = np.zeros(nframes)
    for k in range(nframes):
        run(nsave)
        st = sim.context.getState(getPositions=True)
        pos = st.getPositions(asNumpy=True).value_in_unit(u.nanometer)[None]
        phi[k] = [S.dihedral(pos, q)[0] for q in phis]
        psi[k] = [S.dihedral(pos, q)[0] for q in psis]
        if a.mode == "prod" and a.bias:
            V[k] = sim.context.getState(getEnergy=True, groups={2}).getPotentialEnergy().value_in_unit(u.kilojoules_per_mole)
    wall = time.time() - t0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    s = S.states(phi)
    flips = int((np.diff((np.degrees(phi) > 0) & (np.degrees(phi) < 120), axis=0) != 0).sum())
    diag = dict(mode=a.mode, x=x.tolist(), bias=a.bias, ns=a.ns, save_ps=a.save_ps, T=a.T, seed=a.seed,
                frames=nframes, wall_s=wall, ns_per_day=a.ns / wall * 86400,
                state_counts=np.bincount(s, minlength=8).tolist(), phi_sign_flips=flips,
                platform="Reference", dt_ps=dt, friction_per_ps=1.0, ff="amber14-all.xml (ff14SB), NoCutoff, HBonds")
    if meta is not None:
        grid = meta._totalBias.copy()
        diag.update(bias_factor=a.bias_factor, height=a.height, width=a.width, pace=a.pace)
        np.savez(a.out.replace(".npz", "_bias.npz"), grid=grid, self_grid=meta._selfBias.copy(), T=a.T,
                 bias_factor=a.bias_factor)
        diag["bias_max_kJ"] = float(grid.max())
    np.savez_compressed(a.out, phi=phi, psi=psi, V=V, x=x, T=a.T)
    json.dump(diag, open(a.out.replace(".npz", ".json"), "w"), indent=1)
    print(json.dumps(diag))


if __name__ == "__main__":
    main()
