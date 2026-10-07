"""Capped trialanine (Ace-Ala3-Nme) in vacuum with a linear torsion design.

U(conf; x) = U0(conf) + sum_j x_j B_j(conf), with U0 = Amber ff14SB (openmm amber14-all.xml), NoCutoff,
no solvent. The design x (kJ/mol) is an additive Fourier correction to the three backbone phi torsions:
    B = [cos phi_i, sin phi_i, cos 2phi_i, sin 2phi_i]  for i = 1, 2, 3   (12 dims, x0 = 0 is ff14SB).
The basis is linear in x, so any x is evaluated from frames saved at another x by reweighting with B.
States: s_i = 1 ('+', the phi ~ +60 basin) if 0 < phi_i < 120 deg, else 0 ('-', the phi ~ -70 basin);
state index = 4 s_1 + 2 s_2 + s_3 (8 states, Tang/Chen/Roux/Chipot arXiv:2609.04449 Fig. 2B).
"""
import numpy as np
try:   # openmm is needed only to build / simulate; the analysis helpers below are numpy-only
    import openmm as mm
    import openmm.app as app
    import openmm.unit as u
except ImportError:
    mm = app = u = None

KB = 0.0083144626  # kJ/mol/K
NBASIS_PER_RES = 4
D = 3 * NBASIS_PER_RES
BASIS_NAMES = [f"{f}(phi{i+1})" for i in range(3) for f in ("cos", "sin", "cos2", "sin2")]


def build_pdb(path):
    """Ace-Ala3-Nme heavy atoms from a 5-residue Ala chain (PeptideBuilder, extended-ish start), caps
    taken from residues 1 and 5; hydrogens added by openmm Modeller."""
    import PeptideBuilder
    from PeptideBuilder import Geometry
    geos = []
    for k in range(5):
        g = Geometry.geometry("A")
        g.phi, g.psi_im1 = -75.0, 140.0
        geos.append(g)
    st = PeptideBuilder.make_structure_from_geos(geos)
    res = list(st.get_residues())
    lines, n = [], 0

    def atom(name, resn, resi, xyz, el):
        nonlocal n
        n += 1
        lines.append("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s" %
                     (n, name if len(name) == 4 else " " + name, resn, resi, *xyz, el))
    atom("CH3", "ACE", 1, res[0]["CA"].coord, "C")
    atom("C", "ACE", 1, res[0]["C"].coord, "C")
    atom("O", "ACE", 1, res[0]["O"].coord, "O")
    for k in (1, 2, 3):
        for nm in ("N", "CA", "C", "O", "CB"):
            atom(nm, "ALA", k + 1, res[k][nm].coord, nm[0])
    atom("N", "NME", 5, res[4]["N"].coord, "N")
    atom("C", "NME", 5, res[4]["CA"].coord, "C")
    lines += ["TER", "END"]
    tmp = path + ".heavy.pdb"
    open(tmp, "w").write("\n".join(lines) + "\n")
    pdb = app.PDBFile(tmp)
    ff = app.ForceField("amber14-all.xml")
    mod = app.Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff)
    with open(path, "w") as f:
        app.PDBFile.writeFile(mod.topology, mod.positions, f)
    return path


def phi_indices(topology):
    """(C_{i-1}, N_i, CA_i, C_i) for the three Ala residues, and psi (N_i, CA_i, C_i, N_{i+1})."""
    res = list(topology.residues())
    idx = lambda r, nm: [a.index for a in r.atoms() if a.name == nm][0]
    phis = [(idx(res[k - 1], "C"), idx(res[k], "N"), idx(res[k], "CA"), idx(res[k], "C")) for k in (1, 2, 3)]
    psis = [(idx(res[k], "N"), idx(res[k], "CA"), idx(res[k], "C"), idx(res[k + 1], "N")) for k in (1, 2, 3)]
    return phis, psis


def make_system(pdb_path, x=None, flatten=None):
    """System for design x (kJ/mol, len 12; None -> x0 = 0). The design enters through a CustomTorsionForce
    with global parameters x0..x11 (force group 1). `flatten` = optional static sampling bias on phi
    (callable added separately by the sampler), not used here."""
    pdb = app.PDBFile(pdb_path)
    ff = app.ForceField("amber14-all.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    phis, psis = phi_indices(pdb.topology)
    x = np.zeros(D) if x is None else np.asarray(x, float)
    for i, q in enumerate(phis):
        b = NBASIS_PER_RES * i
        expr = (f"xa{i}*cos(theta)+xb{i}*sin(theta)+xc{i}*cos(2*theta)+xd{i}*sin(2*theta)")
        f = mm.CustomTorsionForce(expr)
        for k, nm in enumerate("abcd"):
            f.addGlobalParameter(f"x{nm}{i}", float(x[b + k]))
        f.addTorsion(*q, [])
        f.setForceGroup(1)
        system.addForce(f)
    return pdb, system, phis, psis


def basis(phi):
    """phi: (N, 3) radians -> B (N, 12)."""
    out = np.empty((phi.shape[0], D))
    for i in range(3):
        p = phi[:, i]
        out[:, 4 * i:4 * i + 4] = np.stack([np.cos(p), np.sin(p), np.cos(2 * p), np.sin(2 * p)], 1)
    return out


def states(phi):
    """phi: (N, 3) radians -> state index 0..7 (bit i = 1 if phi_i in (0, 120) deg)."""
    d = np.degrees(phi)
    plus = (d > 0) & (d < 120)
    return (plus[:, 0] * 4 + plus[:, 1] * 2 + plus[:, 2]).astype(int)


STATE_LABELS = ["".join("+" if (s >> (2 - i)) & 1 else "-" for i in range(3)) for s in range(8)]


def dihedral(pos, q):
    """pos (N, A, 3); q tuple of 4 atom indices -> dihedral (N,) radians (IUPAC sign)."""
    p0, p1, p2, p3 = (pos[:, k] for k in q)
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1n = b1 / np.linalg.norm(b1, axis=1, keepdims=True)
    v = b0 - (b0 * b1n).sum(1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(1, keepdims=True) * b1n
    xx = (v * w).sum(1)
    yy = (np.cross(b1n, v) * w).sum(1)
    return np.arctan2(yy, xx)


def canonical_features(phi, psi):
    """Label-free representation of a frame: the three residues' (phi, psi) pairs sorted by phi (ascending in
    (-pi, pi]), each as (cos phi, sin phi, cos psi, sin psi) -> (N, 12). Invariant to relabeling the residues,
    so the ensemble of these features is the label-free ensemble."""
    order = np.argsort(phi, axis=1)
    ph = np.take_along_axis(phi, order, 1); ps = np.take_along_axis(psi, order, 1)
    return np.stack([np.cos(ph), np.sin(ph), np.cos(ps), np.sin(ps)], 2).reshape(len(phi), 12)


def count_plus(phi):
    """Number of residues in the phi ~ +60 basin (0..3)."""
    d = np.degrees(phi)
    return ((d > 0) & (d < 120)).sum(1)
