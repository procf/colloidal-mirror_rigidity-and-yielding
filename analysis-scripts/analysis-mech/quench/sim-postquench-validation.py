"""
Post-quench validation of the non-affine G_NA decomposition.

Check whether the estimate of G_NA from particle/Hessian route vs. 
harmonic bond energy AGREE. 
They are the same quadratic form (u^T H u = F^res . u^NA) and MUST match for an
exact, well-conditioned Hessian; but they can diverge when the network has floppy
modes (lambda <~ epsilon) with nonzero affine-field overlap (common is sub-isostatic gel systems)

This script reports, per system:
  1. G_NA from the particle route     :  (1/V gamma^2) sum_i F^res_i . u^NA_i
  2. G_NA from the bond-energy route  :  (1/V gamma^2) sum_ij Du^T M Du
  3. their ratio (target -> 1.0 after a good quench)
  4. an epsilon-sweep of that ratio. If the ratio does NOT depend on epsilon,
     the quench worked and any residual gap is physical. If the ratio scales
     with epsilon, you still have lambda <~ epsilon modes and need a
     null-space projection (rather than a smaller epsilon).


NOTE: The parameter-consistency check (Section 4) will abort if the stiffnesses used
here disagree with what built the Hessian

NOTE: This does not invert the full Hessian; it uses spsolve to get u^NA.
"""

import numpy as np
import pandas as pd
import gsd.hoomd
from scipy.sparse import lil_matrix, csr_matrix, eye as speye
from scipy.sparse.linalg import spsolve


# ==========================================================================
# PSF parser for bond data
# ==========================================================================
def load_bonds_from_psf(psf_path):
    with open(psf_path, 'r') as f:
        lines = f.readlines()
    bonds, n_particles, i = [], 0, 0
    while i < len(lines):
        line = lines[i].strip()
        if '!NATOM' in line:
            for part in line.split():
                try:
                    n_particles = int(part); break
                except ValueError:
                    continue
            i += 1 + n_particles
            continue
        if '!NBOND' in line:
            n_bonds_expected = 0
            for part in line.split():
                try:
                    n_bonds_expected = int(part); break
                except ValueError:
                    continue
            i += 1
            collected = 0
            while collected < n_bonds_expected and i < len(lines):
                l = lines[i].strip()
                if not l or l.startswith('!'):
                    i += 1; continue
                try:
                    ints = [int(v) for v in l.split()]
                except ValueError:
                    break
                for k in range(0, len(ints) - 1, 2):
                    bonds.append([ints[k] - 1, ints[k + 1] - 1])  # PSF is 1-indexed
                    collected += 1
                    if collected >= n_bonds_expected:
                        break
                i += 1
            continue
        i += 1
    return np.array(bonds, dtype=int), n_particles


# ==========================================================================
# Morse helpers (F = -U', kappa = U'')
# ==========================================================================
def morse_F_k(r, U0, kap, r0):
    e = np.exp(-kap * (r - r0))
    dU  = 2.0 * U0 * kap    * e * (1.0 - e)       # U'(r)
    d2U = 2.0 * U0 * kap**2 * e * (2.0 * e - 1.0)  # U''(r)
    return -dU, d2U                                # F=-U', k=U''


def build_bond_params(types, bonds, U0_dict, kappa, r0_dict, kT):
    """Per-bond (U0, kappa, r0) -- MUST match what the Hessian solve used."""
    type_map = {1: 'S', 2: 'L'}
    M = len(bonds)
    U0b = np.empty(M); r0b = np.empty(M); kapb = np.full(M, float(kappa))
    for b in range(M):
        ti = type_map[int(types[bonds[b, 0]])]
        tj = type_map[int(types[bonds[b, 1]])]
        pair = ''.join(sorted([ti, tj]))
        U0b[b] = U0_dict[pair] * kT
        r0b[b] = r0_dict[pair]
    return U0b, kapb, r0b


# ==========================================================================
# Geometry
# ==========================================================================
def bond_geometry(positions, bonds, box, lees_edwards_gamma=0.0):
    """
    Minimum-image bond vectors, lengths, unit vectors.

    lees_edwards_gamma: if nonzero, engineering shear strain;
    applies the Lees-Edwards image shift for a box sheared by 
    x -> x + gamma*y and ensures that bonds crossing the y-boundary 
    get the correct image (lees_edwards_gamma=0 for the quiescent gel)
    """
    box = np.asarray(box, float)
    dr = positions[bonds[:, 1]] - positions[bonds[:, 0]]
    if lees_edwards_gamma != 0.0:
        # number of y-images to unwrap, then shift x by gamma*Ly per image
        ny = np.round(dr[:, 1] / box[1])
        dr[:, 0] -= lees_edwards_gamma * box[1] * ny
    dr -= box * np.round(dr / box)
    r = np.linalg.norm(dr, axis=1)
    good = r > 0
    return dr, r, good


# ==========================================================================
# Forces, Hessian, solve  (mirrors your production code)
# ==========================================================================
def compute_forces(positions, types, bonds, box, U0b, kapb, r0b,
                   lees_edwards_gamma=0.0):
    N = len(positions)
    dr, r, good = bond_geometry(positions, bonds, box, lees_edwards_gamma)
    F = np.zeros((N, 3))
    Fmag = np.zeros(len(bonds)); krad = np.zeros(len(bonds))
    n = np.zeros_like(dr)
    n[good] = dr[good] / r[good, None]
    for b in range(len(bonds)):
        if not good[b]:
            continue
        i, j = bonds[b]
        f, k = morse_F_k(r[b], U0b[b], kapb[b], r0b[b])
        Fmag[b] = f; krad[b] = k
        fv = f * n[b]
        F[i] -= fv; F[j] += fv     # F=-U'>0 (repulsive) pushes i,j apart
    return F, r, n, good, Fmag, krad


def build_hessian(positions, bonds, n, r, krad, Fmag, good):
    """H from per-bond M_ij = k nn + (F/r)(I-nn), with F=-U' so F/r = -U'/r.
       (transverse coefficient is U'/r = -F/r; sign carried by F=-U'.)"""
    N = len(positions)
    H = lil_matrix((3 * N, 3 * N))
    I3 = np.eye(3)
    for b in range(len(bonds)):
        if not good[b]:
            continue
        i, j = bonds[b]
        nn = np.outer(n[b], n[b])
        M = krad[b] * nn - (Fmag[b] / r[b]) * (I3 - nn)   # -F/r = U'/r
        for a in range(3):
            for c in range(3):
                H[3*i+a, 3*i+c] += M[a, c]; H[3*j+a, 3*j+c] += M[a, c]
                H[3*i+a, 3*j+c] -= M[a, c]; H[3*j+a, 3*i+c] -= M[a, c]
    return H.tocsr()


def solve_nonaffine(positions, types, bonds, box, U0b, kapb, r0b,
                    gamma, epsilon):
    """Return u^NA (per particle) and F^res, for a given regularization eps."""
    N = len(positions)
    # residual force field: F(gamma) - F(0)
    aff = positions.copy(); aff[:, 0] += gamma * positions[:, 1]
    F_g, r_g, n_g, good_g, Fmag_g, krad_g = compute_forces(
        aff, types, bonds, box, U0b, kapb, r0b, lees_edwards_gamma=gamma)
    F_0, _, _, _, _, _ = compute_forces(
        positions, types, bonds, box, U0b, kapb, r0b, lees_edwards_gamma=0.0)
    Fres = (F_g - F_0)

    # Hessian at the quiescent configuration
    _, r0len, n0, good0, Fmag0, krad0 = compute_forces(
        positions, types, bonds, box, U0b, kapb, r0b, lees_edwards_gamma=0.0)
    H = build_hessian(positions, bonds, n0, r0len, krad0, Fmag0, good0)
    Hreg = H + epsilon * speye(3 * N, format='csr')

    u = spsolve(Hreg, Fres.flatten()).reshape(-1, 3)
    return u, Fres, H, (aff, r_g, n_g, good_g)


# ==========================================================================
# The two routes
# ==========================================================================
def gna_particle_route(Fres, u_NA, V, gamma):
    """(1/V gamma^2) sum_i F^res_i . u^NA_i  = (1/Vg^2) u^T H_reg u."""
    return float(np.sum(Fres * u_NA)) / (V * gamma**2)


def gna_bond_route(u_NA, bonds, n0, r0len, krad0, Fmag0, good0, V, gamma):
    """(1/V gamma^2) sum_ij Du^T M_ij Du  = (1/Vg^2) u^T H u (UNregularized).

    Uses e_ij = k s^2 + (F/r-with-correct-sign) perp^2, i.e. the same M blocks.
    """
    tot = 0.0
    for b in range(len(bonds)):
        if not good0[b]:
            continue
        i, j = bonds[b]
        du = u_NA[j] - u_NA[i]
        s = np.dot(du, n0[b])
        perp2 = np.dot(du, du) - s * s
        kt = -Fmag0[b] / r0len[b]                 # U'/r
        tot += krad0[b] * s * s + kt * perp2
    return tot / (V * gamma**2)


# ==========================================================================
# Driver
# ==========================================================================
def validate_system(label, gsd_path, psf_path, U0_dict, kappa, r0_dict, kT,
                    gamma=1e-4, eps_list=(1e-7, 1e-8, 1e-9)):
    with gsd.hoomd.open(gsd_path, 'r') as traj:
        frame = traj[-1]
        tid_all = np.asarray(frame.particles.typeid)          # ALL particles
        col = np.where((tid_all == 1) | (tid_all == 2))[0]    # colloid subset
        positions = np.array(frame.particles.position[col], float)
        types = tid_all[col]
        box = np.array(frame.configuration.box[:3], float)
    V = float(np.prod(box))

    # --- load PSF bonds (global indices) and remap to the colloid subset ---
    bonds_global, _ = load_bonds_from_psf(psf_path)
    # global particle id -> local index within `col`; -1 for non-colloids
    global_to_local = -np.ones(len(tid_all), dtype=int)
    global_to_local[col] = np.arange(len(col))
    bonds = global_to_local[bonds_global]                     # (M,2), local indices
    # drop any bond touching a non-colloid particle (maps to -1), if present
    keep = (bonds[:, 0] >= 0) & (bonds[:, 1] >= 0)
    if not keep.all():
        print(f"  [{label}] dropped {(~keep).sum()} bonds touching non-colloid "
              f"particles ({keep.sum()} kept)")
    bonds = bonds[keep]

    if bonds.max() >= len(positions):
        raise ValueError(f"[{label}] remap failed: bond index {bonds.max()} >= "
                         f"{len(positions)} colloids.")

    U0b, kapb, r0b = build_bond_params(types, bonds, U0_dict, kappa, r0_dict, kT)

    # --- coordination diagnostic (are we still floppy?) -------------------
    from collections import Counter
    cnt = Counter()
    for i, j in bonds:
        cnt[int(i)] += 1; cnt[int(j)] += 1
    z = np.array([cnt.get(k, 0) for k in range(len(positions))])
    print(f"\n===== {label} =====")
    print(f"  <z> = {z.mean():.2f}   (central-force isostatic in 3D is z=6; "
          f"floppy if <6)")

    # --- epsilon sweep of the ratio --------------------------------------
    print(f"  {'epsilon':>10} {'particle G_NA':>16} {'bond G_NA':>14} "
          f"{'ratio(bond/particle)':>22}")
    rows = []
    for eps in eps_list:
        u, Fres, H, _ = solve_nonaffine(positions, types, bonds, box,
                                        U0b, kapb, r0b, gamma, eps)
        # geometry at quiescent frame for the bond route
        _, r0len, n0, good0, Fmag0, krad0 = compute_forces(
            positions, types, bonds, box, U0b, kapb, r0b, 0.0)
        gp = gna_particle_route(Fres, u, V, gamma)
        gb = gna_bond_route(u, bonds, n0, r0len, krad0, Fmag0, good0, V, gamma)
        ratio = gb / gp if gp != 0 else np.nan
        print(f"  {eps:10.0e} {gp:16.6e} {gb:14.6e} {ratio:22.4f}")
        rows.append(dict(label=label, epsilon=eps, G_NA_particle=gp,
                         G_NA_bond=gb, ratio=ratio, z_mean=z.mean()))

    # --- interpretation ---------------------------------------------------
    ratios = [r['ratio'] for r in rows]
    spread = (max(ratios) - min(ratios)) / np.mean(ratios) if np.mean(ratios) else np.inf
    print("  ---")
    if abs(np.mean(ratios) - 1.0) < 0.05 and spread < 0.05:
        print("  PASS: routes agree and ratio is epsilon-independent. "
              "Sum rule restored -- absolute G_NA is now trustworthy.")
    elif spread > 0.2:
        print("  FAIL: ratio still depends strongly on epsilon. You still have "
              "lambda <~ epsilon modes.\n        -> quench harder, or use "
              "null-space projection instead of eps I.")
    else:
        print("  PARTIAL: ratio epsilon-stable but != 1. Residual gap is likely "
              "physical (genuine soft modes with nonzero Xi-overlap).\n"
              "        Absolute G_NA remains ill-defined; report RELATIVE "
              "per-bond localization only.")
    return pd.DataFrame(rows)


# ==========================================================================
if __name__ == '__main__':

    gsd_paths = ["quench_clean_small.gsd", 
                 "quench_clean_bi.gsd", 
                 "quench_clean_large.gsd"
                 ]
    psf_files = ["../viz/100-0/100-0_psfs_standard/all_bonds.psf",
                 "../viz/50-50/50-50_psfs_standard/all_bonds.psf",
                 "../viz/0-100/0-100_psfs_standard/all_bonds.psf"
                 ]

    labels    = ['small', 'bi', 'large']
    kappa, kT = 60.0, 0.1

    all_rows = []
    for p, label in enumerate(labels):
        if label == 'large':
            U0_dict = {'SS': 24.0, 'SL': 18.0, 'LS':18.0, 'LL': 12.0}
            r0_dict = {'SS': 4.0,  'SL': 3.0,  'LS':3.0, 'LL': 2.0}
        else:
            U0_dict = {'SS': 12.0, 'SL': 18.0, 'LS':18.0, 'LL': 24.0}
            r0_dict = {'SS': 2.0,  'SL': 3.0,  'LS':3.0, 'LL': 4.0}

        df = validate_system(label, gsd_paths[p], psf_files[p],   # <- path, not np.load
                             U0_dict, kappa, r0_dict, kT,
                             gamma=1e-4, eps_list=(1e-7, 1e-8, 1e-9))
        all_rows.append(df)

    pd.concat(all_rows, ignore_index=True).to_csv(
        "postquench_validation.csv", index=False)
    print("\nWrote postquench_validation.csv")
