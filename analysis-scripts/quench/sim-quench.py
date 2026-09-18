"""
FIRE quench of the final frame of a DPD gelation simulation
to produce a saddle-free reference state for the G_A & G_NA solvers.

FIRE removes saddles genuine negative-curvature directions that
are artifacts of an incompletely minimized systems. It does not remove the
physical soft/floppy modes of a sub-isostatic gel. Those modes form a
dense band of small eigenvalues down to the numerical-zero floor and are real physics. 

A clean minimum therefore still has an ill-conditioned absolute G_NA; 
the point of the clean quench is a saddle-free, prestress-consistent 
reference state (better Xi, H, e_ij), NOT a rescued absolute modulus. 

Therefore, we use this to report the RELATIVE per-bond localization 
of G_A and the non-affine harmonic relaxation energy, e_ij

-  Convergence criterion: the number of eigsh eigenvalues
   below a floor (-neg_floor). eigsh returns real eigenvalues 
   for the symmetric (Hermetian) Hessian.

-  Checkpoints after every FIRE round (quench_ckpt_<label>.gsd) and RESUMES from
   the checkpoint if present; stopping cleanly (leaving a resumable
   checkpoint) if it runs out of time before the spectrum is clean.

Conventions 
--------------------------------------------------
  U(r)  = U0 [(1 - exp(-kappa (r-r0)))^2 - 1]
  F     = -U'          (positive = repulsive)
  k     =  U''
  M_ij  = k (n n) + (U'/r)(I - n n) = k (n n) - (F/r)(I - n n) 
"""

import os
import time
import numpy as np
import gsd.hoomd
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import eigsh


# --------------------------------------------------------------------------
# Handling PSF bond data
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Per-bond Morse parameters 
# --------------------------------------------------------------------------
def build_bond_params(types, bonds, U0_dict, kappa, r0_dict, kT):
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


def morse_F_k(r, U0, kap, r0):
    e = np.exp(-kap * (r - r0))
    dU = 2.0 * U0 * kap * e * (1.0 - e)            # U'
    d2U = 2.0 * U0 * kap**2 * e * (2.0 * e - 1.0)  # U''
    return -dU, d2U                                # F = -U', k = U''


# --------------------------------------------------------------------------
# Energy + gradient (fixed bonds)
# --------------------------------------------------------------------------
def energy_and_grad(x_flat, shape, bi, bj, box, U0b, kapb, r0b):
    x = x_flat.reshape(shape)
    dr = x[bj] - x[bi]
    dr -= box * np.round(dr / box)
    r = np.linalg.norm(dr, axis=1)
    r = np.where(r == 0.0, 1e-12, r)
    e = np.exp(-kapb * (r - r0b))
    U = U0b * ((1.0 - e) ** 2 - 1.0)
    E = float(U.sum())
    dUdr = 2.0 * U0b * kapb * e * (1.0 - e)
    gj = (dUdr / r)[:, None] * dr
    N = x.shape[0]
    g = np.empty_like(x)
    for d in range(3):
        g[:, d] = (np.bincount(bj, weights=gj[:, d], minlength=N)
                   - np.bincount(bi, weights=gj[:, d], minlength=N))
    return E, g.ravel()


# --------------------------------------------------------------------------
# FIRE (semi-implicit Euler, dt capped for kappa=60 stiffness). 
# --------------------------------------------------------------------------
def fire_batch(x0, grad_fn, shape, deadline, dt=1e-3, dtmax=5e-3, ftol=1e-10,
               max_steps=30000, alpha0=0.1, falpha=0.99, finc=1.1,
               fdec=0.5, nmin=5, verbose=True):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    alpha = alpha0; npos = 0
    _, g = grad_fn(x); F = -g
    fmax = np.max(np.linalg.norm(F.reshape(shape), axis=1))
    t0 = time.time()
    hit_deadline = False
    for step in range(max_steps):
        if fmax < ftol:
            break
        if time.time() > deadline:
            hit_deadline = True
            break
        P = float(F @ v)
        if P > 0.0:
            nF = np.linalg.norm(F); nv = np.linalg.norm(v)
            v = (1.0 - alpha) * v + alpha * (nv / (nF + 1e-30)) * F
            npos += 1
            if npos > nmin:
                dt = min(dt * finc, dtmax); alpha *= falpha
        else:
            v[:] = 0.0; npos = 0; dt *= fdec; alpha = alpha0
        v = v + dt * F
        x = x + dt * v
        E, g = grad_fn(x); F = -g
        fmax = np.max(np.linalg.norm(F.reshape(shape), axis=1))
        if verbose and step % 5000 == 0:
            print(f"      FIRE {step:6d}  E={E:.6e}  max|F|={fmax:.2e}  "
                  f"dt={dt:.1e}  ({time.time()-t0:.0f}s)", flush=True)
    return x, fmax, step + 1, hit_deadline


# --------------------------------------------------------------------------
# Corrected Hessian (vectorized COO)
# --------------------------------------------------------------------------
def build_hessian(positions, bi, bj, U0b, kapb, r0b, box, N):
    d = positions[bj] - positions[bi]
    d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    good = r > 0
    d, r = d[good], r[good]
    bii, bjj = bi[good], bj[good]
    U0g, kapg, r0g = U0b[good], kapb[good], r0b[good]
    n = d / r[:, None]
    F, k = morse_F_k(r, U0g, kapg, r0g)
    nn = np.einsum('bi,bj->bij', n, n)
    I3 = np.eye(3)
    Mb = k[:, None, None] * nn - (F / r)[:, None, None] * (I3[None] - nn)  # -F/r = U'/r
    bi3, bj3 = 3 * bii, 3 * bjj
    rows, cols, data = [], [], []
    for a in range(3):
        for b in range(3):
            m = Mb[:, a, b]
            rows += [bi3 + a, bj3 + a, bi3 + a, bj3 + a]
            cols += [bi3 + b, bj3 + b, bj3 + b, bi3 + b]
            data += [m,        m,        -m,       -m]
    rows = np.concatenate(rows); cols = np.concatenate(cols); data = np.concatenate(data)
    return coo_matrix((data, (rows, cols)), shape=(3 * N, 3 * N)).tocsr()


# --------------------------------------------------------------------------
# Low spectrum via eigsh (symmetric -> real eigenvalues). Shift-invert near 0
# with a tiny regularizing shift so the factorization of the singular H exists.
# --------------------------------------------------------------------------
def low_spectrum(H, n_ev=300):
    N3 = H.shape[0]
    k = min(n_ev, N3 - 2)
    try:
        vals = eigsh(H + 1e-12 * identity(N3), k=k, sigma=0.0, which='LM',
                     return_eigenvectors=False)
    except Exception:
        vals = eigsh(H, k=k, sigma=1e-10, which='LM', return_eigenvectors=False)
    return np.sort(vals)


def spectrum_report(vals, neg_floor=1e-7, zero_tol=1e-11):
    """Three-bin breakdown. neg_floor separates real saddles from noise;
    zero_tol marks the numerical-zero (translation/floppy) band."""
    n_sig_neg = int(np.sum(vals < -neg_floor))     # real saddles to clear
    n_noise_neg = int(np.sum((vals >= -neg_floor) & (vals < -zero_tol)))
    n_zero = int(np.sum(np.abs(vals) <= zero_tol))  # translations + floppy
    n_pos = int(np.sum(vals > zero_tol))
    return n_sig_neg, n_noise_neg, n_zero, n_pos


# --------------------------------------------------------------------------
# Checkpoint I/O
# --------------------------------------------------------------------------
def write_frame(path, positions, typeids, box6, ptypes):
    box6 = np.asarray(box6, float)
    pos = np.array(positions, float)
    if np.allclose(box6[3:], 0.0):
        L = box6[:3]; pos -= L * np.round(pos / L)
    try:
        frame = gsd.hoomd.Frame()
    except AttributeError:
        frame = gsd.hoomd.Snapshot()
    frame.particles.N = int(len(pos))
    frame.particles.position = pos
    frame.particles.typeid = np.asarray(typeids)
    frame.particles.types = list(ptypes)
    frame.configuration.box = box6
    with gsd.hoomd.open(path, 'w') as t:
        t.append(frame)


def read_frame(path):
    with gsd.hoomd.open(path, 'r') as traj:
        fr = traj[-1]
        tid = np.asarray(fr.particles.typeid)
        col = np.where((tid == 1) | (tid == 2))[0]
        pos = np.array(fr.particles.position[col], float)
        types = tid[col]
        box6 = np.array(fr.configuration.box, float)
        ptypes = list(fr.particles.types)
    return pos, types, box6, ptypes


# --------------------------------------------------------------------------
# Driver: FIRE in rounds, checkpoint each round, stop when clean or out of time
# --------------------------------------------------------------------------
def quench_until_clean(label, orig_gsd, psf_file, U0_dict, kappa, r0_dict, kT,
                       wall_seconds, ckpt_path, clean_path,
                       steps_per_round=30000, max_rounds=1000,
                       n_ev=300, neg_floor=1e-7, audit_every=1):
    t_start = time.time()
    deadline = t_start + wall_seconds

    # resume from checkpoint if present, else start from the original frame
    if os.path.exists(ckpt_path):
        print(f"  resuming from checkpoint {ckpt_path}", flush=True)
        positions, types, box6, ptypes = read_frame(ckpt_path)
    else:
        print(f"  starting fresh from {orig_gsd}", flush=True)
        positions, types, box6, ptypes = read_frame(orig_gsd)

    box = box6[:3]
    N = len(positions)
    bonds, _ = load_bonds_from_psf(psf_file)
    if bonds.max() >= N:
        raise ValueError(f"[{label}] bond index {bonds.max()} >= {N} colloids; "
                         "remap PSF global indices to the colloid subset first.")
    bi, bj = bonds[:, 0], bonds[:, 1]
    U0b, kapb, r0b = build_bond_params(types, bonds, U0_dict, kappa, r0_dict, kT)
    shape = positions.shape
    grad_fn = lambda xf: energy_and_grad(xf, shape, bi, bj, box, U0b, kapb, r0b)

    # coordination, for context
    z = np.bincount(np.concatenate([bi, bj]), minlength=N)
    print(f"  N={N}  <z>={z.mean():.2f}  (isostatic z=6; floppy if <6)", flush=True)

    def audit(tag):
        _, g = grad_fn(positions.ravel())
        fmax = float(np.max(np.linalg.norm(g.reshape(shape), axis=1)))
        H = build_hessian(positions, bi, bj, U0b, kapb, r0b, box, N)
        vals = low_spectrum(H, n_ev=n_ev)
        nsn, nnn, nz, npo = spectrum_report(vals, neg_floor=neg_floor)
        print(f"  [{tag}] max|F|={fmax:.2e}  lowest eig={vals.min():.2e}", flush=True)
        print(f"    significant negatives (< -{neg_floor:.0e}): {nsn}"
              f"  | noise-band negatives: {nnn}  | ~zero: {nz}  | positive: {npo}",
              flush=True)
        return nsn, fmax, vals

    nsn, fmax, vals = audit("initial")
    x = positions.astype(float)
    rnd = 0
    while nsn > 0 and rnd < max_rounds:
        if time.time() > deadline:
            print(f"  wall budget reached before clean; checkpoint saved, "
                  f"resubmit to continue.", flush=True)
            break
        rnd += 1
        print(f"  -- FIRE round {rnd}  ({nsn} significant negatives to clear) --",
              flush=True)
        x, fmax, nst, hit = fire_batch(x.ravel(), grad_fn, shape, deadline,
                                       max_steps=steps_per_round)
        x = x.reshape(shape)
        positions = x
        write_frame(ckpt_path, positions, types, box6, ptypes)   # checkpoint
        print(f"     round {rnd}: {nst} FIRE steps"
              f"{' (deadline hit mid-round)' if hit else ''}; checkpoint written",
              flush=True)
        if rnd % audit_every == 0:
            nsn, fmax, vals = audit(f"after round {rnd}")

    # final audit + write clean frame
    nsn, fmax, vals = audit("final")
    if nsn == 0:
        write_frame(clean_path, positions, types, box6, ptypes)
        print(f"  -> CLEAN: no significant negatives, max|F|={fmax:.2e}. "
              f"Wrote {clean_path}.", flush=True)
        print(f"     (Physical soft modes remain -- see the ~zero/noise bands. "
              f"That is expected for a sub-isostatic gel and does NOT mean the "
              f"quench failed; absolute G_NA stays ill-conditioned, so keep "
              f"reporting RELATIVE per-bond localization.)", flush=True)
    else:
        print(f"  -> NOT yet clean: {nsn} significant negatives remain "
              f"(max|F|={fmax:.2e}). Checkpoint saved; resubmit to continue, or "
              f"raise steps_per_round. If nsn plateaus at a small number across "
              f"many rounds, inspect those eigenvectors -- they may be very flat "
              f"saddles, or your neg_floor may be cutting into the physical band.",
              flush=True)
    return positions, nsn, vals


# ==========================================================================
## CONFIG AND RUN
# ==========================================================================
if __name__ == '__main__':

    # DPD gelation frames 
    orig_gsd = [
        "/projects/props/Rob/colloids/bimodal/r1-0/DPD/L70/mono/seedNone/phi20/potential-morse-bysize/12kT/kappa60/Gelation-lastframe.gsd",
        "/projects/props/Rob/colloids/bimodal/r1-2/DPD/L70/poly0.0/seedNone/phi20/50-50/potential-morse-bysize/12kT/kappa60/Gelation-lastframe.gsd",
        "/projects/props/Rob/colloids/bimodal/r2-0/DPD/L70/phi20/potential-morse-bysize/12kT/kappa60/Gelation-lastframe.gsd",
    ]
    psf_files = [
        "../viz/100-0/100-0_psfs_standard/all_bonds.psf",
        "../viz/50-50/50-50_psfs_standard/all_bonds.psf",
        "../viz/0-100/0-100_psfs_standard/all_bonds.psf",
    ]
    labels = ['small', 'bi', 'large']

    kappa, kT = 60.0, 0.1

    WALL_SECONDS_PER_SYSTEM = 86400          # wall-clock budget per system (seconds)
                                             # defaults to 24 h per system
    STEPS_PER_ROUND = 30000

    NEG_FLOOR = 1e-7                         # saddle vs noise threshold; tuned to
                                             # sit above the numerical-zero band

    for p, label in enumerate(labels):
        print(f"\n========== quenching {label} ==========", flush=True)
        if label == 'large':
            U0_dict = {'SS': 24.0, 'SL': 18.0, 'LS': 18.0, 'LL': 12.0}
            r0_dict = {'SS': 4.0,  'SL': 3.0,  'LS': 3.0,  'LL': 2.0}
        else:
            U0_dict = {'SS': 12.0, 'SL': 18.0, 'LS': 18.0, 'LL': 24.0}
            r0_dict = {'SS': 2.0,  'SL': 3.0,  'LS': 3.0,  'LL': 4.0}

        quench_until_clean(
            label, orig_gsd[p], psf_files[p], U0_dict, kappa, r0_dict, kT,
            wall_seconds=WALL_SECONDS_PER_SYSTEM,
            ckpt_path=f"quench_ckpt_{label}.gsd",
            clean_path=f"quench_clean_{label}.gsd",
            steps_per_round=STEPS_PER_ROUND,
            neg_floor=NEG_FLOOR,
        )
