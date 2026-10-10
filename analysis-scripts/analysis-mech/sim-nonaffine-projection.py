"""
Calculate the non-affine displacement field for a colloidal gel.

Given a quiescent equilibrium configuration, apply a small affine shear
strain and compute the non-affine displacement field by minimizing the
harmonic energy of the bond network. Projects the field onto each bond
to estimate the relative non-affine relaxation per bond.
"""

import numpy as np
import pandas as pd
import gsd.hoomd

# LU solver
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve

# conjugate gradient
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import minres




# create "nonaffine" subfolder if it doesn't exit
data_outpath = "nonaffine"
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)



def load_bonds_from_psf(psf_path):
    """
    Load bond connectivity from a CHARMM PSF file.
    
    PSF files have sections marked with header lines like:
        !NATOM
        !NBOND: bonds
        !NTHETA: angles
        ...
    
    The !NBOND section lists bonds as pairs of atom indices (1-indexed in PSF).
    Bond pairs are typically grouped 4 per line (8 numbers per line), but
    formatting can vary.
    
    Returns:
        bonds: (M, 2) numpy array of 0-indexed particle indices
        n_particles: number of particles (from !NATOM line)
    """
    with open(psf_path, 'r') as f:
        lines = f.readlines()
    
    bonds = []
    n_particles = 0
    
    # Parse line by line, looking for section headers
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # NATOM section header
        if '!NATOM' in line:
            # The number of atoms is typically the first integer on this line
            parts = line.split()
            for part in parts:
                try:
                    n_particles = int(part)
                    break
                except ValueError:
                    continue
            # Skip past the atom records (one per particle)
            i += 1 + n_particles
            continue
        
        # NBOND section header
        if '!NBOND' in line:
            # The number of bonds is on this line
            parts = line.split()
            n_bonds_expected = 0
            for part in parts:
                try:
                    n_bonds_expected = int(part)
                    break
                except ValueError:
                    continue
            
            # Parse bond pairs from subsequent lines
            i += 1
            collected = 0
            while collected < n_bonds_expected and i < len(lines):
                line = lines[i].strip()
                if not line or line.startswith('!'):
                    i += 1
                    continue
                
                # Each line has integers; bonds come in pairs
                values = line.split()
                try:
                    ints = [int(v) for v in values]
                except ValueError:
                    # Not a bond line, must be another section
                    break
                
                # Group into pairs
                for k in range(0, len(ints) - 1, 2):
                    bond_i = ints[k] - 1      # PSF uses 1-indexed
                    bond_j = ints[k + 1] - 1
                    bonds.append([bond_i, bond_j])
                    collected += 1
                    if collected >= n_bonds_expected:
                        break
                i += 1
            continue
        
        i += 1
    
    return np.array(bonds, dtype=int), n_particles

def morse_force_and_stiffness(r, U0, kappa, r0):
    """
    Compute the radial force F(r) and stiffness k(r) for a Morse pair potential.
    
    U(r) = U0 * [(1 - exp(-kappa*(r - r0)))^2 - 1]
    F(r) = -dU/dr
    k(r) = d2U/dr2
    
    Returns:
        F_r: force magnitude (positive = repulsive, negative = attractive)
        k_r: radial stiffness (d2U/dr2)
    """
    exp_term = np.exp(-kappa * (r - r0))
    # First derivative: dU/dr = 2*U0*kappa * exp(-k(r-r0)) * (1 - exp(-k(r-r0)))
    dU_dr = 2.0 * U0 * kappa * exp_term * (1.0 - exp_term)
    # Second derivative
    d2U_dr2 = 2.0 * U0 * kappa**2 * exp_term * (2.0 * exp_term - 1.0)

    F_r = -dU_dr ## FORCE TERM
    k_r = d2U_dr2
    return F_r, k_r


def get_bond_potential_params(type_i, type_j, U0_dict, kappa, r0_dict):
    """
    Given bond endpoint particle types, return the Morse parameters for that bond.
    
    Args:
        type_i, type_j: integer or string particle types (e.g. 0=small, 1=large
                        or 'S'=small, 'L'=large depending on your convention)
        U0_dict: dict like {'SS': 12, 'SL': 18, 'LL': 24} (kT units)
        kappa: scalar, same for all pairs
        r0_dict: dict like {'SS': 2.0, 'SL': 3.0, 'LL': 4.0} (contact distances)
    
    Returns:
        U0, kappa, r0 for this bond
    """
    # Convert numeric types to size label
    type_map = {1: 'S', 2: 'L'}

    if isinstance(type_i, (int, np.integer)):
        ti = type_map[int(type_i)]
    else:
        ti = type_i
    if isinstance(type_j, (int, np.integer)):
        tj = type_map[int(type_j)]
    else:
        tj = type_j

    # Sort to get canonical bond type
    pair_sorted = ''.join(sorted([ti, tj]))

    return U0_dict[pair_sorted], kappa, r0_dict[pair_sorted]


def compute_affine_forces(positions, types, bonds, box_lengths, gamma,
                            U0_dict, kappa, r0_dict, kT=1.0):
    """
    Apply affine shear strain (gamma) to particle positions and compute
    the forces this creates on each particle.
    
    The affine shear maps r_i -> r_i + gamma * y_i * x_hat.
    After this shift, bonds have new lengths that aren't in equilibrium,
    creating forces on each particle.
    
    Args:
        positions: (N, 3) original equilibrium positions
        types: (N,) particle types
        bonds: (M, 2) bond endpoint indices
        box_lengths: (3,) box edge lengths
        gamma: small shear strain magnitude (e.g., 1e-4)
        U0_dict, kappa, r0_dict, kT: Morse parameters
    
    Returns:
        affine_positions: (N, 3) positions after affine shear (no relaxation)
        affine_forces: (N, 3) forces on each particle in affine state
        bond_stiffnesses: (M,) k_radial at affine bond lengths
        bond_directions: (M, 3) bond unit vectors in affine state
    """
    N = len(positions)
    
    # Apply affine shear
    affine_positions = positions.copy()
    affine_positions[:, 0] += gamma * positions[:, 1]
    
    # Compute forces from each bond at affine positions
    affine_forces = np.zeros((N, 3))
    bond_stiffnesses = np.zeros(len(bonds))
    bond_directions = np.zeros((len(bonds), 3))
    bond_lengths = np.zeros(len(bonds))
    bond_forces = np.zeros(len(bonds))  # radial force magnitudes
    
    for bidx in range(len(bonds)):
        i, j = int(bonds[bidx, 0]), int(bonds[bidx, 1])
        
        # Vector with PBC
        dr = affine_positions[j] - affine_positions[i]
        dr -= box_lengths * np.round(dr / box_lengths)
        r = np.linalg.norm(dr)
        
        if r == 0:
            continue
        
        n_ij = dr / r
        bond_directions[bidx] = n_ij
        bond_lengths[bidx] = r
        
        U0, kap, r0 = get_bond_potential_params(
            types[i], types[j], U0_dict, kappa, r0_dict
        )
        U0_phys = U0 * kT
        
        F_r, k_r = morse_force_and_stiffness(r, U0_phys, kap, r0)
        bond_stiffnesses[bidx] = k_r
        bond_forces[bidx] = F_r
        
        # Force on particle i is along -n_ij (pointing toward j),
        # force on particle j is along +n_ij
        # Sign convention: F_r > 0 means repulsive (pushes i and j apart)
        force_vec = F_r * n_ij
        affine_forces[i] -= force_vec   # repulsive pushes i in -n_ij direction
        affine_forces[j] += force_vec   # repulsive pushes j in +n_ij direction
    
    return affine_positions, affine_forces, bond_stiffnesses, bond_directions, bond_lengths, bond_forces


def build_dynamical_matrix(positions, bonds, bond_directions, bond_stiffnesses,
                            bond_lengths, bond_forces):
    """
    Build the harmonic dynamical matrix K (3N x 3N).
    
    For each bond ij with stiffness k_ij and direction n_ij, the contribution
    to the dynamical matrix is:
    
        K^bond_ii = K^bond_jj = M_ij
        K^bond_ij = K^bond_ji = -M_ij
    
    where M_ij = k_ij * n_ij ⊗ n_ij + (F_ij/r_ij) * (I - n_ij ⊗ n_ij)
    
    The first term is the radial stretching stiffness; the second is the
    transverse (rotational) stiffness from the central force at non-equilibrium
    bond length.
    
    Args:
        positions: (N, 3) particle positions
        bonds: (M, 2) bond indices
        bond_directions: (M, 3) bond unit vectors
        bond_stiffnesses: (M,) k_radial values
        bond_lengths: (M,) bond lengths
        bond_forces: (M,) radial forces (positive = repulsive)
    
    Returns:
        K: (3N, 3N) sparse dynamical matrix in CSR format
    """
    N = len(positions)
    K = lil_matrix((3*N, 3*N))
    I3 = np.eye(3)
    
    for bidx in range(len(bonds)):
        i, j = int(bonds[bidx, 0]), int(bonds[bidx, 1])
        n_ij = bond_directions[bidx]
        k_r = bond_stiffnesses[bidx]
        r = bond_lengths[bidx]
        F = bond_forces[bidx]
        
        if r == 0:
            continue
        
        # Radial contribution: k_r * n_ij outer n_ij
        outer = np.outer(n_ij, n_ij)
        # Transverse contribution: (F/r) * (I - outer)
        transverse = - (F / r) * (I3 - outer)
        
        M_ij = k_r * outer + transverse
        # Stretching-only dynamical matrix (no pre-stress / transverse term)
        # "harmonic spring" approximation treats each bond as a Hookean spring 
        # along its current direction, ignoring the rotational softness from 
        # the central-force pre-stress. The non-affine result is then strictly 
        # positive-definite as long as the bond network is rigid
        
        # Place into the dynamical matrix
        for a in range(3):
            for b in range(3):
                K[3*i + a, 3*i + b] += M_ij[a, b]
                K[3*j + a, 3*j + b] += M_ij[a, b]
                K[3*i + a, 3*j + b] -= M_ij[a, b]
                K[3*j + a, 3*i + b] -= M_ij[a, b]
    
    return K.tocsr()

def select_regularization(K, residual_forces, box_lengths, gamma,
                          eps_list=(1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11),
                          flat_tol=1e-3, verbose=True):
    """
    The Hessian K of a free, periodic network has exactly 3 zero modes
    for rigid translation in x, y, z. These are physical zero modes (uniform 
    translation costs no energy) that make K exactly singular and will 
    cause a direct solve to fail. Therefore, after a clean (saddle-free) quench
    that ensures all physical soft modes have positive eigenvalues, we regularize 
    the matrix with epsilon*I to lift the zero modes. 

    This epsilon is selected by solving K_reg u = F_res across a range of values
    and checking that G_NA remains independent of epsilon (G_NA vs. epsilon plateaus). 
    We then select the largest epsilon value in this region to achieve the best conditioned
    solve (i.e. make sure epsilon is large enough to actually lift the zero modes, make 
    K non-singular, and achieve as close to machine precision as possible with the LU solver). 

    This regularization is justified by the fact that F_res is orthogonal to rigid 
    translations (by Newton's third law), so the lifted translational modes should
    not carry physical signal AND the per-bond quantities evaluated in this study 
    depend on relative values, where this regularization cancels out.

    Parameters
    ----------
    K : (3N,3N) sparse, UNregularized Hessian.
    residual_forces : (N,3) F(gamma) - F(0), the affine-induced force field.
    box_lengths : (3,) box edges, for V = prod(box).
    gamma : applied shear strain.
    eps_list : regularization values to sweep (descending).
    flat_tol : relative drift defining the flat window (default 1e-3).

    Returns
    -------
    eps_selected : float, the chosen regularization.
    table : list of (epsilon, G_NA, LU_residual) tuples.
    """

    N3 = K.shape[0]
    V = float(np.prod(box_lengths))
    F_flat = residual_forces.flatten()
    eps_list = sorted(eps_list, reverse=True)   # large -> small

    table = []
    print("  regularization sweep (G_NA vs epsilon):")
    print(f"    {'epsilon':>10} {'G_NA/gamma^2':>16} {'LU residual':>14}")
    for eps in eps_list:
        K_reg = K + eps * speye(N3, format='csr')
        u = spsolve(K_reg, F_flat).reshape(-1, 3)
        G_NA = float(np.sum(residual_forces * u)) / (V * gamma**2)
        res = np.linalg.norm(K_reg @ u.flatten() - F_flat) / np.linalg.norm(F_flat)
        table.append((eps, G_NA, res))
        print(f"    {eps:10.0e} {G_NA:16.6f} {res:14.2e}")

    gna = np.array([t[1] for t in table])
    resid = np.array([t[2] for t in table])

    # Make sure the LU solve is clean (low residual)
    clean = resid < 1e-10
    if not clean.any():
        raise RuntimeError("No epsilon gave a clean LU solve (<1e-10). The "
                           "matrix may be too singular -- check the quench.")
    plateau = gna[clean].mean()

    # Flat window: epsilons that fall within the G_NA plateau to flat_tol 
    #   AND have a clean solve. Pick the LARGEST of these (best conditioned)
    in_window = (np.abs(gna - plateau) / abs(plateau) < flat_tol) & clean
    if not in_window.any():
        raise RuntimeError("G_NA depends on epsilon everywhere in the sweep -- "
                           "no flat window. The quench is not clean enough: "
                           "physical soft modes are falling below epsilon. "
                           "Quench harder before trusting the regularization.")
    eps_selected = max(e for e, flat in zip([t[0] for t in table], in_window) if flat)

    drift = (gna[in_window].max() - gna[in_window].min()) / abs(plateau)
    print(f"  plateau G_NA = {plateau:.6f}   "
              f"flat-window drift = {drift:.2e}  (< {flat_tol:.0e} required)")
    print(f"  -> regularization justified: G_NA is epsilon-independent over "
              f"the flat window.")
    print(f"  -> selected epsilon = {eps_selected:.0e}  "
              f"(largest value still in the physics-independent plateau)")

    return eps_selected, table


def regularize_matrix(K, epsilon=1e-8):
    """
    Add a small regularization to handle the zero modes (translations).
    A 3D system has 3 trivial zero modes from rigid translation. Adding
    epsilon * I to the diagonal lifts these without significantly altering
    the non-trivial response.
    """
    N3 = K.shape[0]
    K_reg = K + epsilon * csr_matrix(np.eye(N3))
    return K_reg


def solve_nonaffine_displacement(positions, types, bonds, box_lengths, gamma,
                                    U0_dict, kappa, r0_dict, kT=1.0,
                                    regularization=1e-8
                                    ):
    """
    Compute the non-affine displacement field under small shear strain.
    
    Returns:
        nonaffine_displacements: (N, 3) non-affine displacement of each particle
        affine_positions: (N, 3) affine-deformed positions
        K: sparse dynamical matrix
    """
    print(f"Computing affine state with gamma = {gamma}...")
    (affine_positions, affine_forces, bond_stiffnesses,
     bond_directions, bond_lengths, bond_forces) = compute_affine_forces(
        positions, types, bonds, box_lengths, gamma,
        U0_dict, kappa, r0_dict, kT=kT
    )

    print(f"Building dynamical matrix ({len(positions)} particles, "
          f"{len(bonds)} bonds)...")
    K = build_dynamical_matrix(
        positions, bonds, bond_directions, bond_stiffnesses,
        bond_lengths, bond_forces
    )

    # build K once (unregularized), then certify + select epsilon
    K = build_dynamical_matrix(positions, bonds, bond_directions,
                               bond_stiffnesses, bond_lengths, bond_forces)

    # Use residual forces (removes pre-stress, if present)
    _, eq_forces, *_ = compute_affine_forces(positions, types, bonds,
                                             box_lengths, 0.0,
                                             U0_dict, kappa, r0_dict, kT=kT)
    _, affine_forces, *_ = compute_affine_forces(positions, types, bonds,
                                                box_lengths, gamma,
                                                U0_dict, kappa, r0_dict, kT=kT)
    residual_forces = affine_forces - eq_forces

    # Regularize the matrix to remove the 3 translational zero modes
    #  - select the largest regularization possible without affecting floppy modes
    #  - ensures the matrix is not singular and we can use spsolve
    regularization, _ = select_regularization(K, residual_forces, box_lengths, gamma)
    K_reg = regularize_matrix(K, epsilon=regularization)
    
    print("Solving for non-affine displacements...")
    # Solve K * u_NA = -F_affine
    F_flat = residual_forces.flatten()
    u_lu  = spsolve(K_reg, F_flat) # sparse LU solver

    ## COMPARE TO STANDARD CONJUGATE-GRADIENT METHOD
    u_cg, info = minres(K_reg, F_flat, rtol=1e-8, maxiter=20000)
    rel_diff = np.linalg.norm(u_cg - u_lu) / np.linalg.norm(u_lu)
    print(f"CG info={info}, ||u_cg - u_lu||/||u_lu|| = {rel_diff:.2e}")

    ## Check LU residual to see if solve is exact
    res = np.linalg.norm(K_reg @ u_lu - F_flat) / np.linalg.norm(F_flat)
    print(f"LU residual ||Ku-F||/||F|| = {res:.2e}")
    u_flat = u_lu.copy()
    nonaffine_displacements = u_flat.reshape(-1, 3)
    
    return nonaffine_displacements, affine_positions, K, residual_forces


def project_nonaffine_onto_bonds(nonaffine_displacements, bonds, bond_directions):
    """
    For each bond, compute how the non-affine displacement field affects it.
    
    Returns per-bond:
        relative_displacement: u_j - u_i (vector difference of non-affine
                                          displacements at the bond endpoints)
        nonaffine_stretch: component of relative displacement along bond
                           (positive = stretching, negative = compression)
        nonaffine_rotation: perpendicular component (shear of bond direction)
        magnitude: total magnitude of relative non-affine displacement
    """
    n_bonds = len(bonds)
    relative_disp = np.zeros((n_bonds, 3))
    stretch = np.zeros(n_bonds)
    rotation = np.zeros(n_bonds)
    magnitude = np.zeros(n_bonds)
    
    for bidx in range(n_bonds):
        i, j = int(bonds[bidx, 0]), int(bonds[bidx, 1])
        delta_u = nonaffine_displacements[j] - nonaffine_displacements[i]
        n_ij = bond_directions[bidx]
        
        relative_disp[bidx] = delta_u
        
        # Project onto bond direction
        stretch_component = np.dot(delta_u, n_ij)
        stretch[bidx] = stretch_component
        
        # Perpendicular component
        perp = delta_u - stretch_component * n_ij
        rotation[bidx] = np.linalg.norm(perp)
        
        magnitude[bidx] = np.linalg.norm(delta_u)
    
    return relative_disp, stretch, rotation, magnitude


def compute_nonaffine_modulus_correction(nonaffine_displacements, residual_forces, 
                                            box_lengths, gamma):
    """
    Compute the non-affine correction to G'.
    
    The non-affine correction is:
        G^NA = (1/(V * gamma^2)) * F_residual_i · u_NA_o
    
    Subtracting this from the affine Born modulus gives the true G':
        G_true = G_affine - G^NA

    Args:
        nonaffine_displacements: (N, 3) particle non-affine displacements
        residual_forces: (N, 3) F_affine - F_equilibrium
        box_lengths: (3,) box edges
        gamma: shear strain magnitude

    Returns:
        G_NA: scalar, positive, with correct gamma normalization
        G_NA_per_particle: (N,) per-particle contributions
    """
    V = float(np.prod(box_lengths))
    
    # The non-affine correction is the work done by the affine forces against
    # the non-affine displacement, normalized by V and gamma^2
    # Per-particle contribution: F_affine_i · u_NA_i
    per_particle = np.sum(residual_forces * nonaffine_displacements, axis=1)
    G_NA_total = np.sum(per_particle) / V ## drop 0.5 because this cancels out in 2nd derivative
    
    return G_NA_total, per_particle


# ============================================================
# Main calculation
# ============================================================

def analyze_nonaffine_field(positions, types, bonds, box_lengths,
                              U0_dict, kappa, r0_dict, kT=1.0,
                              gamma=1e-4, edge_data_df=None):
    """
    Full non-affine analysis pipeline.
    
    Args:
        positions, types, bonds, box_lengths: system data
        U0_dict, kappa, r0_dict, kT: Morse parameters
        gamma: small shear strain (default 1e-4)
        edge_data_df: optional dataframe with EBC, motif, bridge info to merge in
    
    Returns:
        bond_df: per-bond non-affine projection statistics
        particle_df: per-particle non-affine displacement
        summary: dict of overall statistics
    """
    # Solve for non-affine field
    nonaffine_disp, affine_positions, K, residual_forces = solve_nonaffine_displacement(
        positions, types, bonds, box_lengths, gamma,
        U0_dict, kappa, r0_dict, kT=kT
    )
    
    # Recompute affine forces for projection
    _, affine_forces, _, bond_directions, bond_lengths, bond_forces = compute_affine_forces(
        positions, types, bonds, box_lengths, gamma,
        U0_dict, kappa, r0_dict, kT=kT
    )
    
    # Project onto bonds
    relative_disp, stretch, rotation, magnitude = project_nonaffine_onto_bonds(
        nonaffine_disp, bonds, bond_directions
    )
    
    # Compute non-affine modulus correction
    G_NA, G_NA_per_particle = compute_nonaffine_modulus_correction(
        nonaffine_disp, residual_forces, box_lengths, gamma
    )
    G_NA_normalized = G_NA / gamma**2
    
    # Build per-bond dataframe
    type_map = {1: 'S', 2: 'L'}
    bond_types_str = []
    for bidx in range(len(bonds)):
        i, j = int(bonds[bidx, 0]), int(bonds[bidx, 1])
        ti = type_map.get(int(types[i]), str(types[i]))
        tj = type_map.get(int(types[j]), str(types[j]))
        bond_pair = {ti, tj}
        if bond_pair == {'S'}:
            bond_types_str.append('SS')
        elif bond_pair == {'L'}:
            bond_types_str.append('LL')
        else:
            bond_types_str.append('SL')
    
    bond_df = pd.DataFrame({
        'i': bonds[:, 0],
        'j': bonds[:, 1],
        'type': bond_types_str,
        'r': bond_lengths,
        'nonaffine_stretch': stretch,
        'nonaffine_rotation': rotation,
        'nonaffine_magnitude': magnitude,
        # Normalize by gamma to make these scale-independent
        'nonaffine_stretch_per_gamma': stretch / gamma,
        'nonaffine_rotation_per_gamma': rotation / gamma,
        'nonaffine_magnitude_per_gamma': magnitude / gamma,
    })
    
    # Merge with bond properties if provided
    if edge_data_df is not None:
        bond_df = bond_df.merge(
            edge_data_df[['i', 'j', 'ebc', 'ebc_w', 'backbone', 'bridge',
                          'in_isolated', 'in_pair', 'in_chain', 'in_tree',
                          'in_cycle', 'in_dense', 'in_bipyramid',
                          'in_triangle', 'in_triangle_pendant', 'noT']],
            on=['i', 'j'], how='left'
        )
    
    # Per-particle dataframe
    particle_df = pd.DataFrame({
        'particle_id': np.arange(len(positions)),
        'type': [type_map.get(int(t), str(t)) for t in types],
        'pos_x': positions[:, 0],
        'pos_y': positions[:, 1],
        'pos_z': positions[:, 2],
        'nonaffine_ux': nonaffine_disp[:, 0],
        'nonaffine_uy': nonaffine_disp[:, 1],
        'nonaffine_uz': nonaffine_disp[:, 2],
        'nonaffine_magnitude': np.linalg.norm(nonaffine_disp, axis=1),
        'G_NA_contribution': G_NA_per_particle,
    })
    
    summary = {
        'gamma': gamma,
        'G_NA': G_NA,
        'G_NA_per_gamma_squared': G_NA_normalized,
        'mean_particle_nonaffine_magnitude': np.linalg.norm(nonaffine_disp, axis=1).mean(),
        'max_particle_nonaffine_magnitude': np.linalg.norm(nonaffine_disp, axis=1).max(),
        'mean_bond_stretch_per_gamma': np.abs(stretch).mean() / gamma,
        'mean_bond_rotation_per_gamma': rotation.mean() / gamma,
    }
    
    return bond_df, particle_df, summary


# ============================================================
# Rn calculation
# ============================================================

if __name__ == '__main__':
    gsd_paths = ["quench/quench_clean_small.gsd",
                 "quench/quench_clean_bi.gsd",
                 "quench/quench_clean_large.gsd"
                ]
    psf_files = ["../analysis-topology/viz/100-0/100-0_psfs_standard/all_bonds.psf",
                 "../analysis-topology/viz/50-50/50-50_psfs_standard/all_bonds.psf",
                 "../analysis-topology/viz/0-100/0-100_psfs_standard/all_bonds.psf"
                ]
    labels = ['small','bi','large']

    for p in range(len(gsd_paths)):
        gsd_path = gsd_paths[p]
        label = labels[p]
        psf_file = psf_files[p]
        if label == 'large':
          U0_dict = {'SS': 24.0, 'SL': 18.0, 'LS': 18.0, 'LL': 12.0}  # in kT units
          r0_dict = {'SS': 4.0, 'SL': 3.0, 'LS': 3.0, 'LL': 2.0}     # contact distances (radii sum)
        else:
          U0_dict = {'SS': 12.0, 'SL': 18.0, 'LS': 18.0, 'LL': 24.0}  # in kT units
          r0_dict = {'SS': 2.0, 'SL': 3.0, 'LS': 3.0, 'LL': 4.0}     # contact distances (radii sum)

        kappa = 60.0  # attraction range parameter
        kT = 0.1      # your thermal energy
 
        with gsd.hoomd.open(gsd_path, 'r') as traj:
            frame = traj[-1]
            colloids = np.where((frame.particles.typeid == 1) | (frame.particles.typeid == 2))[0]
            positions = np.array(frame.particles.position[colloids])
            types = np.array(frame.particles.typeid[colloids])
            box_lengths = np.array(frame.configuration.box[:3])
    
        # Load bonds from PSF
        bonds, _ = load_bonds_from_psf(psf_file)
     
        # Run non-affine analysis
        bond_df, particle_df, summary = analyze_nonaffine_field(
            positions, types, bonds, box_lengths,
            U0_dict, kappa, r0_dict, kT=kT,
            gamma=1e-4,
            # edge_data_df=your_edge_data_df  # optional
        )
   
        # Compute forces at the ORIGINAL (un-sheared) positions
        _, eq_forces, _, _, _, _ = compute_affine_forces(
            positions, types, bonds, box_lengths,
            0.0,  # gamma, as positional
            U0_dict, kappa, r0_dict, kT=kT
        )

        print("Summary:", summary)
        bond_df.to_csv(f"{data_outpath}/nonaffine_per_bond_{label}.csv", index=False)
        particle_df.to_csv(f"{data_outpath}/nonaffine_per_particle_{label}.csv", index=False)
