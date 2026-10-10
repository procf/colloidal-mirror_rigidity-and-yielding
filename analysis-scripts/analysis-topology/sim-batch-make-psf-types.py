"""
Create VMD-compatible PSF files for different subsets of bonds
from HOOMD-blue GSD files and the all_edge_data.csv file created
with sim-analyze-topology.py 

OUTPUTS:
-------
 all_bonds.psf
 bridge_bonds.psf
 singlebridge.psf
 backbone.psf
 motif.psf
 ebc90.psf
 LL_only.psf
 SL_only.psf
 SS_only.psf
"""

import sys
import numpy as np
import pandas as pd
import gsd.hoomd
import os


small_comps = [100, 50, 0]
compositions = ['100-0', '50-50', '0-100']
size_ratios = ['1-0', '1-2', '0-2']
volume_fractions = ['20']
attraction_strengths = ['12']
depletion_scaling = ['bysize']
attraction_range = ['60']

# choose: 'standard' or 'weighted' GMM clusters
clustering_style = ['standard']

eta0 = 0.3
kT = 0.1
L = 70
system_volume = L**3

# create subfolders if they don't exit
data_outpath = "data"
if os.path.exists(data_outpath) == False:
  os.mkdir(data_outpath)

viz_outpath = "viz"
if os.path.exists(viz_outpath) == False:
  os.mkdir(viz_outpath)


# ======================================================
# READ GSD
# ======================================================
def read_gsd(gsd_file):
    traj = gsd.hoomd.open(gsd_file, mode='r')
    frame = traj[-1]
    N = frame.particles.N
    pos = np.asarray(frame.particles.position, dtype=float)
    typeid = np.asarray(frame.particles.typeid, dtype=int)
    types = list(frame.particles.types)

    return N, pos, typeid, types


# ======================================================
# READ EDGE CSV
# ======================================================
def read_edge_metadata(csvfile):
    df = pd.read_csv(csvfile)

    if "i" not in df.columns or "j" not in df.columns:
        raise ValueError("CSV must contain i,j columns")

    # clean booleans
    structure_class = ["in_isolated", "in_pair", "in_chain", "in_triangle", "in_triangle_pendant", "in_tree", "in_cycle", "in_dense", "in_bipyramid", "backbone"]
    for col in structure_class:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map(
                {"true": True, "false": False}
            )

    if "bridge" in df.columns:
        df["bridge"] = df["bridge"].astype(str).str.lower()

    if "motif" in df.columns:
        df["motif"] = df["motif"].astype(str).str.lower()

    return df


# ======================================================
# MINIMUM IMAGE / PBC FILTER
# ======================================================
def crosses_pbc(ri, rj, L, max_bond=4.1, tol=1e-6):

    dr = rj - ri

    if np.any(np.abs(dr) > (L / 2.0 - tol)):
        return True

    raw_len = np.linalg.norm(dr)

    if raw_len > max_bond:
        return True

    return False


def save_pbc_edges(df, pos, L, max_bond=4.1):

    pbc = []

    for idx, row in df.iterrows():

        i = int(row["i"])
        j = int(row["j"])

        ri = pos[i]
        rj = pos[j]

        bad = crosses_pbc(ri, rj, L, max_bond=max_bond)

        if bad:
            pbc.append(idx)
        else:
            continue

    return df.loc[pbc].copy()

def filter_df_edges(df, pos, L, skip_pbc=True, max_bond=4.1):

    keep = []

    for idx, row in df.iterrows():

        i = int(row["i"])
        j = int(row["j"])

        ri = pos[i]
        rj = pos[j]

        bad = crosses_pbc(ri, rj, L, max_bond=max_bond)

        if bad and skip_pbc:
            continue

        keep.append(idx)

    return df.loc[keep].copy()

# ======================================================
# PSF WRITER
# ======================================================
def write_psf(filename, bonds, N, typeid, types):

    with open(filename, "w") as f:

        f.write("PSF\n\n")
        f.write("       1 !NTITLE\n")
        f.write(" REMARKS generated from GSD + edge metadata\n\n")

        # atoms
        f.write(f"{N:8d} !NATOM\n")

        for idx in range(N):

            atom_id = idx + 1
            segid = "SYS"
            resid = 1
            resname = "COL"
            atomname = f"P{idx}"

            tname = types[typeid[idx]] if typeid[idx] < len(types) else "X"

            charge = 0.0
            mass = 1.0

            f.write(
                f"{atom_id:8d} "
                f"{segid:<4s} "
                f"{resid:4d} "
                f"{resname:<4s} "
                f"{atomname:<6s} "
                f"{tname:<6s} "
                f"{charge:10.6f} "
                f"{mass:10.4f}\n"
            )

        f.write("\n")

        # bonds
        nbond = len(bonds)
        f.write(f"{nbond:8d} !NBOND: bonds\n")

        count = 0

        for i, j in bonds:

            f.write(f"{i+1:8d}{j+1:8d}")
            count += 1

            if count == 4:
                f.write("\n")
                count = 0

        if count != 0:
            f.write("\n")

        f.write("\n")

        sections = [
            "!NTHETA",
            "!NPHI",
            "!NIMPHI",
            "!NDON",
            "!NACC",
            "!NNB",
            "!NGRP",
        ]

        for sec in sections:
            f.write(f"{0:8d} {sec}\n\n")

    print(f"Wrote {filename:<20s} ({nbond} bonds)")


# ======================================================
# BOND HELPERS
# ======================================================
def bonds_from_df(d):
    return list(
        zip(
            d["i"].astype(int).to_numpy(),
            d["j"].astype(int).to_numpy()
        )
    )


# ======================================================
# MAIN
# ======================================================

for size in size_ratios:
  radii = size.split('-')
  R_C1 = float(radii[0])
  R_C2 = float(radii[1])
  for phi in volume_fractions:
      for comp_1 in small_comps:
          comp_2 = 100 - comp_1
          comp = f'{comp_1}-{comp_2}'
          if os.path.exists(f'{viz_outpath}/{comp}') == False:
              os.mkdir(f'{viz_outpath}/{comp}')
          # skip invalid fake mixtures for monodisperse case
          if size != '1-0' and comp == '100-0':
              continue
          if size == '1-0' and comp != '100-0':
              continue
          if size != f'0-{int(R_C2)}' and comp == '0-100':
              continue
          if size == f'0-{int(R_C2)}' and comp != '0-100':
              continue
          if size == f'0-{int(R_C2)}' and comp == '0-100':
              size = f'{int(R_C2)}-0'
              if R_C2 != float(int(R_C2)):
                  print('ERROR: large monomodal data processing script cannot accept float particle size: {R_C2}')
                  continue
          #print(size, comp)
          for D0 in attraction_strengths:
              for scaling in depletion_scaling:
                  for kappa in attraction_range:
                    for cluster_style in clustering_style:
                      out_path = f'{viz_outpath}/{comp}/{comp}_psfs_{cluster_style}'
                      if os.path.exists(out_path) == False:
                          os.mkdir(out_path)

                      if scaling == 'uniform':
                          if (size == '1-0') and (comp == '100-0'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif (size == f'{int(R_C2)}-0') and (comp == '0-100'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif comp != '100-0':
                             data_path = (f'/projects/props/Rob/colloids/bimodal/uniform-potential/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse/'
                                f'{D0}kT/kappa{kappa}/')
                             analysis_path = f"{data_path}analysis-bi-colloids/data/"

                      else:
                          if (size == '1-0') and (comp == '100-0'):
                             if scaling == 'bysize':
                                data_path = (f'/projects/props/Rob/colloids/bimodal/r{size}/DPD/L{L}/mono/seedNone/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/')
                                analysis_path = f"{data_path}analysis-DPD/data/"
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (size == f'{int(R_C2)}-0') and (comp == '0-100'):
                             if scaling == 'bysize':
                                data_path = (f'/projects/props/Rob/colloids/bimodal/r{size}/DPD/L{L}/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/')
                                analysis_path = f"{data_path}analysis-DPD/data/"
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (comp != '100-0') and (comp != '0-100'):
                             data_path = (f'/projects/props/Rob/colloids/bimodal/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse-{scaling}/'
                                f'{D0}kT/kappa{kappa}/')
                             analysis_path = f"{data_path}analysis-bi-colloids/data/"
                          else:
                             continue

                      ### NOTE: Must be colloids only
                      gsd_file = f"{data_path}Gelation_Colloids.gsd"

                      csvfile = f'{data_outpath}/all_edge_data.csv'

                      out_psf = f'{out_path}bond_network.psf' 

                      skip_pbc = True

                      print(f"\nr{size}, phi={phi}, {comp}, D0={D0}kT ({scaling}) kappa={kappa}, {cluster_style} clustering")
                      print(" - Reading GSD...")
                      N, pos, typeid, types = read_gsd(gsd_file)
                      print("   Particles:", N)

                      print(" - Reading edge metadata...")
                      df_all = read_edge_metadata(csvfile)

                      df = df_all.loc[(df_all['size-ratio'] == size) & (df_all['phi'] == float(phi)) & (df_all['composition'] == comp) & 
                                      (df_all['D0'] == float(D0)) & (df_all['scaled_D0'] == scaling) & (df_all['kappa'] == float(kappa)) & 
                                      (df_all['cluster_style'] == cluster_style)
                                      ]

                      print("   Raw edges:", len(df))

                      print("   CSV backbone:", len(df.loc[df["backbone"] == True]))

                      print(" - Filtering edges...")
                      df_keep = filter_df_edges(
                          df,
                          pos,
                          L,
                          skip_pbc=skip_pbc,
                          max_bond=(2*max(R_C1,R_C2))+(3/float(kappa))
                      )
                      df_pbc = save_pbc_edges(
                          df,
                          pos,
                          L,
                          max_bond=(2*max(R_C1,R_C2))+(3/float(kappa))
                      )

                      df_pbc.to_csv(f"{out_path}/cross_boundary_bonds.csv", index=False)

                      print("   Filtered backbone:", len(df_keep.loc[df_keep["backbone"] == True]))

                      print("   Remaining edges:", len(df_keep))

                      # --------------------------------------------------
                      # ALL
                      # --------------------------------------------------

                      write_psf(f"{out_path}/all_bonds.psf", bonds_from_df(df_keep), N, typeid, types)

                      # --------------------------------------------------
                      # BRIDGES
                      # --------------------------------------------------

                      if "bridge" in df_keep.columns:

                          write_psf(
                              f"{out_path}/bridge_bonds.psf",
                              bonds_from_df(df_keep[df_keep["bridge"] != "none"]),
                              N, typeid, types
                          )

                          write_psf(
                              f"{out_path}/singlebridge.psf",
                              bonds_from_df(df_keep[df_keep["bridge"] == "single"]),
                              N, typeid, types
                          )

                      # --------------------------------------------------
                      # BACKBONE
                      # --------------------------------------------------

                      if "backbone" in df_keep.columns:

                          write_psf(
                              f"{out_path}/backbone.psf",
                              bonds_from_df(df_keep[df_keep["backbone"] == True]),
                              N, typeid, types
                          )

                      # --------------------------------------------------
                      # ALL TETRAHEDRA
                      # --------------------------------------------------

                      #motif_cols = ["rho1T", "rho2T", "rho3T", "rho5T", "otherT"]
                      #motif_cols = pd.unique(df_keep['motifs']) 

                      if "motifs" in df_keep.columns:

                         write_psf(
                              f"{out_path}/tetrahedra.psf",
                              bonds_from_df(df_keep[df_keep["motifs"] != 'noT']),
                              N, typeid, types
                         )

                      # --------------------------------------------------
                      # MOTIFS
                      # --------------------------------------------------

                      if "motifs" in df_keep.columns:

                         for motif in pd.unique(df_keep['motifs']):

                              write_psf(
                                  f"{out_path}/motif_{motif}_bonds.psf",
                                  bonds_from_df(df_keep[df_keep["motifs"] == motif]),
                                  N, typeid, types
                              )


                      # --------------------------------------------------
                      # TOPOLOGICAL CLASSES
                      # --------------------------------------------------

                      #topo_cols = ["rho1T", "rho2T", "rho3T", "rho5T", "otherT"]
                      topo_cols = ["in_isolated", "in_pair", "in_chain", "in_triangle", "in_triangle_pendant", "in_tree", "in_cycle", "in_dense", "in_bipyramid"] 

                      if all(col in df_keep.columns for col in topo_cols):

                          mask = False

                          for col in topo_cols:
                              mask |= (df_keep[col] == True)

                              write_psf(
                                  f"{out_path}/topology_{col}.psf",
                                  bonds_from_df(df_keep[mask]),
                                  N, typeid, types
                              )

                      # --------------------------------------------------
                      # TYPE SUBSETS
                      # --------------------------------------------------

                      if "type" in df_keep.columns:

                          for t in ["LL", "SL", "SS"]:

                              write_psf(
                                  f"{out_path}/{t}_only.psf",
                                  bonds_from_df(df_keep[df_keep["type"] == t]),
                                  N, typeid, types
                              )

print("\nDone.")
