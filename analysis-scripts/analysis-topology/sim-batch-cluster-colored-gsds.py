# ============================================================
# Find the minimum number of colors needed to label all GMM clusters
# in the last frame of a GSD file, such that no neighboring clusters
# share the same color
#
# Save separate GSD files for each color-set of clusters so that they
# can be individually loaded in VMD
#
# These GSD files preserve the original:
#   positions
#   diameters
#   mass
#   velocity
#   types
#   full trajectory (or single-frame lastframe.gsd)
#
# INPUTS:
#   clustered_df_standard.csv
#   lcc-cluster-network_G_clusters_standard.pkl
#   lastframe.gsd   (or Gelation_Colloids.gsd)
#
# OUTPUT:
#   cluster_color_0.gsd
#   cluster_color_1.gsd
#   ...
# ============================================================

import pandas as pd
import numpy as np
import networkx as nx
import pickle
import gsd.hoomd
import os


small_comps = [100, 50, 0]
compositions = ['100-0', '50-50', '0-100']
size_ratios = ['1-0', '1-2', '0-2']
volume_fractions = ['20']

attraction_strengths = ['12']
depletion_scaling = ['bysize']
attraction_range = ['60']

# choose: 'standard' 'weighted'
clustering_style = ['standard']

eta0 = 0.3
kT = 0.1
L = 70
system_volume = L**3


strategies = []
numbers_of_colors = []
for size in size_ratios:
  radii = size.split('-')
  R_C1 = float(radii[0])
  R_C2 = float(radii[1])
  for phi in volume_fractions:
      for comp_1 in small_comps:
          comp_2 = 100 - comp_1
          comp = f'{comp_1}-{comp_2}' 
          if os.path.exists(f'viz/{comp}') == False:
              os.mkdir(f'viz/{comp}')
          # skip invalid fake mixtures for monodisperse case
          #print(size, comp, R_C1, R_C2, f'0-{int(R_C2)}', (size == f'0-{int(R_C2)}'))
          if size != '1-0' and comp == '100-0':
              #print('...skip bimodal size or mono_large size for mono_small comp')
              continue
          if size == '1-0' and comp != '100-0':
              #print('---skip mono_small size for bimodal comp or mono_large comp')
              continue
          if size != f'0-{int(R_C2)}' and comp == '0-100':
              #print('++++skip bimodal size or mono_small size for mono_large comp')
              continue
          if size == f'0-{int(R_C2)}' and comp != '0-100':
              #print('////skip mono_large size for bimodal comp or mono_small comp')
              continue
          if size == f'0-{int(R_C2)}' and comp == '0-100':
              size = f'{int(R_C2)}-0'
              #print('reformat size for mono_large size and mono_large comp')
              if R_C2 != float(int(R_C2)):
                  print('ERROR: large monomodal data processing script cannot accept float particle size: {R_C2}')
                  continue
          #print(size, comp)
          for D0 in attraction_strengths:
              for scaling in depletion_scaling:
                  for kappa in attraction_range:
                    for cluster_style in clustering_style:
                      out_path = f'viz/{comp}/{comp}_clusters_{cluster_style}'
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

                      #gsd_in = f"{data_path}Gelation_Colloids.gsd"
                      gsd_in = f"{data_path}Gelation-lastframe.gsd"

                      particle_file = f"{analysis_path}GMM/clustered_df_standard.csv"
                      graph_file    = f"{analysis_path}GMM/lcc-cluster-network_G_clusters_standard.pkl"

                      out_prefix = "cluster_color"

                      print(f"r{size}, phi={phi}, {comp}, D0={D0}kT ({scaling}) kappa={kappa}, {cluster_style} clustering")
                      # ------------------------------------------------------------
                      # 1. LOAD PARTICLE CLUSTER DATA
                      # ------------------------------------------------------------

                      df = pd.read_csv(particle_file)
                      df["Cluster"] = df["Cluster"].astype(int)
                      df["Tag#"] = df["Tag#"].astype(int)

                      # ------------------------------------------------------------
                      # 2. LOAD CLUSTER GRAPH
                      # ------------------------------------------------------------

                      with open(graph_file, "rb") as f:
                          G = pickle.load(f)

                      print(" - Particles in CSV:", len(df))
                      print(" - Clusters:", df["Cluster"].nunique())
                      print(" - Graph nodes:", G.number_of_nodes())
                      print(" - Graph edges:", G.number_of_edges(),"\n")

                      # ------------------------------------------------------------
                      # 3. FIND NEAR-MINIMUM GRAPH COLORING
                      # ------------------------------------------------------------

                      strategies = [
                          "largest_first",
                          "smallest_last",
                          "independent_set",
                          "connected_sequential_bfs",
                          "connected_sequential_dfs",
                          "saturation_largest_first",
                      ]

                      best_coloring = None
                      best_n = 10**9
                      best_strategy = None

                      for strat in strategies:
                          try:
                              c = nx.coloring.greedy_color(G, strategy=strat)
                              n = max(c.values()) + 1 if len(c) else 0
                              print(f"   {strat:30s} -> {n} colors")

                              if n < best_n:
                                  best_n = n
                                  best_coloring = c
                                  best_strategy = strat

                          except Exception as e:
                              print("   skip", strat, e)

                      coloring = best_coloring

                      print("\n - BEST STRATEGY:", best_strategy)
                      print(" - COLORS USED:", best_n)

                      # ------------------------------------------------------------
                      # 4. MAP PARTICLE TAG -> COLOR GROUP
                      # ------------------------------------------------------------

                      df["ColorID"] = df["Cluster"].map(coloring).astype(int)

                      particle_groups = {}

                      for c in range(best_n):
                          inds = df.loc[df["ColorID"] == c, "Tag#"].values.astype(int)
                          particle_groups[c] = np.sort(inds)

                      for c in range(best_n):
                          print(f"   Color {c}: {len(particle_groups[c])} particles")

                      # ------------------------------------------------------------
                      # 5. OPEN INPUT GSD
                      # ------------------------------------------------------------

                      traj = gsd.hoomd.open(gsd_in, "r")
                      nframes = len(traj)
                      
                      print("\n - Frames in input:", nframes)

                      # ------------------------------------------------------------
                      # 6. WRITE ONE GSD PER COLOR
                      # ------------------------------------------------------------

                      # remove old outputs
                      for c in range(best_n):
                          fout = f"{out_path}/{out_prefix}_{c}.gsd"
                          if os.path.exists(fout):
                              os.remove(fout)

                      # loop over frames
                      for frame_i in range(nframes):

                          fr = traj[frame_i]

                          box = fr.configuration.box
                          types = list(fr.particles.types)

                          pos = fr.particles.position
                          vel = fr.particles.velocity
                          mass = fr.particles.mass
                          dia = fr.particles.diameter
                          tid = fr.particles.typeid

                          for c in range(best_n):

                              keep = particle_groups[c]

                              snap = gsd.hoomd.Frame()
                              snap.configuration.box = box

                              snap.particles.N = len(keep)
                              snap.particles.types = types
                              snap.particles.position = pos[keep]
                              snap.particles.velocity = vel[keep]
                              snap.particles.mass = mass[keep]
                              snap.particles.diameter = dia[keep]
                              snap.particles.typeid = tid[keep]

                              fout = f"{out_path}/{out_prefix}_{c}.gsd"

                              mode = "a"
                              if frame_i == 0:
                                  mode = "w"

                              with gsd.hoomd.open(name=fout, mode=mode) as f:
                                  f.append(snap)

                          print("   Finished frame", frame_i + 1, "/", nframes)

                      strategies.append(best_strategy)
                      numbers_of_colors.append(best_n)                      

# ------------------------------------------------------------
# 7. REPORT
# ------------------------------------------------------------

print("\n==============================")
print("DONE")
print("==============================")
print("Strategies:", strategies)
print("Colors used:", numbers_of_colors)

#for c in range(best_n):
#    print(f"Wrote {out_prefix}_{c}.gsd   ({len(particle_groups[c])} particles)")
