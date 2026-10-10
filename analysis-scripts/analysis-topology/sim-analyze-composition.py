# Basic stats on each bimodal/monomodal composition 


########################
""" MODULE LIBRARY """
########################
import numpy as np
import pandas as pd

import os
import re
import glob


########################
""" DATA HANDLING """
########################

size_ratios = ['1-2'] 
volume_fractions = [20] 
small_comps = [100, 50, 0] 
compositions = ['100-0', '50-50', '0-100'] 
attraction_strengths = ['12'] 
depletion_scaling = ['bysize'] 
attraction_range = ['60'] 

eta0 = 0.3
kT = 0.1
L = 70
system_volume = L**3

data_outpath = 'data'

system_dfs = []
for size in size_ratios:
  radii = size.split('-')
  R_C1 = float(radii[0])
  R_C2 = float(radii[1])
  for phi in volume_fractions:
      for comp_1 in small_comps:
          comp_2 = 100 - comp_1
          comp = f'{comp_1}-{comp_2}'
          # skip invalid fake mixtures for monodisperse case
          #print(size, comp, R_C1, R_C2, f'0-{int(R_C2)}', (size == f'0-{int(R_C2)}'))
          if comp == '100-0':
              R_C2 = 0.0
              size_tag = f'1-0' 
          elif comp == '0-100':
              R_C1 = 0.0
              size_tag = f'{int(R_C2)}-0'
          else:
              R_C1 = float(radii[0])
              R_C2 = float(radii[1])
              size_tag = size
          for D0 in attraction_strengths:
            for scaling in depletion_scaling:
               for kappa in attraction_range:
                  if scaling == 'uniform':
                          if (comp == '100-0'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif (comp == '0-100'):
                             print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                             continue
                          elif (comp != '100-0') and (comp != '0-100'):
                             data_path = (f'/projects/props/Rob/colloids/bimodal/uniform-potential/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse/'
                                f'{D0}kT/kappa{kappa}/analysis-bi-colloids/data/')
                  else:
                          if (comp == '100-0'):
                             if scaling == 'bysize':
                                data_path = (f'/projects/props/Rob/colloids/bimodal/r{size_tag}/DPD/L{L}/mono/seedNone/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/analysis-DPD/data/')
                                print(data_path)
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (comp == '0-100'):
                             if scaling == 'bysize':
                                data_path = (f'/projects/props/Rob/colloids/bimodal/r{size_tag}/DPD/L{L}/phi{phi}'
                                   f'/potential-morse-bysize/{D0}kT/kappa{kappa}/analysis-DPD/data/')
                             else:
                                print(f'WARNING: monodisperse data only recognizes scaling "bysize", not {scaling}')
                                continue
                          elif (comp != '100-0') and (comp != '0-100'):
                             data_path = (f'/projects/props/Rob/colloids/bimodal/r{size}'
                                f'/DPD/L70/poly0.0/seedNone/phi{phi}/{comp}/potential-morse-{scaling}/'
                                f'{D0}kT/kappa{kappa}/analysis-bi-colloids/data/')
                          else:
                             continue

                  if os.path.exists(data_path) == False:
                      continue

                  Zavg_df = pd.read_csv(data_path+'Zavg.csv')
                  netx_df = pd.read_csv(data_path+'networkx-allframes.csv')

                  # ONLY THE LAST FRAME
                  Zavg_frames = pd.unique(Zavg_df['simframe'])
                  netx_frames = pd.unique(netx_df['frame'])

                  Zavg = Zavg_df.at[max(Zavg_frames), 'Z_any']
                  ncolloids_all = netx_df.at[max(netx_frames), 'ncolloids']

                  # Fraction of particles in the LCC
                  lcc_all = netx_df.at[max(netx_frames), 'lcc_size'] / ncolloids_all
                
                  if (comp == '100-0'):
                     # --------------
                     # Bonds
                     # --------------
                     Zcount_df = pd.read_csv(data_path+'Z-counts.csv')
                     Zcount_frames = pd.unique(Zcount_df['frame'])
                     Zcount_lastframe_df = Zcount_df.loc[Zcount_df['frame'] == max(Zcount_frames)] 
                     Zany_count = Zcount_lastframe_df['Z'].to_numpy()

                     # CONTACT NUMBERS
                     Zavg_1any = Zavg
                     Zavg_11 = Zavg
                     Zavg_12 = 0
                     Zavg_2any = 0
                     Zavg_21 = 0
                     Zavg_22 = 0
                     # NUMBER OF COLLOIDS
                     ncolloids_1 = ncolloids_all
                     ncolloids_2 = 0
                     # NUMBER RATIO
                     # NOTE: this measure is not used in the present study
                     number_ratio_c1 = 1.0
                     number_ratio_c2 = 0
                     Zavg_1any_weighted = Zavg_1any
                     Zavg_2any_weighted = 0
                     # LCC
                     # NOTE: this measure is not used in the present study
                     lcc_c1 = lcc_all
                     lcc_c2 = 0
                     # SURFACE AREA FRACTION
                     # NOTE: this measure is not used in the present study
                     surf_frac_1 = 1.0 
                     surf_frac_2 = 0.0
                     # BOND FRACTION
                     b_11 = sum(Zany_count)
                     b_12 = 0
                     b_21 = 0
                     b_22 = 0
                     b_all = sum(Zany_count) 
                     bond_frac_11 = 1.0
                     bond_frac_12 = 0
                     bond_frac_22 = 0
                     bond_frac_1any = 1.0
                     bond_frac_2any = 0

                  elif (comp == '0-100'):
                     # --------------
                     # Bonds
                     # --------------
                     Zcount_df = pd.read_csv(data_path+'Z-counts.csv')
                     Zcount_frames = pd.unique(Zcount_df['frame'])
                     Zcount_lastframe_df = Zcount_df.loc[Zcount_df['frame'] == max(Zcount_frames)] 
                     Zany_count = Zcount_lastframe_df['Z'].to_numpy()

                     # CONTACT NUMBERS
                     Zavg_1any = 0
                     Zavg_11 = 0
                     Zavg_12 = 0
                     Zavg_2any = Zavg
                     Zavg_21 = 0
                     Zavg_22 = Zavg
                     # NUMBER OF COLLOIDS
                     ncolloids_1 = 0
                     ncolloids_2 = ncolloids_all
                     # PARTICLE NUMBER RATIO
                     # NOTE: this measure is not used in the present study
                     number_ratio_c1 = 0
                     number_ratio_c2 = 1.0
                     # LCC
                     # NOTE: this measure is not used in the present study
                     lcc_c1 = 0
                     lcc_c2 = lcc_all
                     # SURFACE AREA FRACTION
                     # NOTE: this measure is not used in the present study
                     surf_frac_1 = 1.0
                     surf_frac_2 = 0.0
                     # BOND FRACTION
                     b_11 = 0
                     b_12 = 0
                     b_21 = 0
                     b_22 = sum(Zany_count)
                     b_all = sum(Zany_count) 
                     bond_frac_11 = 0.0
                     bond_frac_12 = 0.0
                     bond_frac_22 = 1.0
                     bond_frac_1any = 0.0
                     bond_frac_2any = 1.0

                  elif (comp != '100-0') and (comp != '0-100'):
                    netx_c1_df = pd.read_csv(data_path+'networkx-allframes-c1only.csv')
                    netx_c2_df = pd.read_csv(data_path+'networkx-allframes-c2only.csv')


                    # ONLY THE LAST FRAME
                    netx_c1_frames = pd.unique(netx_c1_df['frame'])
                    netx_c2_frames = pd.unique(netx_c2_df['frame'])

                    # CONTACT NUMBERS
                    Zavg_1any = Zavg_df.at[max(Zavg_frames), 'Z_1any']
                    Zavg_11 = Zavg_df.at[max(Zavg_frames), 'Z_11']
                    Zavg_12 = Zavg_df.at[max(Zavg_frames), 'Z_12']
                    Zavg_2any = Zavg_df.at[max(Zavg_frames), 'Z_2any']
                    Zavg_21 = Zavg_df.at[max(Zavg_frames), 'Z_21']
                    Zavg_22 = Zavg_df.at[max(Zavg_frames), 'Z_22']

                    ncolloids_1 = netx_c1_df.at[max(netx_c1_frames), 'ncolloids']
                    ncolloids_2 = netx_c2_df.at[max(netx_c2_frames), 'ncolloids']                
                    if ncolloids_1 + ncolloids_2 != ncolloids_all:
                        print(f'ERROR: population counts do not sum to all colloids:\n'
                              f'c1={ncolloids_1}, c2={ncolloids_2}, all={ncolloids_all}')
                        exit()


                    # --------------
                    # PARTICLE NUMBER RATIO
                    # NOTE: this measure is not used in the present study
                    # --------------
                    number_ratio_c1 = ncolloids_1 / ncolloids_all 
                    number_ratio_c2 = ncolloids_2 / ncolloids_all 

                    # --------------
                    # LCC
                    # NOTE: this measure is not used in the present study 
                    # --------------
                    lcc_c1 = netx_c1_df.at[max(netx_c1_frames), 'lcc_size'] / ncolloids_1
                    lcc_c2 = netx_c2_df.at[max(netx_c2_frames), 'lcc_size'] / ncolloids_2

                    # --------------
                    # SURFACE AREA FRACTION ~ A_pop/A_total
                    # NOTE: this measure is not used in the present study
                    # --------------
                    phi_1 = (comp_1/100) * (phi/100)
                    phi_2 = (comp_2/100) * (phi/100)

                    N = ncolloids_all / system_volume
                    N_1 = ncolloids_1 / system_volume
                    N_2 = ncolloids_2 / system_volume

                    A_1 = ncolloids_1 * 4*np.pi*R_C1**2
                    A_2 = ncolloids_2 * 4*np.pi*R_C2**2
                    surf_frac_1 = A_1 / (A_1 + A_2)
                    surf_frac_2 = A_2 / (A_1 + A_2)
                    if round(surf_frac_2,3) != round(1-surf_frac_1,3):
                        print(f"ERROR: Surface area fractions do not sum to 1:")
                        print(f"       f_C1={surf_frac_1_rand}, f_C2={surf_frac_2_rand}")

                    # --------------
                    # Fraction of bonds of different types
                    # --------------
                    Zcount_df = pd.read_csv(data_path+'Z-counts.csv')
                    Zcount_frames = pd.unique(Zcount_df['frame'])
                    Zcount_lastframe_df = Zcount_df.loc[Zcount_df['frame'] == max(Zcount_frames)]
                    Z11_count = Zcount_lastframe_df['Z_11'].to_numpy()
                    Z12_count = Zcount_lastframe_df['Z_12'].to_numpy()
                    Z21_count = Zcount_lastframe_df['Z_21'].to_numpy()
                    Z22_count = Zcount_lastframe_df['Z_22'].to_numpy()
                    Zany_count = Zcount_lastframe_df['Z_any'].to_numpy()
                    b_11 = sum(Z11_count) # number of 1-1 bonds
                    b_12 = sum(Z12_count) # number of 1-2 bonds
                    b_21 = sum(Z21_count) # number of 1-2 bonds
                    b_22 = sum(Z22_count) # number of 2-2 bonds
                    b_all = sum(Zany_count)-b_21 # all bonds
                    if b_12 != b_21:
                        print(f"ERROR: Z_12 does not equal Z_21: sum(Z_12)={b_12}, sum(Z_21)={b_21}")
                    if b_all != (b_11+b_12+b_22):
                        print(f"ERROR: Z counts do not sum:")
                        print(f"       Z_11={b_11}, Z_12={b_12}, Z_21={b_21}, Z_22={b_22}, Z_any={b_all}")
                    bond_frac_11 = b_11 / b_all
                    bond_frac_12 = b_12 / b_all
                    bond_frac_22 = b_22 / b_all
                    bond_frac_1any = (b_11+b_12) / b_all
                    bond_frac_2any = (b_12+b_22) / b_all

                  # compile outputs
                  print('---')
                  print(comp)
                  print('---')
                  results_dict = {'size-ratio'          :[size_tag],
                                  'phi'                 :[phi],
                                  'composition'         :[comp],
                                  'D0'                  :[D0],
                                  'scaled_D0'           :[scaling],
                                  'kappa'               :[kappa],
                                  # CONTACT NUMBERS
                                  'Zavg_all'            :[Zavg],
                                  'Zavg_1any'           :[Zavg_1any],
                                  'Zavg_11'             :[Zavg_11],
                                  'Zavg_12'             :[Zavg_12],
                                  'Zavg_2any'           :[Zavg_2any],
                                  'Zavg_21'             :[Zavg_21],
                                  'Zavg_22'             :[Zavg_22],
                                  # NUMBER OF COLLOIDS
                                  'ncolloids_all'       :[ncolloids_all],
                                  'ncolloids_1'         :[ncolloids_1],
                                  'ncolloids_2'         :[ncolloids_2],
                                  # NUMBER RATIO
                                  'number_ratio_c1'     :[number_ratio_c1],
                                  'number_ratio_c2'     :[number_ratio_c2],
                                  'Zavg_1any_weighted'  :[Zavg_1any_weighted],
                                  'Zavg_2any_weighted'  :[Zavg_2any_weighted],
                                  # LCC
                                  'lcc_all'             :[lcc_all],
                                  'lcc_c1'              :[lcc_c1],
                                  'lcc_c2'              :[lcc_c2],
                                  # SURFACE AREA FRACTION
                                  'surf_frac_1'         :[surf_frac_1],
                                  'surf_frac_2'         :[surf_frac_2],
                                  # BOND FRACTION
                                  'b_11'                :[b_11],
                                  'b_12'                :[b_12],
                                  'b_21'                :[b_21],
                                  'b_22'                :[b_22],
                                  'b_all'               :[b_all],
                                  'bond_frac_11'        :[bond_frac_11],
                                  'bond_frac_12'        :[bond_frac_12],
                                  'bond_frac_22'        :[bond_frac_22],
                                  'bond_frac_1any'      :[bond_frac_1any],
                                  'bond_frac_2any'      :[bond_frac_2any],
                                  }

                  res_df = pd.DataFrame(results_dict)
                  system_dfs.append(res_df) 

all_results_df = pd.concat(system_dfs)
all_results_df.to_csv(f'{data_outpath}/composition-and-preference.csv',index = False)

