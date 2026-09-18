## serial simulation to create *-BD.gsd from  *-DPD.gsd
## load position data (from DPD sim or experiment)
## NOTE: for a variable number of colloids and FIXED BOX SIZE
## NOTE: recreates all frames and does not scale the sim size
## (Rob Campbell)

######### Modules
# Use HOOMD-blue
import hoomd
import hoomd.md
import gsd.hoomd # read and write HOOMD schema GSD files
import numpy as np
import math
import random # psuedo-random number generator
import os # miscellaneous operating system interfaces
import sys
import re
import glob

#########  Simulation Inputs

scale_factor = 1 #how many times sim box is extended in each direction

# Input parameters
file_list = []
for filepath in glob.glob('Shear-*.gsd'):
    if not filepath.endswith('lastframe.gsd'):
        file_list.append(filepath)
if not file_list:
  print("ERROR: no GSD files found to convert to BD style")
  exit()
else:
  file_list.sort(key=os.path.getctime) # sort list by file creation time
  in_filepath = file_list[-1]

# convert in_filename into out_filename
sim_style = os.path.basename(in_filepath)
sim_style = os.path.splitext(sim_style)[0]
sim_style = re.sub(r'^Shear(_Colloids)?', '', sim_style)
sim_style = re.sub(r'(DPD|BD)$', '', sim_style)
sim_style = sim_style.strip('-')
if sim_style != "":
  out_file = 'Shear-DPDColloids'+sim_style+'-BD.gsd'
else:
  out_file = 'Shear-DPDColloids-BD.gsd'

### Check if sim already exists:
if os.path.exists(out_file):
  print("Shear-*-BD.gsd file already exists. No new files created.")
  exit()


### Extract DPD sim details 

# load data
traj = gsd.hoomd.open(in_filepath,'r')

if 'DPD' in in_filepath:
  colloid1_type = 1
  colloid2_type = 2
elif 'BD' in in_filepath:
  colloid1_type = 0
  colloid2_type = 1
else:
  colloid1_type = 1
  colloid2_type = 2

ref_frame_choice = 0

Lbox = traj[ref_frame_choice].configuration.box[:3]
V_total_DPD = Lbox[0] * Lbox[1] * Lbox[2]


# convert every frame of the simulation from DPD to BD
for frame_choice in range(len(traj)):

  #colloids = np.where(traj[frame_choice].particles.typeid == [colloid_type])    
  colloids = np.where((traj[frame_choice].particles.typeid == [colloid1_type]) | (traj[frame_choice].particles.typeid == [colloid2_type]))[0]
    
  #N_C1_DPD = len(colloid1)
  # N_C2_DPD = len(colloid2)
  N_C_DPD = len(colloids)

  typeid_DPD = traj[frame_choice].particles.typeid[colloids]
  mass_DPD = traj[frame_choice].particles.mass[colloids]
  diameter_DPD = traj[frame_choice].particles.diameter[colloids]
  velocity_DPD = traj[frame_choice].particles.velocity[colloids]
  position_DPD = traj[frame_choice].particles.position[colloids]
    
  #modify typeid so that minimum typeid is 0
  N_total_BD = N_C_DPD

  if colloid1_type != 0:
    typeid_BD = typeid_DPD - 1
  else:
    typeid_BD = typeid_DPD
  mass_BD = mass_DPD
  diameter_BD = diameter_DPD
  velocity_BD = velocity_DPD
  position_BD = position_DPD


  #########  Create empty large box    
  L_X_BD = Lbox[0]
  L_Y_BD = Lbox[1]
  L_Z_BD = Lbox[2] 

  # double confirm that data is formatted correctly    
  typeid = []
  typeid.extend([0]*(N_total_BD))
  mass = []
  mass.extend([0]*(N_total_BD))
  diameter = []
  diameter.extend([0]*(N_total_BD))
  velocity = np.zeros((N_total_BD,3))
  pos_arr = np.zeros((N_total_BD,3))
  
  j=0
  while j < N_total_BD:
      typeid[j] = typeid_BD[j]
      j += 1
  j=0
  while j < N_total_BD:
      mass[j] = mass_BD[j]
      j += 1
  j=0
  while j < N_total_BD:
      diameter[j] = diameter_BD[j]
      j += 1
  j=0
  while j < N_total_BD:
      velocity[j,0] = velocity_BD[j,0]
      velocity[j,1] = velocity_BD[j,1]
      velocity[j,2] = velocity_BD[j,2]
      j += 1


  j=0
  while j < N_total_BD:
      pos_arr[j,0] = position_BD[j,0]
      pos_arr[j,1] = position_BD[j,1]
      pos_arr[j,2] = position_BD[j,2]
      j += 1
      
  if frame_choice == 0:
      ######### Create snapshot
      snapshot = gsd.hoomd.Frame()
      snapshot.configuration.box = [L_X_BD, L_Y_BD, L_Z_BD, 0, 0, 0] 
      snapshot.particles.N = N_total_BD
      snapshot.particles.types = ['B','C']
      snapshot.particles.typeid = typeid
      snapshot.particles.mass = mass
      snapshot.particles.diameter = diameter
      snapshot.particles.position = pos_arr
      snapshot.particles.velocity = velocity
      with gsd.hoomd.open(name=out_file, mode='w') as f:
        f.append(snapshot)

  else:
      ######### Update GSD file
      snapshot = gsd.hoomd.Frame()
      snapshot.configuration.box = [L_X_BD, L_Y_BD, L_Z_BD, 0, 0, 0]
      snapshot.particles.N = N_total_BD 
      snapshot.particles.types = ['B','C']
      snapshot.particles.typeid = typeid
      snapshot.particles.mass = mass
      snapshot.particles.diameter = diameter
      snapshot.particles.position = pos_arr
      snapshot.particles.velocity = velocity
      with gsd.hoomd.open(name=out_file, mode='a') as f:
        f.append(snapshot)
