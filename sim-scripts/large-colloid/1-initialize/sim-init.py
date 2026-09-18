## serial simulation to create init.gsd
## number of colloids is calculated from a FIXED BOX SIZE
## (Rob Campbell)


######### MODULE LIBRARY
# Use HOOMD-blue
import hoomd
import hoomd.md # molecular dynamics
# Use GSD files
import gsd # provides Python API; MUST import sub packages explicitly (as below)
import gsd.hoomd # read and write HOOMD schema GSD files
# Maths
import numpy
import math
import random # psuedo-random number generator
# Other
import os # miscellaneous operating system interfaces


#########  SIMULATION INPUTS
# Density parameters
phi = 0.2 # volume fraction
percent_C1 = 1.0 # volume percent of type 1 colloid
percent_C2 = 0.0 # volume percent of type 2 colloid
rho = 3 # number density (per unit volume)

# Simulation box size (fixed by L_X)
L_X = 70
L_Y = L_X
L_Z = L_X 
V_total = L_X * L_Y * L_Z # total volume of simulation box (cube)

# Solvent particle details
m_S = 1 # solvent particle mass 
R_S = 0.5 # solvent particle radius 
V_Solvents = (1 - phi) * V_total # total volume of solvent
N_Solvents = math.floor(rho * V_Solvents) # total number of solvent particles

# Colloid particle details
R_C1 = 1  # 1st type colloid particle radius
V_C1 = (4./3.) * math.pi * R_C1 ** 3 # 1st type colloid particle volume (1 particle)
m_C1 = V_C1 * rho # 1st type colloid particle mass
V_Colloids_type1 = percent_C1 * phi * V_total # total volume of type 1 colloids
N_C1 = round(V_Colloids_type1 / V_C1) # number of 1st type of colloid particles (INT)

R_C2 = 2 # 2nd type colloid particle radius
V_C2 = (4./3.) * math.pi * R_C2 ** 3 # 2nd type colloid particle volume (1 particle)
m_C2 = V_C2 * rho # 2nd type colloid particle mass
V_Colloids_type2 = percent_C2 * phi * V_total # total volume of type 2 colloids
N_C2 = round(V_Colloids_type2 / V_C2) # number of 2nd type of colloid particles (INT)

# colloids totals NOTE: ASSUMES 2 colloid types
N_C = N_C1 + N_C2 # total number of colloids
V_Colloids = V_Colloids_type1 + V_Colloids_type2 # total volume of all colloids

# Total number of particles in the simulation
N_total = int(N_Solvents + N_C)


######### SIMULATION
## Checks for existing files. If none are found, creates a new 
## random distribution of particles for use as initial simulation state. 

if os.path.exists('init.gsd'):
	print("Initialization file already exists. No new files created.")
else:
	print("New initialization file is being created")
	## Initialize a snapshot of the system
	snapshot = gsd.hoomd.Snapshot()
	snapshot.configuration.box = [L_X, L_Y, L_Z, 0, 0, 0] # create the sim box
	snapshot.particles.N=N_total # add all particles to the snapshot
	print("Setting particle types")
	# set the particle types
	snapshot.particles.types = ['A','B','C']
	# assign particles to each type
	typeid = []
	typeid.extend([2]*N_C2)
	typeid.extend([1]*N_C1)
	typeid.extend([0]*N_Solvents)
	snapshot.particles.typeid = typeid
	print("Setting particle masses")
	# set a mass for each particle type
	mass = []
	mass.extend([m_C2]*N_C2)
	mass.extend([m_C1]*N_C1)
	mass.extend([m_S]*N_Solvents)
	snapshot.particles.mass = mass
	print("Setting particle diameters")
	# set a diameter for each particle type
	diameter = []
	diameter.extend([2.0*R_C2]*N_C2)
	diameter.extend([2.0*R_C1]*N_C1)
	diameter.extend([2.0*R_S]*N_Solvents)
	snapshot.particles.diameter = diameter
	print("Randomly distributing colloid particles")
	# randomly distribute all the particles in 3D space
	pos_arr = numpy.zeros((N_total,3))

	placed_particles = 0
	while placed_particles < N_C:
		curr_index = placed_particles
		if curr_index > 0:
			overlap = True
			while overlap == True:
				overlap_counter = 0
				#print("Placing particle " + str(curr_index+1))
				# place the next particle
				pos_arr[curr_index,0] = numpy.random.uniform(-0.5*L_X, 0.5*L_X)
				pos_arr[curr_index,1] = numpy.random.uniform(-0.5*L_Y, 0.5*L_Y)
				pos_arr[curr_index,2] = numpy.random.uniform(-0.5*L_Z, 0.5*L_Z)
				# then check for overlap with EACH particle that was already placed
				for i in range(0, placed_particles):
					# find the center-center interparticle distance vector
					dr = pos_arr[curr_index,0:3] - pos_arr[i,0:3]			
					# adjust distances to account for periodic boundaries	
					dr[0] = dr[0] - (L_X * numpy.rint(dr[0]/L_X))
					dr[1] = dr[1] - (L_Y * numpy.rint(dr[1]/L_Y))
					dr[2] = dr[2] - (L_Z * numpy.rint(dr[2]/L_Z))
					# update coordinates and convert to surface-surface distance
					h_ij = math.sqrt(sum(dr*dr)) - (diameter[i] + diameter[curr_index])/2	
					# check for overlap
					if h_ij <= 0:
						overlap_counter += 1
				if overlap_counter == 0:
						#print(h_ij)
						overlap = False
			placed_particles += 1
			
				
		else:
			#print("Placing first particle")
			# place the first particle
			pos_arr[curr_index,0] = numpy.random.uniform(-0.5*L_X, 0.5*L_X)
			pos_arr[curr_index,1] = numpy.random.uniform(-0.5*L_Y, 0.5*L_Y)
			pos_arr[curr_index,2] = numpy.random.uniform(-0.5*L_Z, 0.5*L_Z)
			placed_particles += 1

	print(str(N_C) + " colloid particles have been placed without overlaps")
	print("Randomly distributing solvent particles")
	while placed_particles < N_total:
		curr_index = placed_particles
		if curr_index > 0:
			overlap = True
			while overlap == True:
				#print("Placing particle " + str(curr_index+1))
				# place the next particle
				pos_arr[curr_index,0] = numpy.random.uniform(-0.5*L_X, 0.5*L_X)
				pos_arr[curr_index,1] = numpy.random.uniform(-0.5*L_Y, 0.5*L_Y)
				pos_arr[curr_index,2] = numpy.random.uniform(-0.5*L_Z, 0.5*L_Z)
				# then check for overlap with EACH colloid particle that was already placed
				for i in range(0, N_C):
					# find the center-center interparticle distance vector
					dr = pos_arr[curr_index,0:3] - pos_arr[i,0:3]			
					# adjust distances to account for periodic boundaries	
					dr[0] = dr[0] - (L_X * numpy.rint(dr[0]/L_X))
					dr[1] = dr[1] - (L_Y * numpy.rint(dr[1]/L_Y))
					dr[2] = dr[2] - (L_Z * numpy.rint(dr[2]/L_Z))
					# update coordinates and convert to surface-surface distance
					h_ij = math.sqrt(sum(dr*dr)) - diameter[i]/2	
					# check for overlap
					if h_ij <= 0:
						overlap = True
					else:
						overlap = False
			placed_particles += 1
	print(str(N_Solvents) + " solvent particles have been placed without colloid-solvent overlaps")


	snapshot.particles.position = pos_arr
	# save the snapshot of the initialized system
	with gsd.hoomd.open(name='init.gsd', mode='wb') as f:
       		f.append(snapshot)

	print("New initialization file (init.gsd) created.\n")
	print("Simulation volume: L_X = " + str(L_X) + ", L_Y = " + str(L_Y) + ", L_Z = " + str(L_Z))
	print("Volume fraction: " + str(phi) + "; percent colloid 1: " + str(percent_C1) +
		", percent colloid 2: " + str(percent_C2) + "\n")
	print("Total number of particles: " + str(N_total))
	print("Total number of solvent particles: " + str(N_Solvents))
	print("Total number of colloid particles: " + str(N_C) + 
		"; N_C1 = " + str(N_C1) + "; N_C2 = " + str(N_C2))
