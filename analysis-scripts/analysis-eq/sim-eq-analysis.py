## Analyze the results of a DPD colloid equilibrium simulation
## NOTE: this code assumes 1 solvent type (typeid=0) and 2 colloid type (typeid=1)
##
## Extracts temperature and pressure (-stress) data from a GSD file, 
##
## (Rob Campbell)


######### MODULE LIBRARY
import numpy as np
import gsd.hoomd
import math

######### INPUT PARAMETERS
filepath = '../Equilibrium.gsd'

## general simulation parameters
period = 10000 # from simulation
dt_Integration = 0.001 # from simulation
t1 = period * dt_Integration # timestep conversion factor


######### DEFINE SIM CHECKS
## extract thermodynamic properties (temperature and pressure (AKA -stress) components)
def extract_properties_py(filename):
	# open the simulation GSD file as "traj" (trajectory)
	traj = gsd.hoomd.open(filename, 'rb')
	# get the number of frames
	nframe = len(traj)
	
	# create a file to save the thermodynamic properties
	f=open('gsd-properties.txt', 'w')
	f.write('DPD-time Virial-Pressure Vr_CONS Vr_DISS Vr_RAND Vr_SQUE Vr_CONT PE kT tps\n')

	# for each frame
	for i in range(0, nframe):
		DPDtime = (i+1)*t1
		# extract the total "force virial contribution" of the pressure tensor
		Pr=float(traj[i].log['md/compute/ThermodynamicQuantities/pressure_tensor'][1])
		# extract the decomposed virial compoenents of the pressure tensor
		Vr=np.asarray(traj[i].log['md/compute/ThermodynamicQuantities/virial_ind_tensor'],dtype=float)
		# extrac the potential energy and scale to kilo-units (1/1000)
		Pe=float(traj[i].log['md/compute/ThermodynamicQuantities/potential_energy'][0])*0.001
		# extract the kinetic temperature (kT)
		KT=float(traj[i].log['md/compute/ThermodynamicQuantities/kinetic_temperature'][0])
		# extract the transactios per secont (tps) for assessing speed/efficiency of the simulation
		tps=int(traj[i].log['Simulation/tps'][0])

		# write these values to the output file:
		# raw values
		f.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9}\n'.format(DPDtime, Pr, 
			Vr[0], Vr[1], Vr[2], Vr[3], Vr[4], Pe, KT, tps))
		
		# rounded values
		#f.write('%f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %d %0.2f %d\n'%((i+1)*t1, Pr, 
		#	Vr[0], Vr[1], Vr[2], Vr[3], Vr[4], Pe, KT, tps))


######### RUN CHECKS ON A SIMULATION

if __name__ == '__main__':	
	extract_properties_py(filepath)
