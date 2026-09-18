## MPI simulation to shear a DPD attractive colloidal gel
## NOTE: requires matching GSD gelation file
##       for a fixed number of colloids and FIXED BOX SIZE 
## (Rob Campbell)

######### MODULE LIBRARY 
# Use HOOMD-blue
import hoomd

# Use GSD files
import gsd.hoomd # read and write HOOMD schema GSD files

# Maths
import numpy as np
import math
import random # pseudo-random number generator

# for tracking MPI processes
##pip install mpi4py
from mpi4py import MPI 
# track MPI rank and comm.size (1 if serial)
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if comm.size > 1:
  print("MPI rank: "+str(rank))


######### SIMULATION INPUTS
# General parameters
rho = 3.0    # number density (per unit volume)
kT = 0.1     # system temperature
D0 = 12 * kT # attraction strength (gels at >=4kT)
D0_BB = D0
D0_BC = 1.5*D0
D0_CC = 2*D0
kappa = 10   # range of attraction (4 (long range)- 30 (short range))
             # distance in DPD units is approx 3/kappa

# Colloid particle details
R_C1 = 1     # 1st type colloid particle radius
R_C2 = 2     # 2nd type colloid particle radius

# use higher viscosity to see shear behavior better
eta0 = 1.1   # background viscosity
gamma = 45   # DPD controlling parameter for viscous resistance (dissipative force)

# Particle interaction parameters
r_c = 1.0    # cut-off radius parameter, r_c>=3/kappa (r_cut = # * r_c) 
if rank == 0 or comm.size == 1:
  if r_c < (3/kappa):
    print('WARNING: r_c is less than range of attraction. Increase r_c')
r0 = 0.0             # minimum inter-particle distance
f_contact = 10000.0  # magnitude of contact force
bond_calc = True     # activate/deactive bond form/break tracking 

# modified center-center cut-off radius for solvent-colloid interactions
r_sc1_cut = (r_c**3 + R_C1**3) ** (1/3)
r_sc2_cut = (r_c**3 + R_C2**3) ** (1/3)

# shear flow details
shear_style = 'const' # 'const' for constant or 'cosin' for cosinusoid

n_strains = 1 # number of strains or oscillation cycles
              # ex: 10 for constant strain, 5 for oscillatory, etc.
frames_per_strain = 1000               # how many frames you want in each cycle
n_frames = frames_per_strain*n_strains # total simulation frames

init_velocity = True   # initialize with or without a linear velocity profile

# box size
#L_X = 70 # box size in flow direction
L_Y = 70 # box size in gradient direction
#L_Z = 70 # box size in vorticity direction

# NOTE: reduce timestep 0.001->0.000001 as SR increases 0.001->1.0
dt_Integration = 1e-4 # DPD timestep

# Constant
if shear_style == 'const':
  shear_rate_const = 0.5
  shear_rate = shear_rate_const
  vinf = shear_rate * L_Y   # dimensional param for the maximum flow velocity
  theta = 1.0               # desired xy tilt factor (0.0 to 1.0)
  t_ramp = theta/shear_rate # length of each shear pass in BD-time

  file_tag = shear_style+'-SR'+str(shear_rate)+'-'+str(n_strains)+'strains'

else:
  Amp = 0.0    # ex: 0.005 0.01 0.05 0.1
  omega = 0.0  #2*np.pi/t_ramp # frequency (1/DPD-time)
  omega_timesteps = omega*dt_Integration # frequency (1/N_timesteps)
  vinf = Amp * omega * L_Y # dimensional param for the maximum flow velocity
  shear_rate = Amp * omega
  t_ramp = Amp/shear_rate  # length of each strain in BD-time

  file_tag = shear_style+'-SR'+str(round(shear_rate,2))+"-A"+str(round(Amp,3))+"-w"+str(round(omega,1))+'-'+str(n_strains)+'strains'

Iter_cycle = int(t_ramp / dt_Integration)  # length of each shear pass (timesteps) 
period= int(Iter_cycle/frames_per_strain)  # recording interval (ex: 20 frames in theta strains or omega sweeps)


# set the random seed for reproducibility
seed_value = 42


######### SIMULATION
# Checks for existing shear flow files. If none exist, begins shearing 
# from the gelation state
if rank == 0 or comm.size == 1:
  if os.path.exists('Shear-'+file_tag+'-bi-DPD.gsd'):
    print('Shear flow file already exists. No new files created.')
    exit()
  else:
    print('Shearing the bimodal DPD simulation with '+shear_style+' shear')
    print(" - shear rate:",shear_rate)
    print(" - n_strains:",n_strains)
    print(" - dt_Integration:",dt_Integration)
    print(" - DPD times per strain:",t_ramp)
    print(" - timesteps per strain:",Iter_cycle)
    print(" - run time:",n_strains*Iter_cycle,"timesteps")

## Create a CPU simulation
device = hoomd.device.CPU()
sim = hoomd.Simulation(device=device, seed=seed_value)

# reset timestep to zero and start simulation from the GSD file
sim.timestep=0
sim.create_state_from_gsd(filename='../potential-morse-bysize/Gelation-lastframe.gsd')

# assign particle types to groups
# (in case we want to integrate over subpopulations only,
# but would require other mods to source code)
groupA = hoomd.filter.Type(['A'])
groupB = hoomd.filter.Type(['B'])
groupC = hoomd.filter.Type(['C'])
all_colloids = hoomd.filter.Type(['B','C'])
all_ = hoomd.filter.Type(['A','B','C'])

# DON'T thermalize the system (shearing is not at thermal equilibrium!)
#sim.state.thermalize_particle_momenta(filter=groupA, kT=kT)

# [optional] apply initial velocity field
if init_velocity == True:
  # apply an initial velocity field to all solvent particles
  snap = sim.state.get_snapshot()
  if(snap.communicator.rank == 0):    # if this is the start of the simulation
    for i in range(snap.particles.N): # for all particles
        # ALL PARTICLES ARE INITIALIZED
        #if(snap.particles.typeid[i]==0): # if that particle is a solvent particle
        v_x = snap.particles.position[i,1]*shear_rate # apply a linear velocity profile
        snap.particles.velocity[i][0] += v_x
  # apply the linear velocity profile to the initial simulation state
  sim.state.set_snapshot(snap)
  file_tag = file_tag+"-vinit"+str(shear_rate)


# set shear style Variant class
# Constant | value
if shear_style == 'const':
  flip = True
  shear_style_vinf = hoomd.variant.Constant(vinf)

else:
  flip = False
  # Cosinusoid | value, t_start, omega
  if shear_style == 'cosinusoid' :
    shear_style_vinf = hoomd.variant.Cosinusoid(vinf, sim.timestep, omega_timesteps)


# create neighboring list
nl = hoomd.md.nlist.Tree(buffer=0.05);

# define Morse force (attraction) interactions
morse = hoomd.md.pair.DPDMorse(nlist=nl, kT=kT, default_r_cut=1.0 * r_c, period=period, bond_calc=bond_calc)

# solvent-solvent: soft particles (allow deformation/overlap)
morse.params[('A','A')] = dict(A0=25.0 * kT / r_c, gamma=gamma,
  D0=0, alpha=kappa, r0=r0, eta=0.0, f_contact=0.0,
  a1=0.0, a2=0.0, rcut=r_c) # force calc
morse.r_cut[('A','A')] = r_c # used to assemble nl

# solvent-colloid: soft particles (allow deformation/overlap)
morse.params[('A','B')] = dict(A0=25.0 * kT / r_sc1_cut, gamma=gamma,
  D0=0, alpha=kappa, r0=r0, eta=0.0, f_contact=0.0, a1=0.0, a2=R_C1,
  rcut=r_sc1_cut - (0 + R_C1)) # force calc
morse.r_cut[('A','B')] = r_sc1_cut # used to assemble nl

# solvent-colloid: soft particles (allow deformation/overlap)
morse.params[('A','C')] = dict(A0=25.0 * kT / r_sc2_cut, gamma=gamma,
  D0=0, alpha=kappa, r0=r0, eta=0.0, f_contact=0.0, a1=0.0, a2=R_C2,
  rcut=r_sc2_cut - (0 + R_C2)) # force calc
morse.r_cut[('A','C')] = r_sc2_cut # used to assemble nl

# colloid-colloid: hard particles (no deformation/overlap)
morse.params[('B','B')] = dict(A0=0.0, gamma=gamma,
  D0=D0_BB, alpha=kappa, r0=r0, eta=eta0, f_contact=f_contact,
  a1=R_C1, a2=R_C1, rcut=r_c) # force calc
morse.r_cut[('B','B')] = (r_c + 2.0 * R_C1) # used to assemble nl

# colloid-colloid: hard particles (no deformation/overlap)
morse.params[('B','C')] = dict(A0=0.0, gamma=gamma,
  D0=D0_BC, alpha=kappa, r0=r0, eta=eta0, f_contact=f_contact,
  a1=R_C1, a2=R_C2, rcut=r_c) # force calc
morse.r_cut[('B','C')] = (r_c + R_C1 + R_C2) # used to assemble nl

# colloid-colloid: hard particles (no deformation/overlap)
morse.params[('C','C')] = dict(A0=0.0, gamma=gamma,
  D0=D0_CC, alpha=kappa, r0=r0, eta=eta0, f_contact=f_contact,
  a1=R_C2, a2=R_C2, rcut=r_c) # force calc
morse.r_cut[('C','C')] = (r_c + 2.0 * R_C2) # used to assemble nl

# choose integration method for the end of each timestep
nve = hoomd.md.methods.ConstantVolume(filter=all_, thermostat=False)
integrator=hoomd.md.Integrator(dt=dt_Integration, vinf=shear_style_vinf, forces=[morse], methods=[nve])
sim.operations.integrator = integrator

# set the simulation to log certain values
logger = hoomd.logging.Logger()
thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=all_)
sim.operations.computes.append(thermodynamic_properties)
logger.add(thermodynamic_properties,quantities=['kinetic_temperature','pressure_tensor','virial_ind_tensor','potential_energy'])
logger.add(sim,quantities=['tps'])

# set output file
gsd_writer = hoomd.write.GSD(trigger=period, filename='Shear-'+file_tag+'-bi-DPD.gsd', filter=all_, mode='wb', dynamic=['property','momentum','attribute'])
gsd_writer.write_diameter = True
sim.operations.writers.append(gsd_writer)
gsd_writer.logger = logger

# shear the system
# set the box resize style
box_resize=hoomd.update.BoxShear(trigger=1, vinf=shear_style_vinf, deltaT=dt_Integration, flip=flip)
sim.operations += box_resize

# run the simulation!
sim.run(Iter_cycle*n_strains, write_at_start=True)

if rank == 0 or comm.size == 1:	
  print('\nNew bimodal '+shear_style+' shear DPD state (Shear-'+file_tag+'-bi-DPD.gsd) created.')
  print("("+str(n_frames)+" frames)")

  # record sim parameters
  print("\nR_C1:", R_C1)
  print("R_C2:", R_C2)
  print("seed_value:", seed_value)
  print("period:", period)
  print("D0/kT:", round(D0/kT))
  print("kappa:", kappa, "\n")
  print("L_Y:", L_Y)
  print("shear_style:", shear_style)
  if shear_style == "const":
    print("vinf:", vinf)
    print("shear_rate:", shear_rate)
    print(str(n_strains)+" strains")
  else: 
    print("amplitude:", A)
    print("frequency:", omega)
    print("t_ramp:", t_ramp)
    print("vinf:", vinf)
    print("shear_rate:", shear_rate)
    print(str(n_strains)+" cycles")
