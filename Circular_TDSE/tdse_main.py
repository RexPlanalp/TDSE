import sys
import os
import time

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *
from Atomic import *

from Laser import *
from Psi import *
from Interaction import *
from Propagator import *

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

if rank == 0:
    if not os.path.exists("TDSE_files"):
        os.mkdir("TDSE_files")
    if not os.path.exists("temp"):
        os.mkdir("temp")
    if not os.path.exists("images"):
        os.mkdir("images")
comm.barrier()

if rank == 0:   
    sim_start = time.time()
    print("Setting up Simulation...")
    print("\n")
simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 
simInstance.printGrid()
if rank == 0:   
    sim_end = time.time()
    print(f"Finished Setting up in {round(sim_end-sim_start,6)} seconds")
    print("\n")

if rank == 0:
    basis_start = time.time()
    print("Creating Bspline Knot Vector")
    print("\n")
basisInstance = basis()
basisInstance.createKnots(simInstance)
if rank == 0:
    basis_end = time.time()
    print(f"Finished Constructing Knot Vector in {round(basis_end-basis_start,6)} seconds")
    print("\n")

if rank == 0:   
    laser_start = time.time()
    print("Defining Laser")
    print("\n")
laserInstance = laser()
laserInstance.createEnvelope(simInstance)
laserInstance.createCarrierX(simInstance)
laserInstance.createCarrierY(simInstance)
laserInstance.createAmplitude(simInstance)
laserInstance.createPulse(simInstance)
laserInstance.plotPulse(simInstance)
if rank == 0:
    laser_end = time.time()
    print(f"Finished defining laser in {round(laser_end-laser_start,6)} seconds")
    print("\n")



if rank == 0:
    state_start = time.time()
    print("Defining Initial State")
    print("\n")
psiInstance = psi(simInstance)
if rank == 0:
    state_end = time.time()
    print(f"Finished constructing initial state in {round(state_end-state_start,6)} seconds")
    print("\n")



if rank == 0:
    int_start = time.time()
    print("Constructing Interaction Matrices")
    print("\n")
interactionInstance = interaction()
interactionInstance.H_INV_R(simInstance,basisInstance)
interactionInstance.H_DER_R(simInstance,basisInstance)
interactionInstance.H_INT_1(simInstance)
interactionInstance.H_INT_2(simInstance)
if rank == 0:
    int_end = time.time()
    print(f"Finished constructing Interaction matrices in {round(int_end-int_start,6)} seconds")
    print("\n")



if rank == 0:
    prop_start = time.time()
    print("Starting wavefunction propagation")
propagatorInstance = propagator()
propagatorInstance.partialAtomic(simInstance)
propagatorInstance.partialInteraction(simInstance,interactionInstance)
propagatorInstance.propagateCN(simInstance,psiInstance,laserInstance)
if rank == 0:
    prop_end = time.time()
    print(f"Time to propagate:{round(prop_end-prop_start,6)} seconds")

if rank == 0:
    print("Cleaning up...")
    os.system("rm -rf temp")
    os.system("mv TDSE.h5 TDSE_files")


