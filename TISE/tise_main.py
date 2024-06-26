import sys
import os
import time

os.environ['PETSC_VIEWER_STDOUT_WORLD'] = "/dev/null"

from TISE import *
sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *
from Atomic import *

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

EMBED = True
SOLVE = True

if rank == 0:
    if not os.path.exists("TISE_files"):
        os.mkdir("TISE_files")
    if not os.path.exists("temp"):
        os.mkdir("temp")


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
    atomic_start = time.time()
    print(f"Constructing Atomic Matrices: EMBED={EMBED}")
    print("\n")
atomicInstance = atomic(simInstance,basisInstance)
atomicInstance.createS(simInstance,basisInstance)



if EMBED:
    atomicInstance.embedS(simInstance)
   


if rank == 0:
    atomic_end = time.time()
    print(f"Finished Constructing Atomic Matrices in {round(atomic_end-atomic_start,6)} seconds")
    print("\n")


if rank == 0:
    TISE_start = time.time()
    print(f"Solving TISE:EMBED={EMBED}")
    print("\n")
tiseInstance = tise()
tiseInstance.solveEigensystem(simInstance,basisInstance,atomicInstance,EMBED)
if rank == 0:
    TISE_end = time.time()
    print(f"Finished Solving TISE in {TISE_end-TISE_start} seconds")
    print("\n")

if rank == 0:
    print("Cleaning up...")
    os.system("rm -rf temp")
    os.system(f"mv {atomicInstance.pot_func.__name__}.h5 TISE_files")




