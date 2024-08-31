import numpy as np
import time
import sys
import os

from petsc4py import PETSc
from slepc4py import SLEPc

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *
from Atomic import *

comm = PETSc.COMM_WORLD
rank = comm.rank




def EVSolver(H, S, num_of_energies):
    E = SLEPc.EPS().create()
    E.setOperators(H, S)
    E.setDimensions(nev=num_of_energies)
    E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
    E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    E.setTolerances(1e-8, max_it=2500)
    E.solve()
    nconv = E.getConverged()
    return E, nconv

def solveEigensystem(simInstance, basisInstance, atomicInstance):
    lmax = simInstance.lm["lmax"]
    
    ViewTISE = PETSc.Viewer().createHDF5(f"{atomicInstance.pot_func.__name__}_Cont.h5", mode=PETSc.Viewer.Mode.WRITE, comm=comm)
    
    K = atomicInstance.createK(simInstance, basisInstance)
    
    for l in range(lmax + 1):
        V_l = atomicInstance.createV_l(simInstance, basisInstance, l)
        H_l = K + V_l
        
        num_of_energies = simInstance.splines["n_basis"]

        if rank == 0:
            print("Working on subsystem with l = ", l)
        E, nconv = EVSolver(H_l, atomicInstance.S, num_of_energies)
        if rank == 0:
            print(f"l = {l}, Requested: {num_of_energies}, Converged: {nconv}")

        for i in range(nconv):
            eigenvalue = E.getEigenvalue(i) 
            if np.real(eigenvalue) < 0:
                continue
            eigen_vector = H_l.getVecLeft()  
            E.getEigenvector(i, eigen_vector)  
                    
            Sv = atomicInstance.S.createVecRight()
            atomicInstance.S.mult(eigen_vector, Sv)
            norm = eigen_vector.dot(Sv)

            eigen_vector.scale(1/np.sqrt(norm))
            eigen_vector.setName(f"Psi_{l}_{np.real(eigenvalue)}")
            ViewTISE.view(eigen_vector)
        H_l.destroy()
        V_l.destroy()
        E.destroy()
    
    K.destroy()
    ViewTISE.destroy()

    return None

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
    print(f"Constructing Atomic Matrices")
    print("\n")
atomicInstance = atomic(simInstance)
atomicInstance.createS(simInstance,basisInstance)

solveEigensystem(simInstance,basisInstance,atomicInstance)


