import numpy as np
import time
import h5py
import sys
import os

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *

if rank == 0:
    if not os.path.exists("PES_files"):
        os.mkdir("PES_files")
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
if rank == 0:   
    sim_end = time.time()
    print(f"Finished Setting up in {round(sim_end-sim_start,6)} seconds")
    print("\n")

if True:  
    
    gamma,E_max = simInstance.E
    E_range = np.arange(0, E_max + 2 * gamma, 2 * gamma)
    np.save("PES_files/E.npy",E_range)

    n_basis = simInstance.splines["n_basis"]
    order = simInstance.splines["order"]
    lmax = simInstance.lm["lmax"]
    n_block = simInstance.n_block
    
    total_size = n_basis * n_block
    lm_dict,block_dict = simInstance.lm_dict,simInstance.block_dict

    def q_nk(k):
        return ((2 * k - 1) * np.pi) / 4


if True:  
    with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:, 0]
        imaginary_part = data[:, 1]
        wavefunction = real_part + 1j * imaginary_part
    psi_final = PETSc.Vec().createMPI(n_basis*n_block,comm = PETSc.COMM_WORLD)
    global_indices = np.array(range(total_size))
    global_indices = global_indices.astype("int32")
    psi_final.setValues(global_indices, wavefunction)
    psi_final.assemble()


if True:  
    H_0 = PETSc.Mat().createAIJ([len(wavefunction), len(wavefunction)], nnz=(2 * (order - 1) + 1), comm=comm)
    viewer = PETSc.Viewer().createBinary('TISE_files/H.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()

    S = PETSc.Mat().createAIJ([len(wavefunction), len(wavefunction)], nnz=(2 * (order - 1) + 1), comm=comm)
    viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
    S.load(viewer)
    viewer.destroy()





if rank == 0:
    start = time.time()

ksp_solver = PETSc.KSP().create(comm=comm)


exp1 = np.exp(1j * q_nk(1))
exp2 = np.exp(1j * q_nk(2))

PES_vals = []
intermediate_vector = psi_final.copy() 
result_vector = psi_final.copy()  

ViewPES = PETSc.Viewer().createHDF5("PES_files/PES_Vecs.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)


psi_final.scale(gamma**4)
for E in E_range:
    if rank == 0:
        print(E)
    A = H_0.copy()
    A.axpy(-E + gamma * exp1, S)
    ksp_solver.setOperators(A)
    S.mult(psi_final, result_vector) 
    ksp_solver.solve(result_vector, intermediate_vector)
    A.destroy()

    B = H_0.copy()
    B.axpy(-E - gamma * exp1, S)
    ksp_solver.setOperators(B)
    S.mult(intermediate_vector, result_vector)
    ksp_solver.solve(result_vector, intermediate_vector)
    B.destroy()

    C = H_0.copy()
    C.axpy(-E + gamma * exp2, S)
    ksp_solver.setOperators(C)
    S.mult(intermediate_vector, result_vector)
    ksp_solver.solve(result_vector, intermediate_vector)
    C.destroy()

    D = H_0.copy()
    D.axpy(-E - gamma * exp2, S)
    ksp_solver.setOperators(D)
    S.mult(intermediate_vector, result_vector)
    ksp_solver.solve(result_vector, intermediate_vector)  
    D.destroy()

    Sv = S.createVecRight()
    S.mult(intermediate_vector, Sv)
    PES_val = intermediate_vector.dot(Sv)
    PES_vals.append(PES_val)
    
    intermediate_vector.setName(f"{E}")
    ViewPES.view(intermediate_vector)


if rank == 0:
    np.save("PES_files/PES.npy",PES_vals)
    end = time.time()
    print("Cleaning up ...")
    print(f"Time to compute PES:{end-start}")
comm.barrier()
ViewPES.destroy()  
H_0.destroy()
S.destroy()
psi_final.destroy()
intermediate_vector.destroy()
result_vector.destroy()

