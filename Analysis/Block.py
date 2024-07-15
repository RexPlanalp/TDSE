import numpy as np
import time
import h5py
import sys
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_SELF
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

n_basis = simInstance.splines["n_basis"]
order = simInstance.splines["order"]
lmax = simInstance.lm["lmax"]
n_block = simInstance.n_block
pot = simInstance.box["pot"]
    
total_size = n_basis * n_block
lm_dict,block_dict = simInstance.lm_dict,simInstance.block_dict
if rank == 0:   
    sim_end = time.time()
    print(f"Finished Setting up in {round(sim_end-sim_start,6)} seconds")
    print("\n")

if True:
    S = PETSc.Mat().createAIJ([total_size, total_size], nnz=(2 * (order - 1) + 1), comm=comm)
    viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    Si, Sj, Sv = S.getValuesCSR()
    S.destroy()
    S_array = csr_matrix((Sv, Sj, Si))
    S_R= S_array[:n_basis, :n_basis]
    S.destroy()

if True:  
    with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:, 0]
        imaginary_part = data[:, 1]
        wavefunction = real_part + 1j * imaginary_part

lm_list = []

CONT = True


for (l,m),block_index in lm_dict.items():
    if l != m:
        continue
    print(f"Computing Probability of having l,m = {l,m}")
    wavefunction_block = wavefunction[block_index*n_basis:(block_index+1)*n_basis]

    if CONT:
        with h5py.File(f'TISE_files/{pot}.h5', 'r') as f:
            datasets = list(f.keys())
            for dataset_name in datasets:
                if dataset_name.startswith('Psi_'):
                    parts = dataset_name.split('_')
                    current_n = int(parts[1])
                    current_l = int(parts[2])
                    
                    if current_l == l:
                       
                        data = f[dataset_name][:]
                        real_part = data[:, 0]
                        imaginary_part = data[:, 1]
                        bound_state = real_part + 1j * imaginary_part

                    
                        inner_product = bound_state.conj().dot(S_R.dot(wavefunction_block))
                        wavefunction_block -= inner_product * bound_state
                        
                    
    probability = wavefunction_block.conj().dot(S_R.dot(wavefunction_block))
    lm_list.append(probability)


    

l_array = [l for l,m in lm_dict.keys() if l == m]

plt.bar(l_array,lm_list,color = "k")
plt.yscale('log')
if CONT:
    plt.title("Cont")
else:
    plt.title("Bound + Cont")
plt.savefig("images/blocks.png")

print(np.argmax(lm_list))

        
    