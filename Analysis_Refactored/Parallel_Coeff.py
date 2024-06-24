# Additional imports
import time
import pickle
import numpy as np
import h5py
import sys

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append('/users/becker/dopl4670/Research/TDSE_refactored/Common')
from Sim import *
from Basis import *
from Atomic import *

simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 

basisInstance = basis()
basisInstance.createKnots(simInstance)

atomicInstance = atomic(simInstance,basisInstance)
atomicInstance.createS(simInstance,basisInstance)
S = atomicInstance.S



if True: 
    gamma,E_max = simInstance.E
    E_range = np.arange(0, E_max + 2 * gamma, 2 * gamma)

    n_basis = simInstance.splines["n_basis"]
    order = simInstance.splines["order"]
    lmax = simInstance.lm["lmax"]
    n_block = simInstance.n_block
    
    total_size = n_basis * n_block
    lm_dict,block_dict = simInstance.lm_dict,simInstance.block_dict

   

coeff_dict = {}
for l,m in lm_dict.keys():
    coeff_dict[(l,m)] = []

for E in E_range:
    print(E)
    with h5py.File(f'PES_files/PES_Vecs.h5', 'r') as f:
        data = f[f"{E}"][:]
        real_part = data[:, 0]
        imaginary_part = data[:, 1]
        wavefunction = real_part + 1j * imaginary_part
    state = PETSc.Vec().createMPI(n_basis*n_block,comm = PETSc.COMM_WORLD)
    global_indices = np.array(range(total_size))
    global_indices = global_indices.astype("int32")
    state.setValues(global_indices, wavefunction)
    state.assemble()

    for i in range(n_block):

        l,m = block_dict[i]
        IS = PETSc.IS().createGeneral(range(i * n_basis, (i + 1) * n_basis))
        subvec = state.getSubVector(IS)
        Sv = S.createVecRight()
        S.mult(subvec, Sv)
        probability = subvec.dot(Sv)
        coeff_dict[(l,m)].append(np.real(probability))
if rank == 0:
    with open("PES_files/coeff_dict.json", "wb") as fp:
        pickle.dump(coeff_dict , fp)
IS.destroy()  



