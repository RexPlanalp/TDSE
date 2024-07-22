import numpy as np
import time
import sys
from scipy.special import sph_harm
import h5py

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
 
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
    SLICE = simInstance.SLICE
    if SLICE == "XZ":
        phi_range = np.array([0,np.pi])
        theta_range = np.arange(0,np.pi+0.01,0.01)
    elif SLICE == "XY":
        phi_range = np.arange(0,2*np.pi+0.01,0.01)
        theta_range = np.array([np.pi/2])
    elif SLICE == "YZ":
        phi_range = np.array([np.pi/2,3*np.pi/2])
        theta_range = np.arange(0,np.pi+0.01,0.01)

if True:
        S = PETSc.Mat().createAIJ([total_size, total_size], nnz=(2 * (order - 1) + 1), comm=MPI.COMM_SELF)
        viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
        S.load(viewer)
        viewer.destroy()

        from scipy.sparse import csr_matrix
        Si, Sj, Sv = S.getValuesCSR()
        S.destroy()
        S_array = csr_matrix((Sv, Sj, Si))
        S_R= S_array[:n_basis, :n_basis]
        S.destroy()

        
        

E_vals = []
phi_vals = []
theta_vals = []
phi_vals = []
pad_vals = []

def distribute_array(E_max, gamma):
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine the full range of the array
    full_array = np.arange(0, E_max + 2 * gamma, 2 * gamma)

    # Calculate the portion size each rank will handle
    n_elements = len(full_array)
    elements_per_rank = n_elements // size
    remainder = n_elements % size

    # Determine the start and end indices for each rank
    start_index = rank * elements_per_rank + min(rank, remainder)
    end_index = start_index + elements_per_rank + (1 if rank < remainder else 0)

    # Each rank computes its portion of the array
    local_array = full_array[start_index:end_index]

    return local_array
local_E_range = distribute_array(E_max,gamma)

m_list = np.array([m for _,m in block_dict.values()])
l_list = np.array([l for l,_ in block_dict.values()])

for E in local_E_range:
    if E>0:
        if rank == 0:
            print(E,np.max(local_E_range))
        with h5py.File('PES_files/PES_Vecs.h5', 'r') as f:
            data = f[f"{E}"][:]
            real_part = data[:, 0]
            imaginary_part = data[:, 1]
            wavefunction = real_part + 1j * imaginary_part
        for theta_val in theta_range:
            for phi_val in phi_range:
                harmonics = sph_harm(m_list, l_list, phi_val, theta_val)
                spherical_vector = np.tile(harmonics, (n_basis, 1)).T.flatten()

                

                scaled_vector = spherical_vector * wavefunction
                scaled_vector = scaled_vector.reshape(n_block,n_basis)

                total_sum = scaled_vector.sum(axis = 0)
                value = total_sum.conj().dot(S_R.dot(total_sum))

                # top_lm = [(26,26),(25,25)]
                # top_indices = [lm_dict[(l,m)] for l,m in top_lm]
                # partial_sum = scaled_vector[top_indices, :].sum(axis=0)
                # value = partial_sum.conj().dot(S_R.dot(partial_sum))
                
                E_vals.append(E)
                theta_vals.append(theta_val)
                phi_vals.append(phi_val)
                pad_vals.append(value)

PAD_vals = np.vstack((E_vals,theta_vals,phi_vals,pad_vals))
PAD_arrays = MPI.COMM_WORLD.gather(PAD_vals, root=0)
if rank == 0:
    PAD = np.hstack(PAD_arrays)
    np.save("PES_files/PAD.npy",PAD)
    print("Finished Computing PAD")
    print("\n")
    print("Total Time:")
    print(time.time()-sim_start)
    print("\n")
    print("All Done!")

