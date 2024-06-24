import h5py
import numpy as np

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank


class psi:
    def __init__(self,simInstance):
        n_basis = simInstance.splines["n_basis"]
        n_block = simInstance.n_block

        potential = simInstance.box["pot"]
        n_value,l_value,m_value = simInstance.state

        block_index = simInstance.lm_dict[(l_value,m_value)]
        


        psi_initial = PETSc.Vec().createMPI(n_basis*n_block,comm = comm)
        
        with h5py.File(f'TISE_files/{potential}.h5', 'r') as f:
            data = f[f"/Psi_{n_value}_{l_value}"][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        global_indices = np.array(range(n_basis))+block_index*n_basis
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        psi_initial.assemble()
        
        self.state = psi_initial
        return None
    
    