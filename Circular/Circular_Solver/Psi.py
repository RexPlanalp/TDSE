from petsc4py import PETSc
import h5py
import numpy as np
import json 


class psi:
    def __init__(self,n_blocks,lm_dict):
        self.n_blocks = n_blocks
        self.lm_dict = lm_dict
        pass

    def createInitial(self,basisInstance): 
        n_basis = basisInstance.n_basis
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        lmax = input_par["lm"]["lmax"]
        l_value = input_par["state"][1]
        m_value = input_par["state"][2]

        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{l_value+1}_{l_value}"][:]

            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        psi_initial = PETSc.Vec().createMPI(n_basis*self.n_blocks,comm = PETSc.COMM_WORLD)


        block_index = self.lm_dict[(l_value),m_value]


        global_indices = np.array(range(n_basis))+block_index*n_basis
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        
        
       
        psi_initial.assemble()
        
        self.psi_initial = psi_initial

        return None
    
    def createFinal(self,vec):
        self.psi_final = vec
        return None