from petsc4py import PETSc
import h5py
import numpy as np
import json 


class psi:
    def __init__(self):
        pass

    def createInitial(self,basisInstance,lm_map): 
        n_basis = basisInstance.n_basis
        n_blocks = basisInstance.n_blocks
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        lmax = input_par["lm"]["lmax"]
        l = input_par["state"][1]
        m = input_par["state"][2]

        psi_initial = PETSc.Vec().createMPI(n_basis*n_blocks,comm = PETSc.COMM_WORLD)
        
        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{l+1}_{l}"][:]

            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        block_index = lm_map[(l,m)]

        global_indices = np.array(range(n_basis))+block_index*n_basis
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        
       
        psi_initial.assemble()
        
        self.psi_initial = psi_initial

        return None
    
    def createFinal(self,vec):
        self.psi_final = vec
        return None