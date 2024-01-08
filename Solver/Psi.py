from petsc4py import PETSc
import h5py
import numpy as np
import json 


class psi:
    def __init__(self):
        pass

    def createInitial(self,basisInstance): 
        n_basis = basisInstance.n_basis
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        lmax = input_par["lm"]["lmax"]
        l = input_par["state"][1]

        psi_initial = PETSc.Vec().createMPI(n_basis*(lmax+1),comm = PETSc.COMM_WORLD)
        
        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{l+1}_{l}"][:]

            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part

        global_indices = np.array(range(n_basis))+l*n_basis
        global_indices = global_indices.astype("int32")
        psi_initial.setValues(global_indices,total)
        
        psi_array = np.pad(total,(l*n_basis,(lmax-l)*n_basis),constant_values= (0,0))
       
        psi_initial.assemble()
        
        self.psi_initial = psi_initial

        return None
    
    def createFinal(self,vec):
        self.psi_final = vec
        return None