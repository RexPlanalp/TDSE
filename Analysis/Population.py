import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import sys
import h5py
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt




def computePopulation():
    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    with h5py.File('TDSE.h5', 'r') as f:
        data = f[f"/psi_final"][:]

        real_part = data[:,0]
        imaginary_part = data[:,1]
        psi_final = real_part + 1j*imaginary_part
    N = len(psi_final)
    psi_final = PETSc.Vec().createWithArray(comm = PETSc.COMM_WORLD,size = N,array = psi_final)
    


    lmax = 50
    nmax = 15
    n_basis = int(N/(lmax+1))
    
    
    
    total_pop = 0

    for n in range(nmax-1):
         for l in range(n):
            print(n,l)
            with h5py.File('Hydrogen.h5', 'r') as f:
            
                data = f[f"/Psi_{n}_{l}"][:]

                real_part = data[:,0]
                imaginary_part = data[:,1]
                total = real_part + 1j*imaginary_part
        
            psi_array = np.pad(total,(l*n_basis,(lmax-l)*n_basis),constant_values= (0,0))


            psi_bound = PETSc.Vec().createWithArray(comm = PETSc.COMM_WORLD,size = len(psi_array),array = psi_array)


            Sv = S.getVecRight()
            S.mult(psi_final,Sv)
            amp = psi_bound.dot(Sv)
            total_pop += np.abs(amp)**2
    print(total_pop)
    return None
computePopulation()



    

