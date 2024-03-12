import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import sys
import h5py
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import json

module_path = "/users/becker/dopl4670/Research/TDSE/Circular/Circular_Solver"

from Module import *


def checkCircularNorm():
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    order = input_par["splines"]["order"]
    n_basis = N_knots - order -2

    n_blocks = calc_n_block(lmax)



    S = PETSc.Mat()
   

    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    
    

    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    psi_final = PETSc.Vec().createWithArray(wavefunction)


    
    with h5py.File('Hydrogen.h5', 'r') as f:
            data = f["/Psi_1_0"][:]
            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part
    psi_initial = PETSc.Vec().createWithArray(np.pad(total,(0,(n_blocks-1)*n_basis),constant_values= (0,0)))
    

    Sv = S.createVecRight()
    S.mult(psi_final,Sv)
    final_prod = psi_final.dot(Sv)

    Sv = S.createVecRight()
    S.mult(psi_initial,Sv)
    initial_prod = psi_initial.dot(Sv)

    print("Norm of Initial State:",np.sqrt(np.real(initial_prod)))
    print("Norm of Final State:",np.sqrt(np.real(final_prod)))
    
    
    return

checkCircularNorm()