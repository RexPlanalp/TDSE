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





# New
def checkNorm():
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    order = input_par["splines"]["order"]
    n_basis = N_knots - order -2

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
    psi_initial = PETSc.Vec().createWithArray(np.pad(total,(0,(lmax)*n_basis),constant_values= (0,0)))

    Sv = S.createVecRight()
    S.mult(psi_final,Sv)
    final_prod = psi_final.dot(Sv)

    Sv = S.createVecRight()
    S.mult(psi_initial,Sv)
    initial_prod = psi_initial.dot(Sv)

    print("Norm of Initial State:",np.sqrt(np.real(initial_prod)))
    print("Norm of Final State:",np.sqrt(np.real(final_prod)))
    
    
    return
def probDisribution():
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    order = input_par["splines"]["order"]
    n_basis = N_knots - order -2

    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    prob_list = []
    norm_list = []
    for l in range(lmax+1):
        partial_wavefunction = wavefunction[l*n_basis:(l+1)*n_basis]
        phi_lm = PETSc.Vec().createWithArray(np.pad(partial_wavefunction,(l*n_basis,(lmax-l)*n_basis),constant_values= (0,0)))

       
        Sv = S.createVecRight()
        S.mult(phi_lm,Sv)
        final_prod = Sv.dot(phi_lm)
        
        # Stipulate that floating error give small imaginary piece, throwaway!!!
        print(f"Norm of {l} block", np.real(np.sqrt(final_prod)))


        prob_list.append(final_prod)
        norm_list.append(np.real(np.sqrt(final_prod)))
    print(np.real(np.sqrt(np.sum(prob_list))))
    
    plt.bar(range(lmax+1),norm_list)
    plt.savefig("images/prob_dist.png")
def groundStatePop():


    with open('input.json', 'r') as file:
            input_par = json.load(file)
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    order = input_par["splines"]["order"]
    n_basis = N_knots - order -2
    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        psi_final = real_part + 1j*imaginary_part
    psi_final = PETSc.Vec().createWithArray(psi_final)

    with h5py.File('Hydrogen.h5', 'r') as f:
        data = f["Psi_1_0"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        ground_state = real_part + 1j*imaginary_part
    ground_state = PETSc.Vec().createWithArray(np.pad(ground_state,(0,(lmax)*n_basis),constant_values= (0,0)))


    Sv = S.createVecRight()
    S.mult(psi_final,Sv)
    prod = ground_state.dot(Sv)
    print(np.abs(prod)**2)
    return

#checkNorm()
#probDisribution()
groundStatePop()








    

