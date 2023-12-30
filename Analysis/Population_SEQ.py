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


# Old
def plotWavefunction():
    
    basis_array = np.load("basis.npy")
    grid_size = 1000
    grid_spacing = 0.01

    r = np.linspace(0,grid_size,int(grid_size/grid_spacing)+1)

    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    pos_space_wavefunction = 0
    L =0
    for i in range(L*338,(L+1)*338):
        j = i% 338
        pos_space_wavefunction+= basis_array[:,j]*wavefunction[i]

    pos_space_bound = 0
    with h5py.File("Hydrogen.h5","r") as f:
        data = f["Psi_1_0"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    for i in range(338):
        pos_space_bound += basis_array[:,i]*wavefunction[i]

    plt.plot(r,np.abs(pos_space_wavefunction)**2,label = "final")
    plt.plot(r,np.abs(pos_space_bound)**2,label = "bound")
    plt.legend()
    plt.xlim([0,10])
    plt.savefig("test.png")
    plt.clf()
def checkPhase():
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    with h5py.File('Hydrogen.h5', 'r') as f:
        data = f["Psi_1_0"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        groundstate = real_part + 1j*imaginary_part
    
    val1 = wavefunction[0]
    val2 = groundstate[0]
    print(np.angle(val1/val2))
    print(np.abs(val1/val2))



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

    print("Norm of Initial State:",np.real(initial_prod))
    print("Norm of Final State:",np.real(final_prod))
    
    
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
    for l in range(lmax+1):
        partial_wavefunction = wavefunction[l*n_basis:(l+1)*n_basis]
        phi_lm = PETSc.Vec().createWithArray(np.pad(partial_wavefunction,(l*n_basis,(lmax-l)*n_basis),constant_values= (0,0)))

       
        Sv = S.createVecRight()
        S.mult(phi_lm,Sv)
        final_prod = phi_lm.dot(Sv)

        print(f"Norm of {l} block", np.real(final_prod))


        prob_list.append(final_prod)
    print(np.real(np.sum(prob_list)))

    plt.bar(range(lmax+1),np.real(prob_list))
    plt.savefig("images/prob_dist.png")


# Not working
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
        data = f["Psi_2_0"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        ground_state = real_part + 1j*imaginary_part
    ground_state = PETSc.Vec().createWithArray(np.pad(ground_state,(0,(lmax)*n_basis),constant_values= (0,0)))


    Sv = S.createVecRight()
    S.mult(psi_final,Sv)
    prod = ground_state.dot(Sv)
    print(np.abs(prod))
    return

checkNorm()
#probDisribution()
#groundStatePop()







    

