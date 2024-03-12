import numpy as np
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
import json
from scipy.special import sph_harm
import os  

from Module import *

comm = PETSc.COMM_WORLD

def create_array_with_ones(N, M):
    # Create an array of zeros of length N * M
    array = np.zeros(N * M)

    # Set the first element to 1
    array[0] = 1

    # Set every N-th element to 1
    for i in range(N, N * M, N):
        array[i] = 1

    return array
def q_nk(n,k):
        return ((2*k-1)*np.pi)/4


LOAD_EXT = True


if LOAD_EXT:

    if comm.rank == 0:
        print("Reading Input File")

    with open('input.json', 'r') as file:
            input_par = json.load(file)
    order = input_par["splines"]["order"]
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    n_basis = N_knots - order - 2
    n_blocks = calc_n_block(lmax)
    total_size = n_basis * n_blocks
    
    
    
    lm_map = lm_to_block(lmax)
    block_map = {value: key for key, value in lm_map.items()}
    #lm_values = [(l, m) for (l, m) in [block_map[i] for i in range(n_blocks)] for _ in range(n_basis)]
    lm_values = [(l, m) for (l, m) in [block_map[i] for i in range(n_blocks)]]

    

    PES = input_par["PES"][0]
    PAD = input_par["PAD"][0]

    if PES:
        E_min,E_max = input_par["PES"][1],input_par["PES"][2]
    if PAD:
        E_min,E_max = input_par["PAD"][1],input_par["PAD"][2]

    if comm.rank == 0:
        print("Reading Final State")
    
    
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
            
        wavefunction = real_part + 1j*imaginary_part

    
    psi_final = PETSc.Vec().createMPI(total_size,comm = comm)
    global_indices = np.arange(total_size)
    global_indices = global_indices.astype("int32")
    psi_final.setValues(global_indices,wavefunction)
    psi_final.assemble()

    

    if comm.rank == 0:
        print("Reading Matrices")
    
    S = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    S_R = PETSc.Mat().createAIJ([n_basis,n_basis],nnz =(2*order + 1))
    viewer = PETSc.Viewer().createBinary('matrix_files/S_R.bin', 'r')
    S_R.load(viewer)
    viewer.destroy()

    H_0 = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()
    rows,cols = H_0.getSize()


gamma = 0.001

E_range = np.arange(E_min,E_max+2*gamma,2*gamma)
dphi = 0.01
phi_range = np.array([0,np.pi])
theta_range = np.arange(0,np.pi+dphi,dphi)

if PAD:
    if comm.rank == 0:
        print("Computing Spherical Harmonics")
    
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
    sph_harmonics = np.zeros((n_blocks * n_basis, len(phi_range), len(theta_range)), dtype=complex)
    for block_index, (l, m) in enumerate([block_map[i] for i in range(n_blocks)]):
        harmonic_grid = sph_harm(m, l, phi_grid, theta_grid)
        for basis_index in range(n_basis):
            index = block_index * n_basis + basis_index
            sph_harmonics[index, :, :] = harmonic_grid
    
    scaled_vector = PETSc.Vec().createMPI(size = total_size,comm = comm)


def photoAngularV2(E_range,theta_range,phi_range,PAD,PES):
    PES_vals = []
    E_vals = []
    theta_vals = []
    phi_vals = []
    pad_vals = []

    # Create KSP solvers for each equation
    ksp_A = PETSc.KSP().create(comm = comm)
    ksp_B = PETSc.KSP().create(comm = comm)
    ksp_C = PETSc.KSP().create(comm = comm)
    ksp_D = PETSc.KSP().create(comm = comm)

    
        
    
        
    if comm.rank == 0:
        print("Starting Energy Loop")

    for E in E_range:
        if comm.rank == 0:
            print(E)

        A = H_0.copy()
        A.axpy(-E+gamma*np.exp(1j*q_nk(2,1)),S)
        B = H_0.copy()
        B.axpy(-E-gamma*np.exp(1j*q_nk(2,1)),S)
        C = H_0.copy()
        C.axpy(-E+gamma*np.exp(1j*q_nk(2,2)),S)
        D = H_0.copy()
        D.axpy(-E-gamma*np.exp(1j*q_nk(2,2)),S)

    
        psi_final_prime = psi_final.copy()
        S.mult(psi_final,psi_final_prime)


        z = psi_final.copy()
        z_prime = psi_final.copy()
        w = psi_final.copy()
        w_prime = psi_final.copy()
        v = psi_final.copy()
        v_prime = psi_final.copy()


        x = psi_final.copy()


        # Set operators for each solver
        ksp_A.setOperators(A)
        ksp_B.setOperators(B)
        ksp_C.setOperators(C)
        ksp_D.setOperators(D)

        # Solve Az = y and then multiply by S
        ksp_A.solve(psi_final_prime, z)
        S.mult(z, z_prime)

        # Solve Bw = z_prime and then multiply by S
        ksp_B.solve(z_prime, w)
        S.mult(w, w_prime)

        # Solve Cv = w_prime and then multiply by S
        ksp_C.solve(w_prime, v)
        S.mult(v, v_prime)

        # Finally, solve Dx = v_prime (the final result is x)
        ksp_D.solve(v_prime, x)

        A.destroy()
        B.destroy()
        C.destroy()
        D.destroy()

        if PES:
            Sv = S.createVecRight()
            S.mult(x,Sv)
            PES_val = x.dot(Sv)
            PES_vals.append(PES_val*gamma**8)

            Sv.destroy()

        if PAD:
            for i,theta_val in enumerate(theta_range):
                for j,phi_val in enumerate(phi_range):
                    
                    ##UNCOMMENT##
                    #y = x.copy()
                    ##UNCOMMENT#

                    values = sph_harmonics[:,j,i]

                    ##UNCOMMENT,WORKS FOR PARALLEL##
                    #spherical_vector = PETSc.Vec().createMPI(total_size,comm = comm)
                    #global_indices = np.arange(total_size)
                    #global_indices = global_indices.astype("int32")
                    #spherical_vector.setValues(global_indices,values)
                    #spherical_vector.assemble()
                    #####################

                    spherical_vector = PETSc.Vec().createWithArray(values,size = total_size,comm = comm)


                    scaled_vector.pointwiseMult(x,spherical_vector)

                    #UNCOMMENT#
                    #y.pointwiseMult(y, spherical_vector)
                    #UNCOMMENT#
                    
                    ##TESTING##
                    #sumVec = PETSc.Vec().createMPI(n_basis)
                    #for t in range(n_blocks):
                        #indices = np.arange(t * n_basis, (t + 1) * n_basis, dtype=np.int32)
                        #index_set = PETSc.IS().createGeneral(indices, comm=comm)
                        #block = x.getSubVector(index_set)
                        #sumVec.axpy(1, block)  
                        #x.restoreSubVector(index_set, block)
                    ##TESTING##

                    ##TESTING##
                    y_array = scaled_vector.getArray()
                    y_array = y_array.reshape(n_blocks,n_basis)
                    total_sum = y_array.sum(axis = 0)
                    sumVec = PETSc.Vec().createWithArray(total_sum,size = n_basis,comm = comm)
                    ##TESTING##

                    Su = S_R.createVecRight()
                    S_R.mult(sumVec,Su)
                    value = Su.dot(sumVec)


                    ##TESTING##

                    

                    E_vals.append(E)
                    theta_vals.append(theta_val)
                    phi_vals.append(phi_val)
                    pad_vals.append(value * gamma**8)

                    #y.destroy()
                    sumVec.destroy()
                    spherical_vector.destroy()
                     
    

    if PES:
        np.save("PES.npy",PES_vals)
    if PAD:
        PAD = np.vstack((E_vals,theta_vals,phi_vals,pad_vals))
        np.save("PAD.npy",PAD)
    return

        
photoAngularV2(E_range,theta_range,phi_range,PAD,PES)


    



                

    


            







 









