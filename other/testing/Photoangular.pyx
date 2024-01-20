import numpy as np
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.integrate import trapz
from scipy.linalg import block_diag
import json
import matplotlib.cm as cm
from Module import *
import time
from mpi4py import MPI

comm = PETSc.COMM_WORLD
INITIALIZE = True

if INITIALIZE:
    with open('input.json', 'r') as file:
            input_par = json.load(file)

    order = input_par["splines"]["order"]
    lmax = input_par["lm"]["lmax"]


    
    def q_nk(n,k):
        return ((2*k-1)*np.pi)/4

    
    with h5py.File('../Sample/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
            
        wavefunction = real_part + 1j*imaginary_part

    psi_final = PETSc.Vec().createWithArray(wavefunction,size = len(wavefunction),comm = comm)

    basis_array = np.load("../Sample/basis/basis.npy")

    n_basis,n_basis = np.shape(basis_array)
    l_total = int(len(wavefunction)/n_basis)
    
   
    S = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('../Sample/matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    S_R = PETSc.Mat().createAIJ([n_basis,n_basis],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('../Sample/matrix_files/S_R.bin', 'r')
    S_R.load(viewer)
    viewer.destroy()

    H_0 = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('../Sample/matrix_files/H_0.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()
    rows,cols = H_0.getSize()

    S_TILE = PETSc.Mat().createAIJ([n_basis*(lmax+1),n_basis*(lmax+1)],nnz =n_basis*(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('../Sample/matrix_files/S_T.bin', 'r')
    S_TILE.load(viewer)
    viewer.destroy()

    



gamma = 0.001
E_min = 0.001
E_max = 0.5
E_range = np.arange(E_min,E_max+2*gamma,2*gamma)




def photoAngularV1(E_range,dphi,dtheta):
    phi = np.array([0,np.pi])
    theta = np.arange(0,np.pi,dtheta)
    

    r_data = []
    theta_data = []
    phi_data = []
    z_data = []


     # Create KSP solvers for each equation
    ksp_A = PETSc.KSP().create(comm = comm)
    ksp_B = PETSc.KSP().create(comm = comm)
    ksp_C = PETSc.KSP().create(comm = comm)
    ksp_D = PETSc.KSP().create(comm = comm)
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


        
        vectorized = True
        sum = False

        for theta_val in theta:
            for phi_val in phi:
            
            
                if vectorized:
                    y = x.copy()
                    

                    l_values = [[i]*n_basis for i in range(lmax+1)]
                    values = sph_harm(0,l_values,phi_val,theta_val)

                    spherical_vector = PETSc.Vec().createWithArray(values,size = len(wavefunction),comm = comm)

                    y.pointwiseMult(y, spherical_vector)


                    Sv = S_TILE.createVecRight()
                    S_TILE.mult(y,Sv)
                    value = Sv.dot(y)
                    

                    r_data.append(E)
                    theta_data.append(theta_val)
                    phi_data.append(phi_val)
                    z_data.append(value)


                    Sv.destroy()
                    y.destroy()
                    spherical_vector.destroy()
            
                if sum:
                    total = 0

                    for l in range(l_total):
                        global_l = np.array(range(n_basis))+l*n_basis
                        global_l = global_l.astype("int32")
                        global_l_IS = PETSc.IS().createGeneral(global_l)
                        vec_l = x.getSubVector(global_l_IS)
                        vec_l.scale(sph_harm(0,l,phi_val,theta_val))

                        Sv = S_R.createVecRight()
                        S_R.mult(vec_l,Sv)
                        for lprime in range(l_total):
                                if l >= lprime:
                                    global_lprime = np.array(range(n_basis))+lprime*n_basis
                                    global_lprime = global_lprime.astype("int32")
                                    global_lprime_IS = PETSc.IS().createGeneral(global_lprime)
                                    vec_lprime = x.getSubVector(global_lprime_IS)
                                    vec_lprime.scale(sph_harm(0,lprime,phi_val,theta_val))

                                    value = Sv.dot(vec_lprime)
                        
                                    total += value

                                    if l!=lprime:
                                        total+=np.conjugate(value)
                        
                        
                    r_data.append(E)
                    theta_data.append(theta_val)
                    z_data.append(total)
            


    
    
    np.save("r.npy",r_data)
    np.save("theta.npy",theta_data)
    np.save("phi.npy",phi_data)
    np.save("z.npy",z_data)



start = time.time()       
photoAngularV1(E_range,0.01,0.01)
end = time.time()
print(end-start)

