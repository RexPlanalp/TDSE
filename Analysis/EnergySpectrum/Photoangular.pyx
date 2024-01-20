import numpy as np
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
import json
from scipy.special import sph_harm


gamma = 0.001

# Use these for PAD
E_min = 0.001
E_max = 0.5

# Use these for PES
#E_min = -0.6
#E_max = 1

E_range = np.arange(E_min,E_max+2*gamma,2*gamma)

dphi = 0.01
phi_range = np.array([0,np.pi])
theta_range = np.arange(0,np.pi+dphi,dphi)

comm = PETSc.COMM_WORLD

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
    total_size = n_basis * (lmax+1)
    l_values = np.repeat(np.arange(lmax+1),n_basis)


    if comm.rank == 0:
        print("Reading Final State")
    
    
    with h5py.File('../Sample/TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
            
        wavefunction = real_part + 1j*imaginary_part
    psi_final = PETSc.Vec().createWithArray(wavefunction,size = total_size,comm = comm)

    if comm.rank == 0:
        print("Reading Matrices")
    
    S = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('../Sample/matrix_files/overlap.bin', 'r')
    S.load(viewer)
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
    

PES = False
PLOT_PES = False

PAD = False


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

    if PAD:
        if comm.rank == 0:
            print("Computing Spherical Harmonics")
    
        theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
        sph_harmonics = np.zeros((len(l_values), len(phi_range), len(theta_range)), dtype=complex)

        for i, l in enumerate(l_values):
            sph_harmonics[i, :, :] = sph_harm(0, l, phi_grid, theta_grid)

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

        if PAD:
            for i,theta_val in enumerate(theta_range):
                for j,phi_val in enumerate(phi_range):
                    y = x.copy()
                    
                    


                    values = sph_harmonics[:, j, i]

                    spherical_vector = PETSc.Vec().createWithArray(values,size = total_size,comm = comm)

                    y.pointwiseMult(y, spherical_vector)

                    Su = S_TILE.createVecRight()
                    S_TILE.mult(y,Su)
                    value = Su.dot(y)
                    

                    E_vals.append(E)
                    theta_vals.append(theta_val)
                    phi_vals.append(phi_val)
                    pad_vals.append(value * gamma**8)

                    
                    Su.destroy()
                    y.destroy()
                    spherical_vector.destroy()
                     
    

    if PES:
        np.save("PES.npy",PES_vals)
    if PAD:
        PAD = np.vstack((E_vals,theta_vals,phi_vals,pad_vals))
        np.save("PAD.npy",PAD)
    return

        
photoAngularV2(E_range,theta_range,phi_range,PAD,PES)

if PLOT_PES:
    PES_vals = np.load("PES.npy")
    plt.ylim([1E-15,1E0])
    plt.semilogy(E_range,np.real(PES_vals))
    plt.savefig("images/energy.png")
    print(np.max(np.real(PES_vals)))
    



                

    


            







 









