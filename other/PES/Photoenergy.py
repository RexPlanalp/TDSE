import numpy as np
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.integrate import trapz


gamma = 0.001
E_min = -0.6
E_max = 1
E_range = np.arange(E_min,E_max+2*gamma,2*gamma)


def photoEnergyV1(E_range):
    photo_energy = []
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    psi_final = PETSc.Vec().createWithArray(wavefunction)
   
    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    H_0 = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()
    rows,cols = H_0.getSize()

    for E in E_range:
        print(E)

        H_0_1 = H_0.copy()
        H_0_2 = H_0.copy()

        H_0_1.axpy(-E+np.sqrt(1j)*gamma,S)
        H_0_2.axpy(-E-np.sqrt(1j)*gamma,S)

        # Create Vector to store solution
        x = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)

        # Create and population RHS vector
        a = S.createVecRight()
        S.mult(psi_final,a)

        # Set Operator as H_0_2 and solve
        ksp = PETSc.KSP().create()
        ksp.setTolerances(rtol = 1E-15)
        ksp.setOperators(H_0_2)
        ksp.solve(a, x)

        # Create Vector to store solution
        y = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)

        # Create and population RHS vector
        b = S.createVecRight()
        S.mult(x,b)
        
        
       
        
        # Set operator as H_0_1 and solve
        ksp.setOperators(H_0_1)
        ksp.solve(b, y)

        Sv = S.createVecRight()
        S.mult(y,Sv)
        val = y.dot(Sv)

        photo_energy.append(val)

    final = np.array(photo_energy) * gamma**4
    plt.semilogy(E_range,np.real(final))
    
    print(np.max(np.real(final)))
    plt.savefig("images/energy.png")

    np.save("PES.npy",final)
    return


def photoEnergyV2(E_range):
    def q_nk(n,k):
        return ((2*k-1)*np.pi)/4

    photo_energy = []
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    psi_final = PETSc.Vec().createWithArray(wavefunction)
   
    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    H_0 = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()
    rows,cols = H_0.getSize()

    for E in E_range:
        print(E)
        A = H_0.copy()
        A.axpy(-E+gamma*np.exp(1j*q_nk(2,1)),S)
        B = H_0.copy()
        B.axpy(-E-gamma*np.exp(1j*q_nk(2,1)),S)
        C = H_0.copy()
        C.axpy(-E+gamma*np.exp(1j*q_nk(2,2)),S)
        D = H_0.copy()
        D.axpy(-E-gamma*np.exp(1j*q_nk(2,2)),S)

    
        psi_final = PETSc.Vec().createWithArray(wavefunction)
        psi_final_prime = psi_final.copy()
        S.mult(psi_final,psi_final_prime)


        z = psi_final.copy()
        z_prime = psi_final.copy()
        w = psi_final.copy()
        w_prime = psi_final.copy()
        v = psi_final.copy()
        v_prime = psi_final.copy()


        x = psi_final.copy()



        # Create KSP solvers for each equation
        ksp_A = PETSc.KSP().create()
        ksp_B = PETSc.KSP().create()
        ksp_C = PETSc.KSP().create()
        ksp_D = PETSc.KSP().create()

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

        Sv = S.createVecRight()
        S.mult(x,Sv)
        val = x.dot(Sv)

        photo_energy.append(val)
    final = np.array(photo_energy) * gamma**8
    plt.ylim([1E-15,1E0])
    plt.semilogy(E_range,np.real(final))
    
    
    print(np.max(np.real(final)))
    np.save("PES.npy",np.real(final))
    plt.savefig("images/energy.png")
    return

        
photoEnergyV2(E_range)




                

    


            







 









