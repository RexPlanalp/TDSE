import numpy as np
from petsc4py import PETSc
import h5py
import matplotlib.pyplot as plt



gamma = 0.001
E_min = -0.6
E_range = np.arange(E_min,1+2*gamma,2*gamma)



    

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
    print(np.max(final))
    #print(np.max(final))

    plt.semilogy(E_range,final)
    #plt.yticks([10**0, 10**-5, 10**-10, 10**-15])
    #plt.ylim([1e-15, 1e0])
    #plt.axvline([-0.5])
    #plt.axvline([-0.125])
    #plt.axvline([-0.05555555])
    #plt.axvline([0])
    plt.savefig("images/energy.png")

    return
#photoEnergyV1(E_range)

         
photoEnergyV1(E_range)






                

    


            







 









