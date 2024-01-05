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
    

    plt.semilogy(E_range,np.real(final))
    #plt.yticks([10**0, 10**-5, 10**-10, 10**-15])
    #plt.ylim([1e-15, 1e0])
    #plt.axvline([-0.5])
    #plt.axvline([-0.125])
    #plt.axvline([-0.05555555])
    #plt.axvline([0])
    plt.savefig("images/energy.png")

    return
#photoEnergyV1(E_range)


def photoEnergyV2(E_range):
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

        H_0_3 = H_0.copy()
        H_0_4 = H_0.copy()

        H_0_5 = H_0.copy()
        H_0_6 = H_0.copy()

        H_0_7 = H_0.copy()
        H_0_8 = H_0.copy()

        nu_34 = (2*4-1)*np.pi /(2**3)
        H_0_1.axpy(-E-gamma*np.exp(1j*nu_34),S)
        H_0_2.axpy(-E+gamma*np.exp(1j*nu_34),S)

        nu_33 = (2*3-1)*np.pi / (2**3)
        H_0_3.axpy(-E-gamma*np.exp(1j*nu_33),S)
        H_0_4.axpy(-E+gamma*np.exp(1j*nu_33),S)

        nu_32 = (2*2-1)*np.pi / (2**3)
        H_0_5.axpy(-E-gamma*np.exp(1j*nu_32),S)
        H_0_6.axpy(-E+gamma*np.exp(1j*nu_32),S)

        nu_31 = (2*1-1)*np.pi / (2**3)
        H_0_7.axpy(-E-gamma*np.exp(1j*nu_31),S)
        H_0_8.axpy(-E+gamma*np.exp(1j*nu_31),S)

        O_1 = H_0_1.matMult(H_0_2)
        O_2 = H_0_3.matMult(H_0_4)
        O_3 = H_0_5.matMult(H_0_6)
        O_4 = H_0_7.matMult(H_0_8)
        


        



        first = psi_final.copy()
        first.scale(gamma**8)

        


        # Step 1:
        first_RHS = S.createVecRight()
        S.mult(first,first_RHS)
        second = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)
        ksp = PETSc.KSP().create()
        ksp.setOperators(O_1)
        ksp.solve(first_RHS, second)

        # Step 2:
        second_RHS = S.createVecRight()
        S.mult(second,second_RHS)
        third = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)
        ksp = PETSc.KSP().create()
        ksp.setOperators(O_2)
        ksp.solve(second_RHS, third)
        
        # Step 3:
        third_RHS = S.createVecRight()
        S.mult(third,third_RHS)
        fourth = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)
        ksp = PETSc.KSP().create()
        ksp.setOperators(O_3)
        ksp.solve(third_RHS, fourth)

        # Step 4:
        fourth_RHS = S.createVecRight()
        S.mult(fourth,fourth_RHS)
        fifth = PETSc.Vec().createMPI(psi_final.getSize(), comm=PETSc.COMM_WORLD)
        ksp = PETSc.KSP().create()
        ksp.setOperators(O_4)
        ksp.solve(fourth_RHS, fifth)

        
        

        Sv = S.createVecRight()
        S.mult(fifth,Sv)
        val = fifth.dot(Sv)

        photo_energy.append(val)
    plt.semilogy(E_range,photo_energy)

    plt.savefig("images/energy.png")
    print(np.max(photo_energy))
    return



photoEnergyV1(E_range)



                

    


            







 









