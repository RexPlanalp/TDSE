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
        #b = S.createVecRight()
        #S.mult(x,b)
        b = x
        
       
        
        # Set operator as H_0_1 and solve
        ksp.setOperators(H_0_1)
        ksp.solve(b, y)

        Sv = S.createVecRight()
        S.mult(y,Sv)
        val = y.dot(Sv)

        photo_energy.append(val)

    final = np.array(photo_energy) * gamma**4
    plt.semilogy(E_range,np.real(final))
    plt.ylim(1E-30,1E0)
    print(np.max(np.real(final)))
    plt.savefig("images/energy.png")
    return


dp = 0.1
pmax = 1
def photoAngularV1(pmax,dp,slice):
    
    p_x = np.arange(-pmax,pmax+dp,dp)
    p_y = np.arange(-pmax,pmax+dp,dp)
    p_z = np.arange(-pmax,pmax+dp,dp)

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

    basis_array = np.load("basis/basis.npy")

    if slice == "xz":
        pad = np.empty((len(p_x),len(p_z)))
        py = 0 
        for i,px in enumerate(p_x):
            for j,pz in enumerate(p_z):
                E = (px**2+pz**2+pz**2)/2
                print(i,j)
                theta = np.arccos(pz/np.sqrt(px**2+py**2+pz**2))
                phi = np.arctan2(py,px)

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

                total = 0
                for l in range(14+1):
                    wavefunction = 0
                    for n in range(740):
                        index = l * 740 + n
                        wavefunction += y.getValue(index) * basis_array[:,n]
                    total += wavefunction * sph_harm(0,l,phi,theta)
                I = trapz(np.abs(total)**2,dx =0.01)
                pad[i,j] = I
        plt.imshow(pad)
        plt.savefig("PAD.png")


                                



photoEnergyV1(E_range)
#photoAngularV1(pmax,dp,"xz")



                

    


            







 









