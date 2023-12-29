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



def computePopulation():
    S = PETSc.Mat()
    viewer = PETSc.Viewer().createBinary('overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    with h5py.File('TDSE.h5', 'r') as f:
        data = f[f"/psi_final"][:]

        real_part = data[:,0]
        imaginary_part = data[:,1]
        psi_final = real_part + 1j*imaginary_part
    lmax = 50
    for n in range(lmax+1):
        for l in range(n):
            u1 = psi_final[l*206:(l+1)*206]
            with h5py.File('Hydrogen.h5', 'r') as f:
                data = f[f"Psi_{n}_{l}"][:]

                real_part = data[:,0]
                imaginary_part = data[:,1]
                u2 = real_part + 1j*imaginary_part
            inner_prod = 0
            for i in range(len(u1)):
                for j in range(len(u2)):
                    inner_prod += np.conjugate(u2[i])*S.getValue(i,j)*u1[j]
            pop = np.abs(inner_prod)**2 * (2*l+1)
            print(f"The population of the {n}_{l} state is {pop}")
            
    
    
    
    
    
    return None
def probDistribution():
    basis_array = np.load("basis.npy")
    print(np.shape(basis_array))
    grid_size = 1000
    grid_spacing = 0.01
    
    r = np.linspace(0,grid_size,int(grid_size/grid_spacing)+1)
    total = 0
    #with h5py.File('Hydrogen.h5', 'r') as f:
        #data = f["Psi_2_1"][:]
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    prob_list = []
    for l in range(51):
        partial_wavefunction = wavefunction[l*338:(l+1)*338]
        
        pos_space = 0
        for i in range(338):
            pos_space += partial_wavefunction[i]*basis_array[:,i]
        N = trapz(np.abs(pos_space)**2,r)
        print(f"The Norm of the l = {l} block is {N}")
        total+= N
        prob_list.append(N)
    print(f"Total Norm of State:{total}")

    plt.bar(range(51),prob_list)
    plt.savefig("prob_dist.png")
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
    L =2
    for i in range(L*298,(L+1)*298):
        j = i% 298
        pos_space_wavefunction+= basis_array[:,j]*wavefunction[i]

    pos_space_bound = 0
    with h5py.File("Hydrogen.h5","r") as f:
        data = f["Psi_1_0"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part
    for i in range(298):
        pos_space_bound += basis_array[:,i]*wavefunction[i]

    plt.plot(r,np.abs(pos_space_wavefunction)**2,label = "final")
    #plt.plot(r,np.abs(pos_space_bound)**2,label = "bound")
    plt.legend()
    #plt.xlim([0,10])
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


#computePopulation()
probDistribution()
#checkPhase()
#plotWavefunction()


    

