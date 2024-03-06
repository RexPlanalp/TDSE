import numpy as np
from petsc4py import PETSc
import h5py
import json
from scipy.special import sph_harm
from Module import *
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import time
import sys

comm = PETSc.COMM_SELF

INIT = True
ENERGY = True
STATE = True
MAT = True
SPECTRUM = True







if INIT:
    
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
    lm_values = [(l, m) for (l, m) in [block_map[i] for i in range(n_blocks)]]

    def q_nk(n,k):
        return ((2*k-1)*np.pi)/4
  
if ENERGY:
    gamma = input_par["E"][2]
    if len(sys.argv) == 4:
        E_min = float(sys.argv[1])
        E_max = float(sys.argv[2])
        chunk = sys.argv[3]
        print(f"Running for chunk {chunk} = E{E_min,E_max-2*gamma}")
    else:
        E_min = input_par["E"][0]
        E_max = input_par["E"][1]
   


    PES = input_par["PES"][0]
    PES_min = input_par["PES"][1]
    PES_max = input_par["PES"][2]

    PAD = input_par["PAD"][0]
    PAD_min = input_par["PAD"][1]
    PAD_max = input_par["PAD"][2]

    E_range = np.arange(E_min,E_max,2*gamma)

    SLICE = input_par["SLICE"]

    if SLICE == "XZ":
        phi_range = np.array([0,np.pi])
        theta_range = np.arange(0,np.pi+0.01,0.01)
    elif SLICE == "XY":
        phi_range = np.arange(0,2*np.pi+0.01,0.01)
        theta_range = np.array([np.pi/2])
    elif SLICE == "YZ":
        phi_range = np.array([np.pi/2,3*np.pi/2])
        theta_range = np.arange(0,np.pi+0.01,0.01)
    
        
    
    print("Computing Spherical Harmonics")
    
    theta_grid, phi_grid = np.meshgrid(theta_range, phi_range)
    sph_harmonics = np.zeros((n_blocks * n_basis, len(phi_range), len(theta_range)), dtype=complex)
    for block_index, (l, m) in enumerate([block_map[i] for i in range(n_blocks)]):
        harmonic_grid = sph_harm(m, l, phi_grid, theta_grid)
        for basis_index in range(n_basis):
            index = block_index * n_basis + basis_index
            sph_harmonics[index, :, :] = harmonic_grid
if STATE:
    
    print("Reading Final State")
    
    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
            
        wavefunction = real_part + 1j*imaginary_part
if MAT:
    
    print("Reading Matrices")
    
    S = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    Si, Sj, Sv = S.getValuesCSR()
    S.destroy()
    S = csr_matrix((Sv, Sj, Si))

    S_R= S[:n_basis, :n_basis]

    H_0 = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
    H_0.load(viewer)
    viewer.destroy()
    rows,cols = H_0.getSize()

    H_0i, H_0j, H_0v = H_0.getValuesCSR()
    H_0.destroy()
    H_0 = csr_matrix((H_0v, H_0j, H_0i))

if SPECTRUM:
    PES_vals = []
    E_vals = []
    theta_vals = []
    phi_vals = []
    pad_vals = []

    print("Starting Energy Loop")

    for E in E_range:
        in_PES = PES_min<=E<=PES_max
        in_PAD = PAD_min<= E<=PAD_max
        
        print(f"E={E}:PES:{bool(in_PES and PES)},PAD:{bool(in_PAD and PAD)}")

        A = H_0 + S*(-E+gamma*np.exp(1j*q_nk(2,1)))
        B = H_0 + S*(-E-gamma*np.exp(1j*q_nk(2,1)))
        C = H_0 + S*(-E+gamma*np.exp(1j*q_nk(2,2)))
        D = H_0 + S*(-E-gamma*np.exp(1j*q_nk(2,2)))
    
        psi_final_prime = S.dot(wavefunction)

        z = spsolve(A,psi_final_prime)
        z_prime = S.dot(z)

        w = spsolve(B,z_prime)
        w_prime = S.dot(w)

        v = spsolve(C,w_prime)
        v_prime = S.dot(v)

        x = spsolve(D,v_prime)
    
        if PES and in_PES:
            PES_val = x.conj().dot(S.dot(x))
            PES_vals.append(PES_val*gamma**8)
        if PAD and in_PAD:
            for i,theta_val in enumerate(theta_range):
                for j,phi_val in enumerate(phi_range):
                    spherical_vector = sph_harmonics[:,j,i]
                    
                    scaled_vector = spherical_vector * x
        
                    scaled_vector = scaled_vector.reshape(n_blocks,n_basis)
                    total_sum = scaled_vector.sum(axis = 0)

                    value = total_sum.conj().dot(S_R.dot(total_sum))
                    
                    E_vals.append(E)
                    theta_vals.append(theta_val)
                    phi_vals.append(phi_val)
                    pad_vals.append(value * gamma**8)
    if PES:
        np.save(f"photo_files/PES{chunk}.npy",PES_vals)
    if PAD:
        PAD = np.vstack((E_vals,theta_vals,phi_vals,pad_vals))
        np.save(f"photo_files/PAD{chunk}.npy",PAD)
    

start = time.time()
end = time.time()
print(f"Finished Computing Energy Spectrum {end-start}")


    



                

    


            







 









