
from petsc4py import PETSc
import h5py
import numpy as np
import json
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from Module import *

comm = PETSc.COMM_SELF

def probDisribution():
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    lmax = input_par["lm"]["lmax"]
    N_knots = input_par["splines"]["N_knots"]
    order = input_par["splines"]["order"]
    n_basis = N_knots - order -2
    n_blocks = calc_n_block(lmax)

    lm_map = lm_to_block(lmax)
    block_map = {value: key for key, value in lm_map.items()}
    lm_values = [(l, m) for (l, m) in [block_map[i] for i in range(n_blocks)]]

    with h5py.File('TDSE.h5', 'r') as f:
        data = f["psi_final"][:]
        real_part = data[:,0]
        imaginary_part = data[:,1]
        wavefunction = real_part + 1j*imaginary_part

    S = PETSc.Mat().createAIJ([len(wavefunction),len(wavefunction)],nnz =(2*order + 1),comm = comm)
    viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    Si, Sj, Sv = S.getValuesCSR()
    S.destroy()
    S = csr_matrix((Sv, Sj, Si))

    S_R= S[:n_basis, :n_basis]

    

    prob_list = []
    norm_list = []


    for i in range(n_blocks):
        l,m = block_map[i]
        partial_wavefunction = wavefunction[i*n_basis:(i+1)*n_basis]
        
        final_prod = partial_wavefunction.conj().dot(S_R.dot(partial_wavefunction))
        
        # Stipulate that floating error give small imaginary piece, throwaway!!!
        if l !=m and l>0 and m>0:
            print(f"Norm of {l,m} block is", np.real(np.sqrt(final_prod)))


        prob_list.append(np.real(final_prod))
        norm_list.append(np.real(np.sqrt(final_prod)))


    
    num_rows = 2 * lmax + 1 
    heatmap_array = np.zeros((lmax+1,num_rows))



    for i in range(len(prob_list)):
        l, m = block_map[i]
        row = m + 50  # Shift to center m=0 in the middle of the array
        heatmap_array[l,row] = prob_list[i]
    heatmap_array[heatmap_array<10**(-15)] = 0
    plt.imshow(heatmap_array, cmap='viridis', interpolation='nearest', aspect='equal',norm = LogNorm())
    plt.xlabel("m")
    plt.ylabel("l")
    plt.xticks([])  # Suppresses x-axis tick marks
    plt.yticks([])
    plt.colorbar(label='Probability')

    plt.savefig("heatmap.png")

    print(sum(prob_list))

    

    
probDisribution()



   



