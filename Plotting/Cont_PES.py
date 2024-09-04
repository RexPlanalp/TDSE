import numpy as np
import h5py
import json
import sys
import matplotlib.pyplot as plt

import matplotlib.colors as mcolors

from scipy.special import sph_harm
from scipy.special import gamma

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *


E_max = 1
E_res = 0.01

simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 
n_basis = simInstance.splines["n_basis"]
order= simInstance.splines["order"]
total_size = n_basis * simInstance.n_block

basisInstance = basis()
basisInstance.createKnots(simInstance)
B = basisInstance.B
knots = basisInstance.knots

if simInstance.laser["polarization"] == "linear":
    theta_range = np.arange(0,np.pi,0.01)
    phi_range = np.array([0,np.pi])

  
elif simInstance.laser["polarization"] == "elliptical":
    theta_range = np.array([np.pi/2])
    phi_range = np.arange(0,2*np.pi,0.01)

    


LOAD_S = True
if LOAD_S:
    S = PETSc.Mat().createAIJ([total_size, total_size], nnz=(2 * (order - 1) + 1), comm=MPI.COMM_SELF)
    viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
    S.load(viewer)
    viewer.destroy()

    from scipy.sparse import csr_matrix
    Si, Sj, Sv = S.getValuesCSR()
    S.destroy()
    S_array = csr_matrix((Sv, Sj, Si))
    S_R= S_array[:n_basis, :n_basis]
    S.destroy()

with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
    file_data = f["psi_final"][:]
    real_part = file_data[:, 0]
    imaginary_part = file_data[:, 1]
    total = real_part + 1j * imaginary_part


# Load the input data
with open("input.json", 'r') as file:
    data = json.load(file)

# The specified energy range
E_interpolate = np.arange(0, E_max + E_res, E_res)

# Dictionary to store the cleaned states
cleaned_states = {}

# Process each l value
for l in range(data["lm"]["lmax"]+1):
    print(l)
    with h5py.File("H_Cont.h5", 'r') as file:  # Open the original file in read-only mode
        
        # Get a list of dataset names for the specific l
        datasets = list(file.keys())
        filtered_datasets = [name for name in datasets if name.startswith(f'Psi_{l}_')]

        # Extract energies and corresponding datasets
        energies = []
        dataset_names = []
        for dataset_name in filtered_datasets:
            parts = dataset_name.split('_')
            energy = float(parts[2])
            energies.append(energy)
            dataset_names.append(dataset_name)
        energies = np.array(energies)

        # Loop through each specified energy in the range
        for energy in E_interpolate:
            index_closest = np.argmin(np.abs(energies - energy))
            closest_energy = energies[index_closest]
            closest_dataset = dataset_names[index_closest]
            
            # Create new dataset name with interpolated energy
            new_dataset_name = f'Psi_{l}_{energy:.2f}'  # Format to 2 decimal places
            
            # Store in the dictionary instead of writing to a new file
            if new_dataset_name not in cleaned_states:
                # Retrieve data and convert to complex array
                data_array = file[closest_dataset][...]
                if data_array.shape[1] == 2:
                    # Convert two-column real/imag data to complex array
                    psi_complex = data_array[:, 0] + 1j * data_array[:, 1]
                else:
                    # Assume data is already in complex format
                    psi_complex = data_array
                
                # Copy data to the dictionary as complex numbers
                cleaned_states[new_dataset_name] = psi_complex

partial_spectra = {}


for l,m in simInstance.lm_dict:
    print(f'l: {l}, m: {m}')
    vals = []

    block_idx = simInstance.lm_dict[(l, m)]
    block = total[block_idx*n_basis:(block_idx+1)*n_basis]

    for energy in E_interpolate:
        dataset_name = f'Psi_{l}_{energy:.2f}'
        if dataset_name in cleaned_states:
            

            psi_complex = cleaned_states[dataset_name]
            inner_product = psi_complex.conj().dot(S_R.dot(block))
            vals.append(inner_product)

    partial_spectra[(l, m)] = vals

total = 0
for key, value in partial_spectra.items():
    total += np.abs(value)**2

plt.semilogy(E_interpolate, total)
plt.savefig("Cont_PES.png")
plt.clf()


# Computing PAD
k_interpolate = np.sqrt(2 * E_interpolate)

k_vals = []
theta_vals = []
phi_vals = []
pad_vals = []

for i,k in enumerate(k_interpolate):
    print(i,len(k_interpolate))
    if k == 0:
        continue
    E_idx = np.argmin(np.abs(k_interpolate - k))
    for t in theta_range:
        for p in phi_range:

            k_vals.append(k)
            theta_vals.append(t)
            phi_vals.append(p)

            pad_amp = 0
            for key, value in partial_spectra.items():
                
                l,m = key
                #pad_amp += (-1j)**l * np.exp(1j*np.angle(gamma(l + 1 -1j/k))) * sph_harm(m, l, p, t) * value[E_idx]
                pad_amp += (1j)**l * np.exp(-1j*np.angle(gamma(l + 1 -1j/k))) * sph_harm(m, l, p, t) * value[E_idx]
            pad_vals.append(np.abs(pad_amp)**2)


px_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
py_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
pz_vals = k_vals * np.cos(theta_vals)

max_mom = np.max(np.real(pad_vals))
min_mom = np.max(np.real(pad_vals))*10**-2
plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r",norm=mcolors.LogNorm(vmin=min_mom,vmax=max_mom))
#plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r")
plt.colorbar()
plt.savefig("Cont_PAD.png")
            
            
                    
                   



        


