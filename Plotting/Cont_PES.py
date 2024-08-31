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

# Load the input data
with open("input.json", 'r') as file:
    data = json.load(file)

# The specified energy range
E_interpolate = np.arange(0, 1 + 0.01, 0.01)

# Path for the new HDF5 file
new_file_path = 'H_Cont_Cleaned.h5'

# Process each l value
for l in range(data["lm"]["lmax"]+1):
    print(l)
    with h5py.File("H_Cont.h5", 'r') as file:  # Open the original file in read-only mode
        
        # Prepare to write to a new file
        with h5py.File(new_file_path, 'a') as new_file:  # Open or create the new file in append mode

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
                
                # Check if the new name already exists in the new file to avoid overwriting
                if new_dataset_name not in new_file:
                    # Copy data to new file under new name
                    new_file[new_dataset_name] = file[closest_dataset][...]


with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
    file_data = f["psi_final"][:]
    real_part = file_data[:, 0]
    imaginary_part = file_data[:, 1]
    total = real_part + 1j * imaginary_part

#m = 0
partial_spectra = {}
for l in range(data["lm"]["lmax"]+1):
    for m in range(-l,l+1):
        print(l)
        vals = []

        block_idx = simInstance.lm_dict[(l,m)]
        block = total[block_idx*n_basis:(block_idx+1)*n_basis]

        with h5py.File(new_file_path, 'r') as new_file:
            for energy in E_interpolate:
                dataset_name = f'Psi_{l}_{energy:.2f}'
                if dataset_name in new_file:
                    psi = new_file[dataset_name][:]
                    psi_complex = psi[:, 0] + 1j * psi[:, 1]

                    inner_product = psi_complex.conj().dot(S_R.dot(block))
                    vals.append(inner_product)
        partial_spectra[(l,m)] = vals


# Computing Total photoelectron spectrum
total = 0
for key, value in partial_spectra.items():
    total += np.abs(value)**2

plt.semilogy(E_interpolate, total)
plt.savefig("Cont_PES.png")
plt.clf()


# Computing PAD
k_interpolate = np.sqrt(2 * E_interpolate)
#theta = np.arange(0,np.pi+0.01,0.01)
#phi = np.array([0,np.pi])

theta = np.array([np.pi/2])
phi = np.arange(0,2*np.pi,0.01)

k_vals = []
theta_vals = []
phi_vals = []
pad_vals = []

for k in k_interpolate:
    if k == 0:
        continue
    print(k,np.max(k_interpolate))
    E_idx = np.argmin(np.abs(k_interpolate - k))
    for t in theta:
        for p in phi:

            k_vals.append(k)
            theta_vals.append(t)
            phi_vals.append(p)

            pad_amp = 0
            for key, value in partial_spectra.items():
                l,m = key
                pad_amp += (-1j)**l * np.exp(1j*np.angle(gamma(l + 1 -1j/k))) * sph_harm(m, l, p, t) * value[E_idx]
            pad_vals.append(np.abs(pad_amp)**2)


px_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
py_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
pz_vals = k_vals * np.cos(theta_vals)

max_mom = np.max(np.real(pad_vals))
min_mom = np.max(np.real(pad_vals))*10**-6
plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r",norm=mcolors.LogNorm(vmin=min_mom,vmax=max_mom))
plt.savefig("Cont_PAD.png")
            
                    
                   



        


