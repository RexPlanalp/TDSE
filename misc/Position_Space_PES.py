# Additional imports
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys
import time
from scipy.special import gamma,sph_harm
import matplotlib.colors as mcolors

sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *
from Atomic import *

simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 

basisInstance = basis()
basisInstance.createKnots(simInstance)

n_basis = simInstance.splines["n_basis"]
order = simInstance.splines["order"]
lmax = simInstance.lm["lmax"]
n_block = simInstance.n_block
knots = basisInstance.knots
pot = simInstance.box["pot"]

total_size = n_basis * n_block
lm_dict,block_dict = simInstance.lm_dict,simInstance.block_dict

from mpi4py import MPI
from petsc4py import PETSc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
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

E_max = 1
dE = 0.001

E_range = np.arange(dE, E_max + dE, dE)


with h5py.File('TDSE_files/TDSE.h5', 'r') as f:
    tdse_data = f["psi_final"][:]
    real_part = tdse_data[:, 0]
    imaginary_part = tdse_data[:, 1]
    psi_final_bspline = real_part + 1j * imaginary_part


for (l,m) in lm_dict.keys():
    block_idx = lm_dict[(l, m)]
    wavefunction_block = psi_final_bspline[block_idx*n_basis:(block_idx+1)*n_basis]
    with h5py.File(f'TISE_files/{pot}.h5', 'r') as f:
        datasets = list(f.keys())
        for dataset_name in datasets:
            if dataset_name.startswith('Psi_'):
                parts = dataset_name.split('_')
                current_n = int(parts[1])
                current_l = int(parts[2])
                
                if current_l == l:
                    
                    data = f[dataset_name][:]
                    real_part = data[:, 0]
                    imaginary_part = data[:, 1]
                    bound_state = real_part + 1j * imaginary_part

                
                    inner_product = bound_state.conj().dot(S_R.dot(wavefunction_block))
                    wavefunction_block -= inner_product * bound_state

    psi_final_bspline[block_idx*n_basis:(block_idx+1)*n_basis] = wavefunction_block

rmax = simInstance.box["grid_size"]

dr = 0.01

r_range = np.arange(0, rmax + dr, dr)

Nr = len(r_range)

start = time.time()

print("Expanding Wavefunction")

psi_final_pos = np.zeros(Nr*n_block, dtype=np.complex128)
for l,m in simInstance.lm_dict.keys():
    print(l,m)
    block_idx = simInstance.lm_dict[(l, m)]
    block = psi_final_bspline[block_idx*n_basis:(block_idx+1)*n_basis]
    wavefunction = np.zeros_like(r_range,dtype=complex)

    for i in range(simInstance.splines["n_basis"]):
        # Get the range where the B-spline is nonzero
        start = knots[i]
        end = knots[i + order]
        
        # Find the indices in `x` that fall within the nonzero range of the B-spline
        valid_indices = np.where((r_range >= start) & (r_range < end))[0]
        
        if valid_indices.size > 0:
            # Evaluate the B-spline only at the valid points
            wavefunction[valid_indices] += block[i] * basisInstance.B(i, order, r_range[valid_indices], knots)
    start_idx = block_idx * len(r_range)
    end_idx = (block_idx + 1) * len(r_range)
    
    # Assign values directly to the appropriate slice of the final array
    psi_final_pos[start_idx:end_idx] = wavefunction
    
end = time.time()
print(f"Time to expand wavefunction: {end-start}")

def Shooting_Method_faster(r_range, l, E):
    r_range2 = r_range**2
    dr = r_range[1] - r_range[0]
    dr2 = dr * dr
    
    l_term = l * (l + 1)
    k = np.sqrt(2 * E)
    potential = np.empty_like(r_range)
    potential[0] = np.inf  # Set the potential at r=0 to a high value to avoid division by zero.
    potential[1:] = -1 / r_range[1:]

    coul_wave = np.zeros_like(r_range)
    coul_wave[0] = 1.0
    # Adjust initialization for the second point if r_range starts from 0
    coul_wave[1] = coul_wave[0] * (dr2 * (l_term / r_range2[1] + 2 * potential[1] + 2 * E) + 2)

    # Compute wave function values
    for idx in range(2, len(r_range)):
        term = dr2 * (l_term / r_range2[idx-1] + 2 * potential[idx-1] - 2 * E)
        coul_wave[idx] = coul_wave[idx-1] * (term + 2) - coul_wave[idx-2]

    # Final values and phase computation
    r_val = r_range[-2]
    coul_wave_r = coul_wave[-2]
    dcoul_wave_r = (coul_wave[-1] - coul_wave[-3]) / (2 * dr)
    
    norm = np.sqrt(np.abs(coul_wave_r)**2 + (np.abs(dcoul_wave_r) / (k + 1 / (k * r_val)))**2)
    phase = np.angle((1.j * coul_wave_r + dcoul_wave_r / (k + 1 / (k * r_val))) /
                     (2 * k * r_val)**(1.j * 1 / k)) - k * r_val + l * np.pi / 2

    coul_wave /= norm
    return phase, coul_wave

print("Computing Cont States")
start = time.time()
cont_states = {}
phases = {}

for E in E_range:
    print(E)
    for l in range(lmax+1):
        phase, coul_wave = Shooting_Method_faster(r_range, l, E)
        cont_states[(E,l)] = coul_wave
        phases[(E,l)] = phase

end = time.time()

print(f"Time to compute cont states: {end-start}")

print("Computing partial spectra")

partial_spectra = {}

for l,m in lm_dict.keys():
    print(l,m)
    spectra = []
    for E in E_range:
        block_idx = lm_dict[(l, m)]
        block = psi_final_pos[block_idx*Nr:(block_idx+1)*Nr]


        y = np.array(cont_states[(E,l)].conj()) * (block)

        inner_product = np.trapz(y, r_range)
        spectra.append(inner_product)
    partial_spectra[(l, m)] = spectra

total = 0
for key, value in partial_spectra.items():
    total += np.abs(value)**2
plt.semilogy(E_range, total)
plt.savefig("Cont_PES.png")
plt.clf()

np.save("Cont_PES.npy", total)

# Computing PAD
k_range = np.sqrt(2 * E_range)

if simInstance.laser["polarization"] == "linear":
    theta_range = np.arange(0,np.pi,0.01)
    phi_range = np.array([0,np.pi])

elif simInstance.laser["polarization"] == "elliptical":
    theta_range = np.array([np.pi/2])
    phi_range = np.arange(0,2*np.pi,0.01)

k_vals = []
theta_vals = []
phi_vals = []
pad_vals = []

for i,k in enumerate(k_range):
    print(i,len(k_range))
    E = E_range[i]
    if k == 0:
        continue
    E_idx = np.argmin(np.abs(k_range - k))
    for t in theta_range:
        for p in phi_range:

            k_vals.append(k)
            theta_vals.append(t)
            phi_vals.append(p)

            pad_amp = 0
            for key, value in partial_spectra.items():
                
                l,m = key
                pad_amp += (-1j)**l * np.exp(1j**phases[(E,l)]) * sph_harm(m, l, p, t) * value[E_idx]
                #pad_amp += (1j)**l * np.exp(-1j*phases[(E,l)]) * sph_harm(m, l, p, t) * value[E_idx]
            pad_vals.append(np.abs(pad_amp)**2)

k_vals = np.array(k_vals)
theta_vals = np.array(theta_vals)
phi_vals = np.array(phi_vals)
pad_vals = np.array(pad_vals)

px_vals = k_vals* np.sin(theta_vals) * np.cos(phi_vals)
py_vals = k_vals * np.sin(theta_vals) * np.sin(phi_vals)
pz_vals = k_vals * np.cos(theta_vals)

pad_vals = np.array(pad_vals)/ (k_vals**2)

max_mom = np.max(np.real(pad_vals))
min_mom = np.max(np.real(pad_vals))*10**-6
if simInstance.laser["polarization"] == "elliptical":
    #plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r",norm=mcolors.LogNorm(vmin=min_mom,vmax=max_mom))
    plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r")
    plt.colorbar()
    plt.savefig("Cont_PAD.png")
    plt.clf()
elif simInstance.laser["polarization"] == "linear":
    plt.scatter(pz_vals, px_vals, c=pad_vals, cmap="hot_r",norm=mcolors.LogNorm(vmin=min_mom,vmax=max_mom))
    plt.colorbar()
    plt.savefig("Cont_PAD.png")
    plt.clf()


PAD = np.vstack((k_vals,theta_vals,phi_vals,pad_vals))
np.save("Cont_PAD.npy", PAD)