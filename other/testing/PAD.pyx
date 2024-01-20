import h5py
import numpy as np
from scipy.special import gamma
import mpmath as mp
from scipy.integrate import trapz
from scipy.special import sph_harm
from scipy.integrate import fixed_quad
import sys
import json
import multiprocessing as multi
import time
from mpi4py import MPI

with open('input.json', 'r') as file:
    input_par = json.load(file)
lmax = input_par["lm"]["lmax"]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Reading Wavefunction")
with h5py.File('TDSE.h5', 'r') as f:
    data = f["psi_final"][:]
    real_part = data[:,0]
    imaginary_part = data[:,1]
    wavefunction = real_part + 1j*imaginary_part
if rank == 0:
    print("Importing Modules")
module_directory = "../Solver"
if module_directory not in sys.path:
    sys.path.append(module_directory)
from Basis import *
from Grid import *

def findCoulombPhase(l,k):
    return gamma(l+1-1j/k)

def findCoulombF(r,k,l):
    F = []
    for r_val in r:
        F_val = mp.coulombf(l,-1/k,k*r_val)
        F.append(r_val)
    return np.array(F)
    


def findCoulombG(r,k,l):
    return mp.coulombg(l,-1/k,k*r)

def I(k,i,l):
    print(i,l)
    def integrand(r,i):
        return findCoulombF(r,k,l)*basisInstance.basis_funcs[i](r)
    return complex(fixed_quad(integrand,basisInstance.knots[i+1],basisInstance.knots[i+basisInstance.order+1],n = 100,args=(i,))[0])

def find_closest(k_array, p):
    """
    Find the value in k_array closest to p.

    Parameters:
    k_array (np.array): An array of values.
    p (float): The value to find the closest to in k_array.

    Returns:
    float: The value in k_array closest to p.
    """
    # Compute the absolute difference
    differences = np.abs(k_array - p)

    # Find the index of the minimum difference
    index_of_min = np.argmin(differences)

    # Return the closest value
    return k_array[index_of_min]

if rank == 0:
    print("Constructing Grid and Basis")
gridInstance = grid()
basisInstance = basis()
basisInstance.createBasis(gridInstance)
basisInstance.saveBasis(gridInstance,save = False,plot = False)
basis_array = np.load("../Sample/basis/basis.npy")
Nr,n_basis = np.shape(basis_array)

lmax = 2
n_basis = 10

start = time.time()
if rank == 0:
    print("Computing Coefficients")
#k_range = np.arange(0.001,1,0.001)
k = 1
start = time.time()

def worker(args):
    k, n, l = args
    return I(k, n, l)

def parallel_execution(k, lmax, n_basis):
    # Create a list of arguments for each function call
    args_list = [(k, n, l) for l in range(lmax + 1) for n in range(n_basis)]

    # Create a multiprocessing pool
    with multi.Pool(32) as pool:
        results = pool.map(worker, args_list)

    return results
I_list = parallel_execution(k, lmax, n_basis)
end = time.time()

if rank == 0:
    print(end-start)










