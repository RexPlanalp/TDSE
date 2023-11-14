import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import sys
import h5py
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt




with h5py.File('TDSE.h5', 'r') as f:
    data = f["psi_final"][:]

    real_part = data[:,0]
    imaginary_part = data[:,1]
    total = real_part + 1j*imaginary_part

with h5py.File('Overlap.h5', 'r') as f:
    matrix = f["overlap_matrix"][...]
    

