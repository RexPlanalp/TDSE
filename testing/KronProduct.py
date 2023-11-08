import numpy as np
import json
import time
import matplotlib.pyplot as plt
import h5py
import os
from scipy.interpolate import BSpline
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import kron
import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

from numpy import pi


A = PETSc.Mat().createAIJ([2,2],comm = PETSc.COMM_WORLD)
rowstart,rowend = A.getOwnershipRange()
rows,cols = A.getSize()
for i in range(rowstart,rowend):
    for j in range(cols):
        A.setValue(i,j,2)
A.assemblyBegin()
A.assemblyEnd()

B = PETSc.Mat().createAIJ([2,2],comm = PETSc.COMM_WORLD)
rowstart,rowend = B.getOwnershipRange()
rows,cols = B.getSize()
for i in range(rowstart,rowend):
    for j in range(cols):
        B.setValue(i,j,2)
B.assemblyBegin()
B.assemblyEnd()

def kron(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)

    