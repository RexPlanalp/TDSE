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


A = PETSc.Mat().createAIJ([50,50],comm = PETSc.COMM_WORLD)
rowstart,rowend = A.getOwnershipRange()
rows,cols = A.getSize()
for i in range(rowstart,rowend):
    for j in range(cols):
        A.setValue(i,j,1)
A.assemblyBegin()
A.assemblyEnd()

#B = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_WORLD)
# Set the values for the part of the matrix that this processor owns
#if PETSc.COMM_WORLD.rank == 0:
    #B.setValue(0, 0, 6)
    #B.setValue(0, 1, 2)
#elif PETSc.COMM_WORLD.rank == 1:
    #B.setValue(1, 0, 1)
    #B.setValue(1, 1, 4)

# Assemble the matrix
#B.assemblyBegin()
#B.assemblyEnd()

B = PETSc.Mat().createAIJ([50,50],comm = PETSc.COMM_WORLD)
rowstart,rowend = B.getOwnershipRange()
rows,cols = B.getSize()
for i in range(rowstart,rowend):
    for j in range(cols):
        B.setValue(i,j,2)
B.assemblyBegin()
B.assemblyEnd()




import time


if PETSc.COMM_WORLD.rank == 0:
    start = time.time()
def kron(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    from scipy.sparse import lil_matrix

    ownershipA = A.getOwnershipRange()
    A_range = range(ownershipA[0],ownershipA[1])
    A_lil = lil_matrix((A.getSize()),dtype = "complex")
    for i in A_range:
        for j in range(ca):
            A_lil[i,j] = A.getValue(i,j)
          

    ownershipB = B.getOwnershipRange()
    B_range = range(ownershipB[0],ownershipB[1])
    B_lil = lil_matrix((B.getSize()),dtype = "complex")
    for i in B_range:
        for j in range(cb):
            B_lil[i,j] = B.getValue(i,j)
    
    
    A_array = A_lil.toarray()
    B_array = B_lil.toarray()

    



    global_A = np.zeros_like(A_array)
    global_B = np.zeros_like(B_array)
    MPI.COMM_WORLD.Allreduce(A_array, global_A, op=MPI.SUM)
    MPI.COMM_WORLD.Allreduce(B_array, global_B, op=MPI.SUM)

   

    
    

    for i in C_range:
        for j in range(C.getSize()[1]):
            
            value = global_A[i//rb,j//cb] * global_B[i%rb,j%cb]
            C.setValue(i,j,value)

    C.assemblyBegin()
    C.assemblyEnd()

    #for i in C_range:
        #index,val = C.getRow(i)
        #print("Row",i,val)
    return C
    
def kron(A,B):
    C = PETSc.Mat().create()
    C.kron(A,B)
    return C

#idx = [0]  # The indices you want in your index set
#iset = PETSc.IS().createGeneral(idx, comm=PETSc.COMM_WORLD)



kron(A,B)
if PETSc.COMM_WORLD.rank == 0:

    end = time.time()
    print(end-start)


      



        
            





    