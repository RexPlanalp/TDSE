import kron
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np
import gc
comm = PETSc.COMM_WORLD
import time

A = PETSc.Mat().createAIJ([10,10],comm = comm)
istart,iend = A.getOwnershipRange()
rows,cols = A.getSize()
for i in range(istart,iend):
    for j in range(cols):
        if i == j:
            A.setValue(i,j,1)   
A.assemble()

B = PETSc.Mat().createAIJ([10,10],comm = comm)
istart,iend = B.getOwnershipRange()
rows,cols = B.getSize()
for i in range(istart,iend):
    for j in range(cols):
        
        B.setValue(i,j,2)   
B.assemble()

if PETSc.COMM_WORLD.rank == 0:
    start = time.time()

C = kron.kronV3(A,B)

if PETSc.COMM_WORLD.rank == 0:
    print(time.time()-start)

#istart,iend = C.getOwnershipRange()

#for i in range(istart,iend):
    #index,value = C.getRow(i)
    #print(len(value))


