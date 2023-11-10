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

if True:

    A = PETSc.Mat().createAIJ([2,2],comm = PETSc.COMM_WORLD)
    rowstart,rowend = A.getOwnershipRange()
    rows,cols = A.getSize()
    for i in range(rowstart,rowend):
        for j in range(cols):
            A.setValue(i,j,1)
    A.assemblyBegin()
    A.assemblyEnd()

    B = PETSc.Mat().createAIJ([2, 2], comm=PETSc.COMM_WORLD)
    # Set the values for the part of the matrix that this processor owns
    if PETSc.COMM_WORLD.rank == 0:
        B.setValue(0, 0, 2)
        B.setValue(0, 1, 3)
    elif PETSc.COMM_WORLD.rank == 1:
        B.setValue(1, 0, 3)
        B.setValue(1, 1, 2)

    # Assemble the matrix
    B.assemblyBegin()
    B.assemblyEnd()

    #B = PETSc.Mat().createAIJ([128,128],comm = PETSc.COMM_WORLD)
    #rowstart,rowend = B.getOwnershipRange()
    #rows,cols = B.getSize()
    #for i in range(rowstart,rowend):
        #for j in range(cols):
           # B.setValue(i,j,2)
    #B.assemblyBegin()
    #B.assemblyEnd()




import time

import kron

def kronV1(A,B):
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
    
def kronV2(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    
    
    def gather_csr(local_csr_part):
  
        gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_csr_part)
        
        return np.concatenate(gathered)
    def gather_indpr(local_indptr):
        gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_indptr)
        global_indptr = list(gathered[0])
        offset = global_indptr[-1]  # Start with the last element of the first indptr
        for proc_indptr in gathered[1:]:
            # Offset the local indptr (excluding the first element) and extend the global indptr
            global_indptr.extend(proc_indptr[1:] + offset)
            # Update the offset for the next iteration
            offset += proc_indptr[-1] - proc_indptr[0]  # Adjust for the overlapping indices
        return global_indptr
    def getLocal(M):
        local_csr = M.getValuesCSR()
        local_indptr, local_indices, local_data = local_csr
        global_indices = gather_csr(local_indices).astype(np.int32)
        global_data = gather_csr(local_data)
        global_indptr = gather_indpr(local_indptr)
        seq_M = PETSc.Mat().createAIJWithArrays([M.getSize()[0],M.getSize()[1]],(global_indptr,global_indices,global_data),comm = PETSc.COMM_SELF)
        return seq_M
    
    if PETSc.COMM_WORLD.rank == 0:
        start = time.time()
    seq_A = getLocal(A)
    seq_B = getLocal(B)
    if PETSc.COMM_WORLD.rank == 0:
        print(time.time()-start)
    for i in C_range:
        for j in range(C.getSize()[1]):
            
            value = seq_A[i//rb,j//cb] * seq_B[i%rb,j%cb]
            C.setValue(i,j,value)

    C.assemblyBegin()
    C.assemblyEnd()
   
    return C

 
   

    
    

    #for i in C_range:
        #for j in range(C.getSize()[1]):
            
            #value = global_A[i//rb,j//cb] * global_B[i%rb,j%cb]
            #C.setValue(i,j,value)

    #C.assemblyBegin()
    #C.assemblyEnd()

   
    print("DONE")
    return C


#if PETSc.COMM_WORLD.rank == 0:
    start = time.time()

def kronV3(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    
    
    def gather_csr(local_csr_part):
  
        gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_csr_part)
        
        return np.concatenate(gathered)
    def gather_indpr(local_indptr):
        gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_indptr)
        global_indptr = list(gathered[0])
        offset = global_indptr[-1]  # Start with the last element of the first indptr
        for proc_indptr in gathered[1:]:
            # Offset the local indptr (excluding the first element) and extend the global indptr
            global_indptr.extend(proc_indptr[1:] + offset)
            # Update the offset for the next iteration
            offset += proc_indptr[-1] - proc_indptr[0]  # Adjust for the overlapping indices
        return global_indptr
    def getLocal(M):
        local_csr = M.getValuesCSR()
        local_indptr, local_indices, local_data = local_csr
        global_indices = gather_csr(local_indices).astype(np.int32)
        global_data = gather_csr(local_data)
        global_indptr = gather_indpr(local_indptr)
        seq_M = PETSc.Mat().createAIJWithArrays([M.getSize()[0],M.getSize()[1]],(global_indptr,global_indices,global_data),comm = PETSc.COMM_SELF)
        return seq_M
    
   
    seq_A = getLocal(A)
    seq_B = getLocal(B)
    
    for i in C_range:
        A_ind = i//rb
        index,A_row = seq_A.getRow(A_ind)

        B_ind = i%rb
        index,B_row = seq_B.getRow(B_ind)
        C_row = [a * b for a in A_row for b in B_row]
        C_indices = list(range(C.getSize()[1]))
        C.setValues(i,C_indices,C_row)

    C.assemblyBegin()
    C.assemblyEnd()
   
    return C




if PETSc.COMM_WORLD.rank == 0:
    start = time.time()
C = kron.kronV1(A,B)

if PETSc.COMM_WORLD.rank == 0:
    print("Version 1",time.time()-start)

if PETSc.COMM_WORLD.rank == 0:
    start = time.time()
C = kron.kronV2(A,B)

if PETSc.COMM_WORLD.rank == 0:
    print("Version 2",time.time()-start)


if PETSc.COMM_WORLD.rank == 0:
    start = time.time()
C = kron.kronV3(A,B)

if PETSc.COMM_WORLD.rank == 0:
    print("Version 3",time.time()-start)




      



        
            





    