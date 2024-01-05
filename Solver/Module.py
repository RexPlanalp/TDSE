from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
import numpy as np
import gc

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


def kronV4(A,B,nonzeros):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz = nonzeros)
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    seq_A = getLocal(A)
    seq_B = getLocal(B)
    
    for i in C_range:
        A_ind = i//rb
        A_indices,A_row = seq_A.getRow(A_ind)

        B_ind = i%rb
        B_indices,B_row = seq_B.getRow(B_ind)
        
        column_indices = []
        values = []

        outer_product = np.outer(A_row, B_row)
        values = outer_product.flatten()
        column_indices = np.add.outer(A_indices * cb, B_indices).flatten()
        column_indices = column_indices.astype("int32")

        


        C.setValues(i,column_indices,values)
    C.assemble()
    return C



# Even faster, but breaks sparcity
# If I can use the sparse vals/indices of A,B to set
# the rows of C, this might be the best, but currently broken
def kronV3(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)
    C.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
    C.setUp()
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    seq_A = getLocal(A)
    seq_B = getLocal(B)
    
    for i in C_range:

        A_ind = i//rb
        indexA,A_row = seq_A.getRow(A_ind)
        A_temp = np.zeros(ca,dtype = "complex")
        A_temp[indexA] = A_row

        B_ind = i%rb
        indexB,B_row = seq_B.getRow(B_ind)
        B_temp = np.zeros(cb,dtype = "complex")
        B_temp[indexB] = B_row



        C_row = [a * b for a in A_temp for b in B_temp]
        C_indices = list(range(C.getSize()[1]))
        C.setValues(i,C_indices,C_row)

        

    C.assemblyBegin()
    C.assemblyEnd()
    seq_A.destroy()
    seq_B.destroy()
    return C

# Faster, doesn't break sparcity
def kronV2(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz = 2*(2*7+1))
    C.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
    ownershipC = C.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    seq_A = getLocal(A)
    seq_B = getLocal(B)
    
    for i in C_range:
        for j in range(C.getSize()[1]):

            if (np.abs(seq_A[i//rb,j//cb]) == 0) or (np.abs(seq_B[i%rb,j%cb]) == 0):
                continue
            
            
            value = seq_A[i//rb,j//cb] * seq_B[i%rb,j%cb]
            C.setValue(i,j,value)

    C.assemblyBegin()
    C.assemblyEnd()
   
    return C


# Slow, not sure if breaks sparcity
def kronV1(A,B):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD)
    C.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
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
        gc.collect()
        for j in range(C.getSize()[1]):
            
            value = global_A[i//rb,j//cb] * global_B[i%rb,j%cb]
            C.setValue(i,j,value)

    C.assemblyBegin()
    C.assemblyEnd()

    #for i in C_range:
        #index,val = C.getRow(i)
        #print("Row",i,val)
    
    return C


