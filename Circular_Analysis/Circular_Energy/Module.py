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

def kronV6(A,B,nonzeros):
    seq_A = getLocal(A)
    seq_B = getLocal(B)

    if PETSc.COMM_WORLD.rank == 0:
        C_seq = seq_A.kron(seq_B)
        seq_A.destroy()
        seq_B.destroy()
        viewer = PETSc.Viewer().createBinary("matrix_files/temp.bin","w",comm = PETSc.COMM_SELF)
        #viewer.view(C_seq)
        C_seq.view(viewer)
        viewer.destroy()
    else:
        seq_A.destroy()
        seq_B.destroy()

    PETSc.COMM_WORLD.barrier()
    
    
    C_par = PETSc.Mat(comm = PETSc.COMM_WORLD)

    viewer = PETSc.Viewer().createBinary('matrix_files/temp.bin', 'r',comm = PETSc.COMM_WORLD)
    C_par.load(viewer)
    viewer.destroy()

    return C_par



def kronV5(A,B,nonzeros):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C_par = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz = nonzeros)
    ownershipC = C_par.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    seq_A = getLocal(A)
    seq_B = getLocal(B)

    C_seq = seq_A.kron(seq_B)
    
    
    for i in C_range:
        indices,values = C_seq.getRow(i)
        C_par.setValues(i,indices,values)
   
    
    C_par.assemble()

    seq_A.destroy()
    seq_B.destroy()
    C_seq.destroy()

    return C_par

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

def lm_to_block(lmax):
    def block_number(l, m):
        sum_blocks = sum(2*i + 1 for i in range(l))
        m_offset = m + l
        return sum_blocks + m_offset
    lm_dict = {}
    for l in range(lmax+1):
        for m in range(-l,l+1):
            lm_dict[(l,m)] = block_number(l,m)
    return lm_dict
         
def calc_n_block(lmax):
    total = 0
    for l in range(lmax+1):
        manifold = 2*l+1
        total += manifold
    return total