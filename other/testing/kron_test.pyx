from petsc4py import PETSc
import sys

sys.path.append("../Solver")

from Module import *

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

    return C_par

def kronV6(A,B,nonzeros):
    ra,ca = A.getSize()
    rb,cb = B.getSize()

    C_par = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz = nonzeros)
    ownershipC = C_par.getOwnershipRange()
    C_range = range(ownershipC[0],ownershipC[1])

    seq_A = getLocal(A)
    seq_B = getLocal(B)

    C_seq = seq_A.kron(seq_B)

    row_pointers = [0]
    column_indices = []
    value_list = []

    for i in C_range:
        indices,values = C_seq.getRow(i)
        row_pointers.append(row_pointers[-1] + len(indices))
        column_indices.append(indices)
        value_list.append(values)
    
    column_indices = np.concatenate(column_indices)
    value_list = np.concatenate(value_list)
    
    C_par.setValuesCSR(row_pointers,column_indices,value_list)

    C_par.assemble()
        
    seq_A.destroy()
    seq_B.destroy()

    return C_par



