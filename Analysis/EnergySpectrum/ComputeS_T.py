from petsc4py import PETSc
import json
from Module import *


with open('input.json', 'r') as file:
    input_par = json.load(file)

order = input_par["splines"]["order"]
N_knots = input_par["splines"]["N_knots"]
lmax = input_par["lm"]["lmax"]
n_basis = N_knots - order - 2


S = PETSc.Mat().createAIJ([n_basis*(lmax+1),n_basis*(lmax+1)],nnz =(2*order + 1) )
viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
S.load(viewer)
viewer.destroy()

S_R = PETSc.Mat().createAIJ([n_basis,n_basis],nnz =(2*order + 1) )
for i in range(n_basis):
    for j in range(n_basis):
        value = S.getValue(i,j)
        if value == 0:
            continue
        S_R.setValue(i, j, value)
S_R.assemble()

ones = PETSc.Mat().createAIJ([lmax+1,lmax+1])
istart,iend = ones.getOwnershipRange()
for i in range(istart,iend):
    for j in range(lmax+1):
        ones.setValue(i,j,1)
ones.assemble()

S_T= kronV4(ones,S_R,n_basis*(2*order + 1))

ones.destroy()
S_R.destroy()


viewer = PETSc.Viewer().createBinary("matrix_files/S_T.bin","w")
S_T.view(viewer)
viewer.destroy()

print("Done")