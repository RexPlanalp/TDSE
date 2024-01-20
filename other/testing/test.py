from petsc4py import PETSc
from mpi4py import MPI
import time

start = time.time()

# The size of the linear system
n = 8000000

# Initialize a PETSc matrix and vectors
A = PETSc.Mat().createAIJ([n, n], comm=PETSc.COMM_WORLD)
b = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)
x = PETSc.Vec().createMPI(n, comm=PETSc.COMM_WORLD)

# Set values for the matrix and vectors
A.setPreallocationNNZ(3)
for i in range(n):
    if i > 0:
        A[i, i-1] = -1.0
    if i < n - 1:
        A[i, i+1] = -1.0
    A[i, i] = 2.0
    b[i] = i + 1

# Assemble the matrix and vectors
A.assemblyBegin()
A.assemblyEnd()
b.assemblyBegin()
b.assemblyEnd()

# Create Krylov subspace method (KSP) solver and solve the system
ksp = PETSc.KSP().create()
ksp.setOperators(A)
ksp.setFromOptions()
ksp.solve(b, x)

# Print the solution vector
x.view()

# Destroy objects to clean up PETSc
A.destroy()
b.destroy()
x.destroy()
ksp.destroy()

end = time.time()
print(end-start)