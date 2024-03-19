
from petsc4py import PETSc
import petsc4py
comm = PETSc.COMM_WORLD
import gc

class propagator:
    def __init__(self,tol):
        self.tol = tol
        
    def propagateCN(self,gridInstance,psiInstance,laserInstance,hamiltonianInstance):
        t = gridInstance.t
        dt = gridInstance.dt
        L = len(t)
        psi_initial = psiInstance.psi_initial.copy()

        ksp = PETSc.KSP().create(comm = comm)

        
        H_int_1 = hamiltonianInstance.H_int_1
        H_int_2 = hamiltonianInstance.H_int_2

        ksp.setTolerances(rtol = self.tol)
        

        for i,t_i in enumerate(t):
            if PETSc.COMM_WORLD.rank == 0:
                print(i,L-1)


            


            A = laserInstance.A_funcX(t_i+dt/2) + 1j * laserInstance.A_funcY(t_i+dt/2)
            Atilde = laserInstance.A_funcX(t_i+dt/2) - 1j * laserInstance.A_funcY(t_i+dt/2)
            
            partial_L_copy = hamiltonianInstance.partial_L.copy()
            partial_R_copy = hamiltonianInstance.partial_R.copy()
            
            partial_L_copy.axpy(Atilde,H_int_1,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_L_copy.axpy(A,H_int_2,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

            
            partial_R_copy.axpy(-Atilde,H_int_1,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_R_copy.axpy(-A,H_int_2,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            
            
            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()

            partial_R_copy.mult(psi_initial,known)

            ksp.setOperators(partial_L_copy)

            
            
            


            ksp.solve(known,solution)
            solution.copy(psi_initial)

            

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()
        psiInstance.createFinal(psi_initial)
        ViewHDF5 = PETSc.Viewer().createHDF5("TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        psi_initial.setName("psi_final")
        ViewHDF5.view(psi_initial)
        ViewHDF5.destroy()

        return None