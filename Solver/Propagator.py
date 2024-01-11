
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

        
        pulseFunc = laserInstance.pulse_func
        

        ksp.setTolerances(rtol = self.tol)
        

        for i,t in enumerate(t):
            if PETSc.COMM_WORLD.rank == 0:
                print(i,L-1)


            t_mid = t + dt /2





            pulse_val = pulseFunc(t_mid)

            partial_L_copy = hamiltonianInstance.partial_L.copy()
            partial_R_copy = hamiltonianInstance.partial_R.copy()
            partial_angular = hamiltonianInstance.partial_angular
            partial_L_copy.axpy(pulse_val,partial_angular,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_R_copy.axpy(-pulse_val,partial_angular,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

            
            
            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()

            partial_R_copy.mult(psi_initial,known)




            ksp.setOperators(partial_L_copy)

            
            #ksp.setType(PETSc.KSP.Type.BICG)
            #pc = ksp.getPC()
            #pc.setType(PETSc.PC.Type.BJACOBI)
            
            


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