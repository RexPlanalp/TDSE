import numpy as np
import json
import sys

from petsc4py import PETSc
from slepc4py import SLEPc

comm = PETSc.COMM_WORLD
rank = comm.rank

class tise:
    def __init__(self):
        None

    

    
    
    def _EVSolver(self,H,S,num_of_energies):
        E = SLEPc.EPS().create()
        E.setOperators(H,S)
        E.setDimensions(nev=num_of_energies)
        E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        E.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
        E.solve()
        nconv = E.getConverged()
        return E,nconv

    
    def solveEigensystem(self,simInstance,basisInstance,atomicInstance,EMBED,SOLVE):
        ViewTISE = PETSc.Viewer().createHDF5(f"{atomicInstance.pot_func.__name__}.h5", mode=PETSc.Viewer.Mode.WRITE, comm= comm)

        K = atomicInstance.K

        for l in range(simInstance.lm["lmax"]+1):

            if rank == 0:
                print(f"Working on subsysem for l = {l}")

            atomicInstance.createV_l(simInstance,basisInstance,l)
            V =  atomicInstance.V
            H_l = K + V

            if EMBED:
                atomicInstance.embedH_l(simInstance,H_l,l)

            if SOLVE:
                num_of_energies = simInstance.lm["nmax"] - l +1
                if num_of_energies > 0:
                
          

                    E,nconv = self._EVSolver(H_l,atomicInstance.S,num_of_energies)

                    for i in range(nconv):
                        eigenvalue = E.getEigenvalue(i) 
                        if np.real(eigenvalue) > 0:
                            continue
                        eigen_vector = H_l.getVecLeft()  
                        E.getEigenvector(i, eigen_vector)  
                            
                        Sv = atomicInstance.S.createVecRight()
                        atomicInstance.S.mult(eigen_vector, Sv)
                        norm = eigen_vector.dot(Sv)

                        eigen_vector.scale(1/np.sqrt(norm))
                        eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                        ViewTISE.view(eigen_vector)
                        
                        energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                        energy.setValue(0,np.real(eigenvalue))
                        energy.setName(f"E_{i+1+l}_{l}")
                        energy.assemblyBegin()
                        energy.assemblyEnd()
                        ViewTISE.view(energy)
                    H_l.destroy()

            ViewTISE.destroy() 

        if EMBED:
            atomicInstance.saveH_atom()   
        return None