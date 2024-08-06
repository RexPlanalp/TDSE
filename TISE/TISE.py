import numpy as np

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
        E.setTolerances(1e-5,max_it=1000)
        E.solve()
        nconv = E.getConverged()
        return E,nconv


    def solveEigensystem(self,simInstance,basisInstance,atomicInstance,EMBED):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        kron = simInstance.kron
        lmax = simInstance.lm["lmax"]
        nmax = simInstance.lm["nmax"]

        polarization = simInstance.laser["polarization"]


        if EMBED:
            H_atom = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = comm,nnz = 2*(order-1)+1)
            H_atom.assemble()


        ViewTISE = PETSc.Viewer().createHDF5(f"{atomicInstance.pot_func.__name__}.h5", mode=PETSc.Viewer.Mode.WRITE, comm= comm)
        
        K = atomicInstance.createK(simInstance,basisInstance)

        for l in range(lmax+1):
            V_l = atomicInstance.createV_l(simInstance,basisInstance,l)
            H_l = K + V_l

            l_indices = []
            for key,value in simInstance.lm_dict.items():
                if value == l:
                    l_indices.append(key)
            
            
            partial_I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 1)
            istart,iend = partial_I.getOwnershipRange()
            if l in l_indices and l in range(istart,iend):
                partial_I.setValue(l,l,1)
            comm.barrier()
            partial_I.assemble()


            partial_H = kron(partial_I,H_l,comm,2*(order-1) +1)
            H_atom.axpy(1,partial_H)
            partial_I.destroy()


            num_of_energies = nmax - l 

            if rank == 0:
                if num_of_energies > 0 and EMBED:
                    print(f"Solving and Embedding for l = {l}")
                elif EMBED:
                    print(f"Embedding for l = {l}")

            if num_of_energies > 0:

                E,nconv = self._EVSolver(H_l,atomicInstance.S,num_of_energies)
                if rank == 0:
                    print(f"l = {l}, Requested:{num_of_energies}, Converged:{nconv}")
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
        

        if EMBED:
            H_viewer = PETSc.Viewer().createBinary("TISE_files/H.bin","w")
            H_atom.view(H_viewer)
            H_viewer.destroy()
        ViewTISE.destroy()    
        return None