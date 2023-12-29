from petsc4py import PETSc
import time
import numpy as np
from slepc4py import SLEPc
import slepc4py
import json
from scipy.integrate import trapz
import os
comm = PETSc.COMM_WORLD

class tise:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.FFH_R_list = []
        self.nmax = input_par["lm"]["nmax"]
        self.lmax = input_par["lm"]["lmax"]

    


    def _createH_l(self,basisInstance,l):
        def _H_element_1(x,i,j):
            return basis_funcs[i](x) * (-1/2) * basis_funcs[j](x,2)
        def _H_element_2(x,i,j):
            return basis_funcs[i](x) * basis_funcs[j](x) * l*(l+1)/(2*np.sqrt(x**4 + 1E-25 ))
        def _H_element_3(x,i,j):
            return basis_funcs[i](x) * basis_funcs[j](x)* (-1/np.sqrt(x**2 + 1E-25))
        
        n_basis = basisInstance.n_basis
        basis_funcs = basisInstance.basis_funcs
        degree = basisInstance.degree

        
        FFH_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*degree +1)
        rowstart,rowend = FFH_R.getOwnershipRange()
        for i in range(rowstart,rowend):
            for j in range(n_basis):
                    H_1 = basisInstance.integrate(_H_element_1,i,j)
                    H_2 = basisInstance.integrate(_H_element_2,i,j)
                    H_3 = basisInstance.integrate(_H_element_3,i,j)
                    H_element = H_1 + H_2 + H_3
                    FFH_R.setValue(i,j,H_element)      
        FFH_R.assemble()

        self.FFH_R_list.append(FFH_R)
        
        return None
    
    def createAllH(self,basisInstance):
        for l in range(self.lmax+1):
            self._createH_l(basisInstance,l)
        return None
    
    def createS_R(self,basisInstance):
        n_basis = basisInstance.n_basis
        basis_funcs = basisInstance.basis_funcs
        degree = basisInstance.degree

        def S_element(x,i,j):
            return basis_funcs[i](x) * basis_funcs[j](x)

        S_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*degree+1)
        rowstart,rowend = S_R.getOwnershipRange()
        for i in range(rowstart,rowend):
            for j in range(n_basis):
                
                
                    S_1 = basisInstance.integrate(S_element,i,j)
                    S_R.setValue(i,j,S_1)    
        S_R.assemble()
        self.S_R = S_R
        return None

    def solveEigensystem(self):
        if os.path.exists("Hydrogen.h5"):
            return
        
        n_basis,_ = self.S_R.getSize()
        ViewHDF5 = PETSc.Viewer().createHDF5("Hydrogen.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
            
        if (self.lmax >= self.nmax):
            self.nmax = self.lmax +1
        

        for i,l in enumerate(range(self.nmax)):
            
            H = self.FFH_R_list[i]

            E = SLEPc.EPS().create()
            E.setOperators(H, self.S_R)
            
            num_of_energies = self.nmax - i
            E.setDimensions(nev=num_of_energies)
            
            E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
            E.setType(slepc4py.SLEPc.EPS.Type.KRYLOVSCHUR)

            
            E.solve()

            nconv = E.getConverged()
            

            
            for i in range(nconv):
                eigenvalue = E.getEigenvalue(i)  # This retrieves the eigenvalue
                
                if np.real(eigenvalue) > 0:
                    continue

                # Creating separate vectors for the real part of the eigenvector
                eigen_vector = H.getVecLeft()  # Assuming H is the correct operator for the matrix H
                E.getEigenvector(i, eigen_vector)  # This retrieves the eigenvector
                        
                
                
                Sv = self.S_R.createVecRight()
                self.S_R.mult(eigen_vector, Sv)



                norm = eigen_vector.dot(Sv)

                
                eigen_vector.scale(1/np.sqrt(norm))
                eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                ViewHDF5.view(eigen_vector)
                    
                ##############
                '''
                Sv = H.createVecRight()
                H.mult(eigen_vector,Sv)

                Su = self.S_R.createVecRight()
                self.S_R.mult(eigen_vector,Su)

                if comm.rank == 0:
                    print(eigen_vector.getValue(0))
                    print(Sv.getValue(0))
                    print(Su.getValue(0)*eigenvalue)
                    print(eigenvalue)
                '''
                ##############

                        
                energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                energy.setValue(0,np.real(eigenvalue))
                energy.setName(f"E_{i+1+l}_{l}")
                energy.assemblyBegin()
                energy.assemblyEnd()
                ViewHDF5.view(energy)
        ViewHDF5.destroy()    
        return None