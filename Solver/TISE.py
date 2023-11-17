from petsc4py import PETSc
import time
import numpy as np
from slepc4py import SLEPc
import slepc4py
import json

comm = PETSc.COMM_WORLD

class tise:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.FFH_R_list = []
        self.nmax = input_par["lm"]["nmax"]
        self.lmax = input_par["lm"]["lmax"]
    def createH_l(self,basisInstance,l):
        
        n_basis = basisInstance.n_basis
        nodes = basisInstance.nodes
        weights = basisInstance.weights
        
        FFH_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)

        rowstart,rowend = FFH_R.getOwnershipRange()
        
        
        for i in range(rowstart,rowend):
            
            for j in range(n_basis):
                if i >= j:

                    H_element_1 = np.sum(weights * basisInstance.barray[:,i] * (-0.5)* basisInstance.second_barray[:,j])
                    H_element_2 = np.sum(weights * basisInstance.barray[:,i] * basisInstance.barray[:,j] * l*(l+1)/(2*np.sqrt(nodes**4 + 1E-25 )))
                    H_element_3 = np.sum(weights * basisInstance.barray[:,i] * basisInstance.barray[:,j] * (-1/np.sqrt(nodes**2 + 1E-25)))

                    
                    
                    H_element = H_element_1 + H_element_2 + H_element_3
                    

                    
                    FFH_R.setValue(i,j,H_element)
                    

                    if i != j:
                        
                        FFH_R.setValue(j,i,np.conjugate(H_element))
                        
                        
        FFH_R.assemble()


        self.FFH_R_list.append(FFH_R)
        
        return None
    def createAllH(self,basisInstance):
        for l in range(self.lmax+1):
            self.createH_l(basisInstance,l)
        return None
    def createS_R(self,basisInstance):
        n_basis = basisInstance.n_basis
        nodes = basisInstance.nodes
        weights = basisInstance.weights

        S_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm)
        rowstart,rowend = S_R.getOwnershipRange()
        for i in range(rowstart,rowend):
            for j in range(n_basis):
                
                if i >= j:
                    S_element = np.sum(weights * basisInstance.barray[:,i] * basisInstance.barray[:,j])
                    S_R.setValue(i,j,S_element)
                    
                    if i != j:
                        S_R.setValue(j,i,np.conjugate(S_element))
        S_R.assemble()
        self.S_R = S_R
        return None





        


    def solveEigensystem(self):
        
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

                eigen_vector.conjugate()
                norm = eigen_vector.dot(Sv)

                
                eigen_vector.scale(1/np.sqrt(norm))
                eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                ViewHDF5.view(eigen_vector)
                    
                
                        
                energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                energy.setValue(0,np.real(eigenvalue))
                energy.setName(f"E_{i+1+l}_{l}")
                energy.assemblyBegin()
                energy.assemblyEnd()
                ViewHDF5.view(energy)
        ViewHDF5.destroy()    
        return None