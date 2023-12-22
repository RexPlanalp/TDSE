import json
from petsc4py import PETSc
import numpy as np
from Module import *


comm = PETSc.COMM_WORLD

class hamiltonian:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        self.m = input_par["state"][2]

        return None
    

    def H_MIX(self,basisInstance,gridInstance):


        n_basis = basisInstance.n_basis
        weights = basisInstance.weights
        nodes = basisInstance.nodes
        dt = gridInstance.dt

        H_mix_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = comm,nnz = 2)
        H_mix_lm.setUp()

        istart,iend = H_mix_lm.getOwnershipRange()
        
        for i in range(istart,iend):
            for j in range(self.lmax+1):
                
                def clm(index):
                    return np.sqrt(((index+1)**2 - self.m**2)/((2*index+1)*(2*index+3))) * 1j * dt /2

                
                if i == j+1:
                    H_mix_lm.setValue(i,j,clm(i-1))
                elif j == i+1:
                    H_mix_lm.setValue(i,j,clm(j-1))
        H_mix_lm.assemble()

        H_mix_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm)
        istart,iend = H_mix_R.getOwnershipRange()
        
        for i in range(istart,iend):
            for j in range(n_basis):
                    #if i>=j:
                        H_element = np.sum(weights * basisInstance.barray[:,i] *  basisInstance.first_barray[:,j])
                        H_mix_R.setValue(i,j,H_element)
                        #if i!=j:
                            #H_mix_R.setValue(j,i,H_element)

        H_mix_R.assemble()

        total = kronV3(H_mix_lm,H_mix_R)
        total.scale(-1j)

        H_mix_lm.destroy()
        H_mix_R.destroy()
        self.H_mix = total
        return None
    

    def H_ANG(self,basisInstance,gridInstance):

        n_basis = basisInstance.n_basis
        weights = basisInstance.weights
        nodes = basisInstance.nodes
        dt = gridInstance.dt


        H_ang_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = comm,nnz = 2)
        istart,iend = H_ang_lm.getOwnershipRange()
        
        for i in range(istart,iend):
            for j in range(self.lmax+1):

                def clm(index):
                    return np.sqrt(((index+1)**2 - self.m**2)/((2*index+1)*(2*index+3))) * 1j * dt /2
                if j == i+1:
                    H_ang_lm.setValue(i,j,(i)*clm(j-1))
                elif i == j+1:
                    H_ang_lm.setValue(i,j,-(j)*clm(i-1))
        H_ang_lm.assemble()

        H_ang_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm)
        istart,iend = H_ang_R.getOwnershipRange()
        
        for i in range(istart,iend):
            for j in range(n_basis):

                #if i >= j:
                    H_element = np.sum(weights * basisInstance.barray[:,i] *  basisInstance.barray[:,j] / (nodes))
                    H_ang_R.setValue(i,j,H_element)

                    #if i != j:
                        #H_ang_R.setValue(j,i,H_element)
        H_ang_R.assemble()

        total = kronV3(H_ang_lm,H_ang_R)
        total.scale(-1j)

        H_ang_lm.destroy()
        H_ang_R.destroy()

        self.H_ang = total
        return None  
    

    def H_ATOM(self,tiseInstance,basisInstance,gridInstance):
        H_list = tiseInstance.FFH_R_list
        n_basis = basisInstance.n_basis

        H_atom = PETSc.Mat().createAIJ([(self.lmax +1)*n_basis,(self.lmax +1)*n_basis],comm = comm)
        H_atom.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)

        local_H = []
        for l in range(self.lmax+1):
            local_H.append(getLocal(H_list[l]))

        istart,iend = H_atom.getOwnershipRange()
        
        for i in range(istart,iend):
            l = i // n_basis
            l_row = i % n_basis

            index,vals = local_H[l].getRow(l_row)

            full_row = np.zeros(n_basis,dtype = "complex")
            full_row[index] = vals
            row_array = np.pad(full_row,(l*n_basis,(self.lmax-l)*n_basis),constant_values= (0,0))


            
            for j in range((self.lmax +1)*n_basis):
                row_element = row_array[j] 
                H_atom.setValue(i,j,row_element)

        H_atom.assemble()
        self.H_atom = H_atom
        return None
    



    def S(self,tiseInstance,basisInstance):
        n_basis = basisInstance.n_basis
        S_R = tiseInstance.S_R

        I = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        I.assemble()

        total = kronV3(I,S_R)

        I.destroy()
        self.S = total

        viewer = PETSc.Viewer().createBinary("overlap.bin","w")
        
        total.view(viewer)
        viewer.destroy()

        return None
    

    def PartialAtomic(self,gridInstance):
        dt = gridInstance.dt
        
        S_copy_L = self.S.copy()
        S_copy_R = self.S.copy()
        
        S_copy_L.axpy(1j*dt/2,self.H_atom) 
        S_copy_R.axpy(-1j*dt/2,self.H_atom) 


        self.partial_L = S_copy_L
        self.partial_R = S_copy_R

        return None
    

    def PartialAngular(self):
        H_mix_copy = self.H_mix.copy()
        H_mix_copy.axpy(1,self.H_ang)

        self.partial_angular = H_mix_copy
        return None