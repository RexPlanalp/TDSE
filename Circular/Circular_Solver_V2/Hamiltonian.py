import json
from petsc4py import PETSc
import numpy as np
from Module import *
import os


comm = PETSc.COMM_WORLD

class hamiltonian:
    def __init__(self,n_blocks,lm_dict):
        self.n_blocks = n_blocks
        self.lm_dict = lm_dict
        self.block_dict = {value: key for key, value in lm_dict.items()}
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        return None
    
    def a(self,l,m):
        f1 = np.sqrt((l+m)/((2*l+1)*(2*l-1)))
        f2 = -m*np.sqrt(l+m-1) - np.sqrt((l-m)*((l-1)-m*(m-1)))
        return f1*f2
       
    def b(self,l,m):
        f1 = np.sqrt((l-m+1)/((2*l+1)*(2*l+3)))
        f2 = m*np.sqrt(l-m+2) - np.sqrt((l+m+1)*((l+1)*(l+2)-m*(m-1)))
        return f1*f2

    def atilde(self,l,m):
        f1 = np.sqrt((l-m)/((2*l+1)*(2*l-1)))
        f2 = -m*np.sqrt(l-m-1) + np.sqrt((l+m)*((l-1)-m*(m+1)))
        return f1*f2

    def btilde(self,l,m):
        f1 = np.sqrt((l+m+1)/((2*l+1)*(2*l+3)))
        f2 = m*np.sqrt(l+m+2) + np.sqrt((l-m+1)*((l+1)*(l+2)-m*(m+1)))
        return f1*f2

    def c(self,l,m):
        f = np.sqrt((l+m)*(l+m-1)/((2*l+1)*(2*l-1)))
        return f
    
    def d(self,l,m):
        f = np.sqrt((l-m+1)*(l-m+2)/((2*l+1)*(2*l+3)))
        return f

    def ctilde(self,l,m):
        f = np.sqrt((l-m)*(l-m-1)/((2*l+1)*(2*l-1)))
        return f

    def dtilde(self,l,m):
        f = np.sqrt((l+m+1)*(l+m+2)/((2*l+1)*(2*l+3)))
        return f


    def H_ANG_R(self,basisInstance,gridInstance):
        n_basis = self.n_basis
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        def H_ang_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x)/ np.sqrt(x**2 + 1E-25)

        H_ang_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_ang_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_ang_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_ang_R_element,i,j)
                    H_ang_R.setValue(i,j,H_element)
        H_ang_R.assemble()

        self.H_ang_R = H_ang_R
        
    def H_ANG_LM_ONE(self,basisInstance,gridInstance):
        n_blocks = self.n_blocks
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        H_ang_lm_one = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_ang_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (l == lprime+1) and (m == mprime+1):
                    H_ang_lm_one.setValue(i,j,self.a(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_ang_lm_one.setValue(i,j,self.b(l,m))
        H_ang_lm_one.assemble()

        self.H_ang_lm_one = H_ang_lm_one

    def H_ANG_LM_TWO(self,basisInstance,gridInstance):
        n_blocks = self.n_blocks
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        H_ang_lm_two = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_ang_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (l == lprime+1) and (m == mprime-1):
                    H_ang_lm_two.setValue(i,j,self.atilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_ang_lm_two.setValue(i,j,self.btilde(l,m))
        H_ang_lm_two.assemble()

        self.H_ang_lm_two = H_ang_lm_two    

    def H_ANG_ONE(self,basisInstance):
        order = basisInstance.order
        H_ang_one = kronV4(self.H_ang_lm_one,self.H_ang_R,4*(2*order+1))
        H_ang_one.scale(1j/2)
        self.H_ang_lm_one.destroy()
        self.H_ang_one = H_ang_one
    
    def H_ANG_TWO(self,basisInstance):
        order = basisInstance.order
        H_ang_two = kronV4(self.H_ang_lm_two,self.H_ang_R,4*(2*order+1)) 
        H_ang_two.scale(1j/2)
        self.H_ang_lm_two.destroy()
        self.H_ang_R.destroy()
        self.H_ang_two = H_ang_two       

    def H_MIX_R(self,basisInstance,gridInstance):
        n_basis = self.n_basis
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        def H_mix_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x,1)

        H_mix_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_mix_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_mix_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_mix_R_element,i,j)
                    H_mix_R.setValue(i,j,H_element)
        H_mix_R.assemble()

        self.H_mix_R = H_mix_R

    def H_MIX_LM_ONE(self,basisInstance,gridInstance):
        n_blocks = self.n_blocks
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        H_mix_lm_one = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_mix_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (l == lprime+1) and (m == mprime+1):
                    H_mix_lm_one.setValue(i,j,self.c(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_mix_lm_one.setValue(i,j,-self.d(l,m))
        H_mix_lm_one.assemble()

        self.H_mix_lm_one = H_mix_lm_one
    
    def H_MIX_LM_TWO(self,basisInstance,gridInstance):
        n_blocks = self.n_blocks
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        H_mix_lm_two = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_mix_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (l == lprime+1) and (m == mprime-1):
                    H_mix_lm_two.setValue(i,j,-self.ctilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_mix_lm_two.setValue(i,j,self.dtilde(l,m))
        H_mix_lm_two.assemble()

        self.H_mix_lm_two = H_mix_lm_two    

    def H_MIX_ONE(self,basisInstance):
        order= basisInstance.order
        H_mix_one = kronV4(self.H_mix_lm_one,self.H_mix_R,4*(2*order+1))
        H_mix_one.scale(1j/2)
        self.H_mix_lm_one.destroy()
        self.H_mix_one = H_mix_one

    def H_MIX_TWO(self,basisInstance):
        order = basisInstance.order
        H_mix_two = kronV4(self.H_mix_lm_two,self.H_mix_R,4*(2*order+1))
        H_mix_two.scale(1j/2)
        self.H_mix_lm_two.destroy()
        self.H_mix_R.destroy()
        self.H_mix_two = H_mix_two

    def H_ATOM(self,tiseInstance,basisInstance,gridInstance):
        if os.path.exists('matrix_files/H_0.bin'):
            n_basis = basisInstance.n_basis
            order = basisInstance.order
            H_atom = PETSc.Mat().createAIJ([self.n_blocks*n_basis,self.n_blocks*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*order+1)
            viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
            H_atom.load(viewer)
            viewer.destroy()
            self.H_atom = H_atom
            return
        H_list = tiseInstance.FFH_R_list
        local_H = []
        for l in range(self.lmax+1):
            local_H.append(getLocal(H_list[l]))

        n_basis = basisInstance.n_basis
        order = basisInstance.order
        
        ra,ca = self.n_blocks,self.n_blocks
        rb,cb = n_basis,n_basis

        H_atom = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz = 2*order+1)
        ownershipH = H_atom.getOwnershipRange()
        H_range = range(ownershipH[0],ownershipH[1])

        
        for i in H_range:

            
            A_ind = i//rb
            A_indices,A_row = np.array(A_ind),np.array(1)

            l,m = self.block_dict[A_ind]

            B = local_H[l]
            B_ind = i%rb
            B_indices,B_row = B.getRow(B_ind)
        
            column_indices = []
            values = []



            outer_product = np.outer(A_row, B_row)
            values = outer_product.flatten()
            column_indices = np.add.outer(A_indices * cb, B_indices).flatten()
            column_indices = column_indices.astype("int32")
        
            H_atom.setValues(i,column_indices,values)

        H_atom.assemble()
        viewer = PETSc.Viewer().createBinary("matrix_files/H_0.bin","w")
        H_atom.view(viewer)
        viewer.destroy()
        self.H_atom = H_atom
        return None

    def S(self,tiseInstance,basisInstance):
        if os.path.exists('matrix_files/overlap.bin'):
            n_basis = basisInstance.n_basis
            order = basisInstance.order
            

            S = PETSc.Mat().createAIJ([self.n_blocks*n_basis,self.n_blocks*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*order+1)
            viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
            S.load(viewer)
            viewer.destroy()
            self.S = S
            return
        n_basis = basisInstance.n_basis
        S_R = tiseInstance.S_R
        order = basisInstance.order

        I = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        I.assemble()

        total = kronV4(I,S_R,(2*order+1))

        I.destroy()
        self.S = total


        viewer = PETSc.Viewer().createBinary("matrix_files/overlap.bin","w")
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
    
    def PartialAngularVelocity(self,gridInstance):
        dt = gridInstance.dt

        H_INT_1 = self.H_ang_one
        H_INT_2 = self.H_ang_two

        H_INT_1.axpy(1,self.H_mix_one)
        H_INT_2.axpy(1,self.H_mix_2)


        self.H_INT_1 = H_INT_1
        self.H_INT_2 = H_INT_2
        return None



        