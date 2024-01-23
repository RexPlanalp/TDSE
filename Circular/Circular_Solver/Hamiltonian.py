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
    
    def alm(self,l,m):
        f1 = np.sqrt((l+m)/((2*l+1)*(2*l-1)))

        f2_1 = -m*np.sqrt(l+m-1)
        
        f2_2 = -np.sqrt((l-m)*(l*(l-1)-m*(m-1)))

        f2 = f2_1+f2_2
        return f1*f2
        
    def atildelm(self,l,m):
        f1 = np.sqrt((l-m)/((2*l+1)*(2*l-1)))

        f2_1 = f2_1 = -m*np.sqrt(l-m-1)
        
        f2_2 = np.sqrt((l+m)*(l*(l-1)-m*(m+1)))

        f2 = f2_1+f2_2

        return f1*f2

    def blm(self,l,m):
        f = -self.atildelm(l+1,m-1)
        return f

    def btildelm(self,l,m):
        f = -self.alm(l+1,m+1)
        return f

    def clm(self,l,m):
        f = self.dtildelm(l-1,m-1)
        return f

    def ctildelm(self,l,m):
        f = self.dlm(l-1,m+1)
        return f

    def dlm(self,l,m):
        f = np.sqrt(((l-m+1)*(l-m+2))/((2*l+1)*(2*l+3)))
        return f

    def dtildelm(self,l,m):
        f = np.sqrt(((l+m+1)*(l+m+2))/((2*l+1)*(2*l+3)))
        return f
        
    def H_ANG(self,basisInstance,gridInstance):
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        degree = basisInstance.degree
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        def H_ang_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x)/ np.sqrt(x**2 + 1E-25)

        H_ang_lm_x = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_ang_lm_x.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (m == mprime+1) and (l == lprime+1):
                    H_ang_lm_x.setValue(i,j,self.alm(l,m))
                elif (m == mprime+1) and (l == lprime-1):
                    H_ang_lm_x.setValue(i,j,self.blm(l,m))
                elif (m == mprime-1) and (l == lprime+1):
                    H_ang_lm_x.setValue(i,j,self.atildelm(l,m))
                elif (m == mprime-1) and (l==lprime-1):
                    H_ang_lm_x.setValue(i,j,self.btildelm(l,m))
        H_ang_lm_x.assemble()
        H_ang_lm_x.scale(1j/2)

        H_ang_lm_y = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_ang_lm_y.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]

                if (m == mprime+1) and (l == lprime+1):
                    H_ang_lm_y.setValue(i,j,self.alm(l,m))
                elif (m == mprime+1) and (l == lprime-1):
                    H_ang_lm_y.setValue(i,j,self.blm(l,m))
                elif (m == mprime-1) and (l == lprime+1):
                    H_ang_lm_y.setValue(i,j,-self.atildelm(l,m))
                elif (m == mprime-1) and (l==lprime-1):
                    H_ang_lm_y.setValue(i,j,-self.btildelm(l,m))
        H_ang_lm_y.assemble()
        H_ang_lm_y.scale(1/2)

        H_ang_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_ang_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_ang_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_ang_R_element,i,j)
                    H_ang_R.setValue(i,j,H_element)
        H_ang_R.assemble()

        H_ang_x = kronV4(H_ang_lm_x,H_ang_R,4*(2*order+1))
        H_ang_y = kronV4(H_ang_lm_y,H_ang_R,4*(2*order+1))

        H_ang_lm_y.destroy()
        H_ang_lm_x.destroy()
        H_ang_R.destroy()

        self.H_ang_x = H_ang_x
        self.H_ang_y = H_ang_y

        return

    def H_MIX(self,basisInstance,gridInstance):
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        degree = basisInstance.degree
        basis_funcs = basisInstance.basis_funcs
        dt = gridInstance.dt

        def H_mix_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x,1)

        H_mix_lm_x = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_mix_lm_x.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]
                if (m == mprime+1) and (l == lprime+1):
                    H_mix_lm_x.setValue(i,j,self.clm(l,m))
                elif (m == mprime+1) and (l == lprime-1):
                    H_mix_lm_x.setValue(i,j,-self.dlm(l,m))
                elif (m == mprime-1) and (l == lprime+1):
                    H_mix_lm_x.setValue(i,j,-self.ctildelm(l,m))
                elif (m == mprime-1) and (l==lprime-1):
                    H_mix_lm_x.setValue(i,j,self.dtildelm(l,m))
        H_mix_lm_x.assemble()
        H_mix_lm_x.scale(1j/2)

        H_mix_lm_y = PETSc.Mat().createAIJ([self.n_blocks,self.n_blocks],comm = comm,nnz = 4)
        istart,iend = H_mix_lm_y.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(self.n_blocks):
                l,m = self.block_dict[i]
                lprime,mprime = self.block_dict[j]


                if (m == mprime+1) and (l == lprime+1):
                    H_mix_lm_y.setValue(i,j,self.clm(l,m))
                elif (m == mprime+1) and (l == lprime-1):
                    H_mix_lm_y.setValue(i,j,-self.dlm(l,m))
                elif (m == mprime-1) and (l == lprime+1):
                    H_mix_lm_y.setValue(i,j,self.ctildelm(l,m))
                elif (m == mprime-1) and (l==lprime-1):
                    H_mix_lm_y.setValue(i,j,-self.btildelm(l,m))
        
        H_mix_lm_y.assemble()
        H_mix_lm_y.scale(1/2)

        H_mix_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_mix_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_mix_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_mix_R_element,i,j)
                    H_mix_R.setValue(i,j,H_element)
        H_mix_R.assemble()

        H_mix_x = kronV4(H_mix_lm_x,H_mix_R,4*(2*order+1))
        H_mix_y = kronV4(H_mix_lm_y,H_mix_R,4*(2*order+1))

        H_mix_lm_y.destroy()
        H_mix_lm_x.destroy()
        H_mix_R.destroy()

        self.H_mix_x = H_mix_x
        self.H_mix_y = H_mix_y

        return

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

        H_ang_x_copy = self.H_ang_x.copy()
        H_ang_y_copy = self.H_ang_y.copy()

        H_mix_x_copy = self.H_mix_x.copy()
        H_mix_y_copy = self.H_mix_y.copy()

        H_ang_x_copy.axpy(1,H_mix_x_copy)
        H_ang_y_copy.axpy(1,H_mix_y_copy)

        H_ang_x_copy.scale(1j*dt/2)
        H_ang_y_copy.scale(1j*dt/2)



        
        self.partial_angular_x = H_ang_x_copy
        self.partial_angular_y = H_ang_y_copy
        return None



        