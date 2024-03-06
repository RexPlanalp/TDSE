import json
from petsc4py import PETSc
import numpy as np
from Module import *
import os


comm = PETSc.COMM_WORLD

class hamiltonian:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        self.m = input_par["state"][2]

        return None
    

    def a(self,l,m):
        f1 = np.sqrt((l+m)/((2*l+1)*(2*l-1)))
        f2 = -m * np.sqrt(l+m-1) - np.sqrt((l-m)*(l*(l-1)- m*(m-1)))
        return f1*f2
    def atilde(self,l,m):
        f1 = np.sqrt((l-m)/((2*l+1)*(2*l-1)))
        f2 = -m * np.sqrt(l-m-1) + np.sqrt((l+m)*(l*(l-1)- m*(m+1)))
        return f1*f2

    def b(self,l,m):
        return -self.atilde(l+1,m-1)
    def btilde(self,l,m):
        return -self.a(l+1,m+1)

    def c(self,l,m):
        return self.dtilde(l-1,m-1)
    def ctilde(self,l,m):
        return self.d(l-1,m+1)

    def d(self,l,m):
        f1 = np.sqrt((l-m+1)*(l-m+2))
        f2 = np.sqrt((2*l+1)*(2*l+3))
        return f1/f2
    def dtilde(self,l,m):
        return self.d(l,-m)

    def H_INV_R(self,basisInstance):
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs

        H_inv_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_inv_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        def H_inv_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x)/ np.sqrt(x**2 + 1E-25)
        
        istart,iend = H_inv_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_inv_R_element,i,j)
                    H_inv_R.setValue(i,j,H_element)
        comm.barrier()
        H_inv_R.assemble()

        self.H_inv_R = H_inv_R

    def H_DER_R(self,basisInstance):
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        basis_funcs = basisInstance.basis_funcs

        H_der_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*order+1)
        H_der_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        def H_der_R_element(x,i,j):
            return basis_funcs[i](x)*basis_funcs[j](x,1)
        
        istart,iend = H_der_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    H_element = basisInstance.integrate(H_der_R_element,i,j)
                    H_der_R.setValue(i,j,H_element)
        comm.barrier()
        H_der_R.assemble()

        self.H_der_R = H_der_R

    def H_INT_1(self,basisInstance,gridInstance,block_map):
        dt = gridInstance.dt
        order = basisInstance.order
        n_block = basisInstance.n_blocks


        H_lm_one = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_map[i]
            for j in range(n_block):
                lprime,mprime = block_map[j]
                if (l == lprime+1) and (m == mprime+1):
                    H_lm_one.setValue(i,j,self.a(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_lm_one.setValue(i,j,self.b(l,m))
        comm.barrier()
        H_lm_one.assemble()

        H_lm_two = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_map[i]
            for j in range(n_block):
                lprime,mprime = block_map[j]
                if (l == lprime+1) and (m == mprime+1):
                    H_lm_two.setValue(i,j,self.c(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_lm_two.setValue(i,j,-self.d(l,m))
        comm.barrier()
        H_lm_two.assemble()

        term1 = kronV6(H_lm_one,self.H_inv_R,2*(2*order + 1))
        term2 = kronV6(H_lm_two,self.H_der_R,2*(2*order + 1))

        term1.axpy(1,term2)
        term1.scale(1j/2)
        term1.scale(1j*dt/2)

        self.H_int_1 = term1
        return None

    def H_INT_2(self,basisInstance,gridInstance,block_map):
        dt = gridInstance.dt
        order = basisInstance.order
        n_block = basisInstance.n_blocks


        H_lm_one = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_map[i]
            for j in range(n_block):
                lprime,mprime = block_map[j]
                if (l == lprime+1) and (m == mprime-1):
                    H_lm_one.setValue(i,j,self.atilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_lm_one.setValue(i,j,self.btilde(l,m))
        comm.barrier()
        H_lm_one.assemble()

        H_lm_two = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_map[i]
            for j in range(n_block):
                lprime,mprime = block_map[j]
                if (l == lprime+1) and (m == mprime-1):
                    H_lm_two.setValue(i,j,-self.ctilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_lm_two.setValue(i,j,self.dtilde(l,m))
        comm.barrier()
        H_lm_two.assemble()

        term1 = kronV6(H_lm_one,self.H_inv_R,2*(2*order + 1))
        term2 = kronV6(H_lm_two,self.H_der_R,2*(2*order + 1))

        term1.axpy(1,term2)
        term1.scale(1j/2)
        term1.scale(1j*dt/2)

        self.H_int_2 = term1
        return None

    def H_ATOM(self,tiseInstance,basisInstance,gridInstance,block_map):
        if os.path.exists('matrix_files/H_0.bin'):
            n_block = basisInstance.n_blocks
            n_basis = basisInstance.n_basis
            order = basisInstance.order
            H_atom = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*order+1)
            viewer = PETSc.Viewer().createBinary('matrix_files/H_0.bin', 'r')
            H_atom.load(viewer)
            viewer.destroy()
            self.H_atom = H_atom
            return
        H_list = tiseInstance.FFH_R_list
        local_H = []
        for l in range(self.lmax+1):
            local_H.append(getLocal(H_list[l]))

        n_block = basisInstance.n_blocks
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        
        ra,ca = n_block,n_block
        rb,cb = n_basis,n_basis

        H_atom = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz =2*order+1)
        ownershipH = H_atom.getOwnershipRange()
        H_range = range(ownershipH[0],ownershipH[1])

        
        for i in H_range:
            A_ind = i//rb
            A_indices,A_row = np.array(A_ind),np.array(1)

            l,m = block_map[A_ind]


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
        comm.barrier()
        H_atom.assemble()
        viewer = PETSc.Viewer().createBinary("matrix_files/H_0.bin","w")
        H_atom.view(viewer)
        viewer.destroy()
        self.H_atom = H_atom
        return None


    def S(self,tiseInstance,basisInstance):
        if os.path.exists('matrix_files/overlap.bin'):
            n_basis = basisInstance.n_basis
            n_block = basisInstance.n_blocks
            order = basisInstance.order
            
            

            S = PETSc.Mat().createAIJ([(self.n_block)*n_basis,(self.n_block)*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*order+1)
            viewer = PETSc.Viewer().createBinary('matrix_files/overlap.bin', 'r')
            S.load(viewer)
            viewer.destroy()
            self.S = S
            return
        n_basis = basisInstance.n_basis
        n_block = basisInstance.n_blocks
        S_R = tiseInstance.S_R
        order = basisInstance.order

        I = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        comm.barrier()
        I.assemble()

        total = kronV6(I,S_R,(2*order+1))

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
    

    


    def H_ATOM(self,tiseInstance,basisInstance,gridInstance,block_map):
        H_list = tiseInstance.FFH_R_list
        
        n_block = basisInstance.n_blocks
        n_basis = basisInstance.n_basis
        order = basisInstance.order
        
        ra,ca = n_block,n_block
        rb,cb = n_basis,n_basis

        H_atom = PETSc.Mat().createAIJ([ra*rb,ca*cb],comm = PETSc.COMM_WORLD,nnz =2*order+1)
        H_atom.assemble()
        
        for l in range(self.lmax + 1):
            partial_I = PETSc.Mat().createAIJ([n_block,n_block],comm = PETSc.COMM_WORLD,nnz = 1)
            istart,iend = partial_I.getOwnershipRange()
            start_index = int(np.sum([2*n +1 for n in range(l)]))
            end_index = int(start_index + 2*l +1)
            index_range = range(start_index,end_index)

            for i in range(istart,iend):
                if i in index_range:
                    partial_I.setValue(i,i,1)
            comm.barrier()
            partial_I.assemble()

            partial_H = kronV6(partial_I,H_list[l],2*order +1)
            H_atom.axpy(1,partial_H)

            H_list[l].destroy()
            partial_I.destroy()
        
        viewer = PETSc.Viewer().createBinary("matrix_files/H_0.bin","w")
        H_atom.view(viewer)
        viewer.destroy()
        self.H_atom = H_atom
        return None



   
        