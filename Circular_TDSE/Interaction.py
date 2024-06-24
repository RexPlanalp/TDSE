import numpy as np

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

class interaction:
    def __init__(self):
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

    def H_INV_R(self,simInstance,basisInstance):
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]

        B = basisInstance.B
        knots = basisInstance.knots
        integrate = basisInstance.integrate

        H_inv_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1)+1)
        H_inv_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        def H_inv_R_element(x,i,j,knots,order):
            return B(i, order, x, knots)*B(j, order, x, knots)/ np.sqrt(x**2 + 1E-25)
        
        istart,iend = H_inv_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    if np.abs(i-j)>=order:
                        continue
                    H_element = integrate(H_inv_R_element,i,j,order,knots)
                    H_inv_R.setValue(i,j,H_element)
        H_inv_R.assemble()
        self.H_inv_R = H_inv_R

    def H_DER_R(self,simInstance,basisInstance):
        n_basis = simInstance.splines["n_basis"]
        order=  simInstance.splines["order"]

        B = basisInstance.B
        dB = basisInstance.dB
        knots = basisInstance.knots
        integrate = basisInstance.integrate
        

        H_der_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1)+1)
        H_der_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        def H_der_R_element(x,i,j,knots,order):
            return B(i, order, x, knots)*dB(j,order,x,knots)
        
        istart,iend = H_der_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    if np.abs(i-j)>=order:
                        continue
                    H_element = integrate(H_der_R_element,i,j,order,knots)
                    H_der_R.setValue(i,j,H_element)
        H_der_R.assemble()

        self.H_der_R = H_der_R

    def H_INT_1(self,simInstance):
        n_block = simInstance.n_block
        block_dict = simInstance.block_dict
        order = simInstance.splines["order"]
        kron = simInstance.kron


        H_lm_one = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if (l == lprime+1) and (m == mprime+1):
                    H_lm_one.setValue(i,j,self.a(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_lm_one.setValue(i,j,self.b(l,m))
        H_lm_one.assemble()

        H_lm_two = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if (l == lprime+1) and (m == mprime+1):
                    H_lm_two.setValue(i,j,self.c(l,m))
                elif (l == lprime-1) and (m == mprime+1):
                    H_lm_two.setValue(i,j,-self.d(l,m))
        H_lm_two.assemble()

        term1 = kron(H_lm_one,self.H_inv_R,comm,2*(2*(order-1) + 1))
        term2 = kron(H_lm_two,self.H_der_R,comm,2*(2*(order-1) + 1))

        term1.axpy(1,term2)
        term1.scale(1j/2)
       
        self.H_int_1 = term1
        return None

    def H_INT_2(self,simInstance):
        n_block = simInstance.n_block
        block_dict = simInstance.block_dict
        order = simInstance.splines["order"]
        kron = simInstance.kron

        

        H_lm_one = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_one.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if (l == lprime+1) and (m == mprime-1):
                    H_lm_one.setValue(i,j,self.atilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_lm_one.setValue(i,j,self.btilde(l,m))
        H_lm_one.assemble()

        H_lm_two = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_lm_two.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if (l == lprime+1) and (m == mprime-1):
                    H_lm_two.setValue(i,j,-self.ctilde(l,m))
                elif (l == lprime-1) and (m == mprime-1):
                    H_lm_two.setValue(i,j,self.dtilde(l,m))
        H_lm_two.assemble()

        term1 = kron(H_lm_one,self.H_inv_R,comm,2*(2*(order-1) + 1))
        term2 = kron(H_lm_two,self.H_der_R,comm,2*(2*(order-1) + 1))

        term1.axpy(1,term2)
        term1.scale(1j/2)
        

        self.H_int_2 = term1
        return None

 
    

  
    


    