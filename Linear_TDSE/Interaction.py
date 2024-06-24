import numpy as np

from petsc4py import PETSc
comm = PETSc.COMM_WORLD
rank = comm.rank

class interaction:
    def __init__(self):
        return None
    

    def clm(self,l,m):
        return np.sqrt(((l+1)**2 - m**2)/((2*l+1)*(2*l+3)))
    def dlm(self,l,m):
        return np.sqrt(((l)**2 - m**2)/((2*l-1)*(2*l+1)))


    def H_mix(self,simInstance,basisInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        block_dict = simInstance.block_dict
        kron = simInstance.kron

        knots = basisInstance.knots
        B = basisInstance.B
        dB = basisInstance.dB
        integrate = basisInstance.integrate



        
        

        

        def _H_mix_R_element(x,i,j,knots,order):
            return (B(i, order, x, knots)*dB(j, order, x, knots))
        
        H_mix_lm = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        
        istart,iend = H_mix_lm.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if l == lprime+1:
                    H_mix_lm.setValue(i,j,self.dlm(l,m))
                elif l == lprime-1:
                    H_mix_lm.setValue(i,j,self.clm(l,m))
        comm.barrier()
        H_mix_lm.assemble()

        H_mix_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1)+1)
        H_mix_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_mix_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                H_element = integrate(_H_mix_R_element,i,j,order,knots)
                H_mix_R.setValue(i,j,H_element)
        comm.barrier()
        H_mix_R.assemble()

        total = kron(H_mix_lm,H_mix_R,comm,2*(2*(order-1)+1))
        total.scale(-1j)

        H_mix_lm.destroy()
        H_mix_R.destroy()
        self.H_mix = total
        return None


    def H_ang(self,simInstance,basisInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        block_dict = simInstance.block_dict
        kron = simInstance.kron

        knots = basisInstance.knots
        B = basisInstance.B
        integrate = basisInstance.integrate
        
        
        def _H_ang_R_element(x,i,j,knots,order):
            return (B(i, order, x, knots)*B(j, order, x, knots)/ (x+1E-25))
            

        H_ang_lm = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_ang_lm.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]
                if l == lprime+1:
                    H_ang_lm.setValue(i,j,-(l)*self.dlm(l,m))
                elif l == lprime-1:
                    H_ang_lm.setValue(i,j,(l+1)*self.clm(l,m))
        comm.barrier()
        H_ang_lm.assemble()

        H_ang_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1)+1)
        H_ang_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_ang_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    if np.abs(i-j)>=order:
                        continue
                    H_element = integrate(_H_ang_R_element,i,j,order,knots)
                    H_ang_R.setValue(i,j,H_element)
        comm.barrier()
        H_ang_R.assemble()

        total = kron(H_ang_lm,H_ang_R,comm,2*(2*(order-1)+1))
        total.scale(-1j)

        H_ang_lm.destroy()
        H_ang_R.destroy()
        self.H_ang = total
        return None  
    

    


   