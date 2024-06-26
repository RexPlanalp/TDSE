import numpy as np

from petsc4py import PETSc

comm = PETSc.COMM_WORLD
rank = comm.rank


class atomic:
    def __init__(self,simInstance,basisInstance):
        def H(x):
            return (-1/(x+1E-25))          
        def He(x):
                return (-1/np.sqrt(x**2 + 1E-25)) + -1.0*np.exp(-2.0329*x)/np.sqrt(x**2 + 1E-25)  - 0.3953*np.exp(-6.1805*x)
        def Ar(x):
            return -1.0 / (x + 1e-25) - 17.0 * np.exp(-0.8103 * x) / (x + 1e-25) \
            - (-15.9583) * np.exp(-1.2305 * x) \
            - (-27.7467) * np.exp(-4.3946 * x) \
            - 2.1768 * np.exp(-86.7179 * x)
        
        if simInstance.box["pot"] == "H":
            self.pot_func = H
        elif simInstance.box["pot"] == "He":
            self.pot_func = He
        elif simInstance.box["pot"] == "Ar":
            self.pot_func = Ar

    def createK(self,simInstance,basisInstance):
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        knots = basisInstance.knots

        dB = basisInstance.dB
       

        def _K_element(x,i,j,knots,order):
            return dB(i, order, x, knots) * (1/2) * dB(j, order, x, knots) 

        K = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1) +1)
        istart,iend = K.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>= order:
                    continue
                K_element = basisInstance.integrate(_K_element,i,j,order,knots)
                K.setValue(i,j,K_element)
        comm.barrier()
        K.assemble() 
        return K 

    def createV_l(self,simInstance,basisInstance,l):
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        knots = basisInstance.knots

        B = basisInstance.B
       
       

        def _V_element(x,i,j,knots,order):
            return (B(i, order, x, knots) * B(j, order,x, knots) * l*(l+1)/(2*np.sqrt(x**4 + 1E-25 )) + B(i, order, x, knots) * B(j, order, x, knots)* self.pot_func(x))

        V = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
        istart,iend = V.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                V_element = basisInstance.integrate(_V_element,i,j,order,knots)
                V.setValue(i,j,V_element)
        comm.barrier()
        V.assemble() 
        return V   

    def createS(self,simInstance,basisInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        knots = basisInstance.knots

        B = basisInstance.B

        def _S_element(x,i,j,knots,order):
            return (B(i, order, x, knots) * B(j, order,x,knots))

        S = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
        S.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = S.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                    if np.abs(i-j)>=order:
                        continue
                    S_element = basisInstance.integrate(_S_element,i,j,order,knots)
                    S.setValue(i,j,S_element)
        comm.barrier()    
        S.assemble()
        self.S = S
        
    def embedS(self,simInstance):
        n_block = simInstance.n_block
        order = simInstance.splines["order"]
        kron = simInstance.kron

        I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        comm.barrier()
        I.assemble()

        S_atom = kron(I,self.S,comm,2*(order-1)+1)
        S_viewer = PETSc.Viewer().createBinary("TISE_files/S.bin","w")
        S_atom.view(S_viewer)
        S_viewer.destroy()







