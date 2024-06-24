import numpy as np

from petsc4py import PETSc

comm = PETSc.COMM_WORLD
rank = comm.rank

class atomic:
    def __init__(self,simInstance,basisInstance):
        def H(x):
            if  basisInstance.RS > np.max(basisInstance.knots):
                return (-1/(x+1E-25))
            else:
                decay_factor = np.ones_like(x)
                mask = (x > basisInstance.RS/2) & (x < basisInstance.RS)
                decay_factor[mask] = 2 - 2 * x[mask] / basisInstance.RS
                decay_factor[x >= basisInstance.RS] = 0
                potential = -1 / np.sqrt(x**2 + 1E-25)
                potential[x > basisInstance.RS] = 0
                return potential * decay_factor
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
            return dB(i, order, x, knots) * (1/2)*dB(j, order, x, knots) 

        K = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
        istart,iend = K.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>= order:
                    continue
                K_element = basisInstance.integrate(_K_element,i,j,order,knots)
                K.setValue(i,j,K_element)
        K.assemble() 
        self.K = K

    def initializeV(self,simInstance):
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]

    
        V = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
         
        self.V = V

    def createV_l(self,simInstance,basisInstance,l):
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        knots = basisInstance.knots

        
        B = basisInstance.B

        def _V_element(x,i,j,knots,order):
            return (B(i, order, x, knots) * B(j, order,x, knots) * l*(l+1)/(2*x**2 + 1E-25) + B(i, order, x, knots) * B(j, order, x, knots)* self.pot_func(x))
        V = self.V
        istart,iend = V.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                V_element = basisInstance.integrate(_V_element,i,j,order,knots)
                V.setValue(i,j,V_element)
        V.assemble() 
        self.V = V

    def createS(self,simInstance,basisInstance):
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
        S.assemble()
        self.S = S

    def embedS(self,simInstance):
        kron = simInstance.kron
        n_block = simInstance.n_block
        order = simInstance.splines["order"]

        I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        I.assemble()
        
        S_atom = kron(I,self.S,comm,2*(order-1)+1)
        S_viewer = PETSc.Viewer().createBinary("TISE_files/S.bin","w")
        S_atom.view(S_viewer)
        S_viewer.destroy()

        return None
        
    def initializeH_atom(self,simInstance):
        n_basis = simInstance.splines["n_basis"]
        n_block = simInstance.n_block
        order = simInstance.splines["order"]


        H_atom = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
        H_atom.assemble()

        self.H_atom = H_atom

    def embedH_l(self,simInstance,H_l,l):

        n_block = simInstance.n_block
        kron = simInstance.kron
        order = simInstance.splines["order"]

        if simInstance.laser["polarization"] == "linear":
            partial_I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 1)
            istart,iend = partial_I.getOwnershipRange()
            if l in range(istart,iend):
                partial_I.setValue(l,l,1)
            comm.barrier()
            partial_I.assemble()
        elif simInstance.laser["polarization"] == "elliptical":
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

        partial_H = kron(partial_I,H_l,comm,2*(order-1) +1)
        self.H_atom.axpy(1,partial_H)
        partial_I.destroy()
        partial_H.destroy()
    
    def saveH_atom(self):
        H_viewer = PETSc.Viewer().createBinary("TISE_files/H.bin","w")
        self.H_atom.view(H_viewer)
        H_viewer.destroy()

    def createA(self,simInstance,basisInstance):
        if simInstance.HHG:
            n_basis = simInstance.splines["n_basis"]
            n_block = simInstance.n_block
            order = simInstance.splines["order"]
            knots = basisInstance.knots
            kron = simInstance.kron
            
            B = basisInstance.B
            
            

            def _A_element(x,i,j,knots,order):
                return (B(i, order, x, knots) * B(j, order,x,knots)/(x**3+1E-25))

            A = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1) +1)
            A.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
            istart,iend = A.getOwnershipRange()
            for i in range(istart,iend):
                for j in range(n_basis):
                        if np.abs(i-j)>=order:
                            continue
                        A_element = basisInstance.integrate(_A_element,i,j,order,knots)
                        A.setValue(i,j,A_element)   
            A.assemble()

            I = PETSc.Mat().createAIJ([n_block,n_block],comm = comm)
            istart,iend = I.getOwnershipRange()
            for i in range(istart,iend):
                I.setValue(i,i,1)
            I.assemble()

            A_atom = kron(I,A,comm,2*(order-1)+1)

            A_viewer = PETSc.Viewer().createBinary("TISE_files/A.bin","w")
            A_atom.view(A_viewer)
            A_viewer.destroy()


