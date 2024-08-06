import numpy as np
import os

from petsc4py import PETSc
import petsc4py
comm = PETSc.COMM_WORLD
rank = comm.rank


class propagator:
    def __init__(self):
        return
    
    def partialAtomic(self,simInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]

        dt = simInstance.box["time_spacing"]

        if os.path.exists('TISE_files/S.bin'):
            S = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
            S_viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
            S.load(S_viewer)
            S_viewer.destroy()
            S.assemble()
            

        if os.path.exists('TISE_files/H.bin'):
            H = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
            H_viewer = PETSc.Viewer().createBinary('TISE_files/H.bin', 'r')
            H.load(H_viewer)
            H_viewer.destroy()
            H.assemble()
            
        
        S_L = S.copy()
        S_R = S.copy()
        
        

        S_L.axpy(1j*dt/2,H)
        S_R.axpy(-1j*dt/2,H)

        
        

        self.atomic_L = S_L
        self.atomic_R = S_R
        S.destroy()
        H.destroy()

        return None
    
    def partialInteraction(self,simInstance,interactionInstance):
        dt = simInstance.box["time_spacing"]
        

        
        H_ang = interactionInstance.H_ang
        H_mix = interactionInstance.H_mix
        

        
        H_ang.axpy(1,H_mix)
        H_ang.scale(1j*dt/2)

        H_mix.destroy()
        
        self.interaction_mat = H_ang

        

        return None

    def constructHHG(self,simInstance,basisInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        block_dict = simInstance.block_dict
        kron = simInstance.kron

        knots = basisInstance.knots
        B = basisInstance.B
        dB = basisInstance.dB
        integrate = basisInstance.integrate

        def _H_hhg_R(x,i,j,knots,order):
            return (B(i, order, x, knots)*B(j, order, x, knots)/(x**2 + 1E-25))


        def alm(l,m):
            f1 = (l-m-1)*(l-m)
            f2 = 2*(2*l-1)*(2*l+1)
            return np.sqrt(f1/f2)
        
        def blm(l,m):
            f1 = (l+m+1)*(l+m+2)*(l+1)
            f2 = (2*l+1)*(2*l+2)*(2*l+3)
            return np.sqrt(f1/f2)
        
        def clm(l,m):
            f1 = (l+m-1)*(l+m)
            f2 = 2*(2*l-1)*(2*l+1)
            return np.sqrt(f1/f2)

        def dlm(l,m):
            f1 = (l-m+1)*(l-m+2)*(l+1)
            f2 = (2*l+1)*(2*l+2)*(2*l+3)

        def elm(l,m):
            f1 = (l+m)*(l-m)
            f2 = (2*l+1)*(2*l-1)
            return np.sqrt(f1/f2)
        
        def flm(l,m):
            f1 = (l+m+1)*(l-m+1)
            f2 = (2*l+1)*(2*l+3)
            return np.sqrt(f1/f2)


        H_hhg_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = comm,nnz = 2*(order-1)+1)
        H_hhg_R.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        istart,iend = H_hhg_R.getOwnershipRange()
        for i in range(istart,iend):
            for j in range(n_basis):
                if np.abs(i-j)>=order:
                    continue
                H_element = integrate(_H_hhg_R,i,j,order,knots)
                H_hhg_R.setValue(i,j,H_element)
        comm.barrier()
        H_hhg_R.assemble()

        H_hhg_lm_x = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_hhg_lm_x.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]

                if m == mprime-1 and l == lprime+1:
                    H_hhg_lm_x.setValue(i,j,(1/np.sqrt(2))*alm(l,m))
                elif m == mprime-1 and l == lprime-1:
                    H_hhg_lm_x.setValue(i,j,-(1/np.sqrt(2))*blm(l,m))
                elif m == mprime+1 and l == lprime+1:
                    H_hhg_lm_x.setValue(i,j,-(1/np.sqrt(2))*clm(l,m))
                elif m == mprime+1 and l == lprime-1:
                    H_hhg_lm_x.setValue(i,j,(1/np.sqrt(2))*dlm(l,m))    
        comm.barrier()
        H_hhg_lm_x.assemble()
        hhg_x = kron(H_hhg_lm_x,H_hhg_R,comm,2*(2*(order-1)+1))
        self.hhg_x = hhg_x
        H_hhg_lm_x.destroy()

        H_hhg_lm_y = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_hhg_lm_y.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]

                if m == mprime-1 and l == lprime+1:
                    H_hhg_lm_y.setValue(i,j,(-1j/np.sqrt(2))*alm(l,m))
                elif m == mprime-1 and l == lprime-1:
                    H_hhg_lm_y.setValue(i,j,(1j/np.sqrt(2))*blm(l,m))
                elif m == mprime+1 and l == lprime+1:
                    H_hhg_lm_y.setValue(i,j,(-1j/np.sqrt(2))*clm(l,m))
                elif m == mprime+1 and l == lprime-1:
                    H_hhg_lm_y.setValue(i,j,(1j/np.sqrt(2))*dlm(l,m))    
        comm.barrier()
        H_hhg_lm_y.assemble()
        hhg_y = kron(H_hhg_lm_y,H_hhg_R,comm,2*(2*(order-1)+1))
        self.hhg_y = hhg_y
        H_hhg_lm_y.destroy()



        H_hhg_lm_z = PETSc.Mat().createAIJ([n_block,n_block],comm = comm,nnz = 2)
        istart,iend = H_hhg_lm_z.getOwnershipRange()
        for i in range(istart,iend):
            l,m = block_dict[i]
            for j in range(n_block):
                lprime,mprime = block_dict[j]

                if m == mprime and l == lprime+1:
                    H_hhg_lm_z.setValue(i,j,elm(l,m))
                elif m == mprime and l == lprime-1:
                    H_hhg_lm_z.setValue(i,j,flm(l,m))
                
        comm.barrier()
        H_hhg_lm_z.assemble()
        hhg_y = kron(H_hhg_lm_z,H_hhg_R,comm,2*(2*(order-1)+1))
        self.hhg_z = hhg_y
        H_hhg_lm_z.destroy()

        H_hhg_R.destroy()
        return None

    def propagateCN(self,simInstance,psiInstance,laserInstance):
        n_block = simInstance.n_block
        n_basis = simInstance.splines["n_basis"]
        order = simInstance.splines["order"]
        Nt = simInstance.Nt
        Nt_post = simInstance.Nt_post
        dt = simInstance.box["time_spacing"]
        
        psi_initial = psiInstance.state
        if os.path.exists('TISE_files/S.bin'):
            S = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = PETSc.COMM_WORLD,nnz = 2*(order-1)+1)
            S_viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
            S.load(S_viewer)
            S_viewer.destroy()
            S.assemble()
        S_norm = S.createVecRight()
        S.mult(psi_initial,S_norm)
        prod = psi_initial.dot(S_norm)
        S_norm.destroy()
        if rank == 0:
            print(f"Norm of Initial State:{np.real(prod)}")

        ksp = PETSc.KSP().create(comm = comm)
        ksp.setTolerances(rtol = 1E-10)

        partial_angular = self.interaction_mat

        
        ax_list = []
        ay_list = []
        az_list = []

        x_vec = self.hhg_x.createVecRight()
        self.hhg_x.mult(psi_initial,x_vec)
        prod = psi_initial.dot(x_vec)
        ax_list.append(prod)

        y_vec = self.hhg_y.createVecRight()
        self.hhg_y.mult(psi_initial,y_vec)
        prod = psi_initial.dot(y_vec)
        ay_list.append(prod)

        z_vec = self.hhg_z.createVecRight()
        self.hhg_z.mult(psi_initial,z_vec)
        prod = psi_initial.dot(z_vec)
        az_list.append(prod)



    
       

        for i in range(Nt-1):
            if PETSc.COMM_WORLD.rank == 0:
                print(i, Nt-1)

            
            

            pulse_val = laserInstance.A_func(i*dt + dt/2)
            
            partial_L_copy = self.atomic_L.copy()
            partial_R_copy = self.atomic_R.copy()
            partial_L_copy.axpy(pulse_val, partial_angular, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_R_copy.axpy(-pulse_val, partial_angular, structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()
            partial_R_copy.mult(psi_initial, known)

            ksp.setOperators(partial_L_copy)
            ksp.solve(known, solution)
            solution.copy(psi_initial)

            # self.hhg_x.mult(psi_initial, x_vec)
            # prod = psi_initial.dot(x_vec)
            # ax_list.append(prod)

            # self.hhg_y.mult(psi_initial, y_vec)
            # prod = psi_initial.dot(y_vec)
            # ay_list.append(prod)

            self.hhg_z.mult(psi_initial, z_vec)
            prod = psi_initial.dot(z_vec)
            az_list.append(prod)

                
           

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()

        if PETSc.COMM_WORLD.rank == 0:
            print("Starting Free Prop Steps")

        # Free Propagation Steps Optimization
        partial_L_copy = self.atomic_L.copy()
        partial_R_copy = self.atomic_R.copy()

        ksp.setOperators(partial_L_copy)

        for i in range(Nt_post-1):
            if PETSc.COMM_WORLD.rank == 0:
                print(i, Nt_post-1)

            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()
            partial_R_copy.mult(psi_initial, known)

            ksp.solve(known, solution)
            solution.copy(psi_initial)

            known.destroy()
            solution.destroy()

        partial_L_copy.destroy()
        partial_R_copy.destroy()
        
        #a_data = np.vstack([ax_list,ay_list,az_list])
        a_data = az_list
        np.save("TDSE_files/HHG.npy",a_data)

        S_norm = S.createVecRight()
        S.mult(psi_initial,S_norm)
        prod = psi_initial.dot(S_norm)
        S_norm.destroy()
        if rank == 0:
            print(f"Norm of Final State:{np.real(prod)}")

        ViewHDF5 = PETSc.Viewer().createHDF5("TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm= comm)
        psi_initial.setName("psi_final")
        ViewHDF5.view(psi_initial)
        ViewHDF5.destroy()
        return None