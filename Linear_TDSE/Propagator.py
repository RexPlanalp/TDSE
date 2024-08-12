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

        if rank == 0:
            norm_file = open("norms.txt","w")
            norm_file.write(f"Norm of Inititial state: {np.real(prod)} \n")
            norm_file.close()
        

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

            S_norm = S.createVecRight()
            S.mult(psi_initial,S_norm)
            prod = psi_initial.dot(S_norm)
            S_norm.destroy()

            if rank == 0:
                norm_file = open("norms.txt","a")
                norm_file.write(f"Norm of state at step {i}: {np.real(prod)} \n")
                norm_file.close()
                
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