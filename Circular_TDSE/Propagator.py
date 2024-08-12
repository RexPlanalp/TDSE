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
            S = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = comm,nnz = 2*(order-1)+1)
            S_viewer = PETSc.Viewer().createBinary('TISE_files/S.bin', 'r')
            S.load(S_viewer)
            S_viewer.destroy()
            S.assemble()
            

        if os.path.exists('TISE_files/H.bin'):
            H = PETSc.Mat().createAIJ([n_block*n_basis,n_block*n_basis],comm = comm,nnz = 2*(order-1)+1)
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

        H_int_1 = interactionInstance.H_int_1
        H_int_2 = interactionInstance.H_int_2
            
        H_int_1.scale(1j*dt/2)
        H_int_2.scale(1j*dt/2)
       
    
        self.H_int_1 = H_int_1
        self.H_int_2 = H_int_2

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

        norm_indices = np.linspace(0, Nt-2, 100, dtype=int)
        if rank == 0:
            norm_file = open("TDSE_files/norms.txt","w")
            norm_file.write(f"Norm of Inititial state: {np.real(prod)} \n")
            norm_file.close()


        ksp = PETSc.KSP().create(comm = comm)
        ksp.setTolerances(rtol = 1E-10)

       

        
        for i in range(Nt-1):
            if PETSc.COMM_WORLD.rank == 0:
                print(i,Nt)

            pulse_val_star = laserInstance.A_funcX(i*dt+dt/2)-1j*laserInstance.A_funcY(i*dt+dt/2)
            pulse_val = laserInstance.A_funcX(i*dt+dt/2)+1j*laserInstance.A_funcY(i*dt+dt/2)
 
            
            partial_L_copy = self.atomic_L.copy()
            partial_R_copy = self.atomic_R.copy()



            partial_L_copy.axpy(pulse_val_star,self.H_int_1,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_L_copy.axpy(pulse_val,self.H_int_2,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)


            partial_R_copy.axpy(-pulse_val_star,self.H_int_1,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_R_copy.axpy(-pulse_val,self.H_int_2,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)

            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()
            partial_R_copy.mult(psi_initial,known)

            ksp.setOperators(partial_L_copy)
            ksp.solve(known,solution)
            solution.copy(psi_initial)

            if rank == 0 and i in norm_indices:
                norm_file = open("TDSE_files/norms.txt","a")
                norm_file.write(f"Norm of state at step {i}: {np.real(prod)} \n")
                norm_file.close()

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()
        
        if PETSc.COMM_WORLD.rank == 0:
            print("Starting Free Prop Steps")
        # Free Prop
        for i in range(Nt_post-1):
            if PETSc.COMM_WORLD.rank == 0:
                print(i,Nt_post)

            partial_L_copy = self.atomic_L.copy()
            partial_R_copy = self.atomic_R.copy()
            
            known = partial_R_copy.createVecRight() 
            solution = partial_L_copy.createVecRight()
            partial_R_copy.mult(psi_initial,known)

            ksp.setOperators(partial_L_copy)
            ksp.solve(known,solution)
            solution.copy(psi_initial)

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()

        S_norm = S.createVecRight()
        S.mult(psi_initial,S_norm)
        prod = psi_initial.dot(S_norm)
        S_norm.destroy()
        if rank == 0:
            print(f"Norm of Final State:{np.real(prod)}")
        ViewHDF5 = PETSc.Viewer().createHDF5("TDSE.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
        psi_initial.setName("psi_final")
        ViewHDF5.view(psi_initial)
        ViewHDF5.destroy()

        return None