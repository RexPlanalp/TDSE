import numpy as np
from numpy import pi
import matplotlib.pyplot as plt

import json
import time
import h5py
import os
import gc
import kron
from scipy.interpolate import BSpline

import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
def gather_csr(local_csr_part):
  
        gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_csr_part)
        
        return np.concatenate(gathered)
def gather_indpr(local_indptr):
    gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_indptr)
    global_indptr = list(gathered[0])
    offset = global_indptr[-1]  # Start with the last element of the first indptr
    for proc_indptr in gathered[1:]:
        # Offset the local indptr (excluding the first element) and extend the global indptr
        global_indptr.extend(proc_indptr[1:] + offset)
        # Update the offset for the next iteration
        offset += proc_indptr[-1] - proc_indptr[0]  # Adjust for the overlapping indices
    return global_indptr
def getLocal(M):
    local_csr = M.getValuesCSR()
    local_indptr, local_indices, local_data = local_csr
    global_indices = gather_csr(local_indices).astype(np.int32)
    global_data = gather_csr(local_data)
    global_indptr = gather_indpr(local_indptr)
    seq_M = PETSc.Mat().createAIJWithArrays([M.getSize()[0],M.getSize()[1]],(global_indptr,global_indices,global_data),comm = PETSc.COMM_SELF)
    return seq_M



class Grid:
    def __init__(self,grid_size,grid_spacing,N,time_spacing,freq):
        self.r = np.linspace(0,grid_size,int(grid_size/grid_spacing)) #Chagned back to grid spacing

        self.tau = 2*pi/freq
        self.tmax = N*self.tau
        self.t = np.arange(-self.tmax/2,self.tmax/2 + time_spacing,time_spacing)

        self.rmax = np.max(self.r)
        self.dr = grid_spacing

       
        self.dt = time_spacing
    def Print(self,bool):
        if bool:
            print(
                """
                Simulation Box:
            
                x = [{},{}], {}
                t = [0,{}], {}
            
                """.format(0,self.rmax,self.dr,self.tmax,self.dt)
            )
        return None
    
class Basis:
    def __init__(self,N_knots,order,N_gauss):
        self.N_knots = N_knots
        self.order = order
        self.N_gauss = N_gauss

    def CreateFuncs(self,rmax,dr):
        
        knots_start_end = np.repeat([0, rmax],self.order)  # Repeats -end and end, order times

        


        knots_middle = np.linspace(0, rmax - 1, self.N_knots - 2 * self.order)  # Creates evenly spaced knots between the start and end
        knots = np.concatenate([knots_start_end[:self.order], knots_middle, knots_start_end[self.order:]])  # Concatenates the start, middle, and end knots
        
        


        basis_funcs = [BSpline(knots, (i == np.arange(len(knots) - self.order - 1 )).astype(float), self.order) for i in range(len(knots) - self.order - 1 )[1:-(self.order+1)]]


        n_basis = len(basis_funcs)

    
        self.n_basis = len(basis_funcs)
        self.bfuncs = basis_funcs

        return None
    
    def PlotFuncs(self,r,bool):
        if bool:
            basis_array = np.empty((len(r),self.n_basis))
            basis_derivative_array = np.empty((len(r),self.n_basis))
            for i in range(self.n_basis):
                basis_array[:,i] = self.bfuncs[i](r)
                basis_derivative_array[:,i] = self.bfuncs[i](r,2)
        
            for i in range(self.n_basis):
                plt.plot(r,basis_array[:,i])
            plt.savefig("b_splines.png")
        return None

    def CreateGauss(self,rmax):
        x,w = np.polynomial.legendre.leggauss(self.N_gauss)

        transformed_nodes = 0.5 * rmax * x + 0.5 * rmax
        adjusted_weights = 0.5 * rmax * w

        self.nodes = transformed_nodes
        self.weights = adjusted_weights

    def EvalGauss(self):

        basis_array_gauss = np.empty((len(self.nodes),self.n_basis))
        basis_derivative_array_gauss = np.empty((len(self.nodes),self.n_basis))

        for i in range(self.n_basis):
            basis_array_gauss[:,i] = self.bfuncs[i](self.nodes)
            basis_derivative_array_gauss[:,i] = self.bfuncs[i](self.nodes,2)
        
        self.barrays_der_gauss = basis_derivative_array_gauss
        self.barrays_gauss = basis_array_gauss

class TISE:
    def __init__(self):
        self.FFH_R_list = []
    def CreateH_l(self,n_basis,barrays,barrays_der,nodes,weights,l):
        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            start = time.time()
        S_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        FFH_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)

        rowstart,rowend = FFH_R.getOwnershipRange()
        columnstart,columnend = FFH_R.getOwnershipRangeColumn()
        for i in range(rowstart,rowend):
            for j in range(columnstart,columnend):
                if i >= j:

                    H_element_1 = np.sum(weights * barrays[:,i] * (-0.5)* barrays_der[:,j])
                    H_element_2 = np.sum(weights * barrays[:,i] * barrays[:,j] * l*(l+1)/(2*np.sqrt(nodes**4 + 1E-25 )))
                    H_element_3 = np.sum(weights * barrays[:,i] * barrays[:,j] * (-1/np.sqrt(nodes**2 + 1E-25)))

                    S_element = np.sum(weights * barrays[:,i] * barrays[:,j])
                    
                    H_element = H_element_1 + H_element_2 + H_element_3
                    

                    
                    FFH_R.setValue(i,j,H_element)
                    S_R.setValue(i,j,S_element)

                    if i != j:
                        
                        FFH_R.setValue(j,i,H_element)
                        S_R.setValue(j,i,S_element)
                        
        FFH_R.assemblyBegin()
        FFH_R.assemblyEnd()
        S_R.assemblyBegin()
        S_R.assemblyEnd()

        self.FFH_R_list.append(FFH_R)
        self.S_R = S_R
        
        return None
    def EvalEigen(self):
        
        ViewHDF5 = PETSc.Viewer().createHDF5("Hydrogen.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
            
        if (input_par["lm"]["nmax"] >= input_par["lm"]["lmax"]):
            nmax = input_par["lm"]["lmax"] +1
        else:
            nmax = input_par["lm"]["nmax"] 

        
        for i,l in enumerate(range(nmax)):


            


            H = self.FFH_R_list[i]



            E = SLEPc.EPS().create()
            E.setOperators(H, self.S_R)
            
            
            E.setDimensions(nev=6)
            
            
            

            E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
            E.setType(slepc4py.SLEPc.EPS.Type.KRYLOVSCHUR)
            
            E.solve()

            nconv = E.getConverged()
            
            for i in range(nconv):
                eigenvalue = E.getEigenvalue(i)  # This retrieves the eigenvalue
                
                # Creating separate vectors for the real part of the eigenvector
                eigen_vector = H.getVecLeft()  # Assuming H is the correct operator for the matrix H
                E.getEigenvector(i, eigen_vector)  # This retrieves the eigenvector
                        
                    
                eigen_vector.setName(f"Psi_{i+1+l}_{l}")
                ViewHDF5.view(eigen_vector)
                    
                # You could save the real part and the imaginary part if they are non-zero,
                # but for SMALLEST_REAL we typically only consider the real part.
                        
                energy = PETSc.Vec().createMPI(1, comm=PETSc.COMM_WORLD)
                energy.setValue(0,np.real(eigenvalue))
                energy.setName(f"E_{i+1+l}_{l}")
                energy.assemblyBegin()
                energy.assemblyEnd()
                ViewHDF5.view(energy)
        ViewHDF5.destroy()    
        return None
#WARNING EDITED BOUND STATE RANGE


class Laser:
    def __init__(self,w,I):
        self.I = I/3.51E16
        self.freq = w
        
        return None
    
    def CreateEnv(self,time,tmax,N):
        if input_par["laser"]["envelope"] == "sinsq":
            self.env = np.power(np.sin(pi*(time-tmax/2) / tmax), 2.0)
            
        
    def CreatePulse(self,time):
        amplitude = pow(self.I, 0.5) / self.freq

        weighted_env = amplitude * self.env

        pulse = weighted_env * np.sin(self.freq * (time))
        self.pulse = pulse
        if PETSc.COMM_WORLD.rank == 0:
            plt.plot(time,self.pulse)
            plt.savefig("pulse.png")


class Psi:
    def __init__(self,n_basis,lmax):
        psi_initial = PETSc.Vec().createMPI(n_basis*(lmax+1))
        l = input_par["state"][1]
        
        
        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{l+1}_{l}"][:]

            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part
        

      

        psi_array = np.pad(total,(l*n_basis,(lmax-l)*n_basis),constant_values= (0,0))
       
        istart,iend = psi_initial.getOwnershipRange()
        for i in range(istart,iend):
            psi_initial.setValue(i,psi_array[i])
        psi_initial.assemblyBegin()
        psi_initial.assemblyEnd()
        
        self.psi_initial = psi_initial
            
class Hamiltonian:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        self.m = input_par["state"][2]
    def H_MIX(self,n_basis,weights,nodes,basis_funcs):
        H_mix_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD,nnz = 2)
        istart,iend = H_mix_lm.getOwnershipRange()
        for i in range(istart,iend-1):

            clm = np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3)))
            H_mix_lm.setValue(i,i+1,clm)
            H_mix_lm.setValue(i+1,i,clm)
        H_mix_lm.assemblyBegin()
        H_mix_lm.assemblyEnd()

        H_mix_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_mix_R.getOwnershipRange()
        rows,cols = H_mix_R.getSize()
        for i in range(rowstart,rowend):
            for j in range(cols):
                H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes,1))
                H_mix_R.setValue(i,j,H_element)
        H_mix_R.assemblyBegin()
        H_mix_R.assemblyEnd()


        #output = PETSc.Mat().createAIJ([(self.lmax+1)*n_basis,(self.lmax+1)*n_basis],comm = PETSc.COMM_WORLD)
        #output.kron(H_mix_lm,H_mix_R)
        #output.assemblyBegin()
        #output.assemblyEnd()
        #output.scale(-1j)
        
        output = kron.kronV3(H_mix_lm,H_mix_R)
        
        output.scale(-1j)

        

        H_mix_lm.destroy()
        H_mix_R.destroy()
        self.H_mix = output

        return None
    def H_ANG(self,n_basis,weights,nodes,basis_funcs):
        H_ang_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        istart,iend = H_ang_lm.getOwnershipRange()
        for i in range(istart,iend-1):

            clm = np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3)))
            H_ang_lm.setValue(i,i+1,(i+i)*clm)
            H_ang_lm.setValue(i+1,i,-(i+1)*clm)
        H_ang_lm.assemblyBegin()
        H_ang_lm.assemblyEnd()

        H_ang_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_ang_R.getOwnershipRange()
        rows,cols = H_ang_R.getSize()
        for i in range(rowstart,rowend):
            for j in range(cols):
                H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes) / (nodes))
                H_ang_R.setValue(i,j,H_element)
        H_ang_R.assemblyBegin()
        H_ang_R.assemblyEnd()

        #output = PETSc.Mat().createAIJ([(self.lmax+1)*n_basis,(self.lmax+1)*n_basis],comm = PETSc.COMM_WORLD)
        #output.kron(H_ang_lm,H_ang_R)
        #output.assemblyBegin()
        #output.assemblyEnd()
        #output.scale(-1j)

        output = kron.kronV3(H_ang_lm,H_ang_R)
        output.scale(-1j)


        H_ang_lm.destroy()
        H_ang_R.destroy()
        
        self.H_ang = output

        return None
    
    def H_TOTAL(self,field_val):

        intermediate_mix = self.H_mix
        intermediate_mix.scale(field_val)

        intermediate_ang = self.H_ang
        intermediate_ang.scale(field_val)

        intermediate_mix.axpy(1,intermediate_ang)
        intermediate_mix.axpy(1,self.H_atom)

        
    
    
        self.H_total = intermediate_mix
        return None
    def S_TOTAL(self,S_R,n_basis):
        I = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        start,end = I.getOwnershipRange()
        for i in range(start,end):
            I.setValue(i,i,1)
        I.assemblyBegin()
        I.assemblyEnd()

        output = kron.kronV3(I,S_R)
       
        I.destroy()

        self.S_total = output

    def H_ATOM(self,H_list,n_basis):
        def gather_csr(local_csr_part):
  
            gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_csr_part)
        
            return np.concatenate(gathered)
        def gather_indpr(local_indptr):
            gathered = PETSc.COMM_WORLD.tompi4py().allgather(local_indptr)
            global_indptr = list(gathered[0])
            offset = global_indptr[-1]  # Start with the last element of the first indptr
            for proc_indptr in gathered[1:]:
                # Offset the local indptr (excluding the first element) and extend the global indptr
                global_indptr.extend(proc_indptr[1:] + offset)
                # Update the offset for the next iteration
                offset += proc_indptr[-1] - proc_indptr[0]  # Adjust for the overlapping indices
            return global_indptr
        def getLocal(M):
            local_csr = M.getValuesCSR()
            local_indptr, local_indices, local_data = local_csr
            global_indices = gather_csr(local_indices).astype(np.int32)
            global_data = gather_csr(local_data)
            global_indptr = gather_indpr(local_indptr)
            seq_M = PETSc.Mat().createAIJWithArrays([M.getSize()[0],M.getSize()[1]],(global_indptr,global_indices,global_data),comm = PETSc.COMM_SELF)
            return seq_M
        
        
        
        
        H_atom = PETSc.Mat().createAIJ([(self.lmax +1)*n_basis,(self.lmax +1)*n_basis],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_atom.getOwnershipRange()
        
        local_H = []
        for l in range(self.lmax+1):
            local_H.append(getLocal(H_list[l]))
        
        
        for i in range(rowstart,rowend):
            
            l = i // n_basis
            row_index = i % n_basis            
            
            
            index,vals = local_H[l].getRow(row_index)
            

            full_row = np.zeros(n_basis,dtype = "complex")
            full_row[index] = vals
            row_array = np.pad(full_row,(l*n_basis,(self.lmax-l)*n_basis),constant_values= (0,0))

            
            H_atom.setValues(i,list(range((self.lmax +1)*n_basis)),row_array)
            
            


        
        
        H_atom.assemblyBegin()
        H_atom.assemblyEnd()
        self.H_atom = H_atom




if __name__ == "__main__":
    if PETSc.COMM_WORLD.rank ==0:
        start = time.time()
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    
    
    box = Grid(input_par["box"]["xmax"],input_par["box"]["dx"],input_par["box"]["N"],input_par["box"]["dt"],input_par["laser"]["w"])
    box.Print(False)

    splines_par = tuple(input_par["splines"].values())
    splines = Basis(*splines_par)
    splines.CreateFuncs(box.rmax,box.dr)
    splines.PlotFuncs(box.r,False)
    splines.CreateGauss(box.rmax)
    splines.EvalGauss()
    
    FieldFreeH = TISE()
    for l in range(input_par["lm"]["lmax"]+1):
        FieldFreeH.CreateH_l(splines.n_basis,splines.barrays_gauss,splines.barrays_der_gauss,splines.nodes,splines.weights,l)
    FieldFreeH.EvalEigen()

    Field = Laser(input_par["laser"]["w"],input_par["laser"]["I"])
    Field.CreateEnv(box.t,box.tmax,input_par["box"]["N"])
    Field.CreatePulse(box.t)

    psi = Psi(splines.n_basis,input_par["lm"]["lmax"])

    
    Int = Hamiltonian()
    Int.H_MIX(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    Int.H_ANG(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    Int.H_ATOM(FieldFreeH.FFH_R_list,splines.n_basis)
    Int.S_TOTAL(FieldFreeH.S_R,splines.n_basis)

    ############
    for l in range(input_par["lm"]["lmax"]+1):
        FieldFreeH.FFH_R_list[l].destroy()
    ##########
    gc.collect()

    test = False
    if test:
        L = Int.H_atom.getVecRight()
        Int.H_atom.mult(psi.psi_initial,L)

        R = Int.S_total.getVecRight()
        Int.S_total.mult(psi.psi_initial,R)

        if PETSc.COMM_WORLD.rank == 0:
            print(L.getValue(0))
            print(R.getValue(0)*-0.5)

    test2 = False
    if test2:
        dt = box.dt
        L = len(box.t)
        structure = structure=PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN
        def makeLeft(S,MIX,ANG,ATOM,i):
            

            S.axpy(-Field.pulse[i],MIX,structure)
            S.axpy(-Field.pulse[i],ANG,structure)
            S.axpy(-1,ATOM,structure)
            return S
        def makeRight(S,MIX,ANG,ATOM,i):
            
            
            
            S.axpy(Field.pulse[i],MIX,structure)
            S.axpy(Field.pulse[i],ANG,structure)
            S.axpy(1,ATOM,structure)
            
            return S

        H_mix = Int.H_mix
        H_mix.scale(1j * dt /2)
  
        H_ang = Int.H_ang
        H_ang.scale(1j * dt /2)

        H_atom = Int.H_atom
        H_atom.scale(1j * dt /2)

        S = Int.S_total

        psi_initial = psi.psi_initial.copy()
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)

   
        for i,t in enumerate(box.t):
            print(i,L)

            

            O_L = makeLeft(S,H_mix,H_ang,H_atom,i)
            O_R = makeRight(S,H_mix,H_ang,H_atom,i)

            

            if i == 0:
                known = O_R.getVecRight()
                sol = O_L.getVecRight()
        
            O_R.mult(psi_initial,known)

            ksp.setOperators(O_L)
        
            
            ksp.solve(known,sol)

            
            
    
            psi_initial.copy(sol)

        

   
   
    

    
        