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
import sys

import petsc4py
petsc4py.init(sys.argv)
petsc4py.init(comm=PETSc.COMM_WORLD)

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
        

        def Linear():
            knots_start_end = np.repeat([0, rmax],self.order)  # Repeats -end and end, order times
            knots_middle = np.linspace(0, rmax - 1, self.N_knots - 2 * self.order)  # Creates evenly spaced knots between the start and end
            knots = np.concatenate([knots_start_end[:self.order], knots_middle, knots_start_end[self.order:]])  # Concatenates the start, middle, and end knots
            basis_funcs = [BSpline(knots, (i == np.arange(len(knots) - self.order - 1 )).astype(float), self.order) for i in range(len(knots) - self.order - 1 )[1:-(self.order+1)]]


            n_basis = len(basis_funcs)

            return basis_funcs,n_basis
        

        def Quadratic():
            linear = np.linspace(0,rmax/3,int(self.N_knots/2))
            a = 10
            x = np.linspace(np.sqrt((linear[-1])*a),np.sqrt(rmax*a),int(self.N_knots/2))
            quadratic = x**2 /a
            knots = np.append(linear,quadratic)
            for _ in range(self.order+1):
                knots = np.insert(knots,0,0)
                knots = np.append(knots,rmax)
            n_basis = len(knots) - self.order - 1

            basis_funcs = [BSpline(knots, (i == np.arange(n_basis)), self.order) for i in range(n_basis)[int(self.order/5):-int(self.order/5)-1]]
    
            n_basis = len(basis_funcs)

            return basis_funcs,n_basis

        basis_funcs,n_basis = Quadratic()
    
        self.n_basis = len(basis_funcs)
        self.bfuncs = basis_funcs

        return None
    
    def PlotFuncs(self,r,plot):
        if plot:
            plt.figure()
            basis_array = np.empty((len(r),self.n_basis))
            for i in range(self.n_basis):
                basis_array[:,i] = self.bfuncs[i](r)
                plt.plot(r,basis_array[:,i])
            plt.savefig("b_splines.png")
            plt.clf()
        
            ViewHDF5 = PETSc.Viewer().createHDF5("Basis.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
            for i in range(self.n_basis):
                inter = np.array(basis_array[:,i])
                basis_vec = PETSc.Vec().createWithArray(inter,comm = PETSc.COMM_WORLD)
                basis_vec.setName(f"B_{i}")
                ViewHDF5.view(basis_vec)
            ViewHDF5.destroy()
        return None

    def CreateGauss(self,rmax):
        x,w = np.polynomial.legendre.leggauss(self.N_gauss)

        transformed_nodes = 0.5 * rmax * x + 0.5 * rmax
        adjusted_weights = 0.5 * rmax * w

        self.nodes = transformed_nodes
        self.weights = adjusted_weights

    def EvalGauss(self):
        
        basis_array = np.empty((len(self.nodes),self.n_basis))
        first_basis_array = np.empty((len(self.nodes),self.n_basis))
        second_basis_array = np.empty((len(self.nodes),self.n_basis))

        for i in range(self.n_basis):
            basis_array[:,i] = self.bfuncs[i](self.nodes)
            first_basis_array[:,i] = self.bfuncs[i](self.nodes,1)
            second_basis_array[:,i] = self.bfuncs[i](self.nodes,2)
        
        self.barray = basis_array
        self.first_barray = first_basis_array
        self.second_barray = second_basis_array
class TISE:
    def __init__(self):
        self.FFH_R_list = []
    def CreateH_l(self,n_basis,basis_object,nodes,weights,l):
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

                    H_element_1 = np.sum(weights * basis_object.barray[:,i] * (-0.5)* basis_object.second_barray[:,j])
                    H_element_2 = np.sum(weights * basis_object.barray[:,i] * basis_object.barray[:,j] * l*(l+1)/(2*np.sqrt(nodes**4 + 1E-25 )))
                    H_element_3 = np.sum(weights * basis_object.barray[:,i] * basis_object.barray[:,j] * (-1/np.sqrt(nodes**2 + 1E-25)))

                    S_element = np.sum(weights * basis_object.barray[:,i] * basis_object.barray[:,j])
                    
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
            
        if (input_par["lm"]["lmax"] >= input_par["lm"]["nmax"]):
            nmax = input_par["lm"]["lmax"] +1
        else:
            nmax = input_par["lm"]["nmax"] 

        for i,l in enumerate(range(nmax)):
            print(nmax)
            H = self.FFH_R_list[i]

            E = SLEPc.EPS().create()
            E.setOperators(H, self.S_R)
            
            num_of_energies = nmax - i
            E.setDimensions(nev=num_of_energies)
            
            E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
            E.setType(slepc4py.SLEPc.EPS.Type.KRYLOVSCHUR)
            
            E.solve()

            nconv = E.getConverged()
            
            for i in range(nconv):
                eigenvalue = E.getEigenvalue(i)  # This retrieves the eigenvalue
                
                if np.real(eigenvalue) > 0:
                    continue

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

    def PlotPulse(self,time,bool):
        if bool:
            plt.figure()
            plt.plot(time,self.pulse)
            plt.savefig("pulse.png")
            plt.clf()
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
    def H_MIX(self,n_basis,weights,nodes,basis_object,dt):
        H_mix_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD,nnz = 2)
        H_mix_lm.setUp()

        istart,iend = H_mix_lm.getOwnershipRange()
        rows,cols = H_mix_lm.getSize()
        for i in range(istart,iend):
            for j in range(cols):
                clm = np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3))) * 1j * dt /2
                if i == j+1:
                    H_mix_lm.setValue(i,j,clm)
                elif j == i+1:
                    H_mix_lm.setValue(i,j,clm)
        H_mix_lm.assemble()

        H_mix_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        istart,iend = H_mix_R.getOwnershipRange()
        rows,cols = H_mix_R.getSize()
        for i in range(istart,iend):
            for j in range(cols):
                H_element = np.sum(weights * basis_object.barray[:,i] *  basis_object.first_barray[:,j])
                H_mix_R.setValue(i,j,H_element)
        H_mix_R.assemble()

        total = kron.kronV3(H_mix_lm,H_mix_R)
        total.scale(-1j)

        H_mix_lm.destroy()
        H_mix_R.destroy()
        self.H_mix = total
        return None
    def H_ANG(self,n_basis,weights,nodes,basis_object,dt):
        H_ang_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD,nnz = 2)
        istart,iend = H_ang_lm.getOwnershipRange()
        rows,cols = H_ang_lm.getSize()
        for i in range(istart,iend):
            for j in range(cols):
                clm = np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3)))* 1j * dt /2
                if j == i+1:
                    H_ang_lm.setValue(i,j,(i+1)*clm)
                elif i == j+1:
                    H_ang_lm.setValue(i,j,-(i+1)*clm)
        H_ang_lm.assemble()

        H_ang_R =  PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        istart,iend = H_ang_R.getOwnershipRange()
        rows,cols = H_ang_R.getSize()
        for i in range(istart,iend):
            for j in range(cols):
                H_element = np.sum(weights * basis_object.barray[:,i] *  basis_object.barray[:,j] / (nodes))
                H_ang_R.setValue(i,j,H_element)
        H_ang_R.assemble()

        total = kron.kronV3(H_ang_lm,H_ang_R)
        total.scale(-1j)

        H_ang_lm.destroy()
        H_ang_R.destroy()

        self.H_ang = total
        return None  
    def H_ATOM(self,H_list,n_basis,dt):
        H_atom = PETSc.Mat().createAIJ([(self.lmax +1)*n_basis,(self.lmax +1)*n_basis],comm = PETSc.COMM_WORLD)
        H_atom.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES,True)
        local_H = []
        for l in range(self.lmax+1):
            local_H.append(getLocal(H_list[l]))

        istart,iend = H_atom.getOwnershipRange()
        rows,cols = H_atom.getSize()
        for i in range(istart,iend):
            l = i // n_basis
            l_row = i % n_basis

            index,vals = local_H[l].getRow(l_row)

            full_row = np.zeros(n_basis,dtype = "complex")
            full_row[index] = vals
            row_array = np.pad(full_row,(l*n_basis,(self.lmax-l)*n_basis),constant_values= (0,0))

            for j in range(cols):

                row_element = row_array[j] 
                #if np.abs(row_element)<= 1E-10:
                    #continue
                H_atom.setValue(i,j,row_element)
        H_atom.assemble()
        self.H_atom = H_atom
        return None
    def S(self,S_R,n_basis):
        I = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        istart,iend = I.getOwnershipRange()
        for i in range(istart,iend):
            I.setValue(i,i,1)
        I.assemble()

        total = kron.kronV3(I,S_R)

        I.destroy()
        self.S = total
        return None
    def PartialAtomic(self,dt):

        S_copy_L = self.S.copy()
        S_copy_R = self.S.copy()
        
        S_copy_L.axpy(-1j*dt/2,self.H_atom)
        S_copy_R.axpy(1j*dt/2,self.H_atom)


        self.partial_L = S_copy_L
        self.partial_R = S_copy_R
    def PartialAngular(self):
        H_mix_copy = self.H_mix.copy()
        H_mix_copy.axpy(1,self.H_ang)

        self.partial_angular = H_mix_copy
        return None


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
    splines.PlotFuncs(box.r,plot = False)
    splines.CreateGauss(box.rmax)
    splines.EvalGauss()
    
    FieldFreeH = TISE()
    for l in range(input_par["lm"]["lmax"]+1):
        FieldFreeH.CreateH_l(splines.n_basis,splines,splines.nodes,splines.weights,l)
    FieldFreeH.EvalEigen()

    Field = Laser(input_par["laser"]["w"],input_par["laser"]["I"])
    Field.CreateEnv(box.t,box.tmax,input_par["box"]["N"])
    Field.CreatePulse(box.t)
    Field.PlotPulse(box.t,False)

    psi = Psi(splines.n_basis,input_par["lm"]["lmax"])

    
    Int = Hamiltonian()
    Int.H_MIX(splines.n_basis,splines.weights,splines.nodes,splines,box.dt)
    Int.H_ANG(splines.n_basis,splines.weights,splines.nodes,splines,box.dt)
    Int.H_ATOM(FieldFreeH.FFH_R_list,splines.n_basis,box.dt)
    Int.S(FieldFreeH.S_R,splines.n_basis)
    Int.PartialAtomic(box.dt)
    Int.PartialAngular()

    
    for l in range(input_par["lm"]["lmax"]+1):
        FieldFreeH.FFH_R_list[l].destroy()
    

    

    


    #PETSc.Log.begin()
    
    
    
    #PETSc.Log.view()
    
    gc.collect()

    test = False
    if test:
        L = Int.H_atom.getVecRight()
        Int.H_atom.mult(psi.psi_initial,L)

        R = Int.S.getVecRight()
        Int.S.mult(psi.psi_initial,R)

        if PETSc.COMM_WORLD.rank == 0:
            print(L.getValue(0))
            print(R.getValue(0)*-0.5)

    test2 = False
    if test2:
        
        L = len(box.t)
       

        psi_initial = psi.psi_initial.copy()
        ksp = PETSc.KSP().create(PETSc.COMM_WORLD)
        pulse = Field.pulse

        PETSc.Log.begin()
        #for i,t in enumerate([0]):
        for i,t in enumerate(box.t):
            if PETSc.COMM_WORLD.rank == 0:

                print(i,L)
            partial_L_copy = Int.partial_L.copy()
            partial_R_copy = Int.partial_R.copy()

            partial_L_copy.axpy(-pulse[i],Int.partial_angular,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)
            partial_R_copy.axpy(pulse[i],Int.partial_angular,structure =petsc4py.PETSc.Mat.Structure.DIFFERENT_NONZERO_PATTERN)


            
            known = partial_R_copy.getVecRight()
            solution = partial_L_copy.getVecRight()

            partial_R_copy.mult(psi_initial,known)

            ksp.setOperators(partial_L_copy)

            ksp.solve(known,solution)

            psi_initial.copy(solution)

            partial_L_copy.destroy()
            partial_R_copy.destroy()
            known.destroy()
            solution.destroy()
    
    test3 = False
    if test3:
        from scipy.integrate import trapz
        with h5py.File('Hydrogen.h5', 'r') as f:
            data = f[f"/Psi_{2}_{0}"][:]

            real_part = data[:,0]
            imaginary_part = data[:,1]
            total = real_part + 1j*imaginary_part
        psi_r = np.zeros(len(box.r),dtype = "complex")
        for i,coeff in enumerate(total):
            psi_r += coeff*splines.bfuncs[i](box.r)

        if PETSc.COMM_WORLD.rank == 0:
            print(trapz(np.abs(psi_r)**2,box.r))

        inner_prod = 0
        seq_S = getLocal(FieldFreeH.S_R)
        for i,ci in enumerate(total):
            for j,cj in enumerate(total):
                inner_prod += np.conjugate(ci)*cj * seq_S.getValue(i,j)
        if PETSc.COMM_WORLD.rank == 0:
            print(inner_prod)

    if PETSc.COMM_WORLD.rank ==0:
        end = time.time()
        print(f"Total Time:{end-start}")

            

        

   
   
    

    
        