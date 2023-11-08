import numpy as np
import json
import time
import matplotlib.pyplot as plt
import h5py
import os
from scipy.interpolate import BSpline
from scipy.sparse import spdiags
from scipy.sparse import lil_matrix
from scipy.sparse import kron
import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI

from numpy import pi


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
            
            
            
        for l,H in enumerate(self.FFH_R_list):
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

class Laser:
    def __init__(self,w,I):
        self.I = I/3.51E16
        self.freq = w
        
        return None
    
    def CreateEnv(self,time,tmax):
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
        H_mix_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        istart,iend = H_mix_lm.getOwnershipRange()
        for i in range(istart,iend-1):

            clm = ((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3))
            H_mix_lm.setValue(i,i+1,clm)
            H_mix_lm.setValue(i+1,i,clm)
        H_mix_lm.assemblyBegin()
        H_mix_lm.assemblyEnd()

        H_mix_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_mix_R.getOwnershipRange()
        columnstart,columnend = H_mix_R.getOwnershipRangeColumn()
        for i in range(rowstart,rowend):
            for j in range(columnstart,columnend):
                H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes,1))
                H_mix_R.setValue(i,j,H_element)
        H_mix_R.assemblyBegin()
        H_mix_R.assemblyEnd()

        output = self.kron_product(H_mix_lm,H_mix_R)
        output.scale(-1j)

        self.H_mix = output

        return None
    def H_ANG(self,n_basis,weights,nodes,basis_funcs):
        H_ang_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        istart,iend = H_ang_lm.getOwnershipRange()
        for i in range(istart,iend-1):

            clm = ((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3))
            H_ang_lm.setValue(i,i+1,(i+i)*clm)
            H_ang_lm.setValue(i+1,i,-(i+1)*clm)
        H_ang_lm.assemblyBegin()
        H_ang_lm.assemblyEnd()

        H_ang_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_ang_R.getOwnershipRange()
        columnstart,columnend = H_ang_R.getOwnershipRangeColumn()
        for i in range(rowstart,rowend):
            for j in range(columnstart,columnend):
                H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes) / (nodes))
                H_ang_R.setValue(i,j,H_element)
        H_ang_R.assemblyBegin()
        H_ang_R.assemblyEnd()

        output = self.kron_product(H_ang_lm,H_ang_R)
        output.scale(-1j)

        
        self.H_ang = output

        return None
    def H_ATOM(self,H_list):
        H_atom_lm = PETSc.Mat().createAIJ([self.lmax+1,self.lmax+1],comm = PETSc.COMM_WORLD)
        H_atom_lm.setValue(0,0,1)
        H_atom_lm.assemblyBegin()
        H_atom_lm.assemblyEnd()

        H_atom = self.kron_product(H_atom_lm,H_list[0])
        for l in range(1,self.lmax+1):
            
            intermediate = self.kron_product(H_atom_lm,H_list[l])
            H_atom.axpy(1.0, intermediate)
       
        self.H_atom = H_atom



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


class Hamiltonian:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        self.m = input_par["state"][2]
    def H_MIX(self,n_basis,weights,nodes,basis_funcs):
        
        data = -1j * np.array([np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3))) for i in range(self.lmax+1)])
        H_mix_csr = spdiags([data,data],[1,-1],n = self.lmax+1,m = self.lmax+1,format = "csr")
        
        H_R_lil = lil_matrix((n_basis,n_basis),dtype = "complex")
        for i in range(n_basis):
            for j in range(n_basis):
                if i>=j:
                    H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes,1))
                    H_R_lil[i,j] = H_element
                    if i!=j:
                        H_R_lil[j,i] = H_element
        H_R_mix_csr = H_R_lil.tocsr()

        H_mix_csr = kron(H_mix_csr,H_R_mix_csr)
        H_mix_lil = H_mix_csr.tolil()

        H_mix_petsc = PETSc.Mat().createAIJ([n_basis*(self.lmax+1),n_basis*(self.lmax+1)],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_mix_petsc.getOwnershipRange()
        row,col = H_mix_petsc.getSize()

        for i in range(rowstart,rowend):
            for j in range(col):
                H_mix_petsc.setValue(i,j,H_mix_lil[i,j])
        H_mix_petsc.assemblyBegin()
        H_mix_petsc.assemblyEnd()

        self.H_MIX = H_mix_petsc
    def H_ANG(self,n_basis,weights,nodes,basis_funcs):
        data1 = -1j * np.array([(i+1)*np.sqrt(((i+1)**2 - self.m**2)/((2*i+1)*(2*i+3))) for i in range(self.lmax+1)])
        data2 = -data1

        H_ang_csr = spdiags([data1,data2],[1,-1],n = self.lmax+1,m = self.lmax+1,format = "csr")

        H_R_lil = lil_matrix((n_basis,n_basis),dtype = "complex")
        for i in range(n_basis):
            for j in range(n_basis):
                if i>=j:
                    H_element = np.sum(weights * basis_funcs[j](nodes) *  basis_funcs[j](nodes) / (nodes))
                    H_R_lil[i,j] = H_element
                    if i!=j:
                        H_R_lil[j,i] = H_element
        H_R_ang_csr = H_R_lil.tocsr()

        H_ang_csr = kron(H_ang_csr,H_R_ang_csr)
        H_ang_lil = H_ang_csr.tolil()

        H_ang_petsc = PETSc.Mat().createAIJ([n_basis*(self.lmax+1),n_basis*(self.lmax+1)],comm = PETSc.COMM_WORLD)
        rowstart,rowend = H_ang_petsc.getOwnershipRange()
        row,col = H_ang_petsc.getSize()
        for i in range(rowstart,rowend):
            for j in range(col):
                H_ang_petsc.setValue(i,j,H_ang_lil[i,j])
        H_ang_petsc.assemblyBegin()
        H_ang_petsc.assemblyEnd()

        self.H_ANG = H_ang_petsc












        

        
        

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

    #Field = Laser(input_par["laser"]["w"],input_par["laser"]["I"])
    #Field.CreateEnv(box.t,box.tmax)
    #Field.CreatePulse(box.t)

    #psi = Psi(splines.n_basis,input_par["lm"]["lmax"])

    
    Int = Hamiltonian()
    Int.H_MIX(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    Int.H_ANG(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    #Int.H_ATOM(FieldFreeH.FFH_R_list)


    
 



    
    
    
   
    

    
    
            



   

   
        
    

    



    

  
    
  
    

  





    
   

   
    

    '''
    psi_initial = psi.psi_initial
    ksp = PETSc.KSP().create(comm=PETSc.COMM_WORLD)
    y = PETSc.Vec().createMPI(psi_initial.getSize(), comm=PETSc.COMM_WORLD)
    b = y.duplicate()
    for i,t in enumerate(box.t):
        print(f"Step {i} of {len(box.t)}")
        Int.H_TOTAL(Field.pulse[i])
        INTERACTION = Int.H_total
        INTERACTION.scale(box.dt / 2)

        I = PETSc.Mat().createAIJ(size=INTERACTION.getSize(), comm=PETSc.COMM_WORLD)
        jstart,jend = I.getOwnershipRange()
        for j in range(jstart,jend):
            I.setValue(j,j,1)
        I.assemblyBegin()
        I.assemblyEnd()

        H_LEFT = I.copy()  # Creates a duplicate of identity matrix with a separate memory
        H_RIGHT = I.copy() # Same here

        H_LEFT.axpy(1, INTERACTION)  # H_LEFT = I + dt/2 * H
        H_RIGHT.axpy(-1, INTERACTION) # H_RIGHT = I - dt/2 * H

        H_RIGHT.mult(psi_initial, b) # b = (I - dt/2 * H) * psi_initial

        ksp.setOperators(H_LEFT) # Set the operator for the linear solve
        ksp.solve(b, y) # Solve H_LEFT * y = b

        y.copy(psi_initial) # Update psi_initial for the next iteration
    ksp.destroy()
    H_LEFT.destroy()
    H_RIGHT.destroy()
    y.destroy()
    b.destroy()
    print(psi_initial.getArray())
    '''
    if PETSc.COMM_WORLD.rank ==0:
        end = time.time()
        print(f"Total Time:{end-start}")
    

    
        