import numpy as np
import json
import time
import matplotlib.pyplot as plt
import h5py
import os
from scipy.interpolate import BSpline


import slepc4py
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI



class Grid:
    def __init__(self,grid_size,grid_spacing,time_size,time_spacing):
        self.r = np.linspace(0,grid_size,int(grid_size/grid_spacing)) #Chagned back to grid spacing
        self.t = np.arange(-time_size/2,time_size/2 + time_spacing,time_spacing)

        self.rmax = np.max(self.r)
        self.dr = grid_spacing

        self.tmax = time_size
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
        FFH_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        S_R = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)

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


class Hamiltonian:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)
        self.lmax = input_par["lm"]["lmax"]
        self.m = input_par["state"][2]
    def kron_product(self,A, B):
            """
            Compute the Kronecker product of two PETSc matrices.
            """
            m, n = A.getSize()
            p, q = B.getSize()
            r, s = m * p, n * q  # Size of the resulting matrix
           
            # Create the resulting matrix
            C = PETSc.Mat().createAIJ([r, s], comm=PETSc.COMM_WORLD)
            C.setFromOptions()
            C.setUp()

            # Compute the Kronecker product in parallel
            A_ownership_range = A.getOwnershipRange()
            B_ownership_range = B.getOwnershipRange()
    
            for i in range(A_ownership_range[0], A_ownership_range[1]):
                for j in range(n):
                    # Only proceed if the current process owns this row
                    
            
                    # Get the value from matrix A
                    v_A = A.getValue(i, j)
            
                    if v_A != 0:  # Skip if zero to avoid unnecessary computations
                        for k in range(B_ownership_range[0], B_ownership_range[1]):
                            for l in range(q):
                                # Only proceed if the current process owns this row
                                
                        
                        # Get the value from matrix B
                                v_B = B.getValue(k, l)
                        
                        # Compute the new indices for the result matrix
                                new_i = i * p + k
                                new_j = j * q + l
                        
                                # Set the value in the result matrix
                                C[new_i, new_j] = v_A * v_B

            # Assemble the matrix
            C.assemblyBegin()
            C.assemblyEnd()
    
            return C
    def H_MIX(self,n_basis,weights,nodes,basis_funcs):
        H_mix_lm = PETSc.Mat().createAIJ([self.lmax,self.lmax],comm = PETSc.COMM_WORLD)
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
        H_ang_lm = PETSc.Mat().createAIJ([self.lmax,self.lmax],comm = PETSc.COMM_WORLD)
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
        H_atom_lm = PETSc.Mat().createAIJ([self.lmax,self.lmax],comm = PETSc.COMM_WORLD)
        H_atom_lm.setValue(0,0,1)
        H_atom_lm.assemblyBegin()
        H_atom_lm.assemblyEnd()

        H_atom = self.kron_product(H_atom_lm,H_list[0])
        for l in range(1,self.lmax):
            intermediate = self.kron_product(H_atom_lm,H_list[l])
            H_atom.axpy(1.0, intermediate)
        self.H_atom = H_atom



        return None
    def H_TOTAL(self):

        intermediate = self.H_mix
        intermediate.axpy(1,self.H_ang)
        intermediate.axpy(1,self.H_atom)
    
        self.H_total = intermediate
        return None




if __name__ == "__main__":
    if PETSc.COMM_WORLD.rank ==0:
        start = time.time()
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    
    box_par = tuple(input_par["box"].values())
    box = Grid(*box_par)
    box.Print(False)

    splines_par = tuple(input_par["splines"].values())
    splines = Basis(*splines_par)
    splines.CreateFuncs(box.rmax,box.dr)
    splines.PlotFuncs(box.r,False)
    splines.CreateGauss(box.rmax)
    splines.EvalGauss()
    
    FieldFreeH = TISE()
    for l in range(input_par["lm"]["lmax"]):
        FieldFreeH.CreateH_l(splines.n_basis,splines.barrays_gauss,splines.barrays_der_gauss,splines.nodes,splines.weights,l)
    FieldFreeH.EvalEigen()

    
    Int = Hamiltonian()
    Int.H_MIX(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    Int.H_ANG(splines.n_basis,splines.weights,splines.nodes,splines.bfuncs)
    Int.H_ATOM(FieldFreeH.FFH_R_list)
    Int.H_TOTAL()

    if PETSc.COMM_WORLD.rank ==0:
        end = time.time()
        print(f"Total Time:{end-start}")

        