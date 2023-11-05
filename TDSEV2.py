import numpy as np
import json
import time
import matplotlib.pyplot as plt
import h5py
import os
from scipy.interpolate import BSpline



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
    

    def EvalEigen(self,S):
        with h5py.File('Hydrogen.h5', 'w') as hf:
            eigvec_group = hf.require_group("eigvecs")
            eigval_group = hf.require_group("eigvals")
            for l,H in enumerate(self.H_list):
                eigvals, eigvecs = eigsh(H, M=S,which = "SM",k = 10)
                print(eigvals[:3])
                for i in range(10):
                    if not f"eigvecs/Psi_{i+1+l}_{l}" in hf:
                        eigvec_group.create_dataset(f"Psi_{i+1+l}_{l}",data = eigvecs[:,i])
                    if not f"eigvals/{i+1+l}" in hf:
                        eigval_group.create_dataset(f"{i+1+l}",data = eigvals[i])
        return eigvals,eigvecs
    def EvalEigen(self):
        
        ViewHDF5 = PETSc.Viewer().createHDF5("Hydrogen.h5", mode=PETSc.Viewer.Mode.WRITE, comm= PETSc.COMM_WORLD)
            
            
            
        for l,H in enumerate(self.FFH_R_list):
            E = SLEPc.EPS().create()
            E.setOperators(H, self.S_R)
            E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
            E.setDimensions(nev=3)
            E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
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
    for l in range(2):
        FieldFreeH.CreateH_l(splines.n_basis,splines.barrays_gauss,splines.barrays_der_gauss,splines.nodes,splines.weights,l)
    FieldFreeH.EvalEigen()

    if PETSc.COMM_WORLD.rank ==0:
        end = time.time()
   
        print(f"Total Time:{end-start}")

        