import numpy as np
import json
import time
import matplotlib.pyplot as plt
import h5py
import os
from petsc4py import PETSc
from slepc4py import SLEPc
from mpi4py import MPI
from scipy.interpolate import BSpline
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

class Grid:
    def __init__(self,grid_size,grid_spacing,time_size,time_spacing):
        self.r = np.linspace(0,grid_size,int(grid_size/grid_spacing)) #Chagned back to grid spacing
        self.t = np.arange(-time_size/2,time_size/2 + time_spacing,time_spacing)

        self.rmax = np.max(self.r)
        self.dr = grid_spacing

        self.tmax = time_size
        self.dt = time_spacing
    def Print(self):
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
    
    def PlotFuncs(self,r):

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

class FF_Hamiltonian:
    def __init__(self):
        self.H_list = []
    def CreateH_l(self,n_basis,barrays,barrays_der,nodes,weights,l):
        rank = PETSc.COMM_WORLD.getRank()
        if rank == 0:
            start = time.time()
        H_test = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        S_test = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)

        rowstart,rowend = H_test.getOwnershipRange()
        columnstart,columnend = H_test.getOwnershipRangeColumn()
        for i in range(rowstart,rowend):
            for j in range(columnstart,columnend):
                if i >= j:

                    H_element_1 = np.sum(weights * barrays[:,i] * (-0.5)* barrays_der[:,j])
                    H_element_2 = np.sum(weights * barrays[:,i] * barrays[:,j] * l*(l+1)/(2*np.sqrt(nodes**4 + 1E-25 )))
                    H_element_3 = np.sum(weights * barrays[:,i] * barrays[:,j] * (-1/np.sqrt(nodes**2 + 1E-25)))

                    S_element = np.sum(weights * barrays[:,i] * barrays[:,j])

                    H_element = H_element_1 + H_element_2 + H_element_3

                    
                    H_test.setValue(i,j,H_element)
                    S_test.setValue(i,j,S_element)

                    if i != j:
                        
                        H_test.setValue(j,i,H_element)
                        S_test.setValue(j,i,S_element)
                        
        H_test.assemblyBegin()
        H_test.assemblyEnd()

        S_test.assemblyBegin()
        S_test.assemblyEnd()


        E = SLEPc.EPS().create()
        E.setOperators(H_test, S_test)
        E.setProblemType(SLEPc.EPS.ProblemType.GHEP)
        E.setDimensions(nev=3)
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        E.solve()

        nconv = E.getConverged()
        eigenvalues = [E.getEigenvalue(i) for i in range(nconv)]
        
        if rank == 0:
            end = time.time()
            print(eigenvalues)
            print(end-start)
        
        
        
        

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

if __name__ == "__main__":
    
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    box_par = tuple(input_par["box"].values())
    box = Grid(*box_par)
    #box.Print()


    splines_par = tuple(input_par["splines"].values())
    splines = Basis(*splines_par)
    splines.CreateFuncs(box.rmax,box.dr)
    #splines.PlotFuncs(box.r)
    splines.CreateGauss(box.rmax)
    splines.EvalGauss()
    #splines.CreateOverlap()

    start = time.time()
    FieldFreeH = FF_Hamiltonian()
    for l in range(1):
        FieldFreeH.CreateH_l(splines.n_basis,splines.barrays_gauss,splines.barrays_der_gauss,splines.nodes,splines.weights,l)
    #FieldFreeH.EvalEigen(splines.S)
    end = time.time()
    #print(f"Total Time:{end-start}")

        