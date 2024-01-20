import numpy as np
import json
import time
import matplotlib.pyplot as plt
import os

from scipy.integrate import trapz
from scipy.interpolate import BSpline
from scipy.integrate import quad

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

class FF_Hamiltonian:
     def __init__(self,n_basis,basis_funcs,r):
        H_FF = PETSc.Mat().createAIJ([n_basis,n_basis],comm = PETSc.COMM_WORLD)
        H_FF.setFromOptions()
        H_FF.setUp()

        rowstart,rowend = H_FF.getOwnershipRange()
        columnstart,columnend = H_FF.getOwnershipRangeColumn()

        for i in range(rowstart,rowend):
            for j in range(columnstart,columnend):
                    
                T= trapz(-0.5*basis_funcs[i](r),basis_funcs[j](r,2),r)
                V = trapz(basis_funcs[i](r),basis_funcs[j](r) * -1/np.sqrt(r**2 + 1E-25),r)

                H_FF.setValue(i, j, T+V)
        H_FF.assemblyBegin()
        H_FF.assemblyEnd()

        E = SLEPc.EPS()
        E.create(comm=PETSc.COMM_WORLD)
        E.setOperators(H_FF)
        E.setDimensions(nev=3)
        E.setProblemType(SLEPc.EPS.ProblemType.HEP)
        E.setWhichEigenpairs(SLEPc.EPS.Which.SMALLEST_REAL)
        E.solve()


        nconv = E.getConverged()
        eigenvalues = []
        for i in range(nconv):
            eigenvalue = E.getEigenvalue(i)
            eigenvalues.append(eigenvalue)
        print(eigenvalues)
        
    
                    


                         
          
    
   


if __name__ == "__main__":
    
    with open('input.json', 'r') as file:
            input_par = json.load(file)
    
    box_par = tuple(input_par["box"].values())
    box = Grid(*box_par)
    
    splines_par = tuple(input_par["splines"].values())
    splines = Basis(*splines_par)
    splines.CreateFuncs(box.rmax,box.dr)
    
    test = FF_Hamiltonian(splines.n_basis,splines.bfuncs,box.r)

        