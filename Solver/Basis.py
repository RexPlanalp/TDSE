import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from petsc4py import PETSc
from scipy.integrate import fixed_quad
import json

class basis:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)

        self.N_knots = input_par["splines"]["N_knots"]
        self.order = input_par["splines"]["order"]
        self.degree = self.order - 1
        self.knot_spacing = input_par["splines"]["knot_spacing"]

    def _linearKnots(self,rmax):
        N_middle = self.N_knots - 2*(self.degree)
        
        knots_middle = list(np.linspace(0,rmax,N_middle))
        knots_middle[-1] = rmax
        
        knots_start = (self.degree)*[knots_middle[0]]
        knots_end = (self.degree)*[knots_middle[-1]]
        
        knots = knots_start+knots_middle+knots_end
        return np.array(knots)

    def _quadKnots(self,rmax): 
        N_middle = self.N_knots - 2*self.degree
    
        knots_middle_squared = np.linspace(0,np.sqrt(rmax),N_middle)
        knots_middle = list(knots_middle_squared**2)
        knots_middle[-1] = rmax
    
        knots_start = self.degree*[knots_middle[0]]
        knots_end = self.degree*[knots_middle[-1]]
        
        knots = knots_start+knots_middle+knots_end
        return np.array(knots)

    def integrate(self,func,i,j):
        return fixed_quad(func,self.knots[i+1],self.knots[j+self.order+1],n = 100,args = (i,j))[0]
        
    def createBasis(self,gridInstance):
        rmax = gridInstance.rmax
        
        if self.knot_spacing == "linear":
            knots = self._linearKnots(rmax)
        elif self.knot_spacing == "quad":
            knots = self._quadKnots(rmax)

        spline_list = []
        for i in range(1,len(knots)-self.degree-2):
            y = BSpline(knots,[1 if j ==i else 0 for j in range(len(knots)-self.degree-1)],self.degree,extrapolate = False)
            spline_list.append(y)

        self.basis_funcs = np.array(spline_list)
        self.n_basis = len(spline_list)
        self.knots = knots
        return

    def saveBasis(self,gridInstance,plot):
        r = gridInstance.r
        basis_array = np.empty((len(r),self.n_basis))
        for i in range(self.n_basis):
            basis_array[:,i] = self.basis_funcs[i](r)
            if plot:
                plt.plot(r,basis_array[:,i])
        np.save("basis.npy",basis_array)
        if plot:
            plt.savefig("basis.png")
        return

if __name__ == "__main__":
    basisInstance = basis()