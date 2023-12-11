import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from petsc4py import PETSc
import json

class basis:
    def __init__(self):
        with open('input.json', 'r') as file:
            input_par = json.load(file)

        self.N_knots = input_par["splines"]["N_knots"]
        self.order = input_par["splines"]["order"]
        self.N_gauss = input_par["splines"]["N_gauss"]
        self.knot_spacing = input_par["splines"]["knot_spacing"]

    def createFuncs(self,gridInstance):
        dr = gridInstance.dr
        rmax = gridInstance.rmax
        
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

        if self.knot_spacing == "linear":
            basis_funcs,n_basis = Linear()
        elif self.knot_spacing == "quad":
            basis_funcs,n_basis = Quadratic()

    
        self.n_basis = len(basis_funcs)
        self.bfuncs = basis_funcs

        return None
    
    def plotFuncs(self,gridInstance,plot):
        r = gridInstance.r
        if plot:
            plt.figure()
            basis_array = np.empty((len(r),self.n_basis))
            for i in range(self.n_basis):
                basis_array[:,i] = self.bfuncs[i](r)
                plt.plot(r,basis_array[:,i])
            plt.savefig("b_splines.png")
            plt.clf()
        np.save("basis.npy",basis_array)
        return None

    def createGauss(self,gridInstance):
        rmax = gridInstance.rmax
        x,w = np.polynomial.legendre.leggauss(self.N_gauss)

        transformed_nodes = 0.5 * rmax * x + 0.5 * rmax
        adjusted_weights = 0.5 * rmax * w

        self.nodes = transformed_nodes
        self.weights = adjusted_weights

    def evalGauss(self):
        
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


if __name__ == "__main__":
    basisInstance = basis()