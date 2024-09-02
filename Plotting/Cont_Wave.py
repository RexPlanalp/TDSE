import numpy as np
import sys
import h5py 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import gamma


sys.path.append('/users/becker/dopl4670/Research/TDSE/Common')
from Sim import *
from Basis import *

simInstance = Sim("input.json")  
simInstance.lm_block_maps() 
simInstance.calc_n_block()   
simInstance.timeGrid() 
simInstance.spacialGrid() 

order = simInstance.splines["order"]
n_basis = simInstance.splines["n_basis"]

basisInstance = basis()
basisInstance.createKnots(simInstance)
knots = basisInstance.knots

potential = simInstance.box["pot"]


l = 1
E = 0.50
k = np.sqrt(2*E)

with h5py.File(f'{potential}_Cont_Cleaned.h5', 'r') as f:
    data = f[f"Psi_{l}_{E}0"][:]
    real_part = data[:, 0]
    imaginary_part = data[:, 1]
    total = real_part + 1j * imaginary_part

def computePhaseAndNormalizeState(E,total,z,l,knots):
    k = np.sqrt(2*E)

    x_val_idx = np.argmin(np.abs(knots-905))
    x_val = knots[x_val_idx]

    #x_val_idx = np.argmin(np.abs(knots-0.95*np.max(knots)))
    #x_val = knots[x_val_idx]

    val = 0
    der = 0
    for i in range(x_val_idx-order+1,x_val_idx):
        val += total[i] * basisInstance.B(i,order,x_val,knots)
        der += total[i] * basisInstance.dB(i,order,x_val,knots)
    
    #phase = np.angle((1.j*val + der/(k+z/(k*x_val)))/(2*k*x_val)**(1.j*z/k)) - k*x_val + l*np.pi/2
    #A = np.sqrt(np.abs(val)**2+(np.abs(der)/(k+z/(k*x_val))**2))
    #total/= A
    return val,de
#val,der = computePhaseAndNormalizeState(1.000132,total,1,1,knots)

wavefunction = 0
x = np.linspace(1500,1510,1000)
for i in range(simInstance.splines["n_basis"]):
    print(i,n_basis)
    wavefunction += total[i] * basisInstance.B(i,simInstance.splines["order"],x,basisInstance.knots)

PDF = np.abs(wavefunction)**2

plt.plot(x, PDF)
plt.xlim([1000, 1010])
plt.savefig("PDF.png")
plt.clf()

