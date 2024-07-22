import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys

from matplotlib.colors import LinearSegmentedColormap
    
cdict1 = {'red':   ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.1),
                   (1.0, 1.0, 1.0)),
 
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),
 
         'blue':  ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
        }
blue_red1 = LinearSegmentedColormap('BlueRed1',cdict1)
    

ANGULAR = "ANGULAR" in sys.argv
RECT = "RECT" in sys.argv
SLICE = "SLICE" in sys.argv
TEST = "TEST" in sys.argv
ASYM = "ASYM" in sys.argv
PAD_SLICE = 'SLICE' in sys.argv

PAD = np.load("PES_files/PAD.npy")

E_vals = np.real(PAD[0,:])
theta_vals = np.real(PAD[1,:])
phi_vals = np.real(PAD[2,:])
pad_vals = np.real(PAD[3,:])

# Carteisian Energy
x_vals = E_vals * np.sin(theta_vals) * np.cos(phi_vals)
y_vals = E_vals * np.sin(theta_vals) * np.sin(phi_vals)
z_vals = E_vals * np.cos(theta_vals)

# Cartiesian Momentum 
px_vals = np.sqrt(2*E_vals) * np.sin(theta_vals) * np.cos(phi_vals)
py_vals = np.sqrt(2*E_vals) * np.sin(theta_vals) * np.sin(phi_vals)
pz_vals = np.sqrt(2*E_vals) * np.cos(theta_vals)
pad_momentum = pad_vals/np.sqrt(2*E_vals)

max_mom = np.max(np.real(pad_momentum))
min_mom = np.max(np.real(pad_momentum))*10**-2

max_E = np.max(pad_vals)
min_E = np.max(pad_vals)*10**-2

if ANGULAR:
    plt.scatter(px_vals, py_vals, c=pad_vals, cmap="hot_r")
    #plt.scatter(px_vals, py_vals, c=pad_momentum, cmap="hot_r",norm=mcolors.LogNorm(vmin=min_mom,vmax=max_mom))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.savefig("images/PAD.png")

    plt.clf()

    plt.scatter(E_vals,phi_vals,c=pad_vals, cmap=blue_red1)
    #plt.scatter(E_vals,phi_vals,c=pad_vals, cmap=blue_red1,norm = mcolors.LogNorm(vmin=min_E,vmax=max_E))
    plt.colorbar()
    plt.savefig("images/E.png")

if ASYM:
    phi_range = np.unique(phi_vals)

    PAD_dict = {}
    for i,E in enumerate(E_vals):
        PAD_dict[(E,theta_vals[i],phi_vals[i])] = pad_vals[i]
    
    asymmetry_vals = []
    for key,value in PAD_dict.items():
        E,theta,phi = key
        pad_val = value

        target_opposite_phi = (phi+np.pi) % (2*np.pi)
        diff = np.abs(target_opposite_phi-phi_range)
        real_index = np.argmin(diff)
        real_opposite_phi = phi_range[real_index]

        pad_val_opposite = PAD_dict[(E,theta,real_opposite_phi)]
        A = (pad_val-pad_val_opposite)/(pad_val+pad_val_opposite)
        asymmetry_vals.append(A)

    asymmetry_vals = np.array(asymmetry_vals)
    asymmetry_momentum = asymmetry_vals/np.sqrt(2*E_vals)

    plt.scatter(px_vals, py_vals, c=asymmetry_momentum, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.savefig("images/A.png")

    plt.clf()

    plt.scatter(E_vals,phi_vals,c=asymmetry_vals, cmap="bwr",vmin = -1,vmax = 1)
    plt.colorbar()
    plt.savefig("images/A_rect.png")

    plt.clf()

    tol = 0.001
    E = 0.544
    mask = np.abs(E_vals - E) < tol
    x, y = phi_vals[mask], asymmetry_vals[mask]
    sorted_indices = np.argsort(x)
    x, y = x[sorted_indices], y[sorted_indices]

    plt.plot(x,y)
    np.save("TDSE_files/A_slice.npy",y)
    plt.savefig("images/A_slice.png")
    
if PAD_SLICE:
    tol = 0.01
    angle = np.deg2rad(270) 
    mask = np.abs(phi_vals - angle) < tol

    x, y = E_vals[mask], pad_vals[mask]
    sorted_indices = np.argsort(x)
    x, y = x[sorted_indices], y[sorted_indices]

    plt.semilogy(x,y)
    plt.plot("images/PAD_slice.png")



