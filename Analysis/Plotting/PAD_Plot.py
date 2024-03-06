import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys


ANGULAR = "ANGULAR" in sys.argv
RECT = "RECT" in sys.argv

# Use an existing colormap
base_cmap = plt.cm.plasma


# Create a new colormap from the existing colormap
# Start with fully transparent and gradually become opaque
cmap_with_transparency = mcolors.LinearSegmentedColormap.from_list(
    name='transparent_viridis',
    colors=[(0, 0, 0, 0)] + [(c[0], c[1], c[2], i / 256) for i, c in enumerate(base_cmap(np.arange(256)))],
    N=256
)


from matplotlib.colors import LinearSegmentedColormap

# Define the colors for the colormap (white for near zero, black otherwise)
colors = [(0, 0, 0), (1, 1, 1), (0, 0, 0)]  # Black -> White -> Black

# Define the positions of the colors (0.5 is the midpoint)
position = [0, 0.5, 1]

# Create the custom colormap
custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', list(zip(position, colors)))

PAD = np.load("PAD.npy")


E_vals = PAD[0,:]
theta_vals = PAD[1,:]
phi_vals = PAD[2,:]
pad_vals = PAD[3,:]

pad_vals = np.real(pad_vals)
x_vals = E_vals * np.sin(theta_vals) * np.cos(phi_vals)
y_vals = E_vals * np.sin(theta_vals) * np.sin(phi_vals)
z_vals = E_vals * np.cos(theta_vals)

# Momentum Transform
px_vals = np.sqrt(2*E_vals) * np.sin(theta_vals) * np.cos(phi_vals)
py_vals = np.sqrt(2*E_vals) * np.sin(theta_vals) * np.sin(phi_vals)
pz_vals = np.sqrt(2*E_vals) * np.cos(theta_vals)
pad_momentum = pad_vals/np.sqrt(2*E_vals)



if ANGULAR:
    #plt.scatter(pz_vals,px_vals,c=pad_momentum, cmap="binary", norm=mcolors.LogNorm(vmin=10**(-11.13), vmax=10**(-5.13)))
    #plt.scatter(px_vals,py_vals,c=pad_momentum/pad_momentum.max(), cmap=cmap_with_transparency, norm=mcolors.LogNorm(vmin=10**(-0.3), vmax=10**(0)))
    plt.scatter(px_vals,py_vals,c=pad_momentum/np.max(pad_momentum),cmap="binary", norm=mcolors.LogNorm(vmin = 10**(-1.5),vmax = 10**(0)))
    plt.colorbar()
    plt.savefig("PAD.png")

if RECT:
    #plt.scatter(E_vals,theta_vals,c=pad_vals, cmap="binary", norm=mcolors.LogNorm(vmin=10**(-11.13), vmax=10**(-5.13)))
    plt.scatter(E_vals,phi_vals,c=pad_vals, cmap="binary", norm=mcolors.LogNorm())
    plt.savefig("E.png")

    

