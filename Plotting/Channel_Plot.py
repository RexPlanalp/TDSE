import numpy as np 
import matplotlib.pyplot as plt

##########################################################################################################################

Ip = -0.5 # Ionization potential of species in atomic units
N = 10 # Numbe08r of cycles of laser pulse 
w = 0.057 # Central Frequency of laser pulse in atomic units
I = 2.06e14 / 3.51E16 # Intensity of laser pulse in atomic units
E = 0.48 # Energy in atomic units (usually at ATI peak)

tau = 2*np.pi/w # Period of laser pulse in atomic units
t = np.linspace(0,tau,1000) # Time array
envelope = np.sin(np.pi*t/tau)**2 # Envelope function, usually Sin2
I_profile = I * envelope # Intensity profile of laser pulse
Up = I_profile/(4*w**2) # Pondermotive energy in atomic units

############################################################################################################################

def findPhotons(E,Ip,Up,w):
    n = (E-Ip+Up)/w
    return n

n = findPhotons(E,Ip,Up,w)
channel_indices = np.where(np.diff(np.floor(n)))[0]

for idx in channel_indices:
    x_int = t[idx]
    y_int = n[idx]
    plt.plot([x_int, x_int], [np.min(n), y_int], color='k')
    plt.axhline(y_int,linestyle="dashed")

plt.plot(t,n)
plt.savefig("images/channels.png")
plt.clf()
#############################################################################################################################

def findIntensityParameter(Up,w):
    z = Up/w
    return z

z = findIntensityParameter(Up,w)
plt.plot(t,z)
print(f"Max value of Up/w is {np.max(z)}")
plt.savefig("images/z.png")
plt.clf()

##########################################################################################################################################
# total_potential = Up - Ip

# offset = np.max(total_potential)
# n = 25
# print(round(n*w - offset,3))
################################################################################################################################
# w = 0.057  # Central Frequency of laser pulse in atomic units
# import math
# for I_max in np.arange(0.00001, 0.01 + 0.00001, 0.00001):
#     Up = I_max / (4 * w**2)
#     z = Up / w
#     rounded_z = round(z)
#     if math.isclose(z, rounded_z, rel_tol=1e-2):
#         print(f"z value:{z}, and intensity in SI { I_max * 3.51E16:.3e}")

# I_max = (2.08e14 / 3.51E16)
# import math
# for w in np.arange(0.000005, 0.2 + 0.000005, 0.000005):
#     Up = I_max / (4 * w**2)
#     z = Up / w
#     rounded_z = round(z)
#     if math.isclose(z, rounded_z, rel_tol=1e-4) and z<10:
#         print(f"z value:{z}, and wavelength in SI { (0.057/w)*800}")
       