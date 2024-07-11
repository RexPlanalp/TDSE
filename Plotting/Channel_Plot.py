import numpy as np 
import matplotlib.pyplot as plt

Ip = -0.5 # Ionization potential of species in atomic units
N = 10 # Number of cycles of laser pulse 
w = 0.057 # Central Frequency of laser pulse in atomic units
I = 2.0e14 / 3.51E16 # Intensity of laser pulse in atomic units
E = 0.48 # Energy in atomic units (usually at ATI peak)

tau = 2*np.pi/w # Period of laser pulse in atomic units
t = np.linspace(0,tau,1000) # Time array
envelope = np.sin(np.pi*t/tau)**2 # Envelope function, usually Sin2
I_profile = I * envelope # Intensity profile of laser pulse
Up = I_profile/(4*w**2) # Pondermotive energy in atomic units

# Returns the number of photons able to be absorned to read E over time
def findPhotons(E,Ip,Up,w):
    n = (E-Ip+Up)/w
    return n

# Return indices of time where channel closing occurs
def findChannelIndices(n):
    return np.where(np.diff(np.floor(n)))[0]

# Gets number of photons and indices of channel closings
n = findPhotons(E,Ip,Up,w)
channel_indices = findChannelIndices(n)

# Plots photons between channel closings
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
def segment_indices_based_on_nearest_integer(y):
    rounded_y = np.round(y)  # Round the values to the nearest integer
    changes = np.diff(rounded_y)  # Find where the rounded values change
    change_indices = np.where(changes != 0)[0] + 1  # Get indices right after changes

    segments = []
    start_idx = 0
    for idx in change_indices:
        segments.append((start_idx, idx))  # Store start and end index of each segment
        start_idx = idx
    segments.append((start_idx, len(y)))  # Add the last segment

    return segments

segments = segment_indices_based_on_nearest_integer(n)

pulse_energies = []
for i, (start, end) in enumerate(segments):
    plt.plot(t[start:end], n[start:end], label=f"Segment {i+1}")

    pulse_energy = np.trapz(I_profile[start:end], t[start:end])
    pulse_energies.append(pulse_energy)

max_index = np.argmax(pulse_energies)
start_max,end_max = segments[max_index]

second_index= max_index-1
start_second,end_second = segments[second_index]

third_index= max_index-2
start_third,end_third = segments[third_index]

print(f"Segment {max_index} with pulse energy {pulse_energies[max_index]} and value {np.average(np.round(n[start_max:end_max]))}")
print(f"Segment {second_index} with pulse energy {pulse_energies[second_index]} and value {np.average(np.round(n[start_second:end_second]))}")

print(f"Segment {third_index} with pulse energy {pulse_energies[third_index]} and value {np.average(np.round(n[start_third:end_third]))}")

