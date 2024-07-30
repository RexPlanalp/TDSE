import numpy as np 
import matplotlib.pyplot as plt

##########################################################################################################################

Ip = -0.5 # Ionization potential of species in atomic units
N = 10 # Numbe08r of cycles of laser pulse 
w = 0.057 # Central Frequency of laser pulse in atomic units
I = 2e14 / 3.51E16 # Intensity of laser pulse in atomic units
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

def findKeldyshParameter(Ip,Up):
    gamma = np.sqrt(-Ip/(2*np.max(Up)))
    return gamma

gamma = findKeldyshParameter(Ip,Up)
print(f"Keldysh parameter is {gamma}")


##########################################################################################################################################
# nearest_integers = np.round(n).astype(int)

# # Create a dictionary to store indices grouped by nearest integer
# grouped_indices = {}
# for i, ni in enumerate(nearest_integers):
#     if ni not in grouped_indices:
#         grouped_indices[ni] = []
#     grouped_indices[ni].append(i)

# # Use the grouped indices to select elements from x and f
# x_groups = {key: t[grouped_indices[key]] for key in grouped_indices}
# n_groups = {key: n[grouped_indices[key]] for key in grouped_indices}
# I_groups = {key: I_profile[grouped_indices[key]] for key in grouped_indices}


# values = []
# for key in x_groups:
#     # Sort the groups by the x values
#     sorted_indices = np.argsort(x_groups[key])
#     x_sorted = x_groups[key][sorted_indices]
#     I_sorted = I_groups[key][sorted_indices]
    
#     values.append(np.average(I_sorted))
# values = values[::-1]
# values /= np.max(values)
# print(values)


num_samples = 1000000
probability_distribution = n / np.sum(n)
plt.plot(t,probability_distribution)
plt.savefig("test1.png")
plt.clf()
sampled_times = np.random.choice(t, size=num_samples, p=probability_distribution)

result_dict = {}

for time in sampled_times:
    idx = np.where(t == time)
    n_value = n[idx]
    n_int = int(np.round(n_value))

    if n_int in result_dict:
        result_dict[n_int] += 1
    else:
        result_dict[n_int] = 1


x_array = []
y_array = []
for key,value in result_dict.items():
    x_array.append(key)
    y_array.append(value)
x_array = np.array(x_array)
y_array = np.array(y_array)
y_array /= np.max(y_array)

plt.savefig("test2.png")