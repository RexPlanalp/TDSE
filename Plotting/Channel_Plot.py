import numpy as np 
import json
import matplotlib.pyplot as plt



N = 10
w = 0.057
I = 0.26e14
E = 0.48

def findPhotons(w,E,I):
    tau = 2*np.pi/w
    t = np.linspace(0,tau,1000)

    envelope = np.sin(np.pi*t/tau)**2
    I_au = (I/3.51E16)*envelope
    Up = I_au/(4*w**2)
    n = (E+0.5+Up)/w
    return t,n

def findChannelIndices(n):
    return np.where(np.diff(np.floor(n)))[0]


t,n = findPhotons(w,E,I)
channel_indices = findChannelIndices(n)

for idx in channel_indices:
    x_int = t[idx]
    y_int = n[idx]
    plt.plot([x_int, x_int], [np.min(n), y_int], color='k')
    #plt.text(x_int, y_int + 0.25, ha='center',s = f"{round(n[idx])}")
    plt.axhline(y_int,linestyle="dashed")

plt.plot(t,n)
plt.savefig("images/channels.png")
plt.clf()

def findIntensityParameter(w,I):
    tau = 2*np.pi/w
    t = np.linspace(0,tau,1000)

    envelope = np.sin(np.pi*t/tau)**2
    I_au = (I/3.51E16)*envelope
    Up = I_au/(4*w**2)
    z = Up/w

    return t,z


t,z = findIntensityParameter(w,I)
plt.plot(t,z)
print(f"Max value of Up/w is {np.max(z)}")
plt.savefig("images/z.png")