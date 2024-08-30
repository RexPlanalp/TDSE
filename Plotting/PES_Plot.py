import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def findPeakIndices(PES):
    peak_indices = find_peaks(PES,width = 2)[0]
    return peak_indices

def findCentralPeakIndex(PES):
    peak_indices = findPeakIndices(PES)
    max_index = np.argmax(PES[peak_indices])
    return max_index

E_range = np.load("PES_files/E.npy")
PES = np.real(np.load("PES_files/PES.npy"))

plt.figure()
plt.semilogy(E_range,PES,color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield (Log scale)")
plt.savefig("images/log_PES.png")
plt.clf()

plt.figure()
plt.plot(E_range,PES,color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield")
plt.savefig("images/PES.png")
plt.clf()

peak_indices = find_peaks(PES,width = 2)[0]
max_index = findCentralPeakIndex(PES)

max_E = E_range[peak_indices][max_index]
ati_peak_energies = E_range[peak_indices]
print("Max E:",max_E)

plt.semilogy(E_range,PES,color = "k",label = "PES")

for i in peak_indices:
    plt.scatter(E_range[i],PES[i],color = "b",label =  f"E = {E_range[i]}")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 0.175), ncol=3)
plt.savefig("images/peak.png")
plt.clf()


