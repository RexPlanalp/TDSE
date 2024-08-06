import numpy as np 
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

root = "//data/becker/dopl4670/TDSE_Jobs/Project/10_800_2E14/"

total_PES = root + "PES_files/PES.npy"
total_E = root + "PES_files/E.npy"

PES = np.real(np.load(total_PES))
E = np.load(total_E)

def findPeakIndices(PES):
    peak_indices = find_peaks(PES,width = 2)[0]
    return peak_indices

def findCentralPeakIndex(PES):
    peak_indices = findPeakIndices(PES)
    max_index = np.argmax(PES[peak_indices])
    return max_index

peak_indices = find_peaks(PES,width = 2)[0]
max_index = findCentralPeakIndex(PES)


max_E = E[peak_indices][max_index]
ati_peak_energies = E[peak_indices]

print("Max E:",max_E)


plt.semilogy(E,PES,color = "k",label = "PES")
plt.axvline(max_E,color = "r",label = "Central Peak: {:.3f}".format(max_E))

for i in peak_indices:
    if i != peak_indices[max_index]:
        plt.axvline((E[i]),color = "b")
plt.legend()
plt.savefig("images/peak.png")
plt.clf()

print("Peak Energies:",ati_peak_energies)