import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

with open("PES_files/partial_pes.json", "rb") as fp:
    partial_spectra = pickle.load(fp)
E_range = np.load("PES_files/E.npy")

TOP = "TOP" in sys.argv
ALL = "ALL" in sys.argv

if ALL:
    for (l,m),y in partial_spectra.items():
        #plt.semilogy(E_range,np.real(y),label = f"{l,m}")
        plt.plot(E_range,np.real(y),label = f"{l,m}")
    plt.savefig("images/partial.png")

if TOP:
    E = 0.48
    E_index = np.argmin(np.abs(E_range - E))

    contributions = []
    for (l, m), y in partial_spectra.items():
        partial_spectrum = np.real(y)
        contributions.append(partial_spectrum[E_index])

    sorted_indices = np.argsort(contributions)
    top3_indices = sorted_indices[-1:-4:-1]
    
    plt.figure()
    for index in top3_indices:
        l,m = list(partial_spectra.keys())[index]
        y = np.array(partial_spectra[(l, m)])
        plt.plot(E_range, np.real(y), label=f"l={l}, m={m}")
    
    plt.axvline(E)
    plt.xlabel("Energy (E)")
    plt.ylabel("Contribution")
    plt.legend()

   
    plt.savefig("images/top.png")
   
