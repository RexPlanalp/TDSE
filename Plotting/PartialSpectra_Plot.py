import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

with open("PES_files/partial_pes.json", "rb") as fp:
    partial_spectra = pickle.load(fp)
E_range = np.load("PES_files/E.npy")

TOP = "TOP" in sys.argv
ALL = "ALL" in sys.argv

if TOP: 
    E = float(sys.argv[2])
    top = int(sys.argv[3])
if ALL: 
    MODE = sys.argv[2]
    max = int(sys.argv[3])

if ALL:
    for (l,m),y in partial_spectra.items():
        if l < max:
            if MODE == "log":
                plt.semilogy(E_range,np.real(y),label = f"{l,m}")
            elif MODE == "real":
                plt.plot(E_range,np.real(y),label = f"{l,m}")
    plt.savefig("images/partial.png")
if TOP:
    E_index = np.argmin(np.abs(E_range - E))
    contributions = []
    for (l, m), y in partial_spectra.items():
        partial_spectrum = np.real(y)
        contributions.append(partial_spectrum[E_index])

    sorted_indices = np.argsort(contributions)
    top_indices = sorted_indices[-1:-(top+1):-1]
    
    plt.figure()
    set = 0
    ratios = []
    for i,index in enumerate(top_indices):
        l,m = list(partial_spectra.keys())[index]
        y = np.array(partial_spectra[(l, m)])
        plt.plot(E_range, np.abs(y), label=f"l={l}, m={m}")

        if i == 0:
            set = round((np.abs(y[E_index])),8)

        ratio = round((np.abs(y[E_index])),8)/set
        print(f"Top {i+1}, l,m: {l,m}, Value: {ratio}")
        ratios.append(ratio)
        

    plt.axvline(E)
    plt.xlabel("Energy (E)")
    plt.ylabel("Contribution")
    plt.legend()
    plt.savefig("images/top.png")
    plt.clf()

    x_values = np.concatenate((range(len(ratios)), -np.array(range(len(ratios)))))
    y_values = np.concatenate((ratios, ratios))
    plt.plot(x_values,y_values)
    plt.savefig("images/ratios.png")
    plt.clf()

   

