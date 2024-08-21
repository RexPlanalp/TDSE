import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

with open("PES_files/partial_pes.json", "rb") as fp:
    partial_spectra = pickle.load(fp)
E_range = np.load("PES_files/E.npy")

TOP = "TOP" in sys.argv
ALL = "ALL" in sys.argv
SINGLE = "SINGLE" in sys.argv

if TOP: 
    E = float(sys.argv[3])
    top = int(sys.argv[2])
if ALL: 
    MODE = sys.argv[2]

if SINGLE:
    lprime = int(sys.argv[2])
    mprime = int(sys.argv[3])

if ALL:
    total = 0
    total_restricted = 0
    for (l,m),y in partial_spectra.items():
        if MODE == "log":
            plt.semilogy(E_range,np.real(y),label = f"{l,m}")
        elif MODE == "real":
            plt.plot(E_range,np.real(y),label = f"{l,m}")
        if l == m:
            total_restricted += np.real(y)
        total += np.real(y)
    plt.savefig("images/partial.png")
    plt.clf()
    plt.semilogy(E_range,total_restricted,label = "Total Restricted")
    plt.semilogy(E_range,total,label = "Total")
    plt.savefig("images/total.png")
if TOP:
    E_index = np.argmin(np.abs(E_range - E))
    contributions = []
    lm_pairs = []  # To store the (l, m) pairs where l == m

    # Collect contributions where l == m
    for (l, m), y in partial_spectra.items():
        if l == m:
            partial_spectrum = np.real(y)
            contributions.append(partial_spectrum[E_index])
            lm_pairs.append((l, m))

    # Sort contributions and keep track of their indices
    if contributions:  # Proceed only if there are valid contributions
        sorted_indices = np.argsort(contributions)
        top_indices = sorted_indices[-1:-(top+1):-1]
        
        plt.figure()
        set_value = 0
        ratios = []

        # Plot the top contributions
        for i, index in enumerate(top_indices):
            l, m = lm_pairs[index]  # Get the corresponding (l, m) pair
            y = np.array(partial_spectra[(l, m)])
            plt.plot(E_range, np.abs(y), label=f"l={l}, m={m}")

            if i == 0:
                set_value = np.sqrt(np.real(y[E_index]))

            ratio = np.sqrt(np.real(y[E_index]))/set_value
            print(f"Top {i+1}, l,m: {l,m}, Value: {ratio}")
            ratios.append(ratio)
        
        plt.axvline(E)
        plt.xlabel("Energy (E)")
        plt.ylabel("Contribution")
        plt.legend()
        plt.savefig("images/top.png")
        plt.clf()

        # Plot the ratios
        x_values = np.concatenate((range(len(ratios)), -np.array(range(len(ratios)))))
        y_values = np.concatenate((ratios, ratios))
        plt.plot(x_values, y_values)
        plt.savefig("images/ratios.png")
        plt.clf()

if SINGLE:
    plt.clf()
    for (l,m),y in partial_spectra.items():
        if l == lprime and m == mprime:
            E = 0.48
            E_index = np.argmin(np.abs(E_range - E))
            plt.plot(E_range,np.real(y),label = f"{l,m}")
            print((np.real(y[E_index])))
            plt.axvline(E)
    plt.savefig(f"images/{lprime}_{mprime}.png")
    

