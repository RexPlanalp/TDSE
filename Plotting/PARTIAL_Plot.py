import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle

with open("PES_files/partial_pes.json", "rb") as fp:
    coeff_dict = pickle.load(fp)
E_range = np.load("PES_files/E.npy")

TOP = "TOP" in sys.argv
ALL = "ALL" in sys.argv

if ALL:
    for l,m in coeff_dict.keys():
            y = np.array(coeff_dict[(l,m)])
            plt.semilogy(E_range,np.real(y),label = f"{l,m}")
    plt.savefig("images/partial.png")

if TOP:
    E_target = 0.48
    tolerance = 1e-4  # Tolerance for finding the closest energy value
    index = np.argmin(np.abs(E_range - E_target))

    # Calculate the contributions at E = 0.48 for each (l, m) combination
    contributions = {}
    for (l, m), y in coeff_dict.items():
        y_array = np.array(y)
        contributions[(l, m)] = np.real(y_array[index])

    # Sort the (l, m) combinations by their contributions
    sorted_contributions = sorted(contributions.items(), key=lambda item: item[1], reverse=True)

    # Select the top 3 (l, m) combinations
    top_3_contributions = sorted_contributions[:3]

    # Plot the contributions of the top 3 combinations
    plt.figure()
    for (l, m), _ in top_3_contributions:
        y = np.array(coeff_dict[(l, m)])
        plt.plot(E_range, np.real(y), label=f"l={l}, m={m}")

    # Adding labels and legend
    plt.xlabel("Energy (E)")
    plt.ylabel("Contribution")
    plt.legend()

    # Save and show the plot
    plt.savefig("images/top.png")
   
