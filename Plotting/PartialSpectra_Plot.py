import numpy as np
import sys
import matplotlib.pyplot as plt
import pickle
import json

with open("PES_files/partial_pes.json", "rb") as fp:
    partial_spectra = pickle.load(fp)
E_range = np.load("PES_files/E.npy")

# Nothing needs to be specified
if "TOTAL" in sys.argv:
    total = 0
    for (l,m),y in partial_spectra.items():
        total += np.real(y)
    plt.semilogy(E_range,total,label = "Total")
    plt.legend()
    plt.savefig("images/total_log_PES.png")
    plt.clf()

# Must specify condition and change label for condition to avoid confusion
if "PARTIAL" in sys.argv:
    partial = 0
    for (l,m),y in partial_spectra.items():
        condition = l == m
        if condition:
            partial += np.real(y)
    plt.semilogy(E_range,partial,label = "Partial: l = m")
    plt.legend()
    plt.savefig("images/partial.png")
    plt.clf()

# Must specify how many of the top l,m's to find and at what energy via command line
if "TOP" in sys.argv:
    top = int(sys.argv[2])
    E = float(sys.argv[3])
    
   
    E_idx = np.argmin(np.abs(E_range - E))

    values = []
    lm_vals = []
    for (l, m), y in partial_spectra.items():
            partial_spectrum = np.real(y)
            values.append(partial_spectrum[E_idx])
            lm_vals.append((l, m))

    values = np.array(values)
    lm_vals = np.array(lm_vals)

    sorted_indices = np.argsort(values)[::-1]
    values = values[sorted_indices]
    lm_vals = lm_vals[sorted_indices]

    top_values = values[:top]
    top_lm_vals = lm_vals[:top]

    for idx,(l,m) in enumerate(top_lm_vals):
       print(f"{(l,m)}:{np.sqrt(top_values[idx]/np.max(top_values))}")
    
    plt.plot([l for l,_ in top_lm_vals], np.sqrt(top_values/np.max(top_values)), 'o')
    plt.savefig("images/top.png")

# Must specify energy via command line
if "AMP" in sys.argv:
    E = float(sys.argv[2])
   
    E_idx = np.argmin(np.abs(E_range - E))

    amplitude_dict = {}

    for (l, m), y in partial_spectra.items():
        partial_spectrum = np.real(y).tolist()
        key_str = f"({l},{m})"

        amplitude_dict[key_str] = np.sqrt(partial_spectrum[E_idx])
        
    with open('amplitudes.json', 'w') as json_file:
        json.dump(amplitude_dict, json_file, indent=4)

    
    
    


   



    

