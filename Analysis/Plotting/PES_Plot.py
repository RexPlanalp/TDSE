import numpy as np
import matplotlib.pyplot as plt
import json

with open('input.json', 'r') as file:
    input_par = json.load(file)
PES,E_min,E_max = input_par["PES"]

gamma = 0.001
E_range = np.arange(E_min,E_max+2*gamma,2*gamma)
PES = np.load("PES.npy")

plt.ylim([1E-15,1E0])
plt.semilogy(np.real(PES),color = "k")

plt.savefig("PES.png")