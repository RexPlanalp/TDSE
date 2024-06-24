import numpy as np
import matplotlib.pyplot as plt



E_range = np.load("PES_files/E.npy")
PES = np.load("PES_files/PES.npy")

plt.semilogy(E_range,np.real(PES),color = "k")

plt.savefig("images/PES.png")
