import numpy as np
import matplotlib.pyplot as plt



E_range = np.load("PES_files/E.npy")
PES = np.load("PES_files/PES.npy")

plt.figure()
plt.semilogy(E_range,np.real(PES),color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield (Log scale)")
plt.savefig("images/PES_log.png")
plt.clf()

plt.figure()
plt.plot(E_range,np.real(PES),color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield")
plt.savefig("images/PES.png")
plt.clf()
