import numpy as np
import matplotlib.pyplot as plt



E_range = np.load("PES_files/E.npy")
PES = np.load("PES_files/PES.npy")

plt.figure()
plt.semilogy(E_range,np.real(PES),color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield (Log scale)")
plt.savefig("images/log_PES.png")
plt.clf()

plt.figure()
plt.plot(E_range,np.real(PES),color = "k")
plt.xlabel("Energy (au)")
plt.ylabel("Yield")
plt.savefig("images/PES.png")
plt.clf()

I = np.trapz(np.real(PES),E_range)
print(f"Total Ionization:{I}")
