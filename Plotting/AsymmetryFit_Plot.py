import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import sph_harm
from scipy.special import gamma

l_values = [26,25]
m_values = [26,25]

E = 0.48
k = np.sqrt(2 * E)

theta = np.pi / 2
phi = np.arange(0, 2 * np.pi+0.01, 0.01)

phases = [(-1j)**l * np.exp(1j * np.angle(gamma(l + 1 - 1j / k))) for l in l_values]

def PES(phi, *amplitudes):
    PES_amplitude = sum(
        amplitude * phase * sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
    )
    PES_vals = np.abs(PES_amplitude) ** 2
    return PES_vals

def PES_opposite(phi, *amplitudes):
    PES_amplitude_opposite = sum(
        (-1) ** l * amplitude * phase * sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
    )
    PES_vals_opp = np.abs(PES_amplitude_opposite) ** 2
    return PES_vals_opp

def A(phi, *amplitudes):
    PES_vals = PES(phi, *amplitudes)
    PES_vals_opp = PES_opposite(phi, *amplitudes)
    A = (PES_vals - PES_vals_opp) / (PES_vals + PES_vals_opp)
    return A



y_data = np.load("TDSE_files/A_slice.npy")

lower_bounds = [0]*2
upper_bounds = [1]*2
amplitudes = [1]*2

popt, pcov = curve_fit(A, phi, y_data,p0=amplitudes)
print("Optimized parameters:", popt)

plt.plot(phi, y_data, label="Data")
plt.plot(phi, A(phi, *popt), label="Fit")
plt.savefig("images/A_fit.png")





