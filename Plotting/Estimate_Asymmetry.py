import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.special import sph_harm
from scipy.special import gamma

l_values = [25, 24, 23, 22, 21, 20, 19, 18, 17]
m_values = [25, 24, 23, 22, 21, 20, 19, 18, 17]
# l_values = [25, 24, 23]
# m_values = [25, 24, 23]


E = 0.48
k = np.sqrt(2 * E)

theta = np.pi / 2
phi = np.arange(0, 2 * np.pi+0.01, 0.01)

phases = [(-1) ** l * np.exp(1j * np.angle(gamma(l + 1 - 1j / k))) for l in l_values]

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
# y_prediction = A(phi,*[0.08840796672327927,2*0.03697648072296346 ,2*0.02284982509729667])
# plt.plot(phi,y_data)
# plt.plot(phi,y_prediction)
# plt.savefig("images/A_test.png")

lower_bounds = [0]*9
upper_bounds = [1]*9
amplitudes = [1]*9

popt, pcov = curve_fit(A, phi, y_data,p0=amplitudes)
print("Optimized parameters:", popt)

plt.plot(phi, y_data, label="Data")
plt.plot(phi, A(phi, *popt), label="Fit")
plt.savefig("images/A_fit.png")





