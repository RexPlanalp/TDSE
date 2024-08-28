import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import sph_harm, gamma


l_values = np.array([25,24,23,22,21,26,27,26,20,25])
m_values = np.array([25,24,23,22,21,26,25,24,20,23])
fixed_amplitudes = np.array([0.025377139861139313, 0.02289811208166617, 0.011890305712894974, 0.0063552611322039545, 0.003329584625400153, 0.001911661464594239, 0.0019063560149378018, 0.0015955431914372272, 0.0009327722036463356, 0.0008231126715025616])

E = 0.48
k = np.sqrt(2 * E)
theta = np.pi / 2
phi = np.arange(0, 2 * np.pi + 0.01, 0.01)

def A(phi, theta, k, phases):
    PES_amplitude = sum(
        fixed_amplitude * (-1j)**l * np.exp(1j * phase) * np.exp(1j * np.angle(gamma(l + 1 - 1j/k))) * sph_harm(m, l, phi, theta)
        for fixed_amplitude, phase, m, l in zip(fixed_amplitudes, phases, m_values, l_values)
    )

    PES_amplitude_opp = sum(
        fixed_amplitude * (-1j)**l * np.exp(1j * phase) * np.exp(1j * np.angle(gamma(l + 1 - 1j/k))) * (-1)**m * sph_harm(m, l, phi, theta)
        for fixed_amplitude, phase, m, l in zip(fixed_amplitudes, phases, m_values, l_values)
    )

    PES_vals = np.abs(PES_amplitude) ** 2
    PES_vals_opp = np.abs(PES_amplitude_opp) ** 2
    A = (PES_vals - PES_vals_opp) / (PES_vals + PES_vals_opp)
    return A

def A_wrapper(phi, *phases):
    return A(phi, theta, k, phases)

y_data = np.load("TDSE_files/A_slice.npy")  # Adjust path as needed

# Initial guesses for phases only
initial_phases = [0.0] * len(l_values)
lower_bounds = [-np.pi] * len(l_values)
upper_bounds = [np.pi] * len(l_values)
sigma = 0.01 * np.ones_like(y_data)

# Run the fitting algorithm
popt, pcov = curve_fit(A_wrapper, phi, y_data, p0=initial_phases, sigma=sigma, method = "dogbox")

print("Optimized phases:", popt)
plt.plot(phi, y_data, label="Simulation Result")
plt.plot(phi, A_wrapper(phi, *popt), label="Model Fit")
plt.xlabel('Phi')
plt.ylabel('A')
plt.legend()
plt.savefig("images/A_fit.png")
plt.show()
