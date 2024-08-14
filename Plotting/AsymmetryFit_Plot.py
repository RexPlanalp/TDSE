# import numpy as np
# import matplotlib.pyplot as plt
# import sys
# from scipy.optimize import curve_fit
# from scipy.special import sph_harm
# from scipy.special import gamma

# l_values = np.array([25,24,23,22,21,26,27,20])
# m_values = np.array([25,24,23,22,21,26,27,20])

# E = 0.48

# k = np.sqrt(2 * E)
# theta = np.pi / 2
# phi = np.arange(0, 2 * np.pi + 0.01, 0.01)

# def PES(phi, theta, params):
#     n = len(params) // 2
#     amplitudes = params[:n]
#     phases = params[n:]
#     PES_amplitude = sum(
#         amplitude * (-1j)**l *np.exp(1j * phase) * sph_harm(m, l, phi, theta)
#         for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
#     )
#     PES_vals = np.abs(PES_amplitude) ** 2
#     return PES_vals

# def PES_opposite(phi, theta, params):
#     n = len(params) // 2
#     amplitudes = params[:n]
#     phases = params[n:]
#     PES_amplitude_opposite = sum(
#         (-1) ** l * amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
#         for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
#     )
#     PES_vals_opp = np.abs(PES_amplitude_opposite) ** 2
#     return PES_vals_opp

# def A(phi, *params):
#     PES_vals = PES(phi, theta, params)
#     PES_vals_opp = PES_opposite(phi, theta, params)
#     A = (PES_vals - PES_vals_opp) / (PES_vals + PES_vals_opp)
#     return A

# # Load Simulation Result
# y_data = np.load("TDSE_files/A_slice.npy")

# # Sets up initial guesses for amplitudes and phases
# n_params = 2 * len(l_values)
# initial_guess = [0.02] * n_params
# lower_bounds = [0.0] * len(l_values) + [-np.pi] * len(l_values)
# upper_bounds = [np.inf] * len(l_values) + [np.pi] * len(l_values)
# sigma = 0.01 * np.ones_like(y_data)

# # Run the fitting algorithm and normalize relative weights
# popt, pcov = curve_fit(A, phi-0.01, y_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds),sigma = sigma)
# #popt[:len(l_values)] /= popt[0]

# # Print parameters of fit, and plot the result
# print("Optimized parameters:", popt)
# plt.plot(phi, y_data, label="Simulation Result")
# plt.plot(phi, A(phi, *popt)+0.1, label="Model Fit+0.1")
# plt.xlabel('Phi')
# plt.ylabel('A')
# plt.legend()
# plt.savefig("images/A_fit.png")
# plt.show()
###############################################################

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.optimize import curve_fit
from scipy.special import sph_harm
from scipy.special import gamma

l_values = np.array([25,24,23,22,21,26,27,26,20,25])
m_values = np.array([25,24,23,22,21,26,25,24,20,23])
fixed_amplitudes = np.array([0.025377139861139313, 0.02289811208166617, 0.011890305712894974, 0.0063552611322039545, 0.003329584625400153, 0.001911661464594239, 0.0019063560149378018, 0.0015955431914372272, 0.0009327722036463356, 0.0008231126715025616])

# mask = np.array([0,1,2,3,4,5,8])

# l_values = l_values[mask]
# m_values = m_values[mask]
# fixed_amplitudes = fixed_amplitudes[mask]


#E = float(sys.argv[1])
E = 0.48

k = np.sqrt(2 * E)
theta = np.pi / 2
phi = np.arange(0, 2 * np.pi + 0.01, 0.01)


def PES(phi, theta, phases):
    PES_amplitude = sum(
        amplitude * (-1j)**l * np.exp(1j * phase) *  sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(fixed_amplitudes, phases, m_values, l_values)
    )
    PES_vals = np.abs(PES_amplitude) ** 2
    return PES_vals

def PES_opposite(phi, theta, phases):
    PES_amplitude_opposite = sum(
        (-1) ** l * amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(fixed_amplitudes, phases, m_values, l_values)
    )
    PES_vals_opp = np.abs(PES_amplitude_opposite) ** 2
    return PES_vals_opp

def A(phi, *phases):
    PES_vals = PES(phi, theta, phases)
    PES_vals_opp = PES_opposite(phi, theta, phases)
    A = (PES_vals - PES_vals_opp) / (PES_vals + PES_vals_opp)
    return A

# Load Simulation Result
y_data = np.load("TDSE_files/A_slice.npy")

# Sets up initial guesses for phases only
initial_phases_guess = [0.0] * len(l_values)
lower_bounds_phases = [-np.pi] * len(l_values)
upper_bounds_phases = [np.pi] * len(l_values)
sigma = 0.01 * np.ones_like(y_data)

# Run the fitting algorithm and normalize relative weights
popt_phases, pcov_phases = curve_fit(A, phi, y_data, p0=initial_phases_guess, bounds=(lower_bounds_phases, upper_bounds_phases), sigma=sigma)

# Combine fixed amplitudes with optimized phases for full parameter set
popt = np.concatenate((fixed_amplitudes, popt_phases))

# Print parameters of fit, and plot the result
print("Optimized phases:", popt_phases)
plt.plot(phi, y_data, label="Simulation Result")
plt.plot(phi, A(phi, *popt_phases), label="Model Fit+0.1")
plt.xlabel('Phi')
plt.ylabel('A')
plt.legend()
plt.savefig("images/A_fit.png")
plt.show()
