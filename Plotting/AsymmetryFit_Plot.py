import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import sph_harm
from scipy.special import gamma

l_values = [9,8]
m_values = [9,8]
E = 0.304



k = np.sqrt(2 * E)
theta = np.pi / 2
phi = np.arange(0, 2 * np.pi + 0.01, 0.01)

def PES(phi, theta, params):
    n = len(params) // 2
    amplitudes = params[:n]
    phases = params[n:]
    PES_amplitude = sum(
        amplitude * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
    )
    PES_vals = np.abs(PES_amplitude) ** 2
    return PES_vals

def PES_opposite(phi, theta, params):
    n = len(params) // 2
    amplitudes = params[:n]
    phases = params[n:]
    PES_amplitude_opposite = sum(
        (-1) ** l * amplitude * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
        for amplitude, phase, m, l in zip(amplitudes, phases, m_values, l_values)
    )
    PES_vals_opp = np.abs(PES_amplitude_opposite) ** 2
    return PES_vals_opp

def A(phi, *params):
    PES_vals = PES(phi, theta, params)
    PES_vals_opp = PES_opposite(phi, theta, params)
    A = (PES_vals - PES_vals_opp) / (PES_vals + PES_vals_opp)
    return A

# Load Simulation Result
y_data = np.load("TDSE_files/A_slice.npy")

# Sets up initial guesses for amplitudes and phases
n_params = 2 * len(l_values)
initial_guess = [1.0] * n_params
lower_bounds = [0.0] * len(l_values) + [-np.pi] * len(l_values)
upper_bounds = [np.inf] * len(l_values) + [np.pi] * len(l_values)
sigma = 0.01 * np.ones_like(y_data)

# Run the fitting algorithm and normalize relative weights
popt, pcov = curve_fit(A, phi-0.01, y_data, p0=initial_guess, bounds=(lower_bounds, upper_bounds),sigma = sigma)
popt[:len(l_values)] /= popt[0]

# Print parameters of fit, and plot the result
print("Optimized parameters:", popt)
plt.plot(phi, y_data, label="Simulation Result")
plt.plot(phi, A(phi, *popt), label="Model Fit+0.1")
plt.xlabel('Phi')
plt.ylabel('A')
plt.legend()
plt.savefig("images/A_fit.png")
plt.show()
