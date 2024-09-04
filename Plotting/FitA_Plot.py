import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from scipy.optimize import curve_fit
from scipy.special import gamma
import json

P = False
PandA = True
Test1 = False
Test2 = False
Test3 = False


if PandA:
    # Given values
    lm_values = [(25, 25), (24, 24), (23, 23), (22, 22), (21, 21)]
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)

    # Load the A_data
    A_data = np.load("TDSE_files/A_slice.npy")

    # Define the fitting function
    def A(phi, *params):
        amplitudes = params[:len(lm_values)]
        phases = params[len(lm_values):]

        pes_amp = sum(
            amplitude * (1j)**l * np.exp(1j * phase) * np.exp(-1j*np.angle(gamma(l + 1 -1j/k)))* sph_harm(m, l, phi, theta) 
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals = np.abs(pes_amp) ** 2

        pes_amp_opp = sum(
            amplitude * (1j)**l * np.exp(1j * phase) * np.exp(-1j*np.angle(gamma(l + 1 -1j/k)))* sph_harm(m, l, phi+np.pi, theta) 
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals_opp = np.abs(pes_amp_opp) ** 2

        asymmetry = (pes_vals - pes_vals_opp) / (pes_vals + pes_vals_opp)
        return asymmetry

    # Initial guesses for the amplitudes and phases
    initial_guess = np.ones(len(lm_values)) + 1j * np.zeros(len(lm_values))
    initial_guess = np.concatenate([initial_guess.real, initial_guess.imag])

    # Perform the curve fitting
    popt, pcov = curve_fit(A, phi, A_data, p0=initial_guess)

    # Extract the fitted amplitudes and phases
    fitted_amplitudes = popt[:len(lm_values)]/np.max(np.abs(popt[:len(lm_values)]))
    fitted_phases = popt[len(lm_values):]

    # Plotting the results
    plt.plot(phi, A_data, label='Data')
    plt.plot(phi, A(phi, *popt)+0.1, label='Fitted')
    plt.legend()
    plt.savefig("images/A_fit.png")
    plt.clf()

    plt.scatter([l for l, _ in lm_values], fitted_amplitudes/fitted_amplitudes[0], label='Fit Amplitudes')
    plt.legend()
    plt.savefig("images/amplitudes.png")

    print("Fitted amplitudes:", fitted_amplitudes)
    print("Fitted phases:", fitted_phases%(2*np.pi))

if P:
    # Given values
    lm_values = [(25, 25), (24, 24), (23, 23), (22, 22), (21, 21),(26, 26)]
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)

    # Load the A_data
    A_data = np.load("TDSE_files/A_slice.npy")

    with open('amplitudes.json', 'r') as json_file:
        amplitude_dict = json.load(json_file)
    
    amplitudes = amplitude_dict.values()

    


  


    
    # Define the fitting function
    def A(phi, *phases):
        pes_amp = sum(
            np.exp(-1j*(np.angle(gamma(l + 1 -1j/k)))+1j*phase) * amplitude * (1j)**l * sph_harm(m, l, phi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals = np.abs(pes_amp) ** 2

        pes_amp_opp = sum(
            np.exp(-1j*(np.angle(gamma(l + 1 -1j/k)))+1j*phase) * amplitude * (1j)**l * sph_harm(m, l, phi+np.pi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals_opp = np.abs(pes_amp_opp) ** 2

        asymmetry = (pes_vals - pes_vals_opp) / (pes_vals + pes_vals_opp)
        return asymmetry

    # # Initial guesses for the phases
    initial_phases = np.zeros(len(lm_values))

    # Perform the curve fitting
    popt, pcov = curve_fit(A, phi, A_data, p0=initial_phases)

    # The fitted phases are stored in popt
    fitted_phases = popt

    # Plotting the results
    plt.plot(phi, A_data, label='Data')
    plt.plot(phi, A(phi, *popt), label='Fitted')
    plt.legend()
    plt.savefig("images/A_fit.png")
    plt.show()

    print("Fixed amplitudes:", amplitudes)
    print("Fitted phases:", fitted_phases%(2*np.pi))
    

if Test1:
    # Given values
    lm_values = [(i, i) for i in range(51)]  # lm_values from 0 to 50
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)

    # Load the A_data
    A_data = np.load("TDSE_files/A_slice.npy")

    # Predefined amplitudes and phases for all terms (including l = 25, m = 25)
    specified_amplitudes = np.ones(len(lm_values))  # Example: all ones, modify as needed
    specified_phases = np.array([np.angle(gamma(l + 1 - 1j / k)) for l, _ in lm_values])

    # Index for l = 25, m = 25
    index_25_25 = lm_values.index((25, 25))
    
    # Define the fitting function
    def A(phi, amplitude_25_25):
        amplitudes = specified_amplitudes.copy()
        phases = specified_phases.copy()
        amplitudes[index_25_25] = amplitude_25_25  # Fit only the amplitude for l=25, m=25

        pes_amp = sum(
            amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals = np.abs(pes_amp) ** 2

        pes_amp_opp = sum(
            (-1) ** l * amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals_opp = np.abs(pes_amp_opp) ** 2

        asymmetry = (pes_vals - pes_vals_opp) / (pes_vals + pes_vals_opp)
        return asymmetry

    # Initial guess for the amplitude of l=25, m=25
    initial_guess = [specified_amplitudes[index_25_25]]  # Initial amplitude for l=25, m=25

    # Perform the curve fitting
    popt, pcov = curve_fit(A, phi, A_data, p0=initial_guess)

    # Plotting the results
    plt.plot(phi, A_data, label='Data')
    plt.plot(phi, A(phi, *popt) + 0.01, label='Fitted', linestyle='--')
    plt.legend()
    plt.savefig("images/A_fit.png")
    plt.clf()

    print("Fitted amplitude for l=25, m=25:", popt[0])

if Test2:
    lm_values = [(i, i) for i in range(51)]  # lm_values from 0 to 50
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)
    A_data = np.load("TDSE_files/A_slice.npy")

    # Predefined amplitudes and phases for all terms
    specified_amplitudes = np.ones(len(lm_values))  # All ones, modify as needed
    specified_phases = np.array([np.angle(gamma(l + 1 - 1j / k)) for l, _ in lm_values])

    # Index for l = 25, m = 25
    index_25_25 = lm_values.index((25, 25))
    
    # Define the fitting function to only fit the phase for l=25, m=25
    def A(phi, phase_25_25):
        amplitudes = specified_amplitudes.copy()
        phases = specified_phases.copy()
        phases[index_25_25] = phase_25_25  # Fit only the phase for l=25, m=25

        pes_amp = sum(
            amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals = np.abs(pes_amp) ** 2

        pes_amp_opp = sum(
            (-1) ** l * amplitude * (-1j)**l * np.exp(1j * phase) * sph_harm(m, l, phi, theta)
            for amplitude, phase, (m, l) in zip(amplitudes, phases, lm_values)
        )
        pes_vals_opp = np.abs(pes_amp_opp) ** 2

        asymmetry = (pes_vals - pes_vals_opp) / (pes_vals + pes_vals_opp)
        return asymmetry

    # Initial guess for the phase of l=25, m=25
    initial_guess = [specified_phases[index_25_25]]  # Initial phase for l=25, m=25

    # Perform the curve fitting
    popt, pcov = curve_fit(A, phi, A_data, p0=initial_guess)

    # Plotting the results
    plt.plot(phi, A_data, label='Data')
    plt.plot(phi, A(phi, *popt) + 0.01, label='Fitted', linestyle='--')
    plt.legend()
    plt.savefig("images/A_fit.png")
    plt.clf()

    print("Fitted phase for l=25, m=25:", popt[0]%(2*np.pi))
    print("Analytic phase for l=25, m=25:", np.angle(gamma(25 + 1 - 1j / k))%(2*np.pi))

if Test3:
    import json

    E = 0.48
    k = np.sqrt(2 * E)
    theta = np.pi/2
    phi = np.arange(0, 2 * np.pi, 0.01)

    

    # Load the JSON data from the file
    with open('amplitudes.json', 'r') as json_file:
        amplitude_dict = json.load(json_file)

    PAD_amp = 0
    PAD_amp_opp = 0
    for key in amplitude_dict.keys():
        l, m = map(int, key.strip("()").split(','))
        amplitude = amplitude_dict[key]
       

        
        PAD_amp += (1j)**l * np.exp(-1j*np.angle(gamma(l + 1 -1j/k))) * sph_harm(m, l, phi, theta) * amplitude
        PAD_amp_opp +=  (1j)**l * np.exp(-1j*np.angle(gamma(l + 1 -1j/k))) * sph_harm(m, l, phi+np.pi, theta) * amplitude 
        
    PAD = np.abs(PAD_amp)**2
    PAD_opp = np.abs(PAD_amp_opp)**2

    asymmetry = (PAD - PAD_opp) / (PAD + PAD_opp)

    plt.plot(phi, asymmetry)
    plt.savefig("TESTPAD.png")
        







