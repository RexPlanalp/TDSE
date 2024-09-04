FitAmplitudeAndPhases = False
FANDP = FitAmplitudeAndPhases

FitPhaseOnly = True
F = FitPhaseOnly

if FANDP:

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    from scipy.special import sph_harm, gamma

    # Given values
    lm_values = [(25, 25), (24, 24), (23, 23), (22, 22), (21, 21),(26,26)]
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)

    # Load the A_data
    A_data = np.load("TDSE_files/A_slice.npy")

    def model_pes(phi, params, lm_values, k, theta=np.pi/2):
        """
        Compute the Photoelectron Spectrum (PES) for given parameters.
        
        params: array-like, shape (2 * len(lm_values),)
            [amp_real_1, amp_imag_1, ..., phase_1, ...]
        lm_values: list of tuples
            List containing (l, m) pairs.
        phi: array-like
            Array of azimuthal angles.
        theta: float
            Polar angle (fixed at pi/2 by default).
        k: float
            Wave number based on energy.
        """
        num_terms = len(lm_values)
        amplitudes = params[:num_terms] + 1j * params[num_terms:2*num_terms]
        phases = params[2*num_terms:]
        
        pes = np.zeros_like(phi, dtype=complex)
        for i, (l, m) in enumerate(lm_values):
            amp = amplitudes[i] 
            Y_lm = sph_harm(m, l, phi, theta)
            phase_factor = np.exp(-1j * np.angle(gamma(l + 1 - 1j/k))) * np.exp(-1j * phases[i]) * (1j)**l
            pes += amp * phase_factor * Y_lm
        
        return np.abs(pes)**2

    def asymmetry_model(phi, params, lm_values, k, theta=np.pi/2):
        """
        Compute the asymmetry A(phi) based on the PES model.
        """
        pes_phi = model_pes(phi, params, lm_values, k, theta)
        pes_phi_pi = model_pes(phi + np.pi, params, lm_values, k, theta)
        asymmetry = (pes_phi - pes_phi_pi) / (pes_phi + pes_phi_pi)
        return asymmetry

    def residuals(params, phi, A_data, lm_values, k, theta=np.pi/2):
        """
        Residuals function for least_squares optimization.
        """
        A_model = asymmetry_model(phi, params, lm_values, k, theta)
        return A_model - A_data

    def fit_asymmetry_least_squares(A_data, phi_array, lm_values, k, initial_params):
        """
        Fit asymmetry data using least_squares optimization.
        
        Parameters:
        - A_data: array-like
            The asymmetry data to fit.
        - phi_array: array-like
            Array of phi values corresponding to A_data.
        - lm_values: list of tuples
            List of (l, m) pairs used in the model.
        - k: float
            Wave number based on energy.
        - initial_params: array-like
            Initial guesses for parameters [amp_real, amp_imag, ..., phase, ...].
        
        Returns:
        - result: OptimizeResult
            The result of the optimization containing optimized parameters.
        """
        result = least_squares(
            residuals,
            initial_params,
            args=(phi_array, A_data, lm_values, k),
            method='trf',  # Trust Region Reflective algorithm
            jac='2-point',  # Jacobian approximation
            verbose=2  # Print progress
        )
        return result

    # Prepare initial guesses
    num_terms = len(lm_values)
    initial_amplitudes = np.ones(num_terms) + 1j * np.zeros(num_terms)
    initial_phases = np.zeros(num_terms)
    initial_params = np.hstack([initial_amplitudes.real, initial_amplitudes.imag, initial_phases])

    # Perform fitting
    result = fit_asymmetry_least_squares(A_data, phi, lm_values, k, initial_params)

    # Extract fitted parameters
    fitted_params = result.x
    fitted_A = asymmetry_model(phi, fitted_params, lm_values, k)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(phi, A_data, label='A_data', linestyle='--')
    plt.plot(phi, fitted_A+0.01, label='Fitted A', linewidth=2)
    plt.xlabel('Phi')
    plt.ylabel('Asymmetry A')
    plt.legend()
    plt.title('Asymmetry Fit using least_squares')
    plt.savefig("images/A_fit_least_squares.png")
    plt.show()

    # Extract and normalize fitted amplitudes and phases
    fitted_amplitudes = fitted_params[:num_terms] + 1j * fitted_params[num_terms:2*num_terms]
    fitted_amplitudes /= np.max(np.abs(fitted_amplitudes))  # Normalize
    fitted_phases = fitted_params[2*num_terms:] % (2 * np.pi)

    # Plot fitted amplitudes
    plt.figure(figsize=(10, 6))
    plt.scatter([l for l, _ in lm_values], np.abs(fitted_amplitudes), label='Fitted Amplitudes')
    plt.xlabel('l')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.title('Fitted Amplitudes')
    plt.savefig("images/amplitudes_least_squares.png")
    plt.show()

    # Print fitted parameters
    for i, (l, m) in enumerate(lm_values):
        amplitude = fitted_amplitudes[i]/fitted_amplitudes[0]
        phase = fitted_phases[i]
        print(f"(l={l}, m={m}): Amplitude={amplitude}, Phase={phase}")


if F:
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import least_squares
    from scipy.special import sph_harm, gamma
    import json

    # Given values
    lm_values = [(25, 25), (24, 24), (23, 23), (22, 22), (21, 21),(26, 26)]
    E = 0.48
    k = np.sqrt(2 * E)

    theta = np.pi / 2
    phi = np.arange(0, 2 * np.pi, 0.01)

    # Load the A_data
    A_data = np.load("TDSE_files/A_slice.npy")

    def model_pes(phi, amplitudes, phases, lm_values, k, theta=np.pi/2):
        """
        Compute the Photoelectron Spectrum (PES) for given parameters.
        
        amplitudes: list of complex
            List of complex amplitudes provided by the user.
        phases: array-like, shape (len(lm_values),)
            List of phases to be fitted.
        lm_values: list of tuples
            List containing (l, m) pairs.
        phi: array-like
            Array of azimuthal angles.
        theta: float
            Polar angle (fixed at pi/2 by default).
        k: float
            Wave number based on energy.
        """
        pes = np.zeros_like(phi, dtype=complex)
        for i, (l, m) in enumerate(lm_values):
            amp = amplitudes[i] 
            Y_lm = sph_harm(m, l, phi, theta)
            phase_factor = np.exp(-1j * np.angle(gamma(l + 1 - 1j/k))) * np.exp(-1j * phases[i]) * (1j)**l
            pes += amp * phase_factor * Y_lm
        
        return np.abs(pes)**2

    def asymmetry_model(phi, phases, amplitudes, lm_values, k, theta=np.pi/2):
        """
        Compute the asymmetry A(phi) based on the PES model.
        """
        pes_phi = model_pes(phi, amplitudes, phases, lm_values, k, theta)
        pes_phi_pi = model_pes(phi + np.pi, amplitudes, phases, lm_values, k, theta)
        asymmetry = (pes_phi - pes_phi_pi) / (pes_phi + pes_phi_pi)
        return asymmetry

    def residuals(phases, phi, A_data, amplitudes, lm_values, k, theta=np.pi/2):
        """
        Residuals function for least_squares optimization.
        """
        A_model = asymmetry_model(phi, phases, amplitudes, lm_values, k, theta)
        return A_model - A_data

    def fit_asymmetry_least_squares(A_data, phi_array, amplitudes, lm_values, k, initial_phases):
        """
        Fit asymmetry data using least_squares optimization.
        
        Parameters:
        - A_data: array-like
            The asymmetry data to fit.
        - phi_array: array-like
            Array of phi values corresponding to A_data.
        - amplitudes: list of complex
            The amplitudes specified by the user.
        - lm_values: list of tuples
            List of (l, m) pairs used in the model.
        - k: float
            Wave number based on energy.
        - initial_phases: array-like
            Initial guesses for phases.
        
        Returns:
        - result: OptimizeResult
            The result of the optimization containing optimized phases.
        """
        result = least_squares(
            residuals,
            initial_phases,
            args=(phi_array, A_data, amplitudes, lm_values, k),
            method='trf',  # Trust Region Reflective algorithm
            jac='2-point',  # Jacobian approximation
            verbose=2  # Print progress
        )
        return result

    # Specify amplitudes (as a list of complex values)
    
    specified_amplitudes = []

    with open('amplitudes.json', 'r') as json_file:
        amplitude_dict = json.load(json_file)
    
    specified_amplitudes = list(amplitude_dict.values())
    lm_values = [eval(key) for key in amplitude_dict.keys()]

    # Prepare initial guesses for phases
    initial_phases = np.zeros(len(lm_values))

    # Perform fitting
    result = fit_asymmetry_least_squares(A_data, phi, specified_amplitudes, lm_values, k, initial_phases)

    # Extract fitted phases
    fitted_phases = result.x % (2 * np.pi)

    # Calculate the fitted asymmetry
    fitted_A = asymmetry_model(phi, fitted_phases, specified_amplitudes, lm_values, k)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(phi, A_data, label='A_data', linestyle='--')
    plt.plot(phi, fitted_A + 0.01, label='Fitted A', linewidth=2)
    plt.xlabel('Phi')
    plt.ylabel('Asymmetry A')
    plt.legend()
    plt.title('Asymmetry Fit using least_squares')
    plt.savefig("images/A_fit_least_squares_phases.png")
    plt.show()

    # Print fitted phases
    for i, (l, m) in enumerate(lm_values):
        print(f"(l={l}, m={m}): Phase={fitted_phases[i]}")
