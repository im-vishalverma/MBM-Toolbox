import numpy as np
from scipy.fft import fft, ifft

def segmentation(bold_data, window, overlap):
    nAreas = bold_data.shape[1]
    width = window - overlap
    num_series = len(np.arange(0, len(bold_data) - window, width))
    num_steps = int(window)
    segmented_data = np.zeros((num_series, num_steps, nAreas))

    for i in range(nAreas):
        for j in np.arange(0, len(bold_data) - window, width):
            segmented_data[int(j / width), :, i] = bold_data[j:j + window, i]
    
    return segmented_data

def generate_surrogate_data(bold_data_emp, window, overlap):
    # Apply FFT to the BOLD data to get signal in the frequency domain
    signal = np.zeros_like(bold_data_emp, dtype=np.complex128)
    for j in range(68):
        signal[:, j] = fft(bold_data_emp[:, j])

    # Extract phases and amplitudes
    phases = np.angle(signal)
    amplitudes = np.abs(signal)

    # Segment phases and amplitudes
    segmented_data_phases = segmentation(phases, window, overlap)
    segmented_data_amplitudes = segmentation(amplitudes, window, overlap)

    # Generate random phases
    random_phases = np.random.uniform(0, 2 * np.pi, window)

    # Add random phases to the segmented data
    segmented_data_phases_random = segmented_data_phases + random_phases.reshape((window, 1))

    # Reconstruct the surrogate data using inverse FFT
    surrogate_data = np.real(ifft(np.exp(1j * segmented_data_phases_random) * segmented_data_amplitudes, axis=1))
    
    return surrogate_data

# Example usage
path = rf"C:\Users\vhima\Downloads\fMRI_folder\AC_20120917_fMRI_new.mat"
from scipy.io import loadmat
mat = loadmat(path)
bold_data_emp = mat['AC_20120917_ROIts_68']

# Set window and overlap values
window = 30
overlap = 25

# Generate surrogate data
surrogate_data = generate_surrogate_data(bold_data_emp, window, overlap)

