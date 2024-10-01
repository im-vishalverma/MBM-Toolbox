import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def segmentation(bold_data, window, overlap):
    nAreas = 68
    width = window - overlap
    num_series = len(np.arange(0, len(bold_data) - window, width))
    num_steps = int(window)
    segmented_data = np.zeros((num_series, num_steps, nAreas))

    for i in range(nAreas):
        for j in np.arange(0, len(bold_data) - window, width):
            segmented_data[int(j / width), :, i] = bold_data[j:j + window, i]
    
    return segmented_data

def plot_tsne(bold_data_emp, window, overlap):
    # Segment the data
    segmented_data = segmentation(bold_data_emp, window, overlap)

    # Calculate correlation matrices
    width = window - overlap
    num_series = len(np.arange(0, len(bold_data_emp) - window, width))
    correlation_matrix = np.zeros((num_series, 68, 68))
    
    for j in range(num_series):
        correlation_matrix[j, :, :] = np.corrcoef(segmented_data[j, :, :], rowvar=False)

    # Flatten each correlation matrix into a 1D array
    X = np.array([matrix.flatten() for matrix in correlation_matrix])

    # Set perplexity and random state
    perplexity = min(30, X.shape[0] - 1)
    random_state = 42

    # Initialize t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)

    # Fit and transform the data
    X_tsne = tsne.fit_transform(X)

    # Plot the t-SNE results with connecting lines
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='o', c='blue', edgecolor='none', s=30)

    # Draw lines connecting consecutive points
    for i in range(X_tsne.shape[0] - 1):
        plt.plot([X_tsne[i, 0], X_tsne[i + 1, 0]], [X_tsne[i, 1], X_tsne[i + 1, 1]], 'k-', alpha=0.5)

    # Ensure the last point is connected to the first point
    plt.plot([X_tsne[-1, 0], X_tsne[0, 0]], [X_tsne[-1, 1], X_tsne[0, 1]], 'k-', alpha=0.5)

    plt.title("t-SNE Visualization of fMRI Data")
    plt.show()

# Example usage
path = rf"C:\Users\vhima\Downloads\fMRI_folder\AC_20120917_fMRI_new.mat"
from scipy.io import loadmat
mat = loadmat(path)
bold_data_emp = mat['AC_20120917_ROIts_68']

plot_tsne(bold_data_emp, window=30, overlap=25)
