import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorly as tl
from tensorly.decomposition import parafac
from sklearn.cluster import KMeans
from scipy.spatial.distance import squareform, pdist
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score

# segmentation of the data

def segmentation(bold_data,window,overlap):
    x = bold_data
    nAreas = 68
    shift = window-overlap
    num_series = len(np.arange(0,len(x)-window,shift))
    num_steps = window 
    y = np.zeros((num_series,num_steps,nAreas))
    
    for i in np.arange(nAreas):
        for j in np.arange(0,len(x[:,0])-window,shift):
            y[int(j/shift),:,i] = x[j:j+window,i]
    
    return y


window = 60
overlap = 55 

def compute_distances(X, Y):
    points = np.column_stack((X, Y))
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)  # k=2 to include the point itself
    epsilon = distances[:, 1]  # 1st nearest neighbor distances
    return epsilon

def count_neighbors_within_epsilon(X, epsilon):
    N = len(X)
    counts = np.zeros(N)
    for i in range(N):
        counts[i] = np.sum(np.abs(X - X[i]) <= epsilon[i]) - 1  # Exclude the point itself
    return counts

def estimate_mutual_information(X, Y, k=1):
    N = len(X)
    epsilon = compute_distances(X, Y)
    n_x = count_neighbors_within_epsilon(X, epsilon)
    n_y = count_neighbors_within_epsilon(Y, epsilon)
    mi = digamma(k) + digamma(N) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    return mi

def MI_matrix(data):
    n_samples, n_signals = data.shape
    I_XY = np.zeros((n_signals, n_signals))
    k = 1  # consider the first nearest neighbor
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            mi = estimate_mutual_information(data[:, i], data[:, j], k)
            I_XY[i, j] = np.abs(mi)
            I_XY[j, i] = np.abs(mi)  # MI is symmetric
    return I_XY

tensors = []
for subject_id in subject_ids:
    path = r"C:\Users\vhima\Downloads\Jodhpur\Old\Old\FC-old\AC_20120917_fMRI_new.mat"
    mat_data = loadmat(path)
    data_emp = mat_data[f'{subject_id}_ROIts_68']
    data = segmentation(data_emp,window,overlap)
    shape = data.shape
    mi_matrix = np.zeros((shape[0],68,68))
    for i in range(shape[0]):
        data_ = data[i,:,:] 
        mi_matrix[i,:,:] = MI_matrix(data_) 
    tensors.append(mi_matrix)
_ = []

for t in tensors:
    a = np.percentile(t,98)
    tense = np.where(t>a, 1, 0)
    _.append(tense)

F = 4
# decompose and three feature pools
Feature_pool_2 = []
Feature_pool_3 = []

for tensor in _:
    factors = parafac(tensor, rank=F)
    factor_matrices = factors.factors

    factor_matrix_2 = factor_matrices[1]
    factor_matrix_3 = factor_matrices[2]
    Feature_pool_2.append(factor_matrix_2[:,0])    
    Feature_pool_2.append(factor_matrix_2[:,1])    
    Feature_pool_2.append(factor_matrix_2[:,2])    
    Feature_pool_3.append(factor_matrix_3[:,0])    
    Feature_pool_3.append(factor_matrix_3[:,1])    
    Feature_pool_3.append(factor_matrix_3[:,2])

matrix = np.corrcoef(Feature_pool_2)
correlation_matrix = np.corrcoef(matrix, rowvar=False)
distance_matrix = (1 - correlation_matrix) / 2
condensed_distance_matrix = squareform(distance_matrix, checks=False)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
data_transformed = mds.fit_transform(distance_matrix)
# Apply KMeans to the transformed data
k = 4 
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data_transformed)
# Get the cluster labels
labels = kmeans.labels_
# Reorder the correlation matrix according to the cluster labels
sorted_indices = np.argsort(labels)
sorted_correlation_matrix = correlation_matrix[sorted_indices, :][:, sorted_indices]
s_a = silhouette_score(data_transformed, labels)   
print(s_a)
# Plot the reordered correlation matrix with cluster annotations
plt.figure(figsize=(6, 5))
sns.heatmap(sorted_correlation_matrix, cmap='viridis', annot=False, xticklabels=sorted_indices, yticklabels=sorted_indices)
plt.title('Correlation Matrix with K-means Clusters')
plt.axis('off')
plt.show()
