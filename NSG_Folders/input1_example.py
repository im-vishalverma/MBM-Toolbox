# The output script files shall look as:

# Created C:\Users\vhima\Downloads\NSG_Folder\AR_20120813/input1.py

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.signal import filtfilt, butter, hilbert
from scipy.stats import ks_2samp

file_path = r"AR_20120813_SC.mat"
mat_file = loadmat(file_path)
sc = mat_file['SC_cap_agg_bwflav1_norm']
file_path_ = r"AR_20120813_fMRI_new.mat"
mat_file_ = loadmat(file_path_)
key = r'AR_20120813_ROIts_68'
bold_emp = mat_file_[key]

num_minutes = 23
num_sim = int(num_minutes*60*1000) 
nAreas=68

# simulate firing rate and bold data

def firing_rate(Tglu,Tgaba):
    # params
    G = 0.69
    I0 = 0.382
    JNMDA = 0.15
    WE = 1.0
    WI = 0.7
    wplus = 1.4
    aI = 615
    bI = 177
    dI = 0.087
    aE = 310
    bE = 125
    alphaE = 0.072
    betaE = 0.0066
    alphaI = 0.53
    betaI = 0.18
    sigma = 0.001
    gamma = 1
    rho = 3
    dE = 0.16
    nAreas = 68
    
    # equations
    def dSE(SE,SI,J, IE,II,rE,rI):
        return -betaE*SE + (alphaE/1000)*Tglu*rE*(1-SE)+sigma*0
    def dSI(SE,SI,SJ, IE,II,rE,rI):
        return -betaI*SI + (alphaI/1000)*Tgaba*rI*(1-SI)+sigma*0
    def dJ(SE,SI,J, IE,II,rE,rI):
        return gamma*(rI/1000)*(rE-rho)/1000

    # initial conditions
    SE = 0.001*np.ones(nAreas)
    SI = 0.001*np.ones(nAreas)
    J = np.ones(nAreas)
    dt = 0.1

    synaptic_gating_var = np.zeros((num_sim,nAreas))
    excit_firing_rate = np.zeros((num_sim,nAreas))

    
    # simulation
    for i in range(num_sim):            
        IE = WE*I0+wplus*JNMDA*SE+G*JNMDA*np.matmul(sc/np.max(sc),SE) - J*SI
        II = WI*I0+JNMDA*SE-SI
        rE = (aE*IE - bE)/(1-np.exp(-dE*(aE*IE - bE)))
        rI = (aI*II - bI)/(1-np.exp(-dI*(aI*II - bI)))

        SI += dt*dSI(SE,SI,J, IE,II,rE,rI)
        SE += dt*dSE(SE,SI,J, IE,II,rE,rI)
        J  += dt*dJ(SE,SI,J,  IE,II,rE,rI)
        
        SI = np.where(SI>1, 1, SI)
        SI = np.where(SI<0, 0, SI)
        SE = np.where(SE>1, 1, SE)
        SE = np.where(SE<0, 0, SE)

        synaptic_gating_var[i,:] = SE 
        excit_firing_rate[i,:] = rE
        
    return synaptic_gating_var, np.max(np.sum(excit_firing_rate[200000:,:],axis=1)/68)

def bold(r):    
    # params
    taus   = 0.65 #
    tauf   = 0.41 # 0.4
    tauo   = 0.98 # 1
    alpha  = 0.32 # 0.2
    itaus  = 1/taus
    itauf  = 1/tauf
    itauo  = 1/tauo
    ialpha = 1/alpha
    Eo     = 0.34 # 0.8
    vo     = 0.02
    k1     = 7*Eo
    k2     = 2
    k3     = 2*Eo-0.2

    # equations
    def slope_x1(x1,x2,x3,x4):
        return  -itaus*x1 -itauf*(x2-1)
    def slope_x2(x1,x2,x3,x4):
        return x1
    def slope_x3(x1,x2,x3,x4):
        return itauo*(x2-(x3)**ialpha)
    def slope_x4(x1,x2,x3,x4):
        return itauo*(x2*(1-(1-Eo)**(1/x2))/Eo - (x3**ialpha)*x4/x3)
    
    dt = 0.001
    nAreas = 68

    # Initial Conditions
    
    x1 = np.ones(nAreas)
    x2 = np.ones(nAreas)
    x3 = np.ones(nAreas)
    x4 = np.ones(nAreas)

    bold = np.zeros((num_sim//2000, nAreas))

    # simulation
    for i in range(num_sim):
        x1 += dt*slope_x1(x1,x2,x3,x4) + dt*r[i,:]
        x2 += dt*slope_x2(x1,x2,x3,x4)
        x3 += dt*slope_x3(x1,x2,x3,x4)
        x4 += dt*slope_x4(x1,x2,x3,x4)

        if i % 2000 ==1:
            b = 100 / Eo * vo * (k1 * (1 - x4) + k2 * (1 - x4 / x3) + k3 * (1 - x3))
            bold[i//2000, :] = b
        
    return bold[30:,:]

# defining measures

def slicing(bold_data,window,overlap):
    
    number_of_slices = ((bold_data.shape[0]-window)//(window-overlap)) + 1
    slices = np.zeros((number_of_slices, window, bold_data.shape[1]))

    for i in range(number_of_slices):
        slices[i,:,:] = bold_data[(window-overlap)*i:(window-overlap)*i+window,:]

    correlation_matrix = np.zeros((number_of_slices,bold_data.shape[1],bold_data.shape[1]))
    
    for i in range(number_of_slices):
        correlation_matrix[i,:,:] = np.corrcoef(slices[i,:,:], rowvar = False)

    dFC = np.zeros((number_of_slices,number_of_slices))
    
    for i in range(number_of_slices):
        for j in range(number_of_slices):
            m1 = correlation_matrix[i,:,:].flatten()
            m2 = correlation_matrix[j,:,:].flatten()
            dFC[i,j] = np.corrcoef(m1,m2)[0,1]

    return dFC

# find ksd

def ks_distance_between_matrices(matrix1, matrix2):

    flat_matrix1 = matrix1.flatten()
    flat_matrix2 = matrix2.flatten()

    sorted_matrix1 = np.sort(flat_matrix1)
    sorted_matrix2 = np.sort(flat_matrix2)

    ks_distance, _ = ks_2samp(sorted_matrix1, sorted_matrix2)

    return ks_distance

def func_connec(bold_data):
    fc_sim = np.corrcoef(bold_data,rowvar = False)
    return fc_sim

def metastability_index(bold_data):
    PhaseSignal = np.angle(hilbert(bold_data, axis=0))  
    PhaseSignal = np.exp(1j * PhaseSignal)  
    order_parameter = np.mean(PhaseSignal, axis=0)
    m = np.std(np.abs(order_parameter)) 
    return m

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

def MI_matrix(bold_data):
    n_samples, n_signals = bold_data.shape
    I_XY = np.zeros((n_signals, n_signals))
    k = 1  # consider the first nearest neighbor
    for i in range(n_signals):
        for j in range(i+1, n_signals):
            mi = estimate_mutual_information(bold_data[:, i], bold_data[:, j], k)
            I_XY[i, j] = np.abs(mi)
            I_XY[j, i] = np.abs(mi)  # MI is symmetric
    return I_XY

def eucledian_dist(matrix1, matrix2):
    # Flatten the matrices into 1D arrays
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()
    
    # Compute the squared differences element-wise
    squared_diff = (vector1 - vector2) ** 2
    
    # Sum the squared differences and take the square root
    distance = np.sqrt(np.sum(squared_diff))
    return distance/68

def matrix_corrcoef(matrix1, matrix2):
    vector1 = matrix1.flatten()
    vector2 = matrix2.flatten()
    _corr = np.corrcoef(vector1, vector2)[0,1]
    return _corr

Xi = np.zeros((400,7))

s = time.time()
window = 30
overlap = 25
FC_emp = func_connec(bold_emp)
mi_emp = MI_matrix(bold_emp)
dFC_emp = slicing(bold_emp,window,overlap)

i = 0
for Tglu in np.arange(0.25,0.25+5,0.25):
    for Tgaba in np.arange(0.25,0.25+5,0.25):
        r,x = firing_rate(Tglu,Tgaba)
        bold_sim = bold(r)
        FC_sim = func_connec(bold_sim)
        fc_dist = eucledian_dist(FC_sim, FC_emp)
        fc_corr = matrix_corrcoef(FC_sim, FC_emp)
        meta = metastability_index(bold_sim)
        mi_sim = MI_matrix(bold_sim)
        mi_dist = eucledian_dist(mi_sim,mi_emp)
        mi_corr = matrix_corrcoef(mi_sim,mi_emp)
        dFC_sim = slicing(bold_sim,window,overlap)
        ksd = ks_distance_between_matrices(dFC_sim, dFC_emp)
        Xi[i,:] = x,fc_dist,fc_corr,meta,mi_dist,mi_corr,ksd
        i += 1

f = time.time()
print("x")
print("fc_dist")
print("fc_corr")
print("meta")
print("mi_dist")
print("mi_corr")
print("ksd")

print("\n" , f-s , "seconds")

# Convert the array to a DataFrame
df = pd.DataFrame(Xi)
headers = 'x','fc_dist','fc_corr','meta','mi_dist','mi_corr','ksd'
# Save the DataFrame as a CSV file
file_name = f"Xi1.csv"
df.to_csv(file_name, header=headers, index=False)

print(df.head())

------------------------
