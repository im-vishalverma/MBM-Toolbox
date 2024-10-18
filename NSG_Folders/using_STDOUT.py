import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_=[]
for i in np.arange(0.25,15.25,0.25):
        _.append(i)        
titles = ['Mean Firing Rate', 'FC distance', 'FC correlation', 'Metastability', 'MI distance','MI correlation','dFC KS distance']
# Function to extract and parse array data
def extract_arrays(file_content):
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, file_content)
    arrays = []
    for match in matches:
        array = list(map(float, match.split()))
        arrays.append(array)
    
    return arrays
for k in range(7):
    Xi = np.zeros((60,60))
    
    for file_num in range(9):
        with open(fr"C:\Users\Dell\Downloads\STDOUT ({file_num})", 'r') as file:
            file_content = file.read()
        arrays = extract_arrays(file_content)
        df = pd.DataFrame(arrays)
        first_row = df.iloc[:,k] 
        # by changing k to some other integer from 0 to 6, we 7 phase spaces mentioned the paper
        reshaped_array = np.array(first_row).reshape((20, 20))
        arr=reshaped_array
        
        if file_num == 0:
            Xi[:20, :20 ] = arr
    
        elif file_num == 1:
            Xi[:20, 20:40 ] = arr
    
        elif file_num == 2:
            Xi[:20, 40:60 ] = arr
    
        elif file_num == 3:
            Xi[20:40, :20 ] = arr
    
        elif file_num == 4:
            Xi[20:40, 20:40 ] = arr
    
        elif file_num == 5:
            Xi[20:40, 40:60 ] = arr
    
        elif file_num == 6:
            Xi[40:60, :20 ] = arr
    
        elif file_num == 7:
            Xi[40:60, 20:40 ] = arr
    
        elif file_num == 8:
            Xi[40:60, 40:60 ] = arr
    
    plt.subplot(221)
    plt.imshow(Xi, 'nipy_spectral_r', aspect = 'auto',origin = 'lower')
    plt.xticks([0,30,60],[0,7.5,15])
    plt.yticks([0,30,60],[0,7.5,15])
    plt.title(titles[k],fontsize=12)
    plt.xlabel('GABA (mmol)',fontsize=12)
    plt.ylabel('Glutamate (mmol)',fontsize=12)
    plt.colorbar()
    plt.show()
    if k == 1:
        alpha = np.where(Xi==np.min(Xi))
        print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]],'when measure is FC DISTANCE')
    if k == 2:
        alpha = np.where(Xi==np.max(Xi))
        print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]],'when measure is FC CORRELATION')
    if k == 4:
        alpha = np.where(Xi==np.min(Xi))
        print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]],'when measure is MI DISTANCE')
    if k == 5:
        alpha = np.where(Xi==np.max(Xi))
        print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]],'when measure is MI CORRELATION')
    if k == 6:
        alpha = np.where(Xi==np.min(Xi))
        print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]],'when measure is DFC KS DISTANCE')
