
import numpy as np
_=[]
for i in np.arange(0.25,15.25,0.25):
        _.append(i)        


import pandas as pd
import re
import matplotlib.pyplot as plt
# Function to extract and parse array data
def extract_arrays(file_content):
    # Regex pattern to match arrays
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, file_content)
    
    # Convert matches to list of lists of floats
    arrays = []
    for match in matches:
        array = list(map(float, match.split()))
        arrays.append(array)
    
    return arrays

graph=1
Xi = np.zeros((60,60))
# Read the file content

for i in range(9):
    with open(fr"C:\Users\vhima\Downloads\STDOUT ({i+5})", 'r') as file:
        file_content = file.read()

    # Extract arrays
    arrays = extract_arrays(file_content)

    # Create a DataFrame
    df = pd.DataFrame(arrays)

    # Extract the first row
    first_row = df.iloc[:,-3]

    # Reshape the first row into a 20x20 array
    reshaped_array = np.array(first_row).reshape((20, 20))

    # Print the reshaped array
#         print(reshaped_array)
    file_num=i
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
plt.imshow(Xi, 'bwr',aspect = 'auto',origin = 'lower')
plt.axis('off')
plt.title(f'{graph}')
plt.colorbar()
plt.show()
alpha = np.where(Xi==np.min(Xi))
print('glu = ', _[alpha[0][0]], ', gaba = ', _[alpha[1][0]])
graph+=1
