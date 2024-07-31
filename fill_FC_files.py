import shutil
import os

subject_ids = folder_names

# Loop over each subject ID
for subject_id in subject_ids:
    # Construct the file path for the current subject
    file_path = rf"C:\Users\vhima\Downloads\fMRI_Folder\{subject_id}_fMRI_new.mat"
    
    # Construct the folder path for the current subject
    folder_path = rf"C:\Users\vhima\Downloads\NSG_Folder\{subject_id}"
    
    # Ensure the destination folder exists; if not, create it
    os.makedirs(folder_path, exist_ok=True)
    
    # Move the file to the destination folder
    shutil.move(file_path, folder_path)
    
    print(f"File for subject '{subject_id}' has been moved to {folder_path}")
