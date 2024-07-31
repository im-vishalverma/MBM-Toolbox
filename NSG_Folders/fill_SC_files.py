import shutil
import os

# Base directories for source files and destination folders
source_base_dir = r"C:\Users\vhima\Downloads\SC_folder"
destination_base_dir = r"C:\Users\vhima\Downloads\NSG_Folder"

# Iterate over each subject_id
for subject_id in subject_ids:
    # Construct the source file path and the destination folder path
    file_path = rf"{source_base_dir}\{subject_id}_SC.mat"
    folder_path = rf"{destination_base_dir}\{subject_id}"
    
    # Ensure the destination folder exists
    os.makedirs(folder_path, exist_ok=True)
    
    # Construct the destination file path
    destination_file = os.path.join(folder_path, os.path.basename(file_path))
    
    # Move the file
    shutil.move(file_path, destination_file)
    
    print(f"File {file_path} moved to {destination_file}")
