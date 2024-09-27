# STEP FOUR   
import shutil
import os
# example folder names and subject IDs
folder_names = [ 'AA_20120815', 'AY_20111004', 'BQ_20120904', 'DH_20120806', 'EU_20120803', 'FE_20111010', 'FI_20120727', 'FJ_20120808', 'GC_20120803', 'IC_20120810', 'IS_20120809', 'NI_20120831', 'QL_20110925', 'LQ_20120814', 'QR_20111010', 'RF_20120809', 'RI_20110924', 'RI_20120815', 'RS_20120723', 'RT_20110925', 'SE_20110924', 'UB_20120806', 'UK_20110924', 'UK_20111004', 'XB_20120831',
'AC_20120917', 'AR_20120813', 'CN_20120927', 'DA_20120813', 'DG_20120903', 'ER_20120816', 'FR_20120903', 'HA_20120813', 'IQ_20120904', 'JD_20120810', 'JH_20120925', 'JH_20121009', 'JL_20120927', 'JS_20120910', 'JZ_20120824', 'KI_20121009', 'NN_20120824', 'NN_20120831', 'OG_20120917', 'OK_20121011', 'OQ_20120925', 'RQ_20120903', 'RQ_20120917', 'YE_20120910' ]
subject_ids = folder_names

# Example Base directories for source files and destination folders
source_base_dir = r"C:\Users\vhima\Downloads\fMRI_folder"
destination_base_dir = r"C:\Users\vhima\Downloads\NSG_Folder"

# First, copy and then paste all the files containing the bold data into the freshly created fMRI_Folder
# Iterate over each subject_id to send those files subject wise into their respective subject wise "empty folder"
for subject_id in subject_ids:
    # Construct the source file path and the destination folder path
    file_path = rf"{source_base_dir}\{subject_id}_fMRI_Folder.mat"
    folder_path = rf"{destination_base_dir}\{subject_id}"    
    # Ensure the destination folder exists
    os.makedirs(folder_path, exist_ok=True)
    # Construct the destination file path
    destination_file = os.path.join(folder_path, os.path.basename(file_path))
    # Move the file
    shutil.move(file_path, destination_file)
    print(f"File {file_path} moved to {destination_file}")
