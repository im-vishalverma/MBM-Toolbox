import os
# code to create empty folders for each subject of a dataset
# Create subject-wise folder in the list
def empty_folders(base_directory, folder_names):
    for folder_name in folder_names:
        folder_path = os.path.join(base_directory, folder_name)
        try:
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
        except FileExistsError:
            print(f"Folder already exists: {folder_path}")
        except Exception as e:
            print(f"Error creating folder {folder_name}: {e}")  
        
# example folder_names:
folder_names = [ 'AA_20120815', 'AY_20111004', 'BQ_20120904', 'DH_20120806', 'EU_20120803', 'FE_20111010', 'FI_20120727', 'FJ_20120808', 'GC_20120803', 'IC_20120810', 'IS_20120809', 'NI_20120831', 'QL_20110925', 'LQ_20120814', 'QR_20111010', 'RF_20120809', 'RI_20110924', 'RI_20120815', 'RS_20120723', 'RT_20110925', 'SE_20110924', 'UB_20120806', 'UK_20110924', 'UK_20111004', 'XB_20120831',
'AC_20120917', 'AR_20120813', 'CN_20120927', 'DA_20120813', 'DG_20120903', 'ER_20120816', 'FR_20120903', 'HA_20120813', 'IQ_20120904', 'JD_20120810', 'JH_20120925', 'JH_20121009', 'JL_20120927', 'JS_20120910', 'JZ_20120824', 'KI_20121009', 'NN_20120824', 'NN_20120831', 'OG_20120917', 'OK_20121011', 'OQ_20120925', 'RQ_20120903', 'RQ_20120917', 'YE_20120910' ]
# example base directory
base_directory = r"C:\Users\vhima\Downloads\NSG_Folder" # example path
empty_folders(base_directory, folder_names)
