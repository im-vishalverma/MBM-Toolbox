from scipy.io import loadmat
# example folder names
folder_names = [ 'AA_20120815', 'AY_20111004', 'BQ_20120904', 'DH_20120806', 'EU_20120803', 'FE_20111010', 'FI_20120727', 'FJ_20120808', 'GC_20120803', 'IC_20120810', 'IS_20120809', 'NI_20120831', 'QL_20110925', 'LQ_20120814', 'QR_20111010', 'RF_20120809', 'RI_20110924', 'RI_20120815', 'RS_20120723', 'RT_20110925', 'SE_20110924', 'UB_20120806', 'UK_20110924', 'UK_20111004', 'XB_20120831',
'AC_20120917', 'AR_20120813', 'CN_20120927', 'DA_20120813', 'DG_20120903', 'ER_20120816', 'FR_20120903', 'HA_20120813', 'IQ_20120904', 'JD_20120810', 'JH_20120925', 'JH_20121009', 'JL_20120927', 'JS_20120910', 'JZ_20120824', 'KI_20121009', 'NN_20120824', 'NN_20120831', 'OG_20120917', 'OK_20121011', 'OQ_20120925', 'RQ_20120903', 'RQ_20120917', 'YE_20120910' ]
# example subject_ids
subject_ids = folder_names
# first place all files containing sc files in SC_Folder and all files containing bold-data files in fMRI_Folder
# Then, run the following code to check if the folders contain all the required files
i=0
for subject_id in subject_ids:
    scpath = rf"C:\Users\vhima\Downloads\SC_Folder\{subject_id}_SC.mat"
    fcpath = rf"C:\Users\vhima\Downloads\fMRI_Folder\{subject_id}_fMRI_new.mat"
    sc_data =loadmat(scpath)
    fmri_data =loadmat(fcpath)
    SC = sc_data['SC_cap_agg_bwflav1']
    key = rf'{subject_id}_ROIts'
    bold = fmri_data[key]
    i+=1
