pairs = []
for x_ in [0.25, 5.25,10.25]:
    for y_ in [0.25,5.25,10.25]:
        pair = (x_, y_)
        pairs.append(pair)
folder_names = [ 'AA_20120815', 'AY_20111004', 'BQ_20120904', 'DH_20120806', 'EU_20120803', 'FE_20111010', 'FI_20120727', 'FJ_20120808', 'GC_20120803', 'IC_20120810', 'IS_20120809', 'NI_20120831', 'QL_20110925', 'LQ_20120814', 'QR_20111010', 'RF_20120809', 'RI_20110924', 'RI_20120815', 'RS_20120723', 'RT_20110925', 'SE_20110924', 'UB_20120806', 'UK_20110924', 'UK_20111004', 'XB_20120831',
'AC_20120917', 'AR_20120813', 'CN_20120927', 'DA_20120813', 'DG_20120903', 'ER_20120816', 'FR_20120903', 'HA_20120813', 'IQ_20120904', 'JD_20120810', 'JH_20120925', 'JH_20121009', 'JL_20120927', 'JS_20120910', 'JZ_20120824', 'KI_20121009', 'NN_20120824', 'NN_20120831', 'OG_20120917', 'OK_20121011', 'OQ_20120925', 'RQ_20120903', 'RQ_20120917', 'YE_20120910' ]
subject_ids = folder_names      
for subject_id_ in subject_ids:
    output_dir = rf"C:\Users\vhima\Downloads\NSG_Folder\{subject_id_}"
    
    for index, (Tglu_low_, Tgaba_low_) in enumerate(pairs):
        # Replace the placeholder with the actual parameter value
        code_with_param = base_code.format(subject_id = subject_id_, Tglu_low = Tglu_low_, Tgaba_low = Tgaba_low_, ind = index+1)

        # Define the filename
        filename = f"{output_dir}/input{index+1}.py"

        # Write the code to the file
        with open(filename, "w") as file:
            file.write(code_with_param)

        # Print the filename and its contents to verify
        print(f"Created {filename}")
        print(code_with_param)
        print("-" * 24)
        
    print("Scripts created successfully.")
