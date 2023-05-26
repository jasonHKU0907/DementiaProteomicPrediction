

import glob
import os
import numpy as np
import pandas as pd
import re
import time

time_start = time.time()

def get_days_intervel(start_date_var, end_date_var, df):
    start_date = pd.to_datetime(df[start_date_var], dayfirst=True)
    end_date = pd.to_datetime(df[end_date_var], dayfirst=True)
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    my_yrs = [ele/365 for ele in days]
    return pd.DataFrame(my_yrs)

def get_fo_info(target_cancer_f, target_cancer_df, nb_subjects):
    target_date_f = [ele + '_date' for ele in target_cancer_f]
    target_date_df = target_cancer_df[target_date_f].copy()
    fo_date_lst, fo_dis_lst = [], []
    for i in range(nb_subjects):
        ind_target_df = target_date_df.iloc[i, :]
        fo_dis_date = pd.to_datetime(ind_target_df, dayfirst=True).min()
        try:
            fo_dis_FieldID = ind_target_df.index[pd.to_datetime(ind_target_df, dayfirst=True) == fo_dis_date][0]
            fo_dis_FieldID = fo_dis_FieldID.split('_')[0]
        except:
            fo_dis_FieldID = np.nan
        fo_date_lst.append(fo_dis_date)
        fo_dis_lst.append(fo_dis_FieldID)
    return pd.DataFrame(fo_date_lst), pd.DataFrame(fo_dis_lst)

def remove_duplicated_eid(tmpdf, item_date):
    dup_eid_lst = tmpdf.loc[tmpdf.duplicated(subset=['eid']) == True].eid.tolist()
    rm_idx_lst = []
    for eid in dup_eid_lst:
        dup_idx = tmpdf.index[tmpdf.eid == eid].tolist()
        dup_df = tmpdf.iloc[dup_idx]
        earlist_date = dup_df.loc[dup_df.eid == eid][item_date].min()
        keep_idx = dup_df.index[dup_df[item_date] == earlist_date][0]
        dup_idx.remove(keep_idx)
        rm_idx_lst += dup_idx
    return rm_idx_lst

def get_target_cancer_df(cancer_code_df, cancer_date_df, item_code):
    item_date= item_code + '_date'
    index = np.where(cancer_code_df == item_code)
    row_idx = index[0].tolist()
    col_idx = index[1].tolist()
    eid_lst = cancer_date_df.iloc[row_idx].eid.tolist()
    cancer_date_lst = [cancer_date_df.iloc[row_idx[i], col_idx[i]] for i in range(len(row_idx))]
    tmpdf = pd.DataFrame({'eid': eid_lst, item_date: cancer_date_lst})
    rm_idx_lst = remove_duplicated_eid(tmpdf, item_date)
    tmpdf.drop(rm_idx_lst, axis = 0, inplace = True)
    tmpdf.reset_index(inplace = True)
    return tmpdf[['eid', item_date]]


dpath = '/Volumes/JasonWork/Dataset/UKB_20230417/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/OtherNeurDis/'

date_f_lst = ['41280-0.'+str(i) for i in range(259)]
code_f_lst = ['41270-0.'+str(i) for i in range(259)]

cancer_df = pd.read_csv(dpath + 'ukb672277_HospitalInpatient_data.csv', usecols = ['eid'] + date_f_lst + code_f_lst)
target_info_df = pd.read_csv(dpath + 'Target_info_data.csv', usecols=['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs'])
nb_subjects = len(cancer_df)
cancer_df = pd.merge(cancer_df, target_info_df, how = 'left', on = ['eid'])

target_cancer_f = ['G10', 'G11', 'G12', 'G13', 'G14', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G31', 'G32']

cancer_date_df = cancer_df[['eid'] + date_f_lst].copy()
cancer_code_df = cancer_df[['eid'] + code_f_lst].copy()
cancer_code_df.fillna('NA0123', inplace = True)

for col_i in range(1, cancer_code_df.shape[1]):
    cancer_code_df.iloc[:, col_i] = cancer_code_df.iloc[:, col_i].apply(lambda x : x[:3])

target_cancer_df = cancer_df[['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs']]

for item_code in target_cancer_f:
    tmpdf = get_target_cancer_df(cancer_code_df, cancer_date_df, item_code)
    target_cancer_df = pd.merge(target_cancer_df, tmpdf, how = 'left', on = ['eid'])

target_cancer_df['target_date'], target_cancer_df['target_cancer'] = get_fo_info(target_cancer_f, target_cancer_df, nb_subjects)

target_cancer_df['target_y'] = 1
target_cancer_df['target_y'][target_cancer_df['target_date'].isnull() == True] = 0
target_cancer_df['BL2Target_yrs_raw'] = get_days_intervel('BL_date', 'target_date', target_cancer_df)
tmpdf = target_cancer_df[['BL2Target_yrs_raw', 'BL2End_yrs', 'BL2Death_yrs']]
target_cancer_df['BL2Target_yrs'] = pd.DataFrame([tmpdf.iloc[i,:].min() for i in range(nb_subjects)])

out_cols = ['eid', 'target_cancer', 'target_date', 'BL2Target_yrs_raw', 'target_y', 'BL2Target_yrs']
target_df_out = target_cancer_df[out_cols]
target_df_out.to_csv(outpath + 'ONeurD_InHosp.csv', index = False)

time_end2 = time.time()
print('Finish and it total cost ' + str(np.round(time_end2 - time_start, 1)) + ' seconds')




