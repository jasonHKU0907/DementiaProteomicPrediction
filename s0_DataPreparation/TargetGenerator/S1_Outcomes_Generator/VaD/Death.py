

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

def get_fo_info(target_death_f, target_death_df, nb_subjects):
    target_date_f = [ele + '_date' for ele in target_death_f]
    target_date_df = target_death_df[target_date_f].copy()
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


def get_target_death_df(death_code_df, death_date_df, item_code):
    item_date = item_code + '_date'
    index = np.where(death_code_df == item_code)
    row_idx = index[0].tolist()
    eid_lst = death_code_df.iloc[row_idx].eid.tolist()
    death_date_lst = [death_date_df[['40000-0.0']].iloc[row_idx[i], 0] for i in range(len(row_idx))]
    tmpdf = pd.DataFrame({'eid': eid_lst, item_date: death_date_lst})
    rm_idx_lst = remove_duplicated_eid(tmpdf, item_date)
    tmpdf.drop(rm_idx_lst, axis=0, inplace=True)
    tmpdf.reset_index(inplace=True)
    return tmpdf[['eid', item_date]]

dpath = '/Volumes/JasonWork/Dataset/UKB_20230417/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/VaD/'

code_f_lst = ['40001-0.0','40001-1.0'] + ['40002-0.' + str(i) for i in range(15)]

death_df = pd.read_csv(dpath + 'ukb672277_death_data.csv', usecols = ['eid', '40000-0.0'] + code_f_lst)
target_info_df = pd.read_csv(dpath + 'Target_info_data.csv', usecols=['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs'])
nb_subjects = len(death_df)
death_df = pd.merge(death_df, target_info_df, how = 'left', on = ['eid'])

target_death_f = ['F01']

death_date_df = death_df[['eid', '40000-0.0']].copy()
death_code_df = death_df[['eid'] + code_f_lst].copy()
death_code_df.fillna('NA0123', inplace = True)

for col_i in range(1, death_code_df.shape[1]):
    death_code_df.iloc[:, col_i] = death_code_df.iloc[:, col_i].apply(lambda x : x[:3])

target_death_df = death_df[['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs']]

for item_code in target_death_f:
    tmpdf = get_target_death_df(death_code_df, death_date_df, item_code)
    target_death_df = pd.merge(target_death_df, tmpdf, how = 'left', on = ['eid'])

target_death_df['target_date'], target_death_df['target_death'] = get_fo_info(target_death_f, target_death_df, nb_subjects)

target_death_df['target_y'] = 1
target_death_df['target_y'][target_death_df['target_date'].isnull() == True] = 0
target_death_df['BL2Target_yrs_raw'] = get_days_intervel('BL_date', 'target_date', target_death_df)
tmpdf = target_death_df[['BL2Target_yrs_raw', 'BL2End_yrs', 'BL2Death_yrs']]
target_death_df['BL2Target_yrs'] = pd.DataFrame([tmpdf.iloc[i,:].min() for i in range(nb_subjects)])

out_cols = ['eid', 'target_death', 'target_date', 'BL2Target_yrs_raw', 'target_y', 'BL2Target_yrs']
target_df_out = target_death_df[out_cols]
target_df_out.to_csv(outpath + 'VaD_Death.csv', index = False)

time_end2 = time.time()
print('Finish and it total cost ' + str(np.round(time_end2 - time_start, 1)) + ' seconds')




