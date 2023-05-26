

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
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/OtherMBDis/'

code_f_lst = ['40001-0.0','40001-1.0'] + ['40002-0.' + str(i) for i in range(15)]

death_df = pd.read_csv(dpath + 'ukb672277_death_data.csv', usecols = ['eid', '40000-0.0'] + code_f_lst)
target_info_df = pd.read_csv(dpath + 'Target_info_data.csv', usecols=['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs'])
nb_subjects = len(death_df)
death_df = pd.merge(death_df, target_info_df, how = 'left', on = ['eid'])

tmp = "F04;F05.0;F05.1;F05.8;F05.9;F06.0;F06.1;F06.2;F06.3;F06.4;F06.5;F06.6;F06.7;F06.8;F06.9;" \
      "F07.0;F07.1;F07.2;F07.8;F07.9;F09;F10.0;F10.1;F10.2;F10.3;F10.4;F10.5;F10.6;F10.7;F10.8;F10.9;" \
      "F11.0;F11.1;F11.2;F11.3;F11.4;F11.5;F11.7;F11.9;F12.0;F12.1;F12.2;F12.3;F12.5;F12.8;F12.9;" \
      "F13.0;F13.1;F13.2;F13.3;F13.4;F13.9;F14.0;F14.1;F14.2;F14.5;F14.9;F15.0;F15.1;F15.2;F15.3;F15.5;F15.8;F15.9;" \
      "F16.1;F16.2;F16.3;F16.5;F16.7;F16.8;F16.9;F17.0;F17.1;F17.2;F17.3;F17.4;F17.7;F17.9;" \
      "F18.1;F18.2;F18.3;F18.5;F18.9;F19.0;F19.1;F19.2;F19.3;F19.4;F19.5;F19.8;F19.9;" \
      "F20.0;F20.1;F20.2;F20.3;F20.4;F20.5;F20.6;F20.8;F20.9;F21;F22.0;F22.8;F22.9;" \
      "F23.0;F23.1;F23.2;F23.3;F23.8;F23.9;F24;F25.0;F25.1;F256.2;F25.8;F25.9;F28;F29;F30.0;F30.1;F30.2;F30.8;F30.9;" \
      "F31.0;F31.1;F31.2;F31.3;F31.4;F31.5;F31.6;F31.7;F31.8;F31.9;F32.0;F32.1;F32.2;F32.3;F32.8;F32.9;" \
      "F33.0;F33.1;F33.2;F33.3;F33.4;F33.8;F33.9;F34.0;F34.1;F34.8;F34.9;F38.0;F38.1;F38.8;F39;" \
      "F40.0;F40.1;F40.2;F40.8;F40.9;F41.0;F41.1;F41.2;F41.3;F41.8;F41.9;F42.0;F42.1;F42.2;F42.8;F42.9;" \
      "F43.0;F43.1;F43.2;F43.8;F43.9;F44.0;F44.1;F44.2;F44.3;F44.4;F44.5;F44.6;F44.7;F44.8;F44.9;" \
      "F45.0;F45.1;F45.2;F45.3;F45.4;F45.8;F45.9;F48.0;F48.1;F48.8;F48.9;F50.0;F50.1;F50.2;F50.5;F50.8;F50.9;" \
      "F51.0;F51.1;F51.2;F51.3;F51.4;F51.5;F51.8;F51.9;F52.0;F52.1;F52.2;F52.3;F52.4;F52.5;F52.6;F52.7;F52.8;F52.9;" \
      "F53.0;F53.1;F53.8;F53.9;F54;F55;F59;F60.0;F60.1;F60.2;F60.3;F60.4;F60.5;F60.6;F60.7;F60.8;F60.9;F61;" \
      "F62.0;F62.1;F62.8;F62.9;F63.0;F63.1;F63.2;F63.3;F63.8;F63.9;F64.0;F64.1;F64.8;F64.9;F65.0;F65.5;F65.8;F65.9;" \
      "F66.0;F66.1;F66.2;F66.9;F68.0;F68.1;F68.8;F69;F70.0;F70.1;F70.8;F70.9;F71.0;F71.1;F71.9;F72.9;F78.0;F78.9;" \
      "F79.0;F79.8;F79.9;F80.0;F80.1;F80.2;F80.3;F80.8;F80.9;F81.0;F81.2;F81.8;F81.9;F82;F83;" \
      "F84.0;F84.1;F84.3;F84.4;F84.5;F84.9;F89;F90.0;F90.9;F91.1;F91.8;F91.9;F92.0;F92.9;F93.0;F94.0;F94.1;" \
      "F95.0;F95.1;F95.2;F95.8;F95.9;F98.1;F98.5;F98.6;F98.8;F99"

tmp1 = tmp.split(';')
tmp1 = [ele[:3] for ele in tmp1]
target_death_f = list(set(tmp1))


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
target_df_out.to_csv(outpath + 'OMBD_Death.csv', index = False)

time_end2 = time.time()
print('Finish and it total cost ' + str(np.round(time_end2 - time_start, 1)) + ' seconds')





