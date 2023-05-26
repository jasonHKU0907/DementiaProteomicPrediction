

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
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/OtherNervDis/'

code_f_lst = ['40001-0.0','40001-1.0'] + ['40002-0.' + str(i) for i in range(15)]

death_df = pd.read_csv(dpath + 'ukb672277_death_data.csv', usecols = ['eid', '40000-0.0'] + code_f_lst)
target_info_df = pd.read_csv(dpath + 'Target_info_data.csv', usecols=['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs'])
nb_subjects = len(death_df)
death_df = pd.merge(death_df, target_info_df, how = 'left', on = ['eid'])

tmp = "G00.0;G00.1;G00.2;G00.3;G00.8;G00.9;G01;G02.0;G02.1;G03.0;G03.1;G03.2;G03.8;G03.9;" \
      "G04.0;G04.2;G04.8;G04.9;G05.0;G05.1;G05.2;G05.8;G06.0;G06.1;G06.2;G07;G08;G09;G35;G36.0;G36.8;G36.9;" \
      "G37.0;G37.1;G37.2;G37.3;G37.4;G37.8;G37.9;G40.0;G40.1;G40.2;G40.3;G40.4;G40.5;G40.6;G40.7;G40.8;G40.9;" \
      "G41.0;G41.1;G41.2;G41.8;G41.9;G43.0;G43.1;G43.2;G43.3;G43.8;G43.9;G44.0;G44.1;G44.2;G44.3;G44.4;G44.8;" \
      "G45.0;G45.1;G45.2;G45.3;G45.4;G45.8;G45.9;G46.0;G46.1;G46.2;G46.3;G46.4;G46.5;G46.6;G46.7;G46.8;" \
      "G47.0;G47.1;G47.2;G47.3;G47.4;G47.8;G47.9;G50.0;G50.1;G50.8;G50.9;G51.0;G51.1;G51.2;G51.3;G51.4;G51.8;G51.9;" \
      "G52.0;G52.1;G52.2;G52.3;G52.7;G52.8;G52.9;G53.0;G53.1;G53.2;G53.3;G53.8;G54.0;G54.1;G54.2;G54.3;G54.4;G54.5;" \
      "G54.6;G54.7;G54.8;G54.9;G55.0;G55.1;G55.2;G55.3;G55.8;G56.0;G56.1;G56.2;G56.3;G56.4;G56.8;G56.9;G57.0;G57.1;" \
      "G57.2;G57.3;G57.4;G57.5;G57.6;G57.8;G57.9;G58.0;G58.7;G58.8;G58.9;G59.0;G59.8;G60.0;G60.2;G60.3;G60.8;G60.9;" \
      "G61.0;G61.1;G61.8;G61.9;G62.0;G62.1;G62.2;G62.8;G62.9;G63.0;G63.1;G63.2;G63.3;G63.4;G63.5;G63.6;G63.8;G64;" \
      "G70.0;G70.2;G70.8;G70.9;G71.0;G71.1;G71.2;G71.3;G71.8;G71.9;G72.0;G72.1;G72.2;G72.3;G72.4;G72.8;G72.9;G73.0;" \
      "G73.1;G73.5;G73.6;G73.7;G80.0;G80.1;G80.2;G80.3;G80.4;G80.8;G80.9;G81.0;G81.1;G81.9;G82.0;G82.1;G82.2;G82.3;" \
      "G82.4;G82.5;G83.0;G83.1;G83.2;G83.3;G83.4;G83.5;G83.8;G83.9;G90.0;G90.1;G90.2;G90.3;G90.4;G90.8;G90.9;G91.0;" \
      "G91.1;G91.2;G91.3;G91.8;G91.9;G92;G93.0;G93.1;G93.2;G93.3;G93.4;G93.5;G93.6;G93.7;G93.8;G93.9;G94.0;G94.1;" \
      "G94.2;G94.8;G95.0;G95.1;G95.2;G95.8;G95.9;G96.0;G96.1;G96.8;G96.9;G97.0;G97.1;G97.2;G97.8;G97.9;G98;G99.0;" \
      "G99.1;G99.2;G99.8"

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
target_df_out.to_csv(outpath + 'ONervD_Death.csv', index = False)

time_end2 = time.time()
print('Finish and it total cost ' + str(np.round(time_end2 - time_start, 1)) + ' seconds')





