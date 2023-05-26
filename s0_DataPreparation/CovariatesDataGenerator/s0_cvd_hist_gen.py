

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

def get_fo_info(target_date_f, target_source_f, target_df, nb_subjects):
    target_date_df = target_df[target_date_f].copy()
    target_source_df = target_df[target_source_f].copy()
    fo_date_lst, fo_dis_lst, fo_source_lst = [], [], []
    for i in range(nb_subjects):
        ind_target_df = target_date_df.iloc[i, :]
        ind_source_df = target_source_df.iloc[i, :]
        fo_dis_date = pd.to_datetime(ind_target_df, dayfirst=True).min()
        try:
            fo_dis_FieldID = ind_target_df.index[pd.to_datetime(ind_target_df, dayfirst=True) == fo_dis_date][0]
            fo_dis_FieldID = str(int(fo_dis_FieldID.split('-')[0]))
            fo_dis_source = ind_source_df.loc[str(int(fo_dis_FieldID) + 1) + '-0.0']
        except:
            fo_dis_FieldID, fo_dis_source = np.nan, np.nan
        fo_date_lst.append(fo_dis_date)
        fo_dis_lst.append(fo_dis_FieldID)
        fo_source_lst.append(fo_dis_source)
    return pd.DataFrame(fo_date_lst), pd.DataFrame(fo_dis_lst), pd.DataFrame(fo_source_lst)


dpath = '/Volumes/JasonWork/Dataset/UKB_20230417/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Covariates/Raw/'

f_dictionary = pd.read_csv(dpath + 'ukb672277_FO_info.csv', usecols = ['FieldID', 'Field'])
f_dictionary['FieldID'] = f_dictionary['FieldID'].astype(str)
f_dictionary.rename(columns = {'FieldID': 'target_FO_FieldID'}, inplace = True)
target_info_df = pd.read_csv(dpath + 'Target_info_data.csv', usecols=['eid', 'BL_date', 'BL2End_yrs', 'BL2Death_yrs'])

af_date_f = [str(i) + '-0.0' for i in range(131350, 131351) if i%2==0]
chd_date_f = [str(i) + '-0.0' for i in range(131296, 131307) if i%2==0]
hf_date_f = [str(i) + '-0.0' for i in range(131354, 131355) if i%2==0]
str_date_f = [str(i) + '-0.0' for i in range(131360, 131363) if i%2==0] + [str(i) + '-0.0' for i in range(131366, 131369) if i%2==0]
pad_date_f = [str(i) + '-0.0' for i in range(131380, 131395) if i%2==0]
target_date_f = af_date_f + chd_date_f + hf_date_f + str_date_f + pad_date_f

print(len(target_date_f))
target_date_f = [ele for ele in target_date_f if ele.split('-')[0] in f_dictionary.target_FO_FieldID.tolist()]
target_source_f = [str(int(ele.split('-')[0])+1) + '-0.0' for ele in target_date_f]
print(len(target_date_f))

target_f = ['eid'] + target_date_f + target_source_f
target_df = pd.read_csv(dpath + 'ukb672277_FO_data.csv', usecols=target_f)
nb_subjects = len(target_df)
time_end0 = time.time()
print('Finish Reading Data and it cost ' + str(np.round(time_end0 - time_start, 1)) + ' seconds')

target_df[target_date_f] = target_df[target_date_f].replace(['1900-01-01', '1901-01-01', '1902-02-02', '1903-03-03', '2037-07-07'], '1900-01-01')
target_df = pd.merge(target_df, target_info_df, how = 'left', on = ['eid'])
time_end1 = time.time()
print('Finish Preprocessing and it cost ' + str(np.round(time_end1 - time_end0, 1)) + ' seconds')

target_df['target_date'], target_df['target_FO_FieldID'], target_df['Target_source'] = get_fo_info(target_date_f, target_source_f, target_df, nb_subjects)
target_df = pd.merge(target_df, f_dictionary, how = 'left', on=['target_FO_FieldID'])
target_df['BL2Target_yrs'] = get_days_intervel('BL_date', 'target_date', target_df)
target_df['CVD_hist'] = 0
target_df['CVD_hist'].loc[target_df['BL2Target_yrs']<=0] = 1

target_df_out = target_df[['eid', 'CVD_hist']]
target_df_out.to_csv(outpath + 'CVD_FO.csv', index = False)

time_end2 = time.time()
print('Finish and it total cost ' + str(np.round(time_end2 - time_start, 1)) + ' seconds')

