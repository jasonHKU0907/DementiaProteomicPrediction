

import glob
import os
import numpy as np
import pandas as pd
import re

def read_data(FieldID_lst, feature_df, eid_df):
    subset_df = feature_df[feature_df['Field_ID'].isin(FieldID_lst)]
    subset_dict = {k: ['eid'] + g['Field_ID'].tolist() for k, g in subset_df.groupby('Subset_ID')}
    subset_lst = list(subset_dict.keys())
    my_df = eid_df
    for subset_id in subset_lst:
        tmp_dir = dpath + 'UKB_subset_' + str(subset_id) + '.csv'
        tmp_f = subset_dict[subset_id]
        tmp_df = pd.read_csv(tmp_dir, usecols=tmp_f)
        my_df = pd.merge(my_df, tmp_df, how='inner', on=['eid'])
    return my_df

def get_days_intervel(start_date_var, end_date_var, df):
    start_date = pd.to_datetime(df[start_date_var], dayfirst=True)
    end_date = pd.to_datetime(df[end_date_var], dayfirst=True)
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    my_yrs = [ele/365 for ele in days]
    return pd.DataFrame(my_yrs)

def get_binary(var_source, df):
    tmp_binary = df[var_source].copy()
    tmp_binary.loc[tmp_binary >= -1] = 1
    tmp_binary.replace(np.nan, 0, inplace=True)
    return tmp_binary

dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Projects/UKB_DM_Trajectories/Data/DM_Analysis/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')
# age, gender, centers, TDI, Ethnicity,

cognitive_info_lst = ['400-0.1', '10138-0.1', '399-0.1', '10137-0.1',
                      '400-0.2', '10138-0.2', '399-0.2', '10137-0.2',
                      '20018-0.0',
                      '20016-0.0',
                      '404-0.0', '10147-0.0', '404-0.1', '10147-0.1',
                      '404-0.2', '10147-0.2', '404-0.3', '10147-0.3',
                      '404-0.4', '10147-0.4', '20023-0.0']

mydf = read_data(cognitive_info_lst, feature_df, eid_df)

na_idx_4001 = mydf.index[mydf['400-0.1'].isnull()]
mydf.loc[na_idx_4001, '400-0.1'] = mydf.loc[na_idx_4001, '10138-0.1']
na_idx_4002 = mydf.index[mydf['400-0.2'].isnull()]
mydf.loc[na_idx_4002, '400-0.2'] = mydf.loc[na_idx_4002, '10138-0.2']

na_idx_3991 = mydf.index[mydf['399-0.1'].isnull()]
mydf.loc[na_idx_3991, '399-0.1'] = mydf.loc[na_idx_3991, '10137-0.1']
na_idx_3992 = mydf.index[mydf['399-0.2'].isnull()]
mydf.loc[na_idx_3992, '399-0.2'] = mydf.loc[na_idx_3992, '10138-0.2']


na_idx_4040 = mydf.index[mydf['404-0.0'].isnull()]
mydf.loc[na_idx_4040, '404-0.0'] = mydf.loc[na_idx_4040, '10147-0.0']
na_idx_4041 = mydf.index[mydf['404-0.1'].isnull()]
mydf.loc[na_idx_4041, '404-0.1'] = mydf.loc[na_idx_4041, '10147-0.1']
na_idx_4042 = mydf.index[mydf['404-0.2'].isnull()]
mydf.loc[na_idx_4042, '404-0.2'] = mydf.loc[na_idx_4042, '10147-0.2']
na_idx_4043 = mydf.index[mydf['404-0.3'].isnull()]
mydf.loc[na_idx_4043, '404-0.3'] = mydf.loc[na_idx_4043, '10147-0.3']
na_idx_4044 = mydf.index[mydf['404-0.4'].isnull()]
mydf.loc[na_idx_4044, '404-0.4'] = mydf.loc[na_idx_4044, '10147-0.4']

mydf['pm_time'] = (mydf['400-0.1'] + mydf['400-0.2'])/2
mydf['pm_incorrect'] = (mydf['399-0.1'] + mydf['399-0.2'])/2
#mydf['pros_memory'] = mydf['20018-0.0']
#mydf['fi_score'] = mydf['20016-0.0']
mydf['rt_meantime'] = mydf['20023-0.0']
#mydf['rt_dur1snap'] = (mydf['404-0.0'] + mydf['404-0.1'] + mydf['404-0.2'] + mydf['404-0.3'] + mydf['404-0.4'])/2

mydf1 = mydf[['eid', 'pm_time', 'pm_incorrect', 'rt_meantime']]

mydf1['pm_time'].fillna(mydf1['pm_time'].median(), inplace = True)
mydf1['pm_incorrect'].fillna(mydf1['pm_incorrect'].median(), inplace = True)
mydf1['rt_meantime'].fillna(mydf1['rt_meantime'].median(), inplace = True)
mydf1.describe()
mydf1.to_csv('/Volumes/JasonWork/Projects/AD_Proteomics/Data/Covariates/Raw/Cognitive.csv', index = False)