

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

dpath = '/Volumes/JasonWork/Dataset/UKB_Tabular_merged_10/'
outpath = '/Volumes/JasonWork/Dataset/UKB_20230417/'
feature_df = pd.read_csv(dpath + 'UKB_FieldID_Subset.csv')
eid_df = pd.read_csv(dpath + 'UKB_eid.csv')

base_FieldID = ['53-0.0', '21022-0.0', '31-0.0', '54-0.0']
base_df = read_data(base_FieldID, feature_df, eid_df)
death_df = pd.read_csv(outpath + 'ukb672277.csv', usecols = ['eid', '40000-0.0'])
mydf = pd.merge(base_df, death_df, how = 'right', on = ['eid'])

mydf.rename(columns = {'21022-0.0':'Age', '31-0.0':'Gender', '53-0.0': 'BL_date', '54-0.0': 'Site_code', '40000-0.0': 'Death_date'}, inplace = True)

mydf['End_Obs_Date'] = pd.to_datetime('01/03/2023', dayfirst=True)
mydf['BL2End_yrs'] = get_days_intervel('BL_date', 'End_Obs_Date', mydf)
mydf['BL2Death_yrs'] = get_days_intervel('BL_date', 'Death_date', mydf)

'''
"London" = 0
"Wales" = 1
"North-West" = 2
"North-East " = 3
"Yorkshire and Humber" = 4
"West Midlands" = 5
"East Midlands" = 6
"South-East" = 7
"South-West" = 8
"Scotland" = 9
'''

mydf['Region_code'] = mydf['Site_code'].copy()
mydf['Region_code'].replace([10003, 11001, 11002, 11003, 11004, 11005, 11006, 11007, 11008, 11009, 11010,
                             11011, 11012, 11013, 11014, 11016, 11017, 11018, 11020, 11021, 11022, 11023],
                            [2,     2,     7,     1,     9,     9,     5,     7,     2,     3,     4,
                             8,     0,     6,     4,     2,     3,     0,     0,     5,     1,     1], inplace = True)

mydf = mydf[['eid', 'Age', 'Gender', 'Site_code', 'Region_code', 'BL_date',
             'Death_date', 'End_Obs_Date', 'BL2End_yrs', 'BL2Death_yrs']]

mydf.to_csv(outpath + 'Target_info_data.csv', index = False)


