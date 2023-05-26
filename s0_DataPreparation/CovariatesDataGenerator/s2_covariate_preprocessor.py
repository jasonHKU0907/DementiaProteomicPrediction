

import glob
import os
import numpy as np
import pandas as pd
import re
from sklearn.impute import KNNImputer

def get_normalization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        ubd = tmp_df[col].mean() + tmp_df[col].std()*4
        lbd = tmp_df[col].mean() - tmp_df[col].std()*4
        tmp_df[col].iloc[tmp_df[col]>ubd] = ubd
        tmp_df[col].iloc[tmp_df[col]<lbd] = lbd
        tmp_df[col] = (tmp_df[col] - np.mean(tmp_df[col])) / tmp_df[col].std()
    return tmp_df

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'

pro_eid_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'])
region_code_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])
info_df = pd.merge(pro_eid_df, region_code_df, how = 'left', on = 'eid')
nb_ind = len(info_df)
region_code_lst = [ele for ele in range(10)]

cov_df = pd.read_csv(dpath + 'Covariates/Raw/Covariates_full_population.csv')

mydf = pd.merge(info_df, cov_df, how = 'left', on = ['eid'])

mydf[['eid', 'Region_code']] = mydf[['eid', 'Region_code']].astype(int)
mydf.sort_values(by = ['eid'], ascending=True, inplace = True)
mydf.to_csv(dpath + 'Covariates/Raw/Covariates_pro_population.csv.csv', index = False)

categorical_f_lst = ['sex', 'ethn', 'smk', 'reg_pa', 'med_bp', 'diab_hst', 'cvd_hst', 'apoe4']
continuous_f_lst = ['age', 'educ', 'tdi', 'sbp', 'bmi', 'tot_c', 'hdl_c', 'pm_time', 'pm_incor', 'rt_time']

for region_id in region_code_lst:
    for ele in categorical_f_lst:
        train_idx = mydf.index[mydf.Region_code != region_id]
        imp_val = int(mydf.iloc[train_idx][ele].mode())
        test_imp_idx = mydf.index[(mydf.Region_code == region_id) & (mydf[ele].isnull() == True)]
        mydf[ele].iloc[test_imp_idx] = imp_val
    print(region_id)

mydf_out = pd.DataFrame()

import time
time_start = time.time()

for region_id in region_code_lst:
    tmpdf_train = mydf.loc[mydf.Region_code != region_id].copy()
    tmp_cols = tmpdf_train.columns.tolist()[2:]
    tmpdf_test = mydf.loc[mydf.Region_code == region_id].copy()
    tmpdf_test.reset_index(inplace = True)
    knn_imputer = KNNImputer(n_neighbors=50, weights="uniform")
    knn_imputer.fit(tmpdf_train[tmp_cols])
    tmpdf_out = pd.DataFrame(knn_imputer.transform(tmpdf_test[tmp_cols]))
    tmpdf_test[tmp_cols] = tmpdf_out
    mydf_out = pd.concat([mydf_out, tmpdf_test], axis = 0)
    print('Finish and it total cost ' + str(np.round(time.time() - time_start, 1)) + ' seconds')


mydf_out.drop('index', axis = 1, inplace = True)
mydf_out[['eid', 'Region_code']] = mydf_out[['eid', 'Region_code']].astype(int)
mydf_out.sort_values(by = ['eid'], ascending=True, inplace = True)
mydf_out['educ'] = np.round(mydf_out['educ'])
mydf_out['sbp'] = np.round(mydf_out['sbp'])
mydf_out['pm_incor'] = np.round(mydf_out['pm_incor'])


mydf_out.to_csv(dpath + 'Covariates/CovData.csv', index = False)

# Normalize continuous variables
mydf_out[continuous_f_lst] = get_normalization(mydf_out[continuous_f_lst])

mydf_out.to_csv(dpath + 'Covariates/CovData_normalized.csv', index = False)

