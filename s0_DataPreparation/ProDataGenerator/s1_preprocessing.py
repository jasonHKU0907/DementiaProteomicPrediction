

import glob
import os
import numpy as np
import pandas as pd
import re
import math

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Proteomics/'
mydf = pd.read_csv(dpath + 'Raw/ProteomicsData.csv')
pro_f_lst = mydf.columns.tolist()[:-1]

def get_normalization(mydf):
    tmp_df = mydf.copy()
    for col in tmp_df.columns:
        tmp_df[col] = np.log(tmp_df[col])
        ubd = tmp_df[col].mean() + tmp_df[col].std()*4
        lbd = tmp_df[col].mean() - tmp_df[col].std()*4
        tmp_df[col].iloc[tmp_df[col]>ubd] = np.nan
        tmp_df[col].iloc[tmp_df[col]<lbd] = np.nan
        tmp_df[col] = np.round((tmp_df[col] - np.mean(tmp_df[col])) / tmp_df[col].std(), 5)
    return tmp_df

#mydf_out = mydf.copy()
#mydf_out[pro_f_lst] = get_normalization(mydf_out[pro_f_lst])

mydf_out = mydf['eid'].copy()

for col in pro_f_lst:
    tmp_col = mydf[col]/np.abs(mydf[col])*np.log(np.abs(mydf[col]) + 1)
    ubd = tmp_col.mean() + tmp_col.std() * 4
    lbd = tmp_col.mean() - tmp_col.std() * 4
    tmp_col.iloc[tmp_col > ubd] = np.nan
    tmp_col.iloc[tmp_col < lbd] = np.nan
    tmp_col = np.round((tmp_col - np.mean(tmp_col)) / tmp_col.std(), 5)
    mydf_out = pd.concat([mydf_out, tmp_col], axis = 1)

mydf_out.to_csv(dpath + 'ProteomicsData.csv', index = False)
