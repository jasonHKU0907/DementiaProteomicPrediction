

import glob
import os
import numpy as np
import pandas as pd
import re
import time

outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Covariates/Raw/'

target_df_out = target_df[['eid', 'CVD_hist']]
mydf0 = pd.read_csv(outpath + 'CVD_History/CVD_FO_AF.csv')
mydf0.rename(columns = {'CVD_hist': 'af_hist'}, inplace = True)

mydf1 = pd.read_csv(outpath + 'CVD_History/CVD_FO_CHD.csv')
mydf1.rename(columns = {'CVD_hist': 'chd_hist'}, inplace = True)

mydf2 = pd.read_csv(outpath + 'CVD_History/CVD_FO_HF.csv')
mydf2.rename(columns = {'CVD_hist': 'hf_hist'}, inplace = True)

mydf3 = pd.read_csv(outpath + 'CVD_History/CVD_FO_PAD.csv')
mydf3.rename(columns = {'CVD_hist': 'pad_hist'}, inplace = True)

mydf4 = pd.read_csv(outpath + 'CVD_History/CVD_FO_STR.csv')
mydf4.rename(columns = {'CVD_hist': 'str_hist'}, inplace = True)

mydf = pd.merge(mydf0, mydf1, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, mydf2, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, mydf3, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, mydf4, how = 'inner', on = ['eid'])
mydf.to_csv(outpath + 'CVD_History/CVD_Comb.csv', index=False)

