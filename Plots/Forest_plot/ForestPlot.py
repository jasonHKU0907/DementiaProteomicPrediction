

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'

pro_f_lst = ['GFAP', 'NEFL', 'GDF15', 'LTBP2', 'CST5', 'NPTXR', 'BCAN', 'ACTA2', 'EPHA2', 'GFRA1', 'SPON2']

pro_f_acd_df = pd.read_csv(outpath + 'Cox/ACD_M2.csv')
selected_df_acd = pro_f_acd_df[pro_f_acd_df.Pro_code.isin(pro_f_lst)]
selected_df_acd['Target'] = 'ACD'

pro_f_ad_df = pd.read_csv(outpath + 'Cox/AD_M2.csv')
selected_df_ad = pro_f_ad_df[pro_f_ad_df.Pro_code.isin(pro_f_lst)]
selected_df_ad['Target'] = 'AD'

pro_f_vd_df = pd.read_csv(outpath + 'Cox/VaD_M2.csv')
selected_df_vd = pro_f_vd_df[pro_f_vd_df.Pro_code.isin(pro_f_lst)]
selected_df_vd['Target'] = 'VaD'

selected_df = pd.concat([selected_df_acd, selected_df_ad], axis = 0)
selected_df = pd.concat([selected_df, selected_df_vd], axis = 0)

selected_df['HR_rd'] = np.round(selected_df.HR, 2)
selected_df['HR_lbd_rd'] = np.round(selected_df.HR_Lower_CI, 2)
selected_df['HR_ubd_rd'] = np.round(selected_df.HR_Upper_CI, 2)
hr_out_lst = ["%.2f" % selected_df.HR_rd.iloc[i] + ' (' + "%.2f" % selected_df.HR_lbd_rd.iloc[i] + '-' + "%.2f" % selected_df.HR_ubd_rd.iloc[i] + ')' for i in range(len(selected_df))]
selected_df['HR_output'] = hr_out_lst

selected_df['p_val_bfi_corr'] = selected_df['p_val_bfi']*3

pval_lst = []
for i in range(len(selected_df)):
    try:
        str0, str1 = str(selected_df.p_val_bfi_corr.iloc[i]).split('e')
        str00, str01 = str0.split('.')
        pval_lst.append(str00 + '.' + str01[:2] + 'X' + '10' + str1)
    except:
        pval_lst.append('Manual')

selected_df['pval_output'] = pval_lst

selected_df.to_csv(outpath + 'Plots/Figure1/Data/ForestData.csv', index = False)

