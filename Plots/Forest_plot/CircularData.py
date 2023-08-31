

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'

pro_f_m1_df = pd.read_csv(outpath + 'Cox/ACD_M1.csv')
pro_f_m2_df = pd.read_csv(outpath + 'Cox/ACD_M2.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]
selected_df_acd = pro_f_m2_df[pro_f_m2_df.Pro_code.isin(pro_f_lst)]
selected_df_acd['Target'] = 'ACD'

pro_f_m1_df = pd.read_csv(outpath + 'Cox/AD_M1.csv')
pro_f_m2_df = pd.read_csv(outpath + 'Cox/AD_M2.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]
selected_df_ad = pro_f_m2_df[pro_f_m2_df.Pro_code.isin(pro_f_lst)]
selected_df_ad['Target'] = 'AD'

pro_f_m1_df = pd.read_csv(outpath + 'Cox/VaD_M1.csv')
pro_f_m2_df = pd.read_csv(outpath + 'Cox/VaD_M2.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]
selected_df_vd = pro_f_m2_df[pro_f_m2_df.Pro_code.isin(pro_f_lst)]
selected_df_vd['Target'] = 'VaD'

selected_df = pd.concat([selected_df_acd, selected_df_ad], axis = 0)
selected_df = pd.concat([selected_df, selected_df_vd], axis = 0)

selected_df.to_csv(outpath + 'Plots/Figure1/Data/CircularData.csv', index = False)