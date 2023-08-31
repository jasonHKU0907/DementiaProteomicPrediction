

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'

tgt_outcome = 'OMBD'
tgt_outcome1 = 'OtherMBDis'
pro_f_df = pd.read_csv(outpath + 'Plots/Figure4/Data/KM_info_' + tgt_outcome1 + '.csv')
target_df = pd.read_csv(dpath + 'TargetOutcomes/' + tgt_outcome1 + '/' + tgt_outcome + '_outcomes.csv',
                        usecols = ['eid', 'target_y', 'BL2Target_yrs', 'acd_y', 'acd_yrs', 'ad_y', 'vd_y'])
target_df = target_df.loc[(target_df.acd_yrs>0) & (target_df.BL2Target_yrs>0)]
target_df.reset_index(inplace = True, drop = True)
rm_idx = target_df.index[target_df.acd_y == 1]
target_df.drop(rm_idx, axis = 0, inplace = True)
target_df.reset_index(inplace = True, drop = True)

pro_f_df = pro_f_df.loc[pro_f_df.Pro_code == 'GFAP']
pro_f_lst = pro_f_df.Pro_code.tolist()
cut_f_lst = pro_f_df.opt_cutoff.tolist()
riskdir_f_lst = pro_f_df.risk_dir.tolist()
pval_f_lst = pro_f_df.HR_pval_out.tolist()
hrci_f_lst = pro_f_df.HR_out.tolist()

pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols=['eid'] + pro_f_lst)
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
cut_lst = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5]

def get_risk_info(mydf, pro_f, pro_f_cut, cut_lst):
    h_risk_df = mydf.loc[mydf[pro_f] > pro_f_cut]
    h_risk_df.reset_index(inplace = True, drop = True)
    l_risk_df = mydf.loc[mydf[pro_f] <= pro_f_cut]
    l_risk_df.reset_index(inplace = True, drop = True)
    high_risk_lst, low_risk_lst = [], []
    for cut in cut_lst:
        h_tmpdf1 = h_risk_df.loc[h_risk_df.BL2Target_yrs > cut]
        h_tmpdf2 = h_risk_df.loc[h_risk_df.BL2Target_yrs <= cut]
        nb_at_risk_h = len(h_tmpdf1)
        nb_events_h = h_tmpdf2.target_y.sum()
        high_risk_lst.append(str(nb_at_risk_h) + ' (' + str(nb_events_h) + ')')
        l_tmpdf1 = l_risk_df.loc[l_risk_df.BL2Target_yrs > cut]
        l_tmpdf2 = l_risk_df.loc[l_risk_df.BL2Target_yrs <= cut]
        nb_at_risk_l = len(l_tmpdf1)
        nb_events_l = l_tmpdf2.target_y.sum()
        low_risk_lst.append(str(nb_at_risk_l) + ' (' + str(nb_events_l) + ')')
    return (high_risk_lst, low_risk_lst)


myout_df = pd.DataFrame()

for i in range(len(pro_f_lst)):
    pro_f, pro_f_cut = pro_f_lst[i], cut_f_lst[i]
    high_risk_lst, low_risk_lst = get_risk_info(mydf, pro_f, pro_f_cut, cut_lst=cut_lst)
    tmpdf = pd.DataFrame([low_risk_lst, high_risk_lst]).T
    tmpdf.columns = ['lrisk_' + pro_f + '_' + tgt_outcome, 'hrisk_' + pro_f + '_' + tgt_outcome]
    myout_df = pd.concat([myout_df, tmpdf.T], axis = 0)

myout_df.to_csv(outpath + 'Plots/Figure4/Data/'+tgt_outcome+'_info.csv')
