
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Revision/StrictAnalysis/RegionFold/'

pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv')
pro_f_lst = pro_df.columns[1:].tolist()
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv')
m1_f_lst = ['age', 'sex', 'educ', 'apoe4']
m2_f_lst = m1_f_lst + ['sbp', 'med_bp', 'diab_hst', 'smk', 'cvd_hst', 'tot_c', 'hdl_c', 'bmi']

target_df = pd.read_csv(dpath + 'TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
target_df.BL2Target_yrs.describe()

rawdf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
rawdf = pd.merge(rawdf, cov_df, how = 'left', on = ['eid'])

fold_id_lst = list(set(rawdf.Region_code))

for fold_id in fold_id_lst:
    os.mkdir(outpath + 'TestFold' + str(fold_id))
    mydf = rawdf.copy()
    mydf = mydf.loc[mydf.Region_code != fold_id]
    myout_df, pro_out_lst = pd.DataFrame(), []
    for pro_f in tqdm(pro_f_lst):
        tmpdf_f = ['target_y', 'BL2Target_yrs', pro_f] + m1_f_lst
        tmpdf = mydf[tmpdf_f]
        tmpdf.rename(columns={pro_f: "target_pro"}, inplace=True)
        rm_idx = tmpdf.index[tmpdf.target_pro.isnull() == True]
        tmpdf = tmpdf.drop(rm_idx, axis=0)
        tmpdf.reset_index(inplace=True)
        cph = CoxPHFitter()
        my_formula = "age + sex + educ + apoe4 + target_pro"
        try:
            cph.fit(tmpdf, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
            hr = cph.hazard_ratios_.target_pro
            lbd = np.exp(cph.confidence_intervals_).iloc[4, 0]
            ubd = np.exp(cph.confidence_intervals_).iloc[4, 1]
            pval = cph.summary.p.target_pro
            myout = pd.DataFrame([hr, lbd, ubd, pval])
            myout_df = pd.concat((myout_df, myout.T), axis=0)
            pro_out_lst.append(pro_f)
        except:
            print((fold_id, pro_f))
    myout_df.columns = ['HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val']
    myout_df['Pro_code'] = pro_out_lst
    _, p_f_fdr = fdrcorrection(myout_df.HR_p_val.fillna(1))
    _, p_f_bfi = bonferroni_correction(myout_df.HR_p_val.fillna(1), alpha=0.05)
    myout_df['p_val_fdr'] = p_f_fdr
    myout_df['p_val_bfi'] = p_f_bfi
    myout_df = pd.merge(myout_df, pro_dict, how='left', on=['Pro_code'])
    myout_df = myout_df[['Pro_code', 'Pro_definition', 'HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val', 'p_val_fdr', 'p_val_bfi']]
    myout_df.to_csv(outpath + 'TestFold' + str(fold_id) + '/ACD_Cox_M1.csv', index=False)



