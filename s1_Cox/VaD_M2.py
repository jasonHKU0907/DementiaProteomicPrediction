

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import bonferroni_correction

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/Cox/'

pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv')
pro_f_lst = pro_df.columns[1:].tolist()
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])

cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv')
cov_df.columns
m1_f_lst = ['age', 'sex', 'educ', 'apoe4']
m2_f_lst = m1_f_lst + ['smk', 'sbp', 'bmi', 'med_bp', 'diab_hst', 'cvd_hst', 'tot_c', 'hdl_c']

target_df = pd.read_csv(dpath + 'TargetOutcomes/VaD/VaD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
target_df.BL2Target_yrs.describe()

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])

myout_df, pro_out_lst = pd.DataFrame(), []
i=0

for pro_f in pro_f_lst:
    i+=1
    print(i)
    tmpdf_f = ['target_y', 'BL2Target_yrs', pro_f] + m2_f_lst
    tmpdf = mydf[tmpdf_f]
    tmpdf.rename(columns={pro_f: "target_pro"}, inplace=True)
    rm_idx = tmpdf.index[tmpdf.target_pro.isnull() == True]
    tmpdf = tmpdf.drop(rm_idx, axis=0)
    tmpdf.reset_index(inplace=True)
    cph = CoxPHFitter()
    my_formula = "age + sex + educ + apoe4 + smk + sbp + bmi + med_bp + diab_hst + cvd_hst + tot_c + hdl_c + target_pro"
    try:
        cph.fit(tmpdf, duration_col = 'BL2Target_yrs', event_col = 'target_y', formula=my_formula)
        hr = cph.hazard_ratios_.target_pro
        lbd = np.exp(cph.confidence_intervals_).iloc[11, 0]
        ubd = np.exp(cph.confidence_intervals_).iloc[11, 1]
        pval = cph.summary.p.target_pro
        myout = pd.DataFrame([hr, lbd, ubd, pval])
        myout_df = pd.concat((myout_df, myout.T), axis=0)
        pro_out_lst.append(pro_f)
    except:
        print(pro_f)

myout_df.columns = ['HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val']
myout_df['Pro_code'] = pro_out_lst
_, p_f_fdr = fdrcorrection(myout_df.HR_p_val.fillna(1))
_, p_f_bfi = bonferroni_correction(myout_df.HR_p_val.fillna(1), alpha=0.05)

myout_df['p_val_fdr'] = p_f_fdr
myout_df['p_val_bfi'] = p_f_bfi

myout_df = pd.merge(myout_df, pro_dict, how = 'left', on = ['Pro_code'])

myout_df = myout_df[['Pro_code', 'Pro_definition', 'HR', 'HR_Lower_CI', 'HR_Upper_CI', 'HR_p_val', 'p_val_fdr', 'p_val_bfi']]
myout_df.to_csv(outpath + 'VaD_M2.csv', index = False)


