

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Projects/UKB_Proteomics/Data/Covariates/RAW/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Covariates/Raw/'

demo_df = pd.read_csv(dpath + 'DemographicInfo.csv')
ls_df = pd.read_csv(dpath + 'LifeStyle.csv', usecols = ['eid', 'SMK_Status'])
pm_df = pd.read_csv(dpath + 'PhysicalMeasurements.csv', usecols = ['eid', 'SBP', 'BMI'])
med_df = pd.read_csv(dpath + 'MedicationHistory.csv', usecols = ['eid', 'med_bp'])
pa_df = pd.read_csv(dpath + 'Lifestyle_PA.csv', usecols = ['eid', 'RegularPA'])
dis_df = pd.read_csv(dpath + 'DiseaseHistory.csv', usecols = ['eid', 'DIAB_hist'])
bio_df = pd.read_csv(dpath + 'Biofluids.csv', usecols = ['eid', '30690-0.0', '30760-0.0'])
bio_df.rename(columns = {'30690-0.0':'tot_c', '30760-0.0':'hdl_c'}, inplace = True)

cvd_df = pd.read_csv(outpath + 'CVD_FO.csv')
cog_df = pd.read_csv(outpath + 'Cognitive.csv')

apoe_df = pd.read_csv(outpath + 'apoe.csv', usecols= ['eid', 'apoe'])
apoe_df['apoe1'] = pd.DataFrame([1 if 'X' in ele else 0 for ele in apoe_df.apoe])
apoe_df['apoe2'] = pd.DataFrame([1 if 'X_X' in ele else 0 for ele in apoe_df.apoe])
apoe_df['apoe4'] = apoe_df['apoe1'] + apoe_df['apoe2']
apoe_df = apoe_df[['eid', 'apoe4']]


my_cov_df = pd.merge(demo_df, ls_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, pm_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, med_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, pa_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, dis_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, bio_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, cvd_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, cog_df, how = 'left', on = ['eid'])
my_cov_df = pd.merge(my_cov_df, apoe_df, how = 'left', on = ['eid'])

my_cov_df.rename(columns = {'Age': 'age', 'Gender': 'sex', 'Ethnicity': 'ethn', 'Education':'educ', 'TDI': 'tdi',
                            'SMK_Status': 'smk', 'SBP': 'sbp', 'BMI':'bmi', 'RegularPA':'reg_pa',
                            'DIAB_hist': 'diab_hst', 'CVD_hist': 'cvd_hst',
                            'pm_incorrect': 'pm_incor', 'rt_meantime': 'rt_time'}, inplace = True)

my_cov_df_out = my_cov_df[['eid', 'age', 'sex', 'ethn', 'educ', 'tdi', 'smk', 'reg_pa', 'sbp', 'bmi',
                           'med_bp', 'diab_hst', 'cvd_hst', 'pm_time', 'pm_incor', 'rt_time',
                           'tot_c', 'hdl_c', 'apoe4']]

my_cov_df_out.to_csv(outpath + 'Covariates_full_population.csv', index = False)



