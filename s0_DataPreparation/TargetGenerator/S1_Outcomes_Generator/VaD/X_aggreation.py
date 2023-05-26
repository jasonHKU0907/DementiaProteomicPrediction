
import glob
import os
import numpy as np
import pandas as pd
import re
import time

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/VaD/'
alg_df = pd.read_csv(dpath + 'VaD_ALG.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs', 'Target_source'])
alg_df.rename(columns = {'target_y':'alg_y', 'BL2Target_yrs':'alg_yrs', 'Target_source':'alg_source'}, inplace = True)

fo_df = pd.read_csv(dpath + 'VaD_FO.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs', 'Target_source'])
fo_df.rename(columns = {'target_y':'fo_y', 'BL2Target_yrs':'fo_yrs', 'Target_source':'fo_source'}, inplace = True)

death_df = pd.read_csv(dpath + 'VaD_Death.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
death_df.rename(columns = {'target_y':'death_y', 'BL2Target_yrs':'death_yrs'}, inplace = True)

inhosp_df = pd.read_csv(dpath + 'VaD_InHosp.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
inhosp_df.rename(columns = {'target_y':'inhosp_y', 'BL2Target_yrs':'inhosp_yrs'}, inplace = True)

mydf = pd.merge(alg_df, fo_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, death_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, inhosp_df, how = 'left', on = ['eid'])
nb_subjects = len(mydf)

mydf['target_y'] = mydf['alg_y'] + mydf['fo_y'] + mydf['death_y'] + mydf['inhosp_y']
mydf['target_y'][mydf['target_y']>=1] = 1
mydf['target_y'].value_counts()

tmpdf = mydf[['alg_yrs', 'fo_yrs', 'death_yrs', 'inhosp_yrs']]
mydf['BL2Target_yrs'] = pd.DataFrame([tmpdf.iloc[i,:].min() for i in range(nb_subjects)])

alg_sr_idx = mydf.index[mydf.alg_source == 0]
fo_sr_idx = mydf.index[mydf.fo_source>=50]
rm_idx1 = list(set(alg_sr_idx.tolist() + fo_sr_idx.tolist()))
rm_idx2 = list(set(mydf.index[mydf.BL2Target_yrs <= 0]))
rm_idx = list(set(rm_idx1+rm_idx2))
mydf.drop(rm_idx, axis = 0, inplace = True)
mydf.reset_index(inplace = True)

target_df_acd = pd.read_csv('/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid'])
mydf = pd.merge(mydf, target_df_acd, how = 'inner', on = ['eid'])

mydf.to_csv(dpath + 'VaD_outcomes.csv', index = False)

