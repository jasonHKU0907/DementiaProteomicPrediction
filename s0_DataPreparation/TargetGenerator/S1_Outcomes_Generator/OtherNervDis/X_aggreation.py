
import glob
import os
import numpy as np
import pandas as pd
import re
import time

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/OtherNervDis/'

fo_df = pd.read_csv(dpath + 'ONervD_FO.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs', 'Target_source'])
fo_df.rename(columns = {'target_y':'fo_y', 'BL2Target_yrs':'fo_yrs', 'Target_source':'fo_source'}, inplace = True)

death_df = pd.read_csv(dpath + 'ONervD_Death.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
death_df.rename(columns = {'target_y':'death_y', 'BL2Target_yrs':'death_yrs'}, inplace = True)

inhosp_df = pd.read_csv(dpath + 'ONervD_InHosp.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
inhosp_df.rename(columns = {'target_y':'inhosp_y', 'BL2Target_yrs':'inhosp_yrs'}, inplace = True)

mydf = pd.merge(fo_df, death_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, inhosp_df, how = 'left', on = ['eid'])
nb_subjects = len(mydf)

mydf['target_y'] = mydf['fo_y'] + mydf['death_y'] + mydf['inhosp_y']
mydf['target_y'][mydf['target_y']>=1] = 1
mydf['target_y'].value_counts()

tmpdf = mydf[['fo_yrs', 'death_yrs', 'inhosp_yrs']]
mydf['BL2Target_yrs'] = pd.DataFrame([tmpdf.iloc[i,:].min() for i in range(nb_subjects)])

fo_sr_idx = mydf.index[mydf.fo_source>=50]
rm_idx1 = list(set(fo_sr_idx.tolist()))
rm_idx2 = list(set(mydf.index[mydf.BL2Target_yrs <= 0]))
rm_idx = list(set(rm_idx1+rm_idx2))
mydf.drop(rm_idx, axis = 0, inplace = True)
mydf.reset_index(inplace = True, drop = True)


acd_df = pd.read_csv('/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
acd_df.rename(columns = {'target_y': 'acd_y', 'BL2Target_yrs': 'acd_yrs'}, inplace = True)
ad_df = pd.read_csv('/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/AD/AD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
ad_df.rename(columns = {'target_y': 'ad_y', 'BL2Target_yrs': 'ad_yrs'}, inplace = True)
vd_df = pd.read_csv('/Volumes/JasonWork/Projects/AD_Proteomics/Data/TargetOutcomes/VaD/VaD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
vd_df.rename(columns = {'target_y': 'vd_y', 'BL2Target_yrs': 'vd_yrs'}, inplace = True)

mydf = pd.merge(mydf, acd_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, ad_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, vd_df, how = 'left', on = ['eid'])
pd.crosstab(mydf.acd_y, mydf.target_y)
pd.crosstab(mydf.ad_y, mydf.target_y)
pd.crosstab(mydf.vd_y, mydf.target_y)

rm_combid_idx = mydf.index[(mydf.target_y == 1) & (mydf.acd_y == 1)]
rm_combid_idx = rm_combid_idx.tolist()
mydf.drop(rm_combid_idx, axis = 0, inplace = True)
mydf.reset_index(inplace = True, drop = True)

mydf.to_csv(dpath + 'ONervD_outcomes.csv', index = False)
