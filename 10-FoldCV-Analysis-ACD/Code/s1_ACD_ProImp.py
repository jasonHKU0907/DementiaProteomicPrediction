
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
from tqdm import tqdm
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Revision/StrictAnalysis/RegionFold/'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
target_df = pd.read_csv(dpath + 'TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code', 'in_cv_fold'])
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])

fold_id_lst = list(set(mydf.Region_code))
fold_id_lst = [int(ele) for ele in fold_id_lst]
inner_cv_fold_lst = list(set(mydf.in_cv_fold))
inner_cv_fold_lst = [int(ele) for ele in inner_cv_fold_lst]

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

for fold_id in tqdm(fold_id_lst):
    pro_f_m1_df = pd.read_csv(outpath + 'TestFold' + str(fold_id) + '/ACD_Cox_M1.csv')
    pro_f_m2_df = pd.read_csv(outpath + 'TestFold' + str(fold_id) + '/ACD_Cox_M2.csv')
    pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi < 0.05].Pro_code.tolist()
    pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi < 0.05].Pro_code.tolist()
    pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]
    traindf = mydf.copy()
    traindf = traindf.loc[traindf['Region_code'] != fold_id]
    traindf.reset_index(inplace = True, drop = True)
    tg_imp_cv = Counter()
    for inner_cv_fold_id in inner_cv_fold_lst:
        in_train_idx = traindf['in_cv_fold'].index[traindf['in_cv_fold'] != inner_cv_fold_id]
        in_cv_X_train, in_cv_y_train = traindf.iloc[in_train_idx][pro_f_lst], traindf.iloc[in_train_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity= 1, seed=2023)
        my_lgb.set_params(**{'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
                             'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7})
        my_lgb.fit(in_cv_X_train, in_cv_y_train)
        totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
        totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
        tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tg_imp_cv = normal_imp(tg_imp_cv)
    tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()), 'TotalGain_cv': list(tg_imp_cv.values())})
    my_imp_df = pd.merge(tg_imp_df, pro_dict, how = 'left', on=['Pro_code'])
    my_imp_df.sort_values(by = 'TotalGain_cv', ascending=False, inplace=True)
    my_imp_df.to_csv(outpath + 'TestFold' + str(fold_id) + '/ProImportance_cv.csv', index = False)

print('finished')

