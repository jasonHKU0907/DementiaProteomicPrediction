

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

pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv')
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
target_df = pd.read_csv(dpath + 'TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])
rm_idx = mydf.index[(mydf.BL2Target_yrs < 10) & (mydf.target_y == 1)]
mydf.drop(rm_idx, axis = 0, inplace = True)
mydf.reset_index(inplace = True)

pro_f_m1_df = pd.read_csv(outpath + 'Cox/ACD_M1.csv')
pro_f_m2_df = pd.read_csv(outpath + 'Cox/ACD_M2.csv')
pro_f_m1 = pro_f_m1_df.loc[pro_f_m1_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_m2 = pro_f_m2_df.loc[pro_f_m2_df.p_val_bfi*3<0.05].Pro_code.tolist()
pro_f_lst = [ele for ele in pro_f_m2 if ele in pro_f_m1]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(pro_f_lst))
fold_id_lst = [i for i in range(10)]

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][pro_f_lst], mydf.iloc[test_idx][pro_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = 1, seed = 2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))
    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.abs(np.average(shap_values[0], axis=0))
    shap_imp_cv += shap_values / np.sum(shap_values)


shap_imp_df = pd.DataFrame({'Pro_code': pro_f_lst,
                            'ShapValues_cv': shap_imp_cv/10})
shap_imp_df.sort_values(by = 'ShapValues_cv', ascending = False, inplace = True)

tg_imp_cv = normal_imp(tg_imp_cv)
tg_imp_df = pd.DataFrame({'Pro_code': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_cv = normal_imp(tc_imp_cv)
tc_imp_df = pd.DataFrame({'Pro_code': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = shap_imp_df, right = tg_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df = pd.merge(left = my_imp_df, right = tc_imp_df, how = 'left', on = ['Pro_code'])
my_imp_df['Ensemble'] = (my_imp_df['ShapValues_cv'] + my_imp_df['TotalGain_cv'] + my_imp_df['TotalCover_cv'])/3
my_imp_df.sort_values(by = 'Ensemble', ascending = False, inplace = True)
my_imp_df = pd.merge(my_imp_df, pro_dict, how = 'left', on=['Pro_code'])

my_imp_df.to_csv(outpath + 'ML/OVER_10_YEARS/ACD/ProImportance.csv', index = False)

print('finished')



