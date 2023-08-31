

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
outpath1 = outpath + 'ML_Modeling_ProRS/OVER_10_YEARS/AD/'
outputfile = outpath1 + 's1_FeaImpRank.csv'

basic_f_lst = ['age', 'sex', 'educ', 'apoe4']
cog_f_lst = ['pm_time', 'rt_time']
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv', usecols = ['eid'] + basic_f_lst + cog_f_lst)
fold_id_lst = [i for i in range(10)]

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

tmp_f_lst = ['ProRS'] + basic_f_lst + cog_f_lst
tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(tmp_f_lst))

for fold_id in fold_id_lst:
    pro_df = pd.read_csv(outpath1 + 'ProRS/Test_fold' + str(fold_id) + '.csv')
    mydf = pd.merge(pro_df, cov_df, how='inner', on=['eid'])
    rm_idx = mydf.index[(mydf.BL2Target_yrs < 10) & (mydf.target_y == 1)]
    mydf.drop(rm_idx, inplace=True)
    mydf.reset_index(inplace=True, drop=True)
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][tmp_f_lst], mydf.iloc[test_idx][tmp_f_lst]
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


shap_imp_df = pd.DataFrame({'Pro_code': tmp_f_lst,
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
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)

my_imp_df.to_csv(outputfile, index = False)

print('finished')


