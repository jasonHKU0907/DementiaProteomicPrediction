

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
outputfile = outpath + 'ML_Modeling_ProRS/10_YEARS/VaD/ProRS_Data.csv'

pro_f_df = pd.read_csv(outpath + 'ML/10_YEARS/VaD/AccAUC_TotalGain.csv')
nb_top_pros = get_top_pros(pro_f_df)
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)
target_df = pd.read_csv(dpath + 'TargetOutcomes/VaD/VaD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_f_lst = ['age', 'sex', 'educ', 'apoe4', 'pm_time', 'rt_time']
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
mydf['target_y'].loc[mydf.BL2Target_yrs>10] = 0
fold_id_lst = [i for i in range(10)]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

AUC_cv = []
tmp_f = pro_f_lst

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    y_pred_train = my_lgb.predict_proba(X_train)[:, 1]
    prors_train_df = pd.DataFrame({'eid': mydf.iloc[train_idx].eid.tolist(),
                                   'Region_code': mydf.iloc[train_idx].Region_code.tolist(),
                                   'target_y': y_train.tolist(),
                                   'BL2Target_yrs': mydf.iloc[train_idx].BL2Target_yrs.tolist(),
                                   'ProRS': y_pred_train.tolist()})
    y_pred_test = my_lgb.predict_proba(X_test)[:, 1]
    prors_test_df = pd.DataFrame({'eid': mydf.iloc[test_idx].eid.tolist(),
                                   'Region_code': mydf.iloc[test_idx].Region_code.tolist(),
                                   'target_y': y_test.tolist(),
                                   'BL2Target_yrs': mydf.iloc[test_idx].BL2Target_yrs.tolist(),
                                   'ProRS': y_pred_test.tolist()})
    prors_df = pd.concat([prors_test_df, prors_train_df], axis = 0)
    prors_df.to_csv(outpath + 'ML_Modeling_ProRS/10_YEARS/VaD/ProRS/Test_fold' + str(fold_id) + '.csv', index = False)


print('finished')

