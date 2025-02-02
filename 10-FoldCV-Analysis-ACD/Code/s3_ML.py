

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
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Revision/StrictAnalysis/RegionFold/'

pro_df = pd.read_csv(dpath + 'Proteomics/Raw/ProteomicsData.csv')
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
target_df = pd.read_csv(dpath + 'TargetOutcomes/ACD/ACD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code', 'in_cv_fold'])
cov_f_lst = ['age', 'sex', 'educ', 'apoe4', 'pm_time', 'rt_time']
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv', usecols = ['eid'] + cov_f_lst)

mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
mydf = pd.merge(mydf, reg_df, how = 'left', on = ['eid'])

fold_id_lst = list(set(mydf.Region_code))
fold_id_lst = [int(ele) for ele in fold_id_lst]
inner_cv_fold_lst = list(set(mydf.in_cv_fold))
inner_cv_fold_lst = [int(ele) for ele in inner_cv_fold_lst]

for fold_id in fold_id_lst:
    traindf = mydf.loc[mydf.Region_code != fold_id]
    traindf.reset_index(inplace = True, drop = True)
    testdf = mydf.loc[mydf.Region_code == fold_id]
    testdf.reset_index(inplace = True, drop = True)
    auc_imp_df = pd.read_csv(outpath + 'TestFold' + str(fold_id) + '/SFS_cv.csv')
    nb_f = get_top_pros(auc_imp_df)
    pro_f_lst = auc_imp_df.Pro_code.tolist()[:nb_f]
    cv_cov_train,cv_cov_test = traindf[cov_f_lst], testdf[cov_f_lst]
    cv_pro_train,cv_pro_test = traindf[pro_f_lst], testdf[pro_f_lst]
    cv_procov_train,cv_procov_test = traindf[pro_f_lst+cov_f_lst], testdf[pro_f_lst+cov_f_lst]
    cv_y_train, cv_y_test = traindf.target_y, testdf.target_y
    my_lgb_cov = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb_cov.set_params(**{'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
                             'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7})
    my_lgb_cov.fit(cv_cov_train, cv_y_train)
    cv_cov_pred = my_lgb_cov.predict_proba(cv_cov_test)[:, 1].tolist()
    my_lgb_pro = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb_pro.set_params(**{'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
                         'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7})
    my_lgb_pro.fit(cv_pro_train, cv_y_train)
    cv_pro_pred = my_lgb_pro.predict_proba(cv_pro_test)[:, 1].tolist()
    my_lgb_procov = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb_procov.set_params(**{'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
                             'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7})
    my_lgb_procov.fit(cv_procov_train, cv_y_train)
    cv_procov_pred = my_lgb_procov.predict_proba(cv_procov_test)[:, 1].tolist()
    pred_df = pd.DataFrame({'eid':testdf.eid.tolist(), 'target_y':testdf.target_y.tolist(),
                            'y_pred_cov':cv_cov_pred, 'y_pred_pro':cv_pro_pred, 'y_pred_procov':cv_procov_pred})
    pred_df = pd.merge(pred_df, mydf[['eid', 'Region_code', 'in_cv_fold']], how = 'left', on = 'eid')
    print((roc_auc_score(pred_df.target_y, pred_df.y_pred_cov),
           roc_auc_score(pred_df.target_y, pred_df.y_pred_pro),
           roc_auc_score(pred_df.target_y, pred_df.y_pred_procov)))
    pred_df.to_csv(outpath + 'TestFold' + str(fold_id) + '/pred_probs.csv', index = False)


