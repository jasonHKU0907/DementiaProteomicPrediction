

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
outpath1 = outpath + 'ML_Modeling_ProRS/10_YEARS/VaD/'
outputfile = outpath1 + 'ProRS_Results.csv'

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

AUC_cv1, AUC_cv2, AUC_cv3, AUC_cv4 = [], [], [], []
tmp_f1 = ['ProRS']
tmp_f2 = ['ProRS'] + basic_f_lst
tmp_f3 = ['ProRS'] + cog_f_lst
tmp_f4 = ['ProRS'] + basic_f_lst + cog_f_lst

for fold_id in fold_id_lst:
    pro_df = pd.read_csv(outpath1 + 'ProRS/Test_fold' + str(fold_id) + '.csv')
    mydf = pd.merge(pro_df, cov_df, how = 'inner', on = ['eid'])
    mydf['target_y'].loc[mydf.BL2Target_yrs > 10] = 0
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    X_train1, X_test1 = mydf.iloc[train_idx][tmp_f1], mydf.iloc[test_idx][tmp_f1]
    X_train2, X_test2 = mydf.iloc[train_idx][tmp_f2], mydf.iloc[test_idx][tmp_f2]
    X_train3, X_test3 = mydf.iloc[train_idx][tmp_f3], mydf.iloc[test_idx][tmp_f3]
    X_train4, X_test4 = mydf.iloc[train_idx][tmp_f4], mydf.iloc[test_idx][tmp_f4]
    my_lgb1 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb1.set_params(**my_params)
    my_lgb1.fit(X_train1, y_train)
    y_pred_prob1 = my_lgb1.predict_proba(X_test1)[:, 1]
    AUC_cv1.append(np.round(roc_auc_score(y_test, y_pred_prob1), 3))
    my_lgb2 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb2.set_params(**my_params)
    my_lgb2.fit(X_train2, y_train)
    y_pred_prob2 = my_lgb2.predict_proba(X_test2)[:, 1]
    AUC_cv2.append(np.round(roc_auc_score(y_test, y_pred_prob2), 3))
    my_lgb3 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb3.set_params(**my_params)
    my_lgb3.fit(X_train3, y_train)
    y_pred_prob3 = my_lgb3.predict_proba(X_test3)[:, 1]
    AUC_cv3.append(np.round(roc_auc_score(y_test, y_pred_prob3), 3))
    my_lgb4 = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb4.set_params(**my_params)
    my_lgb4.fit(X_train4, y_train)
    y_pred_prob4 = my_lgb4.predict_proba(X_test4)[:, 1]
    AUC_cv4.append(np.round(roc_auc_score(y_test, y_pred_prob4), 3))


tmp_out1 = ['ProRS'] + [np.round(np.mean(AUC_cv1), 3), np.round(np.std(AUC_cv1), 3)] + AUC_cv1
tmp_out2 = ['ProRS+Basic'] + [np.round(np.mean(AUC_cv2), 3), np.round(np.std(AUC_cv2), 3)] + AUC_cv2
tmp_out3 = ['ProRS+Cognitive'] + [np.round(np.mean(AUC_cv3), 3), np.round(np.std(AUC_cv3), 3)] + AUC_cv3
tmp_out4 = ['ProRS+AllCovariate'] + [np.round(np.mean(AUC_cv4), 3), np.round(np.std(AUC_cv4), 3)] + AUC_cv4

AUC_df1 = pd.DataFrame(tmp_out1).T
AUC_df2 = pd.DataFrame(tmp_out2).T
AUC_df3 = pd.DataFrame(tmp_out3).T
AUC_df4 = pd.DataFrame(tmp_out4).T

AUC_df1.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df2.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df3.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]
AUC_df4.columns = ['Model', 'AUC_mean', 'AUC_std'] + ['AUC_' + str(i) for i in range(10)]

myout_df = pd.concat([AUC_df1, AUC_df2], axis = 0)
myout_df = pd.concat([myout_df, AUC_df3], axis = 0)
myout_df = pd.concat([myout_df, AUC_df4], axis = 0)

myout_df.to_csv(outputfile, index = False)

print('finished')

