

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
import shap
pd.options.mode.chained_assignment = None  # default='warn'

def get_top_pros(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)):
        i+=1
    return i


dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
outfile = outpath + 'ML_Modeling/10_YEARS/VaD/s7_SeqSelector.csv'

pro_f_df = pd.read_csv(outpath + 'ML/10_YEARS/VaD/AccAUC_TotalGain.csv')
nb_top_pros = get_top_pros(pro_f_df)
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)
pro_dict = pd.read_csv(dpath + 'Proteomics/Raw/ProCode.csv', usecols = ['Pro_code', 'Pro_definition'])
target_df = pd.read_csv(dpath + 'TargetOutcomes/VaD/VaD_outcomes.csv', usecols = ['eid', 'target_y', 'BL2Target_yrs'])
cov_f_lst = ['age', 'sex', 'educ', 'apoe4', 'pm_time', 'rt_time']
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv', usecols = ['eid', 'Region_code'] + cov_f_lst)
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, cov_df, how = 'left', on = ['eid'])
mydf['target_y'].loc[mydf.BL2Target_yrs > 10] = 0
fold_id_lst = [i for i in range(10)]

f_rank_df = pd.read_csv(outpath + 'ML_Modeling/10_YEARS/VaD/s6_FeaImpRank.csv')
f_rank_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)
my_f_lst = f_rank_df.Pro_code.tolist()

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

my_params = {'n_estimators': 300,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.9,
             'learning_rate': 0.01,
             'colsample_bytree': 0.9}

y_test_full = np.zeros(shape = [1,1])
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_full = np.concatenate([y_test_full, np.expand_dims(mydf.iloc[test_idx].target_y, -1)])

y_pred_full_prev = y_test_full
tmp_f, AUC_cv_lst= [], []

for f in my_f_lst:
    tmp_f.append(f)
    my_X = mydf[tmp_f]
    AUC_cv = []
    y_pred_full = np.zeros(shape = [1,1])
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True,  n_jobs=4, verbosity=-1, seed=2020)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(np.round(roc_auc_score(y_test, y_pred_prob), 3))
        y_pred_full = np.concatenate([y_pred_full, np.expand_dims(y_pred_prob, -1)])
    log10_p = delong_roc_test(y_test_full[:,0], y_pred_full_prev[:,0], y_pred_full[:,0])
    y_pred_full_prev = y_pred_full
    tmp_out = np.array([np.round(np.mean(AUC_cv), 3), np.round(np.std(AUC_cv), 3), 10**log10_p[0][0]] + AUC_cv)
    AUC_cv_lst.append(tmp_out)
    print((f, np.mean(AUC_cv), 10**log10_p[0][0]))

AUC_df = pd.DataFrame(AUC_cv_lst, columns = ['AUC_mean', 'AUC_std', 'p_delong'] + ['AUC_' + str(i) for i in range(10)])

AUC_df = pd.concat((pd.DataFrame({'Pro_code':tmp_f}), AUC_df), axis = 1)
myout = pd.merge(AUC_df, pro_dict, how='left', on=['Pro_code'])
myout.to_csv(outfile, index = False)

print('finished')



