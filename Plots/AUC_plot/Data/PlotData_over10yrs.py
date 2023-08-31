

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

def get_pred_probs(tmp_f, mydf, fold_id_lst, my_params, col_name):
    eid_lst, region_lst = [], []
    y_test_lst, y_pred_lst = [], []
    for fold_id in fold_id_lst:
        train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        X_train, X_test = mydf.iloc[train_idx][tmp_f], mydf.iloc[test_idx][tmp_f]
        y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
        my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, n_jobs=4, verbosity=-1, seed=2020)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1].tolist()
        y_pred_lst += y_pred_prob
        y_test_lst += mydf.target_y.iloc[test_idx].tolist()
        eid_lst += mydf.eid.iloc[test_idx].tolist()
        region_lst += mydf.Region_code.iloc[test_idx].tolist()
    myout_df = pd.DataFrame([eid_lst, region_lst, y_test_lst, y_pred_lst]).T
    myout_df.columns = ['eid', 'Region_code', 'target_y', 'y_pred_'+col_name]
    myout_df[['eid', 'Region_code']] = myout_df[['eid', 'Region_code']].astype('int')
    return myout_df

tgt = 'ACD'
dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
outputfile = outpath + 'Plots/Figure3/Data/' + tgt + '_OVER10YEARS.csv'

target_df = pd.read_csv(dpath + 'TargetOutcomes/' + tgt + '/' + tgt + '_outcomes.csv',
                        usecols = ['eid', 'target_y', 'BL2Target_yrs'])
pro_f_df = pd.read_csv(outpath + 'ML/OVER_10_YEARS/' + tgt + '/AccAUC_TotalGain.csv')
nb_top_pros = get_top_pros(pro_f_df)
pro_f_lst = pro_f_df.Pro_code.tolist()[:nb_top_pros]
pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols = ['eid'] + pro_f_lst)

base_f_lst = ['age', 'sex', 'educ', 'apoe4']
cog_f_lst = ['pm_time', 'rt_time']
cov_df = pd.read_csv(dpath + 'Covariates/CovData_normalized.csv', usecols = ['eid', 'Region_code'] + base_f_lst + cog_f_lst)

mydf = pd.merge(target_df, cov_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, pro_df, how = 'left', on = ['eid'])
rm_idx = mydf.index[(mydf.BL2Target_yrs < 10) & (mydf.target_y == 1)]
mydf.drop(rm_idx, axis = 0, inplace = True)
mydf.reset_index(inplace = True, drop = True)
fold_id_lst = [i for i in range(10)]

my_params1 = {'n_estimators': 100,
             'max_depth': 3,
             'num_leaves': 7,
             'subsample': 1,
             'learning_rate': 0.01,
             'colsample_bytree': 1}

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}

y_test_lst, region_lst = [], []
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_test_lst.append(mydf.target_y.iloc[test_idx].tolist())
    region_lst.append(mydf.Region_code.iloc[test_idx].tolist())

tmp_f1 = ['GFAP']
tmp_f2 = ['GFAP'] + base_f_lst
tmp_f3 = ['GFAP'] + base_f_lst + cog_f_lst
#tmp_f1 = ['GDF15']
#tmp_f2 = ['GDF15'] + base_f_lst
#tmp_f3 = ['GDF15'] + base_f_lst + cog_f_lst
tmp_f4 = pro_f_lst
tmp_f5 = pro_f_lst + base_f_lst
tmp_f6 = pro_f_lst + base_f_lst + cog_f_lst

pred_df1 = get_pred_probs(tmp_f1, mydf, fold_id_lst, my_params1, 'm1')
pred_df2 = get_pred_probs(tmp_f2, mydf, fold_id_lst, my_params, 'm2')
pred_df3 = get_pred_probs(tmp_f3, mydf, fold_id_lst, my_params, 'm3')
pred_df4 = get_pred_probs(tmp_f4, mydf, fold_id_lst, my_params, 'm4')
pred_df5 = get_pred_probs(tmp_f5, mydf, fold_id_lst, my_params, 'm5')
pred_df6 = get_pred_probs(tmp_f6, mydf, fold_id_lst, my_params, 'm6')

myout_df = pd.merge(pred_df1, pred_df2[['eid', 'y_pred_m2']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df3[['eid', 'y_pred_m3']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df4[['eid', 'y_pred_m4']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df5[['eid', 'y_pred_m5']], how = 'inner', on = ['eid'])
myout_df = pd.merge(myout_df, pred_df6[['eid', 'y_pred_m6']], how = 'inner', on = ['eid'])

myout_df.to_csv(outputfile, index = False)




