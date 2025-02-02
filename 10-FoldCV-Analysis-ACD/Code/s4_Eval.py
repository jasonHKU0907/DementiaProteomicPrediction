
import os.path
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import brier_score_loss, average_precision_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import glob

def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1

def Find_Optimal_Cutoff(mydf, target_col, pred_col):
    fpr, tpr, threshold = roc_curve(mydf[target_col], mydf[pred_col])
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    return list(roc_t['threshold'])

def get_eval(y_test, pred_prob, cutoff):
    pred_binary = threshold(pred_prob, cutoff)
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    prec = tp / (tp + fp)
    Youden = sens + spec - 1
    f1 = 2 * prec * sens / (prec + sens)
    auc = roc_auc_score(y_test, pred_prob)
    apr = average_precision_score(y_test, pred_prob)
    brier = brier_score_loss(y_test, pred_prob)
    nnd = 1 / Youden
    evaluations = np.round((cutoff, acc, sens, spec, prec, Youden, f1, auc, apr, nnd, brier), 4)
    evaluations = pd.DataFrame(evaluations).T
    evaluations.columns = ['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND', 'BRIER']
    evaluations = evaluations[['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND', 'BRIER']]
    return evaluations

def get_cv_output(mydf, y_true_col, y_pred_col, fold_col, fold_id_lst, cutoff):
    result_df = pd.DataFrame()
    for fold_id in fold_id_lst:
        tmp_idx = mydf[fold_col].index[mydf[fold_col] == fold_id]
        tmpdf = mydf.iloc[tmp_idx]
        tmpdf.reset_index(inplace = True, drop = True)
        y_test, pred_prob = tmpdf[y_true_col], tmpdf[y_pred_col]
        tmp_result_df = get_eval(y_test, pred_prob, cutoff)
        result_df = pd.concat([result_df, tmp_result_df], axis = 0)
    result_df = result_df.T
    result_df['MEAN'] = result_df.mean(axis=1)
    result_df['STD'] = result_df.std(axis=1)
    output_lst = []
    for i in range(11):
        my_mean = str(np.round(result_df['MEAN'][i], 3))
        my_std = str(np.round(result_df['STD'][i], 3))
        output_lst.append(my_mean + ' +- ' + my_std)
    result_df['output'] = output_lst
    return result_df.T

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Revision/StrictAnalysis/RegionFold/'

reg_df = pd.read_csv(dpath + 'Eid_info_data.csv', usecols = ['eid', 'Region_code', 'in_cv_fold'])

fold_id_lst = [i for i in range(10)]
in_fold_lst = [i for i in range(10)]

pro_results_lst, procov_results_lst = [], []

for fold_id in fold_id_lst:
    mydf = pd.read_csv(outpath + 'TestFold' + str(fold_id) + '/pred_probs_TOP3PRO.csv')
    opt_ct_pro = Find_Optimal_Cutoff(mydf, 'target_y', 'y_pred_pro')[0]
    results_pro = get_cv_output(mydf, 'target_y', 'y_pred_pro', 'in_cv_fold', in_fold_lst, opt_ct_pro)
    results_pro.index = ['inner_cv_iter' + str(in_fold) for in_fold in in_fold_lst] + ['Mean', 'StandarDeviation', 'Output']
    results_pro.to_csv(outpath + 'TestFold' + str(fold_id) + '/Eval_ProPanel_TOP3PRO.csv', index = True)
    pro_results_lst.append(results_pro.iloc[10,:].tolist())
    opt_ct_procov = Find_Optimal_Cutoff(mydf, 'target_y', 'y_pred_procov')[0]
    results_procov = get_cv_output(mydf, 'target_y', 'y_pred_procov', 'in_cv_fold', in_fold_lst, opt_ct_pro)
    results_procov.index = ['inner_cv_iter' + str(in_fold) for in_fold in in_fold_lst] + ['Mean','StandarDeviation',  'Output']
    results_procov.to_csv(outpath + 'TestFold' + str(fold_id) + '/Eval_ProDemo_TOP3PRO.csv', index=True)
    procov_results_lst.append(results_procov.iloc[10,:].tolist())


pro_results_df = pd.DataFrame(pro_results_lst)
pro_results_df = pro_results_df.T
pro_results_df['MEAN'] = pro_results_df.mean(axis=1)
pro_results_df['STD'] = pro_results_df.std(axis=1)
output_lst1 = []
for i in range(11):
    my_mean = str(np.round(pro_results_df['MEAN'][i], 3))
    my_std = str(np.round(pro_results_df['STD'][i], 3))
    output_lst1.append(my_mean + ' +- ' + my_std)
pro_results_df['output'] = output_lst1
pro_results_df = pro_results_df.T
pro_results_df.columns = ['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND', 'BRIER']
pro_results_df.to_csv(outpath + 'CV_Fold_Eval_ProPanel_TOP3PRO.csv')


procov_results_df = pd.DataFrame(procov_results_lst)
procov_results_df = procov_results_df.T
procov_results_df['MEAN'] = procov_results_df.mean(axis=1)
procov_results_df['STD'] = procov_results_df.std(axis=1)
output_lst2 = []
for i in range(11):
    my_mean = str(np.round(procov_results_df['MEAN'][i], 3))
    my_std = str(np.round(procov_results_df['STD'][i], 3))
    output_lst2.append(my_mean + ' +- ' + my_std)
procov_results_df['output'] = output_lst2
procov_results_df = procov_results_df.T
procov_results_df.columns = ['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND', 'BRIER']
procov_results_df.to_csv(outpath + 'CV_Fold_Eval_ProDemo_TOP3PRO.csv')


