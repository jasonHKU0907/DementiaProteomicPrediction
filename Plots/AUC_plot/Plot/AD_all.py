

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, average_precision_score
from Utility.DelongTest import delong_roc_test


mywidth = 4
col1, col2, col3 = 'steelblue', 'deepskyblue', 'yellowgreen'
col4, col5, col6 = 'darkorange', 'lightcoral', 'darkred'
titlename = 'All incident AD'
dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'
output_img = dpath + 'Plots/Figure3/Plot/ALL_AD.png'


mydf = pd.read_csv(dpath + 'Plots/Figure3/Data/AD_ALL.csv')
fold_id_lst = [i for i in range(10)]

y_pred_col1, y_pred_col2, y_pred_col3 = 'y_pred_m1', 'y_pred_m2', 'y_pred_m3'
y_pred_col4, y_pred_col5, y_pred_col6 = 'y_pred_m4', 'y_pred_m5', 'y_pred_m6'

def AUC_calculator(mydf, y_pred_col, fold_id_lst):
    auc_cv_lst = []
    for fold_id in fold_id_lst:
        test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
        y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col].iloc[test_idx]
        auc_cv_lst.append(roc_auc_score(y_true, y_pred))
    auc_mean = np.round(np.mean(auc_cv_lst), 3)
    auc_std = np.round(np.std(auc_cv_lst), 3)
    return auc_mean, auc_std

auc_mean_m1, auc_sd_m1 = AUC_calculator(mydf, y_pred_col1, fold_id_lst)
auc_mean_m2, auc_sd_m2 = AUC_calculator(mydf, y_pred_col2, fold_id_lst)
auc_mean_m3, auc_sd_m3 = AUC_calculator(mydf, y_pred_col3, fold_id_lst)
auc_mean_m4, auc_sd_m4 = AUC_calculator(mydf, y_pred_col4, fold_id_lst)
auc_mean_m5, auc_sd_m5 = AUC_calculator(mydf, y_pred_col5, fold_id_lst)
auc_mean_m6, auc_sd_m6 = AUC_calculator(mydf, y_pred_col6, fold_id_lst)

print((auc_mean_m1, auc_mean_m2, auc_mean_m3, auc_mean_m4, auc_mean_m5 ,auc_mean_m6))

#stat = delong_roc_test(mydf['target_y'], mydf['y_pred_m1'], mydf['y_pred_m2'])
#np.exp(stat)

legend1 = 'Plasma GFAP only (AUC =' +                   str(auc_mean_m1) +')'
legend2 = 'GFAP + demographic  (' +                     str(auc_mean_m2) +')'
legend3 = 'GFAP + demographic + cognition (' +          str(auc_mean_m3) +')'
legend4 = 'Protein panel (' +                           str(auc_mean_m4) +')'
legend5 = 'Protein panel + demographic (' +             str(auc_mean_m5) +')'
legend6 = 'Protein panel + demographic + cognition (' + str(auc_mean_m6) +')'

fig, ax = plt.subplots(figsize = (12, 12))

tprs = []
base_fpr = np.linspace(0, 1, 101)

for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col1].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col1, linewidth = mywidth, label = legend1)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col1, alpha = 0.1)


tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col2].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col2, linewidth = mywidth, label = legend2)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col2, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col3].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col3, linewidth = mywidth, label = legend3)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col3, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col4].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col4, linewidth = mywidth, label = legend4)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col4, alpha = 0.1)



tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col5].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col5, linewidth = mywidth, label = legend5)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col5, alpha = 0.1)




tprs = []
base_fpr = np.linspace(0, 1, 101)
for fold_id in fold_id_lst:
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    y_true, y_pred = mydf.iloc[test_idx].target_y, mydf[y_pred_col6].iloc[test_idx]
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    #plt.plot(fpr, tpr, 'midnightblue', alpha=0.15)
    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0
    tprs.append(tpr)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis = 0)
std = tprs.std(axis = 0)
tprs_upper = np.minimum(mean_tprs + 2*std, 1)
tprs_lower = mean_tprs - 2*std
plt.plot(base_fpr, mean_tprs, col6, linewidth = mywidth, label = legend6)
#plt.fill_between(base_fpr, tprs_lower, tprs_upper, color = col6, alpha = 0.1)

plt.legend(loc=4, fontsize = 20, labelspacing = 1.5, facecolor='gainsboro')

plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.0, 1.0])
plt.ylim([-0.0, 1.0])
plt.ylabel('True Positive Rate', fontsize = 20)
plt.xlabel('False Positive Rate', fontsize = 20)
plt.title(titlename, fontsize = 36, weight='bold')

plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 16)
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 16)
plt.grid(which='minor', alpha=0.2, linestyle=':')
plt.grid(which='major', alpha=0.5,  linestyle='--')
plt.tight_layout()

ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')

plt.savefig(output_img)
