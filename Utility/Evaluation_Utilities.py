
import pickle
import re
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, chisquare, entropy, pointbiserialr
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import log_loss, average_precision_score, f1_score
from itertools import product
import random


def threshold(array, cutoff):
    array1 = array.copy()
    array1[array1 < cutoff] = 0
    array1[array1 >= cutoff] = 1
    return array1


def get_p_value(var, df, y, type):
    '''
    find the p-value of input variable
    var: variabale name
    df: dataframe that contains var
    y: the response variable that we want to test
    type: type of the input variable, whether 'Continuous' or 'Categorical'
    '''
    var_col = df[var]
    try:
        if type == 'Continuous':
            var_yes = var_col[np.where(y == 1)[0]]
            var_no = var_col[np.where(y == 0)[0]]
            p_value = ttest_ind(var_yes, var_no, nan_policy = 'omit')[1]
        elif type == 'Categorical':
            my_tbl = pd.crosstab(var_col.fillna('NA'), y)
            p_value = chi2_contingency(my_tbl)[1]
        else:
            print('Please input the correct Type of your variable: Continuous or Categorical')
    except:
        p_value = 1
    return p_value


def find_sig_pvalue(X, y, pvalue):
    features = X.columns
    nb_levels = [len(set(X[f].fillna('NA'))) for f in features]
    f_type = ['Continuous' if nb_levels[i] > 8 else 'Categorical' for i in range(len(features))]
    pvalues = [get_p_value(features[i], X, y, f_type[i]) for i in range(len(features))]
    sig_f = [features[i] for i in range(len(features)) if pvalues[i] < pvalue]
    sig_p = [pvalues[i] for i in range(len(features)) if pvalues[i] < pvalue]
    return (sig_f, sig_p)


def find_sig_pvalue_top(X, y, top_n):
    features = X.columns
    nb_levels = [len(set(X[f].fillna('NA'))) for f in features]
    f_type = ['Continuous' if nb_levels[i] >8 else 'Categorical' for i in range(len(features))]
    pvalues = [get_p_value(features[i], X, y, f_type[i]) for i in range(len(features))]
    f_dict = dict(zip(features, pvalues))
    f_dict = {k: v for k, v in sorted(f_dict.items(), key = lambda item: item[1], reverse = False)}
    sig_f = [k for k,v in f_dict.items()][:top_n]
    sig_p = [v for k,v in f_dict.items()][:top_n]
    return (sig_f, sig_p)


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/ (n-1))
    rcorr = r - ((r-1)**2)/ (n-1)
    kcorr = k - ((k-1)**2)/ (n-1)
    return np.sqrt(phi2corr/ min((kcorr - 1), (rcorr-1)))


def get_correlation(var, df, y, type):
    var_col = df[var]
    try:
        if type == 'Continuous':
            var_col = var_col.fillna(var_col.mean())
            my_corr = pointbiserialr(y, var_col)[0]
        elif type == 'Categorical':
            #var_col = var_col.fillna(var_col.mode())
            my_corr = cramers_v(var_col, y)
        else:
            print('Please input the correct Type of your variable: Continuous or Categorical')
    except:
        my_corr = 0
    if my_corr >= 0:
        return my_corr
    elif my_corr < 0:
        return -my_corr
    else:
        return 0


def find_sig_corr(X, y, corr):
    features = X.columns
    nb_levels = [len(set(X[f].fillna('NA'))) for f in features]
    f_type = ['Continuous' if nb_levels[i] > 8 else 'Categorical' for i in range(len(features))]
    correlations = [get_correlation(features[i], X, y, f_type[i]) for i in range(len(features))]
    sig_f = [features[i] for i in range(len(features)) if correlations[i] >= corr]
    sig_corr = [correlations[i] for i in range(len(features)) if correlations[i] >= corr]
    return (sig_f, sig_corr)


def find_sig_corr_top(X, y, top_n):
    features = X.columns
    nb_levels = [len(set(X[f].fillna('NA'))) for f in features]
    f_type = ['Continuous' if nb_levels[i] >= 8 else 'Categorical' for i in range(len(features))]
    correlations = [get_correlation(features[i], X, y, f_type[i]) for i in range(len(features))]
    f_dict = dict(zip(features, correlations))
    f_dict = {k: v for k, v in sorted(f_dict.items(), key = lambda item: item[1], reverse = True)}
    sig_f = [k for k,v in f_dict.items()][:top_n]
    sig_cor = [v for k,v in f_dict.items()][:top_n]
    return (sig_f, sig_cor)


def get_pos_ratio(var, df, y):
    var_col = df[var]
    try:
        my_tbl = pd.crosstab(var_col, y)
        nb_pos = my_tbl.iloc[:, 1].sum()
        pos_ratio = nb_pos / np.sum(y)
        return pos_ratio
    except:
        return 0


def find_sig_pos_ratio(X, y, pos_ratio_thres):
    features = X.columns
    pos_ratio = [get_pos_ratio(features[i], X, y) for i in range(len(features))]
    sig_f = [features[i] for i in range(len(features)) if pos_ratio[i] >= pos_ratio_thres]
    sig_pos_ratio = [pos_ratio[i] for i in range(len(features)) if pos_ratio[i] >= pos_ratio_thres]
    return (sig_f, sig_pos_ratio)


def find_sig_f(X, y, corr_thres, pval_thres, pos_ratio_thres):
    my_f = X.columns
    nb_f = len(my_f)
    nb_levels = [len(set(X[f].fillna('NA'))) for f in my_f]
    f_type = ['Continuous' if nb_levels[i] > 8 else 'Categorical' for i in range(nb_f)]
    my_corr = [get_correlation(my_f[i], X, y, f_type[i]) for i in range(nb_f)]
    my_pval = [get_p_value(my_f[i], X, y, f_type[i]) for i in range(nb_f)]
    my_pos_ratio = [get_pos_ratio(my_f[i], X, y) for i in range(nb_f)]
    sig_f = [my_f[i] for i in range(nb_f) if (my_corr[i] >= corr_thres and my_pval[i] <= pval_thres and my_pos_ratio[i] >= pos_ratio_thres)]
    return sig_f



def extend(my_array, nb_points):
    if len(my_array) == nb_points:
        pass
    else:
        nb2impute = nb_points - len(my_array)
        impute_array = np.zeros(nb2impute)
        my_array = np.concatenate((impute_array, my_array), axis=0)
    return np.expand_dims(my_array, -1)


def autolabel(rects, ax, x_move, y_move, fontsize, color, weight = 'bold'):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2 + x_move, height + y_move),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize = fontsize,
                    color = color,
                    weight = weight)

def hl_pvalue(obs_prob, pred_prob, percentage, bin_obs_nb):
    obs_prob = obs_prob / percentage
    pred_prob = pred_prob / percentage
    nominator = (obs_prob-pred_prob)**2*bin_obs_nb
    denominator = pred_prob*(1-pred_prob)
    denominator[denominator == 0] = 1
    stat = nominator/denominator
    stat[stat == np.inf] = 0.0
    stat[stat == -np.inf] = 0.0
    test_stat = np.nansum(stat)
    pvalue = 1-stats.chi2.cdf(test_stat, 8)
    return np.round(pvalue,2)


def round_fun(my_array):
    return [np.round(ele, 2) if ele < 10 else np.round(ele, 1) for ele in my_array]



def my_cali_fun(y_ground_true, y_pred_probs, n_bins):
    y_true, y_pred = np.array(y_ground_true), np.array(y_pred_probs)
    q_cut = np.round(np.linspace(0., 1, n_bins + 1), 3)
    obs_frq_lst, pred_frq_lst = [], []
    for i in range(len(q_cut) -1):
        lower = np.quantile(y_pred, q_cut[i])
        upper = np.quantile(y_pred, q_cut[i+1])
        tmp_idx = np.where((y_pred>=lower) & (y_pred<=upper))[0]
        pred_frq_lst.append(np.sum(y_pred[tmp_idx])/len(tmp_idx))
        obs_frq_lst.append(np.sum(y_true[tmp_idx])/len(tmp_idx))
    return np.array(obs_frq_lst), np.array(pred_frq_lst)
