
import pickle
import re
import numpy as np
import pandas as pd
import warnings
from collections import Counter
from scipy.stats import ttest_ind, chi2_contingency, chisquare, entropy, pointbiserialr
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix
from sklearn.metrics import log_loss, average_precision_score, f1_score
from itertools import product
import random



def get_days_intervel(start_date_var, end_date_var, df):
    '''
    function aiming to find the days intervel between two dates
    Input: start_date_var, variable name of the starting date
           end_date_var, variable name of the ending date
           df, the dataframe that contains the starting & ending date
    '''
    start_date =  pd.to_datetime(df[start_date_var])
    end_date = pd.to_datetime(df[end_date_var])
    nb_of_dates = start_date.shape[0]
    days = [(end_date[i] - start_date[i]).days for i in range(nb_of_dates)]
    days = [item if item >= 0 else -item for item in days ]
    return pd.DataFrame(days, columns = [end_date_var + '_ONSET_Days'])


def BinaryTo01(binary_var, df):
    '''
    For all variables that contain only two levels,
    convert them into binary 0 & 1
    if not contain missing values
    e.g.: (0, 1) to (0, 1); (1, 2) to (0, 1); (1, 3) to (0, 1)
    else if contain missing values:
    e.g. (1, NA) to (1, 0); (2, NA) to (1, 0); (99, NA) to (1, 0)
    '''
    my_col = df[binary_var]
    if my_col.isna().values.any() == False:
        levels = sorted(list(set(my_col)))
        my_col_binary = my_col.map({levels[0]: 0, levels[1]: 1})
    else:
        my_col_fillna = my_col.fillna('NA')
        levels = list(set(my_col_fillna))
        my_col_binary = my_col_fillna.map({levels[0]: 1, levels[1]: 0})
    return my_col_binary



def Convert2Dummies(var, df, datetime_col = None, date_var = False, time_var = False):
    '''
    This function aims to convert features into dummied dataframe
    Aimed variables:
        1st. contain missing values
        2nd. contain equal to or greater than 2 (>=2) levels
    Inputs:
        var: feature name of the variable
        df: pandas dataframe that contains the variable
        datetime_col: if the variable is date or time, input the pd column directly
        date_var: indicator whether the variable is a date variable
        time_var: indicator whther the variable is a time variable
    Steps:
        1st. Filled missing values with string 'NA'
        2nd. Find how many levbeles in the column
        3rd. Dummy the variable
        4th. Rename all the columns with their original variable names plus dummied names
    Outputs:
        Dummied dataframe with renamed columns
    '''
    if (date_var == True) or (time_var == True):
        my_col = datetime_col
    else:
        my_col = df[var]
    my_col_fillna = my_col.fillna('NA')
    levels = list(set(my_col_fillna))
    levels = [int(item)  if type(item) == float else item for item in levels]
    levels =  np.sort(levels).tolist()
    if date_var == True:
        col_names = [var + '-YQ' + str(item) for item in levels]
    elif time_var == True:
        col_names = [var + '-DQ' + str(item) for item in levels]
    else:
        col_names = [var + '-' + str(item) for item in levels]
    dummied_cols = pd.get_dummies(my_col_fillna)
    dummied_cols = dummied_cols.rename(dict(zip(dummied_cols.columns.tolist(), col_names)), axis = 1)
    return dummied_cols


def reformat_date_col(date_var, df):
    '''
    This function aims to convert the date variable to season & weekday
    Season contains four binary indicator:
    YQ1: Jan, Feb, Mar;
    YQ2: Apr, May, Jun;
    YQ3: Jul, Aug, Sep;
    YQ4: Oct, Nov, Dec;
    Weekday contains one binary indicator:
    WD: whether the day is weekday (1) or weekendn (0)
    '''
    date_col = pd.to_datetime(df[date_var], format = '%Y-%m-%d', errors = 'ignore')
    DayOfWeek = [item.dayofweek for item in date_col]
    WeekDay = pd.DataFrame([1 if item <= 4 else 0 if item > 4 else 'NA' for item in DayOfWeek],
                            columns = [date_var + '-WD'])
    if 'NA' in list(WeekDay.iloc[:, 0]):
        WeekDay_df = pd.get_dummies(WeekDay).iloc[:, 1:]
    else:
        WeekDay_df = WeekDay
    month2season = dict(zip(range(1, 13), sorted([1, 2, 3, 4]*3)))
    Season = date_col.dt.month.map(month2season)
    #Season = pd.DataFrame([item.quarter for item in date_col], columns = [date_var])
    Season_df = Convert2Dummies(date_var, df, datetime_col = Season, date_var = True)
    return pd.concat((Season_df, WeekDay_df), axis = 1)


def reformat_time_col(time_var, df):
    '''
    This function aims to convert the time variable to four quarters of the day
    Quarters contains four binary indicator:
    DQ1: 00:00 - 05:59;
    DQ2: 06:00 - 11:59;
    DQ3: 12:00 - 17:59;
    DQ4: 18:00 - 23:59;
    '''
    time_col = pd.to_datetime(df[time_var], errors = 'ignore')
    time2quarter = dict(zip(range(24), sorted([1, 2, 3, 4]*6)))
    day_quarter = time_col.dt.hour.map(time2quarter)
    day_quarter_df = Convert2Dummies(time_var, df, datetime_col = day_quarter, time_var = True)
    return day_quarter_df


