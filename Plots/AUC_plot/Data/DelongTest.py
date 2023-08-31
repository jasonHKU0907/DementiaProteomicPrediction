

import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn.metrics import roc_curve
import pandas as pd
from Utility.DelongTest import delong_roc_test


dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'

mydf = pd.read_csv(dpath + 'Plots/Figure3/Data/ACD_5YEARS.csv')

my_array = np.zeros((7, 7))

for i in range(1,7):
    for j in range(1,7):
        col1 = 'y_pred_m' + str(i)
        col2 = 'y_pred_m' + str(j)
        stat = delong_roc_test(mydf['target_y'], mydf[col1], mydf[col2])
        my_array[i, j] = np.exp(stat)[0][0]

myout_df = pd.DataFrame(my_array)
myout_df.to_csv(dpath + 'Plots/Figure3/Data/DelongTest/ACD_5YEARS_DelongTest.csv', index = False)
