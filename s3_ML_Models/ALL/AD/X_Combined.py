

import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
pd.options.mode.chained_assignment = None  # default='warn'

outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/ML_Modeling/ALL/AD/'

s1_df = pd.read_csv(outpath + 's1_1Pro.csv')
s1_df['Category'] = 'SinglePro'

s20_df = pd.read_csv(outpath + 's20_BasicCov.csv')
s20_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s20_df], axis = 1)
s20_df['Category'] = 'BasicCovar'

s21_df = pd.read_csv(outpath + 's21_BasicCov.csv')
s21_df['Category'] = 'SinglePro+BasicCovar'

s30_df = pd.read_csv(outpath + 's30_CogCov.csv')
s30_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s30_df], axis = 1)
s30_df['Category'] = 'CognitiveCovar'

s31_df = pd.read_csv(outpath + 's31_CogCov.csv')
s31_df['Category'] = 'SinglePro+CognitiveCovar'

s40_df = pd.read_csv(outpath + 's40_FullCov.csv')
s40_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s40_df], axis = 1)
s40_df['Category'] = 'BasicCovar+CognitiveCovar'

s41_df = pd.read_csv(outpath + 's41_FullCov.csv')
s41_df['Category'] = 'SinglePro+BasicCovar+CognitiveCovar'

s50_df = pd.read_csv(outpath + 's50_ProPANEL.csv')
s50_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s50_df], axis = 1)
s50_df['Category'] = 'ProPANEL'

s51_df = pd.read_csv(outpath + 's51_ProPANEL_BasicCov.csv')
s51_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s51_df], axis = 1)
s51_df['Category'] = 'ProPANEL+BasicCovar'

s52_df = pd.read_csv(outpath + 's52_ProPANEL_CogCov.csv')
s52_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s52_df], axis = 1)
s52_df['Category'] = 'ProPANEL+CognitiveCovar'

s53_df = pd.read_csv(outpath + 's53_ProPANEL_FullCov.csv')
s53_df = pd.concat([pd.DataFrame({'Pro_code':[np.nan]}), s53_df], axis = 1)
s53_df['Category'] = 'ProPANEL++BasicCovar+CognitiveCovar'

mydf = pd.concat([s1_df, s20_df, s21_df, s30_df, s31_df, s40_df, s41_df, s50_df, s51_df, s52_df, s53_df], axis = 0)
mydf.to_csv(outpath + 'CombineResults.csv', index = False)

