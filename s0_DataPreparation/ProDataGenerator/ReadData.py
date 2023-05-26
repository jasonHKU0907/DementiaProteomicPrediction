

import glob
import os
import numpy as np
import pandas as pd
import re

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/Proteomics/'
mydf = pd.read_csv(dpath + 'Raw/Proteomics_ins0.csv')
code_dict_df = pd.read_csv(dpath + 'ProCode.csv', usecols=['coding', 'Pro_code'])
code_dict_df['coding'] = code_dict_df['coding'].astype(str)

pro_code_df = pd.DataFrame({'coding':mydf.columns[:-1]})
pro_code_df['coding'] = pro_code_df['coding'].astype(str)

rename_dict_df = pd.merge(pro_code_df, code_dict_df[['coding', 'Pro_code']], how = 'left', on = ['coding'])
rename_dict = rename_dict_df.to_dict()

mydf.rename(columns = rename_dict, inplace = True)
mydf.to_csv(dpath + 'Raw/ProteomicsData.csv', index = False)



