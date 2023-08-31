

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
pd.options.mode.chained_assignment = None  # default='warn'
from lifelines import KaplanMeierFitter as kmf
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'


dpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Data/'
outpath = '/Volumes/JasonWork/Projects/AD_Proteomics/Results/'

tgt_outcome = 'ONeurD'
tgt_outcome1 = 'OtherNeurDis'

tgt_outcome = 'ONervD'
tgt_outcome1 = 'OtherNervDis'

tgt_outcome = 'OMBD'
tgt_outcome1 = 'OtherMBDis'


pro_f_df = pd.read_csv(outpath + 'Plots/Figure4/Data/KM_info_' + tgt_outcome1 + '.csv')
target_df = pd.read_csv(dpath + 'TargetOutcomes/' + tgt_outcome1 + '/' + tgt_outcome + '_outcomes.csv',
                        usecols = ['eid', 'target_y', 'BL2Target_yrs', 'acd_y', 'acd_yrs', 'ad_y', 'vd_y'])
target_df = target_df.loc[(target_df.acd_yrs>0) & (target_df.BL2Target_yrs>0)]
target_df.reset_index(inplace = True, drop = True)
rm_idx = target_df.index[target_df.acd_y == 1]
target_df.drop(rm_idx, axis = 0, inplace = True)
target_df.reset_index(inplace = True, drop = True)

pro_f_df = pro_f_df.loc[pro_f_df.Pro_code == 'GFAP']
pro_f_lst = pro_f_df.Pro_code.tolist()
cut_f_lst = pro_f_df.opt_cutoff.tolist()
riskdir_f_lst = pro_f_df.risk_dir.tolist()
pval_f_lst = pro_f_df.HR_p_val.tolist()
hrci_f_lst = pro_f_df.HR_out.tolist()

pro_df = pd.read_csv(dpath + 'Proteomics/ProteomicsData.csv', usecols=['eid'] + pro_f_lst)
mydf = pd.merge(target_df, pro_df, how = 'inner', on = ['eid'])

for i in range(len(pro_f_lst)):
    pro_f, f_cut, risk_dir = pro_f_lst[i], np.round(cut_f_lst[i], 2), riskdir_f_lst[i]
    f_hrci, f_pval = hrci_f_lst[i], str(np.round(pval_f_lst[i],3))
    plotdf = mydf[['eid', 'target_y', 'BL2Target_yrs'] + [pro_f]]
    plotdf.rename(columns={pro_f: 'target_pro'}, inplace=True)
    rm_idx = plotdf.index[plotdf.target_pro.isnull() == True]
    plotdf = plotdf.drop(rm_idx, axis=0)
    plotdf.reset_index(inplace=True)
    if risk_dir == 1:
        high_risk = (plotdf.target_pro > f_cut)
        prop = np.round(high_risk.sum()/len(plotdf)*100,2)
        high_risk_label = 'High risk group (>' + str(f_cut) + ', ' + str(prop) + '%)'
        low_risk_label = 'Rest control'
    elif risk_dir == 0:
        high_risk = (plotdf.target_pro < f_cut)
        prop = np.round(high_risk.sum() / len(plotdf) * 100, 2)
        high_risk_label = 'High risk group (<' + str(f_cut) + ', ' + str(prop) + '%)'
        low_risk_label = 'Rest control'
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    kmf.fit(durations=plotdf.BL2Target_yrs[~high_risk], event_observed=plotdf.target_y[~high_risk], label='Low')
    kmf.plot_survival_function(ax=ax, color='#abc3f0', linewidth=3)
    kmf.fit(durations=plotdf.BL2Target_yrs[high_risk], event_observed=plotdf.target_y[high_risk], label='High')
    kmf.plot_survival_function(ax=ax, color='red', linewidth=3)
    ax.set_title(pro_f, weight='bold', fontsize=24)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('Survival Probability', weight='bold', fontsize=16)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_xlabel('Timeline', weight='bold', fontsize=16)
    ax.plot([], [], ' ', label="HR = " + f_hrci)
    ax.plot([], [], ' ', label="p value = " + f_pval)
    ax.legend(loc='lower left', fontsize='x-large', labelspacing=1, facecolor='gainsboro')
    plt.subplots_adjust(left=0.2, bottom=0.2)
    fig.tight_layout()
    plt.savefig(outpath + 'Plots/Figure4/Plot/' + pro_f + '_' + tgt_outcome + '.png', bbox_inches='tight',pad_inches = 0.05)

