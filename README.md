Regarding the machine learning pipelines, in the published paper, we performed Cox regression using the whole data. To further alleviate the potential issue of double dipping, we now upload the code and detailed results by rigid performing both Cox and protein selection under strict 10 fold cross-validation. The proteins were initially performed through Cox regressions and those survived multiple comparison tests passed into sequential forward selection. Selected proteins under each cross-validation partition were then used for model development.
The results were largely consistent with those obtained in the published paper. Top selected proteins of NEFL, GFAP, GDF15, BCAN, LTBP2, NPTXR, EDA2R were all selected under each cross-validation partition. The protein panel and protein panel + demographic information were also obtained similar performance of AUCs.
These analysis and corresponding results were uploaded within folder 10-FoldCV-Analysis-ACD.

Code:

Association analysis of Cox regression:

s0_Cox_M1.py

s0_Cox_M2.py

Importance ranking  of proteins:

s1_ACD_ProImp.py

Sequential forward protein selection:

s2_SFS.py

Machine learning model development:

s3_ML.py

Model evaluation:

S4_Eval.py


Results:

CV_Fold_SelectedProteins.csv: Selected proteins under each cross-validation partition

CV_Fold_Eval_ProPanel.csv: Cross-validation model performance of protein panel

CV_Fold_Eval_ProDemo.csv: Cross-validation model performance of protein panel + Demographic

CV_Fold_Eval_Top3ProDemo.csv: Cross-validation model performance of top-3 protein + Demographic; notably, the top-3 proteins were NEFL, GFAP and GDF15


/Results/TestFold*/ : 

cross-validation results under each cross-validation partition

ACD_Cox_M1.csv & ACD_Cox_M2.csv: Cox regression to identify associated proteins

ProImportance_cv.csv: calculate the protein importance and ranking proteins

SFS_cv.csv: sequential forward selection procedure to determine optimal number of proteins

pred_probs.csv: predicted probabilities of selected proteins and predicted probabilities of selected proteins + demographic information

pred_probs_Top3ProDemo.csv: predicted probabilities of top-3 selected proteins + demographic information

Eval_ProPanel.csv & Eval_ProDemo.csv & Eval_Top3ProDemo.csv: cross-validation evaluation of selected protein, selected protein + demographic information and top-3 selected proteins + demographic information

