import pandas as pd 
import os 
from scipy .stats import pearsonr 
import numpy as np 

# 1. Automatic path resolution (fixed in datasets)
current_dir =os .path .dirname (os .path .abspath (__file__ ))
data_path =os .path .join (current_dir ,"..","本分析","03_datasets","dataset_department_N200.csv")# Folder/File: This Analysis | Folder/File: This Analysis

def calculate_icc (df ,metric ):
    """Simplified Calculation of Intra-Class Correlation Coefficient (ICC)"""
    # Average per run_id (group)
    group_means =df .groupby ('run_id')[metric ].transform ('mean')
    overall_mean =df [metric ].mean ()

    # Between-group variance (MSB) and within-group variance (MSW)
    n_groups =df ['run_id'].nunique ()
    k =2 # 2 samples per group (dept 0, 1)

    msb =k *((df .groupby ('run_id')[metric ].mean ()-overall_mean )**2 ).sum ()/(n_groups -1 )
    msw =((df [metric ]-group_means )**2 ).sum ()/(n_groups *(k -1 ))

    # ICC(1) formula
    icc =(msb -msw )/(msb +(k -1 )*msw )
    return max (0 ,icc )

def run_analysis ():
    if not os .path .exists (data_path ):
        print (f"Error: ファイルが見つかりません\n期待したパス: {os .path .abspath (data_path )}")# Folder/File: Error: File not found | Folder/File: Error: File not found
Expected path: 
Expected path: 
        return 

    df =pd .read_csv (data_path )
    metrics =[c for c in df .columns if c not in ['run_id','dept_id','pattern_id']]
    results =[]
  # Folder/File: 
--- Statistical Independence Diagnostic Report ---
    print ("\n--- 統計的独立性の診断レポート ---")# Folder/File: 
--- Statistical Independence Diagnostic Report ---
    for m in metrics :
    # Creating pairs for correlation analysis
        pivot =df .pivot (index ='run_id',columns ='dept_id',values =m ).dropna ()
        if len (pivot )<10 :continue 

        r ,p =pearsonr (pivot [0 ],pivot [1 ])
        icc =calculate_icc (df .dropna (subset =[m ]),m )

        # Determination: Considered 'linked' if the correlation is significant (p<0.05) and the ICC is high (>0.1)
        status ="With interlocking (N = 100 pairs recommended)"if (p <0.05 and icc >0.1 )else "Independent (OK with N=200)"

        results .append ({
        'Indicator':m ,
        'Correlation (r)':round (r ,4 ),
        'P value':f"{p :.4e}"if p <0.001 else round (p ,4 ),
        'ICC':round (icc ,4 ),
        'Judgment':status 
        })

    results_df =pd .DataFrame (results )
    print (results_df .to_string (index =False ))

    output_path =os .path .join (current_dir ,"independence_report.csv")  # Folder/File: 
The results have been saved: 
    results_df .to_csv (output_path ,index =False ,encoding ='utf-8-sig')
    print (f"\n結果を保存しました: {output_path }")# Folder/File: 
The results have been saved: 

if __name__ =="__main__":
    run_analysis ()