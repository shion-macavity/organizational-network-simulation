import os 
import sys 
import pandas as pd 
import numpy as np 

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (os .path .dirname (current_dir ))# analysis/miscellaneous tasks -> analysis -> root
sys .path .append (parent_dir )

# Search for config_analysis using an absolute path in case path resolution is difficult
sys .path .append (os .path .join (parent_dir ,'analysis','本分析'))# Folder/File: This Analysis | Folder/File: This Analysis | Folder/File: 本分析

try :
    import config_analysis as conf 
except ImportError :
# Fallback for when the hierarchy is different
    sys .path .append (os .path .join (parent_dir ,'analysis','本分析'))# Folder/File: This Analysis | Folder/File: This Analysis | Folder/File: 本分析
    import config_analysis as conf 

def diagnose ():
    print ("=== Dept 1 CDU Diagnosis ===")

    # Data Loading
    path_dept =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")
    if not os .path .exists (path_dept ):
        print (f"[ERROR] File not found: {path_dept }")
        return 

    df =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    # Extraction of Dept 1
    df_d1 =df [df ['dept_id']==1 ].copy ()
    print (f"Total Rows for Dept 1: {len (df_d1 )}")

    # Checking the target variable
    target_cols =['ID-Ave CDU','IDIH-CDU','IDTF-CDU','adj-CDC Score']

    for col in target_cols :
        if col not in df_d1 .columns :
            print (f"[CRITICAL] Column '{col }' is MISSING from dataset.")
            return 

            # Check the missing status for each pattern
    print ("\n--- NaN Count by Pattern (Dept 1) ---")
    for pid in [1 ,2 ,3 ,4 ]:
        sub =df_d1 [df_d1 ['pattern_id']==pid ]
        n_total =len (sub )
        print (f"\n[Pattern {pid }] (N={n_total })")

        for col in target_cols :
            n_nan =sub [col ].isna ().sum ()
            n_valid =n_total -n_nan 

            # Sneak peek of the value
            sample_vals =sub [col ].dropna ().head (3 ).tolist ()
            if not sample_vals :
                sample_vals ="All NaN"

                # Variance check (correlation won't appear if all values are the same)
            std_val =sub [col ].std ()if n_valid >1 else 0.0 

            print (f"  - {col :<15}: Valid={n_valid }, NaN={n_nan }, Std={std_val :.4f}, Sample={sample_vals }")

            # Simulation of heat map drawing conditions
    print ("\n--- Correlation Pre-check (adj-CDC vs CDU) ---")
    # Are there five or more rows where adj-CDC is valid (Pattern 2,3,4) and CDU is also valid?

    valid_adj =df_d1 .dropna (subset =['adj-CDC Score'])
    print (f"Rows with valid adj-CDC: {len (valid_adj )}")

    for col in ['ID-Ave CDU','IDIH-CDU','IDTF-CDU']:
    # Rows where both adj-CDC and CDU are valid
        both_valid =valid_adj .dropna (subset =[col ])
        n_overlap =len (both_valid )

        print (f"  Overlap count with {col }: {n_overlap }")

        if n_overlap <5 :
            print (f"  -> [FAIL] Less than 5 samples. Heatmap will show N/A.")
        else :
            print (f"  -> [OK] Sufficient samples for correlation.")

if __name__ =="__main__":
    diagnose ()