import os 
import sys 
import pandas as pd 
import numpy as np 

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Checker Settings
# =========================================================
CHECK_LOGS =[]
ERROR_COUNT =0 
WARNING_COUNT =0 

def log_info (msg ):
    print (f"[INFO] {msg }")

def log_ok (msg ):
    print (f"[OK] {msg }")

def log_warning (msg ):
    global WARNING_COUNT 
    WARNING_COUNT +=1 
    print (f"\033[93m[WARNING] {msg }\033[0m")# Yellow text

def log_error (msg ):
    global ERROR_COUNT 
    ERROR_COUNT +=1 
    print (f"\033[91m[ERROR] {msg }\033[0m")# Red text

    # =========================================================
    # Check function group
    # =========================================================

def load_datasets ():
    """Loading the dataset"""
    path_org =os .path .join (conf .OUTPUT_DATASETS ,"dataset_organization_N100.csv")
    path_dept =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")

    if not os .path .exists (path_org ):
        log_error (f"Organization dataset not found: {path_org }")
        return None ,None 
    if not os .path .exists (path_dept ):
        log_error (f"Department dataset not found: {path_dept }")
        return None ,None 

    df_org =pd .read_csv (path_org ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )
    return df_org ,df_dept 

def check_structure_and_ids (df_org ,df_dept ):
    """Check of structure and ID consistency"""
    log_info ("--- Checking Structure & IDs ---")

    # 1. Line Count Check
    if len (df_org )==100 :
        log_ok (f"Org Dataset Row Count: 100 (Correct)")
    else :
        log_error (f"Org Dataset Row Count: {len (df_org )} (Expected: 100)")

    if len (df_dept )==200 :
        log_ok (f"Dept Dataset Row Count: 200 (Correct)")
    else :
        log_error (f"Dept Dataset Row Count: {len (df_dept )} (Expected: 200)")

        # 2. Run ID Coverage
    missing_runs =set (conf .RUN_IDS )-set (df_org ['run_id'].unique ())
    if not missing_runs :
        log_ok ("All Run IDs (1-100) are present.")
    else :
        log_error (f"Missing Run IDs: {missing_runs }")

        # 3. Consistency of Pattern ID (RunID % 4)
        # Expected value calculation
    df_org ['expected_pattern']=df_org ['run_id'].apply (lambda x :4 if x %4 ==0 else x %4 )
    mismatch =df_org [df_org ['pattern_id']!=df_org ['expected_pattern']]
    if mismatch .empty :
        log_ok ("Pattern IDs are consistently assigned.")
    else :
        log_error (f"Pattern ID mismatches found in {len (mismatch )} rows.")

def check_value_ranges (df_org ,df_dept ):
    """Validation check of numerical range"""
    log_info ("--- Checking Value Ranges ---")

    # SSR Indicator (1.0 - 5.0)
    # ※ Because the data is averaged in the aggregation, there may be slight rounding errors, but it should be 1.0 <= x <= 5.0
    ssr_cols_org =[c for c in df_org .columns if 'Ave'in c or 'Score'in c ]
    # Since things like the CCC Score range from 0 to 1, I want to exclude them, but I can distinguish them by name.

    # 1. SSR System of Organizational Dataset
    # Ave P, Ave CT, ..., CC-Ave ...
    target_ssr =[c for c in df_org .columns if ('Ave'in c )and ('CCC'not in c )and ('CDC'not in c )]
    for col in target_ssr :
        vmin =df_org [col ].min ()
        vmax =df_org [col ].max ()
        if pd .isna (vmin ):continue # Separate check if all are NaN

        if vmin >=1.0 and vmax <=5.0 :
            pass # OK
        else :
            log_warning (f"Column '{col }' out of Likert range (1-5): Min={vmin :.2f}, Max={vmax :.2f}")

            # 2. Conversation Score (CCC): 0.0 - 1.0
    if 'CCC Score'in df_org .columns :
        ccc =df_org ['CCC Score'].dropna ()
        if ccc .min ()>=0.0 and ccc .max ()<=1.0001 :# Floating-point error tolerance
            log_ok ("CCC Score is within range [0, 1].")
        else :
            log_error (f"CCC Score out of range: Min={ccc .min ()}, Max={ccc .max ()}")

            # 3. SSR System of the Department Dataset
    target_ssr_dept =[c for c in df_dept .columns if ('Ave'in c )and ('CDC'not in c )]
    for col in target_ssr_dept :
        vmin =df_dept [col ].min ()
        vmax =df_dept [col ].max ()
        if pd .isna (vmin ):continue 

        if vmin >=1.0 and vmax <=5.0 :
            pass 
        else :
            log_warning (f"Dept Column '{col }' out of Likert range (1-5): Min={vmin :.2f}, Max={vmax :.2f}")

            # 4. CDC Score : 0.0 - 1.0
    if 'CDC Score'in df_dept .columns :
        cdc =df_dept ['CDC Score'].dropna ()
        if cdc .empty :
             log_warning ("CDC Score is all NaN.")
        elif cdc .min ()>=0.0 and cdc .max ()<=1.0001 :
            log_ok ("CDC Score is within range [0, 1].")
        else :
            log_error (f"CDC Score out of range: Min={cdc .min ()}, Max={cdc .max ()}")

            # 5. adj-CDC Score: 0.0 or higher (It is rare to exceed 1.0 in theory, but it is possible by definition)
    if 'adj-CDC Score'in df_dept .columns :
        adj =df_dept ['adj-CDC Score'].dropna ()
        if adj .empty :
            pass # In cases where there is only Pattern 1
        elif adj .min ()<0 :
             log_error (f"adj-CDC Score is negative: Min={adj .min ()}")
        else :
             log_ok (f"adj-CDC Score valid (Min={adj .min ():.2f}, Max={adj .max ():.2f})")

def check_simulation_logic (df_dept ):
    """Logic check of simulation assumptions (especially discontinuities)"""
    log_info ("--- Checking Simulation Logic (Pattern 1 Isolation) ---")

    # Extract data of Pattern 1
    p1_dept =df_dept [df_dept ['pattern_id']==1 ]

    if p1_dept .empty :
        log_warning ("No Pattern 1 data found to check isolation.")
        return 

        # 1. For Pattern 1, the CDC Score must not be 0.
        # (Conversations with outsiders must not occur)
    if 'CDC Score'in p1_dept .columns :
        cdc_p1 =p1_dept ['CDC Score'].fillna (0 )# If it's NaN, it can be regarded as no conversation = 0, but an explicit 0 is preferable.
        if (cdc_p1 ==0 ).all ():
            log_ok ("Pattern 1 Isolation Confirmed: All CDC Scores are 0.")
        else :
            violations =cdc_p1 [cdc_p1 >0 ]
            log_error (f"Pattern 1 Isolation VIOLATION: Found {len (violations )} rows with CDC > 0.")
            print (f"   Example violations (RunID): {p1_dept .loc [violations .index ,'run_id'].values [:5 ]}")

            # 2. For Pattern 1, the adj-CDC Score must be NaN (Undefined)
    if 'adj-CDC Score'in p1_dept .columns :
        adj_p1 =p1_dept ['adj-CDC Score']
        if adj_p1 .isna ().all ():
            log_ok ("Pattern 1 adj-CDC definition Confirmed: All values are NaN.")
        else :
            not_nan =adj_p1 .dropna ()
            log_error (f"Pattern 1 adj-CDC VIOLATION: Found {len (not_nan )} rows with values (should be NaN).")

def check_missing_values (df_org ):
    """Check for unexpected missing values"""
    log_info ("--- Checking Missing Values ---")

    # View the main average columns of organizational data
    cols_to_check =[c for c in df_org .columns if 'Ave'in c ]
    for col in cols_to_check :
        missing =df_org [col ].isna ().sum ()
        if missing >0 :
            log_warning (f"Column '{col }' has {missing } missing values.")
        else :
            pass # OK

            # Check if there are any sequences with a standard deviation (Sigma) of 0 (all values are the same = possibly dead as a simulation)
    cols_sigma =[c for c in df_org .columns if 'CCH'in c or 'CCTF'in c ]
    for col in cols_sigma :
        if (df_org [col ]==0 ).all ():
            log_warning (f"Column '{col }' is ALL ZERO. (No variance detected across all runs)")

def main ():
    print ("==========================================")
    print ("   SSR DATA HEALTH CHECKER v1.0")
    print ("==========================================")

    df_org ,df_dept =load_datasets ()

    if df_org is not None and df_dept is not None :
        check_structure_and_ids (df_org ,df_dept )
        check_value_ranges (df_org ,df_dept )
        check_simulation_logic (df_dept )
        check_missing_values (df_org )

    print ("\n==========================================")
    print (f" CHECK COMPLETED.")
    print (f" Errors:   {ERROR_COUNT }")
    print (f" Warnings: {WARNING_COUNT }")
    print ("==========================================")

    if ERROR_COUNT ==0 :
        print ("\n>>> DATASET STATUS: \033[92mHEALTHY (Ready for Analysis)\033[0m")
    else :
        print ("\n>>> DATASET STATUS: \033[91mCRITICAL ISSUES FOUND\033[0m")
        print ("Please review the errors above.")

if __name__ =="__main__":
    main ()