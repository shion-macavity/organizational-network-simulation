import os 
import sys 
import pandas as pd 
import numpy as np 
from tqdm import tqdm 

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Helper function for statistical calculations
# =========================================================

def safe_mean (series ):
    """Safe average calculation"""
    if series .empty or series .isna ().all ():
        return np .nan 
    return series .mean ()

def safe_std (series ):
    """Safe standard deviation calculation (ddof=1: unbiased variance)"""
    if series .empty or len (series .dropna ())<2 :
        return np .nan 
    return series .std (ddof =1 )

def get_pattern_id (run_id ):
    """Determine the pattern ID (1-4) from the RunID"""
    remainder =run_id %4 
    if remainder ==0 :
        return 4 
    else :
        return remainder 

        # =========================================================
        # Main processing
        # =========================================================

def load_indices ():
    """Load the metric data calculated in Step 2"""
    print (">>> Loading Indices Data...")

    path_ind =os .path .join (conf .OUTPUT_INDICES ,"individual_indices.csv")
    path_comm =os .path .join (conf .OUTPUT_INDICES ,"community_indices.csv")
    path_dept =os .path .join (conf .OUTPUT_INDICES ,"department_indices.csv")

    if not (os .path .exists (path_ind )and os .path .exists (path_comm )and os .path .exists (path_dept )):
        raise FileNotFoundError ("Indices files not found. Run 02_calculate_indices.py first.")

    df_ind =pd .read_csv (path_ind ,encoding =conf .ENCODING )
    df_comm =pd .read_csv (path_comm ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    return df_ind ,df_comm ,df_dept 

def create_organization_dataset (df_ind ,df_comm ):
    """
    Create a dataset from the organizational perspective (N=100)
    Unit: 1 Run
    """
    print (">>> Creating Organization Dataset (N=100)...")

    # Target variable (derived from individual indicators)
    # From an organizational perspective, GT and U use community metrics (CGT, CCU)
    target_vars =['P','CT','RD','CGT','CCU']

    results =[]

    run_ids =df_ind ['run_id'].unique ()

    for run_id in tqdm (run_ids ,desc ="Org Dataset"):
    # Filtering per run
        d_ind =df_ind [df_ind ['run_id']==run_id ]
        d_comm =df_comm [df_comm ['run_id']==run_id ]

        row ={
        'run_id':run_id ,
        'pattern_id':get_pattern_id (run_id )
        }

        # --- 1. Individual-derived Metric Aggregation (X) ---
        for var in target_vars :
            if var not in d_ind .columns :
                continue 

                # Data series
                # All turn data of all individuals
            vals_all =d_ind [var ]

            # a. Ave X (μ_X): Simple average
            row [f'Ave {var }']=safe_mean (vals_all )

            # --- Community-Based Aggregation ---
            # First, calculate the 'time average <X_C>' for each community C and the 'community average X_{C,t}' for each turn t.

            # GroupBy: community_id, turn
            # Average turns per community
            # (community_id, turn) -> mean
            grp_comm_turn =d_ind .groupby (['community_id','turn'])[var ].mean ()

            # Time average for each community <X_C>
            # (community_id) -> mean
            grp_comm_mean =d_ind .groupby ('community_id')[var ].mean ()

            # b. CC-Ave X (μ^{(com)}_{X,cross}): Time average of equally weighted average between communities
            # = (1/N_C) * sum( <X_C> )
            row [f'CC-Ave {var }']=safe_mean (grp_comm_mean )

            # c. CCH-X (\sigma^{(com)}_{X,cross}): Inter-community heterogeneity
            # = std( <X_C> )
            row [f'CCH-{var }']=safe_std (grp_comm_mean )

            # d. CCTF-X (σ^{(com)}_{X,time}): Temporal variability of the community mean
            # First, calculate the 'inter-community average \bar{X}^{(com)}_{t,cross}' for each turn.
            # turn -> mean( of community means )
            mean_cross_t =grp_comm_turn .groupby ('turn').mean ()
            # The standard deviation of that time series
            row [f'CCTF-{var }']=safe_std (mean_cross_t )

            # --- 2. Structural Indicators (CCC) ---
            # CCC Score (μ^{(com)}_{CCC}): The time average of the CCC for each community, averaged across communities
            # In d_comm, there is CCC for each turn and community_id.

            # First, calculate the time average CCC_C for each community.
        ccc_c =d_comm .groupby ('community_id')['CCC'].mean ()
        # That average
        row ['CCC Score']=safe_mean (ccc_c )

        results .append (row )

    df_org =pd .DataFrame (results )
    return df_org 

def create_department_dataset (df_ind ,df_dept ):
    """
    Create a dataset from the department perspective (N=200)
    Unit: 1 Dept (Run x Dept)
    """
    print (">>> Creating Department Dataset (N=200)...")

    # Target variable
    # From a departmental perspective, GT and U use internal departmental indicators (DGT, CDU)
    target_vars =['P','CT','RD','DGT','CDU']

    results =[]

    # Combination of Run x Dept
    # Get existing combinations from df_ind
    pairs =df_ind [['run_id','dept_id']].drop_duplicates ()

    for _ ,pair in tqdm (pairs .iterrows (),total =len (pairs ),desc ="Dept Dataset"):
        run_id =pair ['run_id']
        dept_id =pair ['dept_id']

        # Filtering
        d_ind =df_ind [(df_ind ['run_id']==run_id )&(df_ind ['dept_id']==dept_id )]
        d_dept =df_dept [(df_dept ['run_id']==run_id )&(df_dept ['dept_id']==dept_id )]

        row ={
        'run_id':run_id ,
        'dept_id':dept_id ,
        'pattern_id':get_pattern_id (run_id )
        }

        # --- 1. Individual-derived Metric Aggregation (X) ---
        for var in target_vars :
            if var not in d_ind .columns :
                continue 

            vals_all =d_ind [var ]

            # a. ID-Ave X (\mu^{(dep)}_X): Department-wide overall average
            row [f'ID-Ave {var }']=safe_mean (vals_all )

            # b. IDIH-X (σ^{(dep)}_{X,intra}): Intra-department individual heterogeneity
            # Calculate the time average <X_i> for each individual
            mean_i =d_ind .groupby ('agent_id')[var ].mean ()
            # that variation
            row [f'IDIH-{var }']=safe_std (mean_i )

            # c. IDTF-X (σ^(dep)_{X,time}): Intra-departmental time variability
            # Calculate the departmental average X_{D,t} for each turn
            mean_t =d_ind .groupby ('turn')[var ].mean ()
            # that variation
            row [f'IDTF-{var }']=safe_std (mean_t )

            # --- 2. Structural Indicators (CDC, adj-CDC) ---
            # Take the time average
            # d_dept has CDC and adj_CDC for each turn

        row ['CDC Score']=safe_mean (d_dept ['CDC'])
        row ['adj-CDC Score']=safe_mean (d_dept ['adj_CDC'])

        results .append (row )

    df_final_dept =pd .DataFrame (results )
    return df_final_dept 

def main ():
# Create folder
    os .makedirs (conf .OUTPUT_DATASETS ,exist_ok =True )

    # 1. Data Loading
    df_ind ,df_comm ,df_dept =load_indices ()

    # 2. Creation of Organization Dataset
    df_org =create_organization_dataset (df_ind ,df_comm )
    path_org =os .path .join (conf .OUTPUT_DATASETS ,"dataset_organization_N100.csv")
    df_org .to_csv (path_org ,index =False ,encoding =conf .ENCODING )
    print (f"Saved Organization Dataset to: {path_org }")
    print (f"Columns: {list (df_org .columns )}")

    # 3. Department Dataset Creation
    df_dept_ds =create_department_dataset (df_ind ,df_dept )
    path_dept_ds =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")
    df_dept_ds .to_csv (path_dept_ds ,index =False ,encoding =conf .ENCODING )
    print (f"Saved Department Dataset to: {path_dept_ds }")
    print (f"Columns: {list (df_dept_ds .columns )}")

    print (">>> Step 3 (Final) Completed.")

if __name__ =="__main__":
    main ()