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
# Common function (same logic as Step 1)
# =========================================================

def get_pattern_id (run_id ):
    """Determine the pattern ID (1-4) from the RunID"""
    remainder =run_id %4 
    if remainder ==0 :
        return 4 
    else :
        return remainder 

        # =========================================================
        # Indicator calculation logic
        # =========================================================

def load_data ():
    """Load the preprocessed data"""
    print (">>> Loading Preprocessed Data...")

    # 1. Raw SSR Metrics Data (P, CT, RD, PTS)
    path_metrics =os .path .join (conf .OUTPUT_PREPROCESS ,"all_metrics_raw.csv")
    if not os .path .exists (path_metrics ):
        raise FileNotFoundError (f"{path_metrics } not found. Run 01_preprocess_data.py first.")
    df_metrics =pd .read_csv (path_metrics ,encoding =conf .ENCODING )

    # 2. Conversation Structure Data (I, E counts)
    path_struct =os .path .join (conf .OUTPUT_PREPROCESS ,"conversation_counts.csv")
    if not os .path .exists (path_struct ):
        raise FileNotFoundError (f"{path_struct } not found.")
    df_struct =pd .read_csv (path_struct ,encoding =conf .ENCODING )

    # 3. Community Assignment Table (for Attribute Combination)
    if not os .path .exists (conf .PATH_COMMUNITY ):
        raise FileNotFoundError (f"{conf .PATH_COMMUNITY } not found.")
    df_comm =pd .read_csv (conf .PATH_COMMUNITY ,encoding =conf .ENCODING )

    return df_metrics ,df_struct ,df_comm 

def calculate_individual_indices (df_metrics ,df_comm ):
    """
    Calculate individual unit metrics (P, CT, RD, GT, U).
Return: pd.DataFrame (index: run_id, turn, agent_id)
    """
    print (">>> Calculating Individual Indices (P, CT, RD, GT, U)...")

    # --- 1. Average Turn Score of Basic Scores (P, CT, RD) ---
    # Formula: X_{i,t} = mean(RX_{i,r})

    target_types =['P','CT','RD']
    df_basic =df_metrics [df_metrics ['metric_type'].isin (target_types )].copy ()

    # Group and average
    df_ind_scores =df_basic .groupby (['run_id','turn','agent_id','metric_type'])['ssr_score'].mean ().unstack ()

    for col in target_types :
        if col not in df_ind_scores .columns :
            df_ind_scores [col ]=np .nan 

    df_ind_scores =df_ind_scores .reset_index ()

    # --- 2. Calculation of Similarity Score (GT, U) ---
    df_pts =df_metrics [df_metrics ['metric_type']=='PTS'].copy ()
    # Excluding comparison with oneself (j != i)
    df_pts =df_pts [df_pts ['agent_id']!=df_pts ['target_id']]

    if df_pts .empty :
        print ("Warning: No PTS data found or all filtered out.")
        for col in ['CGT','CCU','DGT','CDU']:
            df_ind_scores [col ]=np .nan 
        return df_ind_scores 

        # Department ID Mapping
    dept_map ={}
    for dept_id ,members in conf .DEPT_DEFINITIONS .items ():
        for m in members :
            dept_map [m ]=dept_id 

            # Formatted for merging
    df_pts =df_pts .merge (df_comm [['run_id','agent_id','community_id']],
    on =['run_id','agent_id'],how ='left')
    df_pts =df_pts .rename (columns ={'community_id':'comm_id_a'})
    df_pts ['dept_id_a']=df_pts ['agent_id'].map (dept_map )

    df_pts =df_pts .merge (df_comm [['run_id','agent_id','community_id']].rename (columns ={'agent_id':'target_id'}),
    on =['run_id','target_id'],how ='left')
    df_pts =df_pts .rename (columns ={'community_id':'comm_id_b'})
    df_pts ['dept_id_b']=df_pts ['target_id'].map (dept_map )

    # --- 2a. Community Perspective (CGT, CCU) ---
    df_pts ['is_same_comm']=(df_pts ['comm_id_a']==df_pts ['comm_id_b'])
    pts_agg_comm =df_pts .groupby (['run_id','turn','agent_id','is_same_comm'])['ssr_score'].mean ().unstack ()

    if True in pts_agg_comm .columns :
        pts_agg_comm ['CGT']=pts_agg_comm [True ]
    else :
        pts_agg_comm ['CGT']=np .nan 

    if False in pts_agg_comm .columns :
    # Formula: U = (L+1) - Ave(External)
        pts_agg_comm ['CCU']=(conf .LIKERT_SCALE_MAX +1 )-pts_agg_comm [False ]
    else :
        pts_agg_comm ['CCU']=np .nan 

    pts_agg_comm =pts_agg_comm [['CGT','CCU']].reset_index ()

    # --- 2b. Department Perspective (DGT, CDU) ---
    df_pts ['is_same_dept']=(df_pts ['dept_id_a']==df_pts ['dept_id_b'])
    pts_agg_dept =df_pts .groupby (['run_id','turn','agent_id','is_same_dept'])['ssr_score'].mean ().unstack ()

    if True in pts_agg_dept .columns :
        pts_agg_dept ['DGT']=pts_agg_dept [True ]
    else :
        pts_agg_dept ['DGT']=np .nan 

    if False in pts_agg_dept .columns :
        pts_agg_dept ['CDU']=(conf .LIKERT_SCALE_MAX +1 )-pts_agg_dept [False ]
    else :
        pts_agg_dept ['CDU']=np .nan 

    pts_agg_dept =pts_agg_dept [['DGT','CDU']].reset_index ()

    # --- Integration ---
    df_final =df_ind_scores .merge (pts_agg_comm ,on =['run_id','turn','agent_id'],how ='left')
    df_final =df_final .merge (pts_agg_dept ,on =['run_id','turn','agent_id'],how ='left')

    df_final =df_final .merge (df_comm ,on =['run_id','agent_id'],how ='left')
    df_final ['dept_id']=df_final ['agent_id'].map (dept_map )

    return df_final 

def calculate_structure_indices (df_struct ):
    """  # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File: 
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    
    構造指標 (CCC, CDC, adj-CDC) を計算する。
    Return: pd.DataFrame (Community/Dept単位)
    """
    print (">>> Calculating Structure Indices (CCC, CDC, adj-CDC)...")

    # Calculate PatternID from RunID
    df_struct ['pattern_id']=df_struct ['run_id'].apply (get_pattern_id )

    # --- 1. Existing Indicators (CCC / CDC) ---
    # Formula: Ratio = E / (E + I)
    df_struct ['total_conv']=df_struct ['E_count']+df_struct ['I_count']

    df_struct ['structure_score']=np .where (
    df_struct ['total_conv']>0 ,
    df_struct ['E_count']/df_struct ['total_conv'],
    np .nan 
    )

    # --- 2. Adjusted Indicators (adj-CDC) ---
    # Definition: adj-CDC = E / E_max_capacity
    # Pattern 1 -> Undefined (NaN)
    # Pattern 2,3 -> 2
    # Pattern 4 -> 4

    def get_max_capacity (pattern_id ):
        if pattern_id ==1 :
            return np .nan 
        elif pattern_id in [2 ,3 ]:
            return 2.0 
        elif pattern_id ==4 :
            return 4.0 
        return np .nan # Unexpected

    df_struct ['E_max_cap']=df_struct ['pattern_id'].apply (get_max_capacity )

    # adj_CDC calculation (only meaningful for departments, but add it as a column anyway)
    # ※ If E_max_cap is NaN, the result will also be NaN, which is convenient.
    df_struct ['adj_score']=df_struct ['E_count']/df_struct ['E_max_cap']

    return df_struct 

def main ():
# Create folder
    os .makedirs (conf .OUTPUT_INDICES ,exist_ok =True )

    # 1. Data Loading
    df_metrics ,df_struct ,df_comm =load_data ()

    # 2. Individual Indicator Calculation
    df_individual =calculate_individual_indices (df_metrics ,df_comm )
    path_ind =os .path .join (conf .OUTPUT_INDICES ,"individual_indices.csv")
    df_individual .to_csv (path_ind ,index =False ,encoding =conf .ENCODING )
    print (f"Saved Individual Indices to: {path_ind }")

    # 3. Structural Indicator Calculation (Community / Department)
    df_structure =calculate_structure_indices (df_struct )

    # Store separately

    # --- For Community Use (CCC) ---
    # Excluded because adj-CDC is not defined in the community
    df_ccc =df_structure [df_structure ['unit_type']=='community'].copy ()
    df_ccc =df_ccc .rename (columns ={'structure_score':'CCC','unit_id':'community_id'})
    # Save only the necessary columns
    cols_ccc =['run_id','turn','community_id','CCC','I_count','E_count']
    path_ccc =os .path .join (conf .OUTPUT_INDICES ,"community_indices.csv")
    df_ccc [cols_ccc ].to_csv (path_ccc ,index =False ,encoding =conf .ENCODING )
    print (f"Saved Community Indices (CCC) to: {path_ccc }")

    # --- For Department Use (CDC, adj-CDC) ---
    df_cdc =df_structure [df_structure ['unit_type']=='department'].copy ()
    df_cdc =df_cdc .rename (columns ={
    'structure_score':'CDC',
    'adj_score':'adj_CDC',
    'unit_id':'dept_id'
    })
    # Save only the necessary columns
    cols_cdc =['run_id','turn','dept_id','CDC','adj_CDC','I_count','E_count','pattern_id']
    path_cdc =os .path .join (conf .OUTPUT_INDICES ,"department_indices.csv")
    df_cdc [cols_cdc ].to_csv (path_cdc ,index =False ,encoding =conf .ENCODING )
    print (f"Saved Department Indices (CDC, adj-CDC) to: {path_cdc }")

    print (">>> Step 2 Completed.")

if __name__ =="__main__":
    main ()