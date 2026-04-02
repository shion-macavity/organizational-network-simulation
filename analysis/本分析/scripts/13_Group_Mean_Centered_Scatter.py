import os 
import sys 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import seaborn as sns 
from scipy import stats 
import warnings 
import matplotlib .font_manager as fm 

warnings .filterwarnings ("ignore")

# =========================================================
# Path Settings
# =========================================================
# analysis/ThisAnalysis/scripts/13_... -> Load analysis/ThisAnalysis/config_analysis.py
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Font Settings
# =========================================================
def setup_japanese_font ():
    """Settings to reliably display Japanese in a Windows environment"""
    try :
        import japanize_matplotlib 
        return 
    except ImportError :
        pass 

    candidate_fonts =[
    'MS Gothic','MS Mincho','Meiryo','Yu Gothic',
    'Hiragino Sans','AppleGothic',
    'Noto Sans CJK JP','IPAGothic'
    ]

    for font_name in candidate_fonts :
        try :
            fm .findfont (font_name ,fallback_to_default =False )
            plt .rcParams ['font.family']=font_name 
            break 
        except :
            continue 

            # =========================================================
            # Data loading and merging process
            # =========================================================
def load_micro_data ():
    print (">>> Loading Indices Datasets...")
    path_comm =os .path .join (conf .OUTPUT_INDICES ,"community_indices.csv")
    path_indiv =os .path .join (conf .OUTPUT_INDICES ,"individual_indices.csv")
    path_dept =os .path .join (conf .OUTPUT_INDICES ,"department_indices.csv")

    if not all (os .path .exists (p )for p in [path_comm ,path_indiv ,path_dept ]):
        print ("[ERROR] Indices csv not found. Please run previous steps.")
        sys .exit (1 )

    df_comm =pd .read_csv (path_comm ,encoding =conf .ENCODING )
    df_indiv =pd .read_csv (path_indiv ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    # --- Department Data Column Name Adjustment and Joining with Personal Data ---
    # Link DGT and CDU in department_indices.csv to personal data
    # Normalization is needed because the column name can be either 'ID-Ave DGT' or 'DGT'.
    rename_map ={}
    target_vars =['DGT','CDU']

    for col in df_dept .columns :
        for v in target_vars :
        # Detect 'ID-Ave DGT' or 'DGT'
            if v in col :
            # If the name is already clean, leave it as is; if there are modifiers, shorten it.
                if col !=v :
                    rename_map [col ]=v 

    if rename_map :
        df_dept =df_dept .rename (columns =rename_map )

        # Join key: run_id, dept_id
        # Extract and combine only the necessary columns
    cols_to_merge =['run_id','dept_id']+[v for v in target_vars if v in df_dept .columns ]

    # Combine personal data with department indicators (this allows you to see correlations such as an individual's P and the department's DGT)
    df_indiv_merged =df_indiv .merge (df_dept [cols_to_merge ],on =['run_id','dept_id'],how ='left')

    return df_comm ,df_indiv_merged 

    # =========================================================
    # Execute analysis
    # =========================================================
OUTPUT_ROOT =os .path .join (conf .ANALYSIS_DIR ,"04_散布図","13_組織内偏差による散布図")# Folder/File: 13_Scatter Plot Due to Organizational Deviations | Folder/File: 13_Scatter Plot Due to Organizational Deviations | Folder/File: 13_組織内偏差による散布図

def plot_centered_scatter (df ,x_col ,y_col ,group_col ,sub_folder ):
    """
    Perform mean centering within the specified group (run_id) and plot a scatter diagram of deviations within the organization.
    """
    # Defect removal
    data =df [[x_col ,y_col ,group_col ,'pattern_id']].dropna ().copy ()
    if len (data )<10 :return 

    # Centering Process (Within-Group Centering)
    # Subtract the average value of that organization from the value
    data [f'{x_col }_centered']=data .groupby (group_col )[x_col ].transform (lambda x :x -x .mean ())
    data [f'{y_col }_centered']=data .groupby (group_col )[y_col ].transform (lambda x :x -x .mean ())

    x_plot =f'{x_col }_centered'
    y_plot =f'{y_col }_centered'

    plt .figure (figsize =(8 ,6 ))

    # Scatter plot
    sns .scatterplot (data =data ,x =x_plot ,y =y_plot ,hue ='pattern_id',palette ='bright',
    style ='pattern_id',alpha =0.5 ,s =60 )

    # Regression line (overall) - the slope here is the 'pure micro correlation'
    sns .regplot (data =data ,x =x_plot ,y =y_plot ,scatter =False ,color ='black',ci =95 )

    # Calculation of statistics
    slope ,intercept ,r_value ,p_value ,std_err =stats .linregress (data [x_plot ],data [y_plot ])
    r_squared =r_value **2 

    significance =""
    if p_value <0.001 :significance ="***"
    elif p_value <0.01 :significance ="**"
    elif p_value <0.05 :significance ="*"
    else :significance ="(n.s.)"
    sign ="+"if intercept >=0 else ""

    stats_text =(f"Within-Group Correlation\n"
    f"$y = {slope :.3f}x {sign } {intercept :.3f}$\n"
    f"$N = {len (data )}$ (Centered)\n"
    f"$r = {r_value :.3f}$ ($R^2={r_squared :.3f}$)\n"
    f"$p = {p_value :.3e}$ {significance }")

    plt .text (0.05 ,0.95 ,stats_text ,transform =plt .gca ().transAxes ,
    fontsize =11 ,verticalalignment ='top',
    bbox =dict (boxstyle ='round',facecolor ='white',alpha =0.9 ))

    plt .title (f"Centered: {y_col } vs {x_col }\n(Removing Organization-Level Differences)",fontsize =13 )
    plt .xlabel (f"Deviation of {x_col } (from Run Mean)")
    plt .ylabel (f"Deviation of {y_col } (from Run Mean)")
    plt .grid (True ,linestyle ='--',alpha =0.5 )
    plt .legend (title ='Pattern',loc ='lower right')

    save_dir =os .path .join (OUTPUT_ROOT ,sub_folder ,f"X_{x_col }")
    os .makedirs (save_dir ,exist_ok =True )

    fname =f"{x_col }_vs_{y_col }_centered.png"
    plt .savefig (os .path .join (save_dir ,fname ),dpi =100 ,bbox_inches ='tight')
    plt .close ()

def main ():
    setup_japanese_font ()
    df_comm ,df_indiv =load_micro_data ()

    # --- 1. Community Level (N ≈ 1000) ---
    print ("--- Processing Community Level (Centered) ---")
    comm_vars =['P','CT','RD','CGT','CCU']
    # Extract only the columns that exist in the dataset
    comm_vars =[v for v in comm_vars if v in df_comm .columns ]

    # Centering each run and doing a round-robin
    for x in comm_vars :
        for y in comm_vars :
            if x ==y :continue 
            plot_centered_scatter (df_comm ,x ,y ,'run_id',"01_Community_Level")

            # --- 2. Individual Level (N ≈ 1400) ---
    print ("--- Processing Individual Level (Centered) ---")
    indiv_vars =['P','CT','RD','DGT','CDU']
    # Extract only the columns that exist in the dataset after merging
    indiv_vars =[v for v in indiv_vars if v in df_indiv .columns ]

    # To observe the macro (organization-average) paradox, individual data is also centered by 'run_id'
    for x in indiv_vars :
        for y in indiv_vars :
            if x ==y :continue 
            plot_centered_scatter (df_indiv ,x ,y ,'run_id',"02_Individual_Level")

    print (f"\n>>> Centered Scatter Plots saved in {OUTPUT_ROOT }")

if __name__ =="__main__":
    main ()