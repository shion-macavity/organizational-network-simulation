import os 
import sys 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import seaborn as sns 
from scipy import stats 
import warnings 
import matplotlib .font_manager as fm 

# Warning suppression
warnings .filterwarnings ("ignore")

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Font Settings
# =========================================================
def setup_japanese_font ():
    try :
        import japanize_matplotlib 
        return 
    except ImportError :
        pass 

    candidate_fonts =['MS Gothic','MS Mincho','Meiryo','Yu Gothic',
    'Hiragino Sans','Noto Sans CJK JP','IPAGothic']
    for font_name in candidate_fonts :
        try :
            fm .findfont (font_name ,fallback_to_default =False )
            plt .rcParams ['font.family']=font_name 
            return 
        except :
            continue 
    plt .rcParams ['font.family']='sans-serif'

    # =========================================================
    # Data loading, merging, and checking
    # =========================================================
def load_and_prep_data ():
    print (">>> [Data Prep] Loading Indices Data...")

    # Check file path
    path_ind =os .path .join (conf .OUTPUT_INDICES ,"individual_indices.csv")
    path_comm =os .path .join (conf .OUTPUT_INDICES ,"community_indices.csv")
    path_dept =os .path .join (conf .OUTPUT_INDICES ,"department_indices.csv")

    if not os .path .exists (path_ind ):
        print (f"[ERROR] File not found: {path_ind }")
        sys .exit (1 )

        # Loading
    df_ind =pd .read_csv (path_ind ,encoding =conf .ENCODING )
    df_comm =pd .read_csv (path_comm ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )if os .path .exists (path_dept )else pd .DataFrame ()

    print (f"    Loaded Individual: {df_ind .shape }")
    print (f"    Loaded Community : {df_comm .shape }")

    # --- 1. Data for Organizational Analysis (Unit: Community) ---
    print (">>> [Data Prep] Aggregating for Organization Analysis (Unit: Community)...")

    # Average individual indicators by 'Run x Community' unit
    # First, take the average per turn, and then take the individual averages within the community,
    # Or should we average all the data at once? Here, since it is a 'score as a community,'
    # Adopt the average of all agents and all turns for each (Run, Community).

    # Required column
    cols_metrics =['P','CT','RD','CGT','CCU']
    available_metrics =[c for c in cols_metrics if c in df_ind .columns ]

    # GroupBy
    df_org_metrics =df_ind .groupby (['run_id','community_id'],as_index =False )[available_metrics ].mean ()

    # Combination of CCC (structural indicators)
    # Since df_comm is (run_id, turn, community_id), this should also be averaged to (run_id, community_id)
    if not df_comm .empty :
        df_comm_grouped =df_comm .groupby (['run_id','community_id'],as_index =False )['CCC'].mean ()
        # Combination
        df_org_analysis =pd .merge (df_org_metrics ,df_comm_grouped ,on =['run_id','community_id'],how ='inner')
    else :
        df_org_analysis =df_org_metrics 
        print ("[WARNING] Community indices (CCC) is empty.")

    print (f"    Organization Analysis Data: {df_org_analysis .shape } records.")

    # --- 2. Data for Department Analysis (Unit: Individual) ---
    print (">>> [Data Prep] Aggregating for Department Analysis (Unit: Agent)...")

    # Average individual indicators per 'Run x Agent' (Turn average)
    cols_ind =['P','CT','RD','DGT','CDU']
    available_ind =[c for c in cols_ind if c in df_ind .columns ]

    # Since we also want to keep dept_id, assume it is uniquely determined by agent_id and either exclude it from the averaging target or include it in the groupby.
    df_ind_analysis =df_ind .groupby (['run_id','agent_id','dept_id'],as_index =False )[available_ind ].mean ()

    # Combination of structural indicators (CDC)
    # df_dept is averaged from (run_id, turn, dept_id) to (run_id, dept_id)
    if not df_dept .empty :
        df_dept_grouped =df_dept .groupby (['run_id','dept_id'],as_index =False )[['CDC','adj_CDC']].mean ()
        # Linking (associating the individual's department CDC)
        df_ind_analysis =pd .merge (df_ind_analysis ,df_dept_grouped ,on =['run_id','dept_id'],how ='left')

        # Create Group ID (for line drawing during plotting: Run_Dept)
    df_ind_analysis ['group_id']=df_ind_analysis ['run_id'].astype (str )+"_d"+df_ind_analysis ['dept_id'].astype (str )

    print (f"    Department Analysis Data: {df_ind_analysis .shape } records.")

    return df_org_analysis ,df_ind_analysis 

    # =========================================================
    # plot function
    # =========================================================
def plot_simpson_v2 (df ,x_col ,y_col ,group_col ,output_dir ,unit_desc =""):
    """
    Plot all the data on a single figure and draw a regression line for each specified group_col.
    """
    # Data Check
    if x_col not in df .columns or y_col not in df .columns :
        return 

    data =df .dropna (subset =[x_col ,y_col ])
    if len (data )<5 :
        return 

        # Create output destination
    os .makedirs (output_dir ,exist_ok =True )

    plt .figure (figsize =(10 ,8 ))

    # 1. Overall scatter plot
    sns .scatterplot (
    data =data ,x =x_col ,y =y_col ,
    color ='blue',s =30 ,alpha =0.4 ,edgecolor =None ,label ='Sample'
    )

    # 2. Regression line for each group (Simpson's Paradox Check)
    # If there are too many lines, it will be heavy, so check the number of data points.
    unique_groups =data [group_col ].unique ()

    # Draw line (loop)
    # The color will be light gray or random, but this time it will be light red to emphasize 'many lines'.
    line_count =0 
    for grp in unique_groups :
        d_sub =data [data [group_col ]==grp ]
        if len (d_sub )<3 :# Do not draw a line if there are too few points.
            continue 

        if d_sub [x_col ].nunique ()<2 :# If X is not dispersed
            continue 

        try :
            slope ,intercept ,_ ,_ ,_ =stats .linregress (d_sub [x_col ],d_sub [y_col ])
            x_vals =np .array ([d_sub [x_col ].min (),d_sub [x_col ].max ()])
            y_vals =slope *x_vals +intercept 
            plt .plot (x_vals ,y_vals ,color ='red',alpha =0.15 ,linewidth =1 )
            line_count +=1 
        except :
            pass 

            # 3. Overall regression line (thick black)
    try :
        slope_all ,intercept_all ,r_val ,p_val ,_ =stats .linregress (data [x_col ],data [y_col ])
        x_range =np .array ([data [x_col ].min (),data [x_col ].max ()])
        plt .plot (x_range ,slope_all *x_range +intercept_all ,
        color ='black',linestyle ='--',linewidth =2.5 ,
        label =f'Global Trend (R={r_val :.2f})')

        # Title Construction
        title_text =f"{y_col } vs {x_col }\n"
        title_text +=f"N={len (data )} ({len (unique_groups )} groups)\n"
        title_text +=f"Simpson's Check: Red lines = Regression per {unit_desc }"

        plt .title (title_text ,fontsize =14 )

    except :
        plt .title (f"{y_col } vs {x_col } (N={len (data )})",fontsize =14 )

    plt .xlabel (x_col ,fontsize =12 )
    plt .ylabel (y_col ,fontsize =12 )
    plt .grid (True ,linestyle ='--',alpha =0.5 )
    plt .legend (loc ='best')

    # Save
    fname =f"{x_col }_vs_{y_col }.png".replace (" ","_")
    plt .savefig (os .path .join (output_dir ,fname ),dpi =100 ,bbox_inches ='tight')
    plt .close ()

    # =========================================================
    # Main processing
    # =========================================================
def main ():
    setup_japanese_font ()

    # Data Preparation
    df_org ,df_dept =load_and_prep_data ()

    # Save Destination Root (analysis/This Analysis/...)
    # This script is in analysis/honbunseki/scripts/,
    # Use parent_dir (analysis/hon bunseki/) as is
    base_dir =parent_dir 

    # =========================================================
    # 1. Intra-organizational community analysis (points represent Community, lines represent Run)
    # =========================================================
    print ("--- Generating Plots: Organization (Communities) ---")
    dir_org =os .path .join (base_dir ,"12_組織内コミュニティ分析")# Folder/File: 12_Analysis of Intra-Organizational Communities | Folder/File: 12_Analysis of Intra-Organizational Communities | Folder/File: 12_組織内コミュニティ分析

    # 1-A. CCC vs Metrics
    dir_ccc =os .path .join (dir_org ,"01_CCC_vs_Metrics")
    metrics_org =['P','CT','RD','CGT','CCU']

    if 'CCC'in df_org .columns :
        for m in metrics_org :
            plot_simpson_v2 (df_org ,'CCC',m ,'run_id',dir_ccc ,unit_desc ="Run")

            # 1-B. Metrics vs Metrics (Round-robin)
    dir_matrix_org =os .path .join (dir_org ,"02_Metrics_vs_Metrics")
    for i ,x in enumerate (metrics_org ):
        for j ,y in enumerate (metrics_org ):
            if i >=j :continue # Deduplication
            plot_simpson_v2 (df_org ,x ,y ,'run_id',dir_matrix_org ,unit_desc ="Run")

            # =========================================================
            # 2. Individual Analysis within the Department (Points represent Agents, Lines represent Dept)
            # =========================================================
    print ("--- Generating Plots: Department (Individuals) ---")
    dir_dept_anal =os .path .join (base_dir ,"13_部署内個人分析")# Folder/File: 13_Individual Analysis within the Department | Folder/File: 13_Individual Analysis within the Department | Folder/File: 13_部署内個人分析

    metrics_dept =['P','CT','RD','DGT','CDU']

    # 2-A. CDC vs Metrics (The relationship between individual metrics and the CDC of the department they belong to)
    dir_cdc =os .path .join (dir_dept_anal ,"01_CDC_vs_Metrics")
    # Only if CDC or adj_CDC exists
    for struct_var in ['CDC','adj_CDC']:
        if struct_var in df_dept .columns :
            for m in metrics_dept :
                plot_simpson_v2 (df_dept ,struct_var ,m ,'group_id',dir_cdc ,unit_desc ="Dept")

                # 2-B. Metrics vs Metrics (Round-robin)
    dir_matrix_dept =os .path .join (dir_dept_anal ,"02_Metrics_vs_Metrics")
    for i ,x in enumerate (metrics_dept ):
        for j ,y in enumerate (metrics_dept ):
            if i >=j :continue 
            plot_simpson_v2 (df_dept ,x ,y ,'group_id',dir_matrix_dept ,unit_desc ="Dept")

    print (f"\n>>> Analysis Completed.")
    print (f"Output Directories:")
    print (f"  - {dir_org }")
    print (f"  - {dir_dept_anal }")

if __name__ =="__main__":
    main ()