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
    candidate_fonts =['MS Gothic','MS Mincho','Meiryo','Yu Gothic','Hiragino Sans','Noto Sans CJK JP']
    for font_name in candidate_fonts :
        try :
            fm .findfont (font_name ,fallback_to_default =False )
            plt .rcParams ['font.family']=font_name 
            break 
        except :
            continue 

            # =========================================================
            # Data Loading
            # =========================================================
def load_data ():
    print (">>> Loading Datasets...")
    path_org =os .path .join (conf .OUTPUT_DATASETS ,"dataset_organization_N100.csv")
    path_dept =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")

    if not os .path .exists (path_org )or not os .path .exists (path_dept ):
        print ("[ERROR] Dataset not found.")
        sys .exit (1 )

    df_org =pd .read_csv (path_org ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    # DGT and CDU integration into organizational data
    dept_cols =[]
    if 'ID-Ave DGT'in df_dept .columns :dept_cols .append ('ID-Ave DGT')
    if 'ID-Ave CDU'in df_dept .columns :dept_cols .append ('ID-Ave CDU')
    if dept_cols :
        df_dept_mean =df_dept .groupby ('run_id')[dept_cols ].mean ().reset_index ()
        rename_map ={c :c .replace ('ID-Ave','Ave')for c in dept_cols }
        df_dept_mean =df_dept_mean .rename (columns =rename_map )
        df_org =df_org .merge (df_dept_mean ,on ='run_id',how ='left')

    return df_org ,df_dept 

    # =========================================================
    # Statistical Calculation Helper
    # =========================================================
def get_anova_label (df ,col ):
    """
    Perform ANOVA on the specified column across pattern_id, and return a label string including the result. Example: "Ave P" -> "Ave P (p<.001 ***)"
    """
    # Defect removal (for adj-CDC, etc.)
    # --- CDU/CDC response: N=200 -> N=100 (average of 2 departments) ---
    current_df =df 
    if ("CDU"in col or "CDC"in col )and 'dept_id'in df .columns and df ['dept_id'].nunique ()>1 :
        if 'run_id'in df .columns :
            current_df =df .groupby (['run_id','pattern_id'],as_index =False )[col ].mean ()
        else :
            current_df =df [df ['dept_id']==0 ]# fallback

    data =current_df [['pattern_id',col ]].dropna ()
    # ---------------------------------------------------------------------------------

    # Cannot calculate unless there are two or more groups to compare
    if len (data ['pattern_id'].unique ())<2 :
        return f"{col } (N/A)"

    groups =[]
    patterns =sorted (data ['pattern_id'].unique ())
    for p in patterns :
        groups .append (data [data ['pattern_id']==p ][col ].values )

    try :
        f_stat ,p_val =stats .f_oneway (*groups )

        # Number of stars
        if p_val <0.001 :sig ="***"
        elif p_val <0.01 :sig ="**"
        elif p_val <0.05 :sig ="*"
        else :sig ="n.s."

        # Notation of p-value
        if p_val <0.001 :
            p_text ="<.001"
        else :
            p_text =f"={p_val :.3f}"

        return f"{col }\n(p{p_text } {sig })"
    except :
        return f"{col } (Err)"

        # =========================================================
        # Transition graph drawing function (with statistics)
        # =========================================================
def create_trend_plot_with_stats (df ,target_cols ,title ,filename ,output_dir ,y_label ="Score"):
    """
    Combine multiple indicators into a line graph and display the ANOVA results in the legend
    """
    valid_cols =[c for c in target_cols if c in df .columns ]
    if not valid_cols :
        return 

        # 1. Create a mapping to replace the column name with 'with statistical information'
    col_mapping ={}
    for col in valid_cols :
        new_label =get_anova_label (df ,col )
        col_mapping [col ]=new_label 

        # 2. Create a data frame for drawing and rename it
        # --- [Modification] Temporarily retain dept_id and run_id for CDU/CDC-related filtering processing ---
    cols_to_use =['pattern_id']+valid_cols 
    if 'dept_id'in df .columns :cols_to_use .append ('dept_id')
    if 'run_id'in df .columns :cols_to_use .append ('run_id')

    df_plot =df [cols_to_use ].copy ()

    # In the case of CDU/CDC-related multiple departments (N=200), store the average value of the two departments in Dept0 and exclude Dept1 (NaN).
    if 'dept_id'in df_plot .columns and df_plot ['dept_id'].nunique ()>1 and 'run_id'in df_plot .columns :
        for col in valid_cols :
            if "CDU"in col or "CDC"in col :
            # Calculate the average for each run (align to the original index with transform)
                means =df_plot .groupby ('run_id')[col ].transform ('mean')

                # Substitute the average value into the Dept0 row
                mask_d0 =df_plot ['dept_id']==0 
                df_plot .loc [mask_d0 ,col ]=means [mask_d0 ]

                # Set the Dept1 row to NaN (it will be excluded from the plot, and N will become 100)
                mask_d1 =df_plot ['dept_id']!=0 
                df_plot .loc [mask_d1 ,col ]=np .nan 

                # Delete unnecessary temporary column
    drop_cols =[c for c in ['dept_id','run_id']if c in df_plot .columns ]
    if drop_cols :
        df_plot =df_plot .drop (columns =drop_cols )
        # ---------------------------------------------------------------------------------

    df_plot =df_plot .rename (columns =col_mapping )

    # New column name list
    new_cols =list (col_mapping .values ())

    # 3. Melt (to Long format)
    df_melt =df_plot .melt (id_vars =['pattern_id'],value_vars =new_cols ,var_name ='Metric',value_name ='Value')
    df_melt =df_melt .dropna ()

    plt .figure (figsize =(12 ,7 ))# Widen slightly for the legend

    # 4. Drawing
    sns .lineplot (
    data =df_melt ,
    x ='pattern_id',
    y ='Value',
    hue ='Metric',
    style ='Metric',
    markers =True ,
    dashes =False ,
    err_style ='bars',# Display error bars (95% confidence interval)
    palette ='bright',
    linewidth =2.5 ,
    markersize =9 
    )

    plt .title (title ,fontsize =16 )
    plt .xlabel ("Pattern ID (1 → 4)",fontsize =12 )
    plt .ylabel (y_label ,fontsize =12 )
    plt .xticks ([1 ,2 ,3 ,4 ])
    plt .grid (True ,linestyle ='--',alpha =0.6 )

    # Place the legend outside the graph (top right)
    plt .legend (bbox_to_anchor =(1.02 ,1 ),loc ='upper left',borderaxespad =0 ,title ="Metrics (ANOVA)")

    # Save
    os .makedirs (output_dir ,exist_ok =True )
    plt .savefig (os .path .join (output_dir ,filename ),dpi =100 ,bbox_inches ='tight')
    plt .close ()

    # =========================================================
    # Main processing
    # =========================================================
SAVE_ROOT =os .path .join (conf .ANALYSIS_DIR ,"07_パターン推移_統計付き")# Folder/File: 07_Pattern Transition_With Statistics | Folder/File: 07_パターン推移_統計付き

def main ():# Folder/File: 07_Pattern Transition_With Statistics
    setup_japanese_font ()
    df_org ,df_dept =load_data ()

    print (f"Output Directory: {SAVE_ROOT }")

    # --- 1. Organization Perspective ---

    # (A) Average Score
    cols_org_mean =['Ave P','Ave CT','Ave RD','Ave CGT','Ave CCU']
    create_trend_plot_with_stats (
    df_org ,cols_org_mean ,
    "Changes in organizational performance (average)",
    "01_Org_Performance_Mean.png",
    os .path .join (SAVE_ROOT ,"01_Organization"),
    y_label ="Mean Score (1-5)"
    )

    # (B) Inter-community Heterogeneity (CCH)
    cols_org_cch =['CCH-P','CCH-CT','CCH-RD','CCH-CGT','CCH-CCU']
    create_trend_plot_with_stats (
    df_org ,cols_org_cch ,
    "Transition of Community-Level Heterogeneity (CCH)",
    "02_Org_Heterogeneity_CCH.png",
    os .path .join (SAVE_ROOT ,"01_Organization"),
    y_label ="Heterogeneity (Std Dev)"
    )

    # (C) Temporal Variability (CCTF)
    cols_org_cctf =['CCTF-P','CCTF-CT','CCTF-RD','CCTF-CGT','CCTF-CCU']
    create_trend_plot_with_stats (
    df_org ,cols_org_cctf ,
    "Transition of temporal variability (CCTF)",
    "03_Org_TemporalFlux_CCTF.png",
    os .path .join (SAVE_ROOT ,"01_Organization"),
    y_label ="Temporal Flux (Std Dev)"
    )

    # (D) Structural Indicators
    create_trend_plot_with_stats (
    df_org ,['CCC Score'],
    "Changes in Organizational Structure Indicators (CCC)",
    "04_Org_Structure_CCC.png",
    os .path .join (SAVE_ROOT ,"01_Organization"),
    y_label ="CCC Score (Ratio)"
    )

    # --- 2. Department Perspective ---

    # (A) Average Score
    cols_dept_mean =['ID-Ave P','ID-Ave CT','ID-Ave RD','ID-Ave DGT','ID-Ave CDU']
    create_trend_plot_with_stats (
    df_dept ,cols_dept_mean ,
    "Transition of Department Performance (Average)",
    "01_Dept_Performance_Mean.png",
    os .path .join (SAVE_ROOT ,"02_Department"),
    y_label ="Mean Score (1-5)"
    )

    # (B) Inter-individual Heterogeneity (IDIH)
    cols_dept_idih =['IDIH-P','IDIH-CT','IDIH-RD','IDIH-DGT','IDIH-CDU']
    create_trend_plot_with_stats (
    df_dept ,cols_dept_idih ,
    "Trends in Individual Differences Within the Department (IDIH)",
    "02_Dept_Heterogeneity_IDIH.png",
    os .path .join (SAVE_ROOT ,"02_Department"),
    y_label ="Heterogeneity (Std Dev)"
    )

    # (C) Structural Indicator (CDC)
    cols_dept_struct =['CDC Score','adj-CDC Score']
    create_trend_plot_with_stats (
    df_dept ,cols_dept_struct ,
    "Transition of Department Structure Index (CDC)",
    "03_Dept_Structure_CDC.png",
    os .path .join (SAVE_ROOT ,"02_Department"),
    y_label ="CDC Score"
    )

    print (">>> Trend Summary with Stats Generated Successfully.")

if __name__ =="__main__":
    main ()