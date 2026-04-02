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

    found_font =None 
    for font_name in candidate_fonts :
        try :
            fm .findfont (font_name ,fallback_to_default =False )
            found_font =font_name 
            break 
        except :
            continue 

    if found_font :
        plt .rcParams ['font.family']=found_font 
    else :
        plt .rcParams ['font.family']='sans-serif'

        # =========================================================
        # Data Loading
        # =========================================================
def load_and_prep_data ():
    print (">>> Loading Datasets...")
    path_org =os .path .join (conf .OUTPUT_DATASETS ,"dataset_organization_N100.csv")
    path_dept =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")

    if not os .path .exists (path_org )or not os .path .exists (path_dept ):
        print (f"[ERROR] Dataset not found. Please run 03_create_datasets.py first.")
        sys .exit (1 )

    df_org =pd .read_csv (path_org ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    # DGT and CDU integration into organizational data
    dept_cols =[]
    if 'ID-Ave DGT'in df_dept .columns :dept_cols .append ('ID-Ave DGT')
    if 'ID-Ave CDU'in df_dept .columns :dept_cols .append ('ID-Ave CDU')

    if dept_cols :
        print (">>> Merging Ave DGT/CDU into Organization Dataset...")
        df_dept_mean =df_dept .groupby ('run_id')[dept_cols ].mean ().reset_index ()
        rename_map ={c :c .replace ('ID-Ave','Ave')for c in dept_cols }
        df_dept_mean =df_dept_mean .rename (columns =rename_map )
        df_org =df_org .merge (df_dept_mean ,on ='run_id',how ='left')

    return df_org ,df_dept 

    # =========================================================
    # Heat map drawing function
    # =========================================================
def create_rich_heatmap (df ,x_vars ,y_vars ,output_dir ,file_name ,title ,figsize =None ):
    """
    Create and save a heatmap including four indicators (significance, R², direction, slope)
    """
    # Matrix for storing results
    slope_mat =np .zeros ((len (y_vars ),len (x_vars )))
    r2_signed_mat =np .zeros ((len (y_vars ),len (x_vars )))# Color use (signed R2)
    annot_mat =np .empty ((len (y_vars ),len (x_vars )),dtype =object )# For text

    # Calculation loop
    for i ,y_col in enumerate (y_vars ):
        for j ,x_col in enumerate (x_vars ):
            if x_col ==y_col :
                slope_mat [i ,j ]=np .nan 
                r2_signed_mat [i ,j ]=0 
                annot_mat [i ,j ]="-"
                continue 

                # Data Preparation (Missing Value Removal)
                # --- CDU support: N=200 -> N=100 (average of 2 departments) ---
            if ("CDU"in x_col or "CDU"in y_col )and 'dept_id'in df .columns and df ['dept_id'].nunique ()>1 :
                current_df =df .groupby (['run_id','pattern_id'],as_index =False ).mean (numeric_only =True )
            else :
                current_df =df 

            sub_df =current_df [[x_col ,y_col ]].dropna ()

            if len (sub_df )<5 :
                slope_mat [i ,j ]=np .nan 
                r2_signed_mat [i ,j ]=0 
                annot_mat [i ,j ]="N/A"
                continue 

            slope ,intercept ,r_val ,p_val ,std_err =stats .linregress (sub_df [x_col ],sub_df [y_col ])

            r2 =r_val **2 
            # For coloring: Signed coefficient of determination (-1 ~ 1)
            # Although R2 itself ranges from 0 to 1, a negative correlation is marked with a minus sign and colored blue.
            sign =1 if slope >=0 else -1 
            r2_signed_mat [i ,j ]=sign *r2 

            # Significance mark
            stars =""
            if p_val <0.001 :stars ="***"
            elif p_val <0.01 :stars ="**"
            elif p_val <0.05 :stars ="*"

            # Display Text: Tilt + Star
            annot_mat [i ,j ]=f"{slope :.2f}\n{stars }"

            # Automatic adjustment of drawing size (according to the number of variables)
    if figsize is None :
        w =max (8 ,len (x_vars )*1.5 )
        h =max (6 ,len (y_vars )*1.0 )
        figsize =(w ,h )

    plt .figure (figsize =figsize )

    # Heat map drawing
    # cmap: 'coolwarm' or 'vlag' (blue-white-red)
    # Set vmin=-1 and vmax=1, then if R2=1 it's bright red/bright blue, and if R2=0 it's white
    ax =sns .heatmap (
    r2_signed_mat ,
    annot =annot_mat ,
    fmt ="",
    cmap ='coolwarm',
    vmin =-0.8 ,vmax =0.8 ,# Color range adjustment (maximum color at R2 ≥ 0.8)
    center =0 ,
    xticklabels =x_vars ,
    yticklabels =y_vars ,
    linewidths =1 ,
    linecolor ='white',
    cbar_kws ={'label':'Signed $R^2$ (Color intensity)'}
    )

    plt .title (title ,fontsize =16 ,pad =20 )
    plt .xticks (rotation =45 ,ha ='right')
    plt .yticks (rotation =0 )

    # Save
    os .makedirs (output_dir ,exist_ok =True )
    plt .savefig (os .path .join (output_dir ,file_name ),dpi =100 ,bbox_inches ='tight')
    plt .close ()

    # =========================================================
    # Analysis Execution Part
    # =========================================================

    # Save route
HEATMAP_ROOT =os .path .join (conf .ANALYSIS_DIR ,"05_仮説検証_ヒートマップ")# Folder/File: 05_Hypothesis Verification_Heat Map | Folder/File: 05_Hypothesis Verification_Heat Map | Folder/File: 05_仮説検証_ヒートマップ

def run_analysis_01_overall (df_org ):
    print ("--- 01_Overall Trends ---")
    vars_base =['P','CT','RD','CGT','CCU','DGT','CDU']
    cols =[f"Ave {v }"for v in vars_base if f"Ave {v }"in df_org .columns ]

    if not cols :return 

    save_dir =os .path .join (HEATMAP_ROOT ,"01_全体傾向")# Folder/File: 01_Overall Trends | Folder/File: 01_Overall Trends | Folder/File: 01_全体傾向
    create_rich_heatmap (
    df_org ,cols ,cols ,
    save_dir ,"Overall_Correlations.png",
    "Overall Trend: Correlation Between Average Scores ($N=100$)"
    )

def run_analysis_02_community (df_org ):
    print ("--- 02_Community Analysis ---")
    vars_base =['P','CT','RD','CGT','CCU']
    save_dir =os .path .join (HEATMAP_ROOT ,"02_コミュニティ分析")# Folder/File: 02_Community Analysis | Folder/File: 02_Community Analysis | Folder/File: 02_コミュニティ分析

    # 02a. Nature Relations
    # (1) CC-Ave vs CC-Ave
    cols_ave =[f"CC-Ave {v }"for v in vars_base if f"CC-Ave {v }"in df_org .columns ]
    if cols_ave :
        create_rich_heatmap (
        df_org ,cols_ave ,cols_ave ,
        os .path .join (save_dir ,"02a_Nature Relations"),"Ave_vs_Ave.png",
        "Community: Relationship Between Average Values"
        )

        # (2) CC-Ave (X) vs CCH (Y)
    cols_hetero =[f"CCH-{v }"for v in vars_base if f"CCH-{v }"in df_org .columns ]
    if cols_ave and cols_hetero :
        create_rich_heatmap (
        df_org ,cols_ave ,cols_hetero ,
        os .path .join (save_dir ,"02a_Nature Relations"),"Ave_vs_Hetero.png",
        "Community: Average (X) vs Heterogeneity (Y)"
        )

        # (3) CC-Ave (X) vs CCTF (Y)
    cols_temp =[f"CCTF-{v }"for v in vars_base if f"CCTF-{v }"in df_org .columns ]
    if cols_ave and cols_temp :
        create_rich_heatmap (
        df_org ,cols_ave ,cols_temp ,
        os .path .join (save_dir ,"02a_Nature Relations"),"Ave_vs_Temporal.png",
        "Community: Average (X) vs Time Variation (Y)"
        )

        # 02b. Causal Analysis (CCC Score)
    x_causal =["CCC Score"]
    if "CCC Score"in df_org .columns :
    # Y-axis: Arrange all indicators by category
        y_all =[]
        y_all .extend ([f"Ave {v }"for v in vars_base if f"Ave {v }"in df_org .columns ])
        y_all .extend ([f"CC-Ave {v }"for v in vars_base if f"CC-Ave {v }"in df_org .columns ])
        y_all .extend ([f"CCH-{v }"for v in vars_base if f"CCH-{v }"in df_org .columns ])
        y_all .extend ([f"CCTF-{v }"for v in vars_base if f"CCTF-{v }"in df_org .columns ])

        if y_all :
            create_rich_heatmap (
            df_org ,x_causal ,y_all ,
            os .path .join (save_dir ,"02b_Causal Analysis"),"Causal_CCC_Impact.png",
            "Causal Analysis: Impact of CCC Score on Each Indicator (Slope & Signed R2)",
            figsize =(6 ,12 )# Make it vertical
            )

def run_analysis_dept_generic (df_target ,folder_name ):
    print (f"--- {folder_name } ---")
    vars_base =['P','CT','RD','DGT','CDU']
    save_dir =os .path .join (HEATMAP_ROOT ,folder_name )

    # 03a. Nature Relations
    cols_ave =[f"ID-Ave {v }"for v in vars_base if f"ID-Ave {v }"in df_target .columns ]

    if cols_ave :
        create_rich_heatmap (
        df_target ,cols_ave ,cols_ave ,
        os .path .join (save_dir ,"03a_Nature Relations"),"Ave_vs_Ave.png",
        f"{folder_name }: Relationship between average values"
        )

        cols_hetero =[f"IDIH-{v }"for v in vars_base if f"IDIH-{v }"in df_target .columns ]
        if cols_hetero :
            create_rich_heatmap (
            df_target ,cols_ave ,cols_hetero ,
            os .path .join (save_dir ,"03a_Nature Relations"),"Ave_vs_IndivHetero.png",
            f"{folder_name }: Average (X) vs Inter-individual variability (Y)"
            )

        cols_temp =[f"IDTF-{v }"for v in vars_base if f"IDTF-{v }"in df_target .columns ]
        if cols_temp :
            create_rich_heatmap (
            df_target ,cols_ave ,cols_temp ,
            os .path .join (save_dir ,"03a_Nature Relations"),"Ave_vs_Temporal.png",
            f"{folder_name }Average (X) vs Time Variation (Y)"
            )

            # 03b. Causal Analysis (CDC, adj-CDC)
    x_causals =[c for c in ["CDC Score","adj-CDC Score"]if c in df_target .columns ]

    if x_causals :
    # Y-axis: All indicators
        y_all =[]
        y_all .extend ([f"ID-Ave {v }"for v in vars_base if f"ID-Ave {v }"in df_target .columns ])
        y_all .extend ([f"IDIH-{v }"for v in vars_base if f"IDIH-{v }"in df_target .columns ])
        y_all .extend ([f"IDTF-{v }"for v in vars_base if f"IDTF-{v }"in df_target .columns ])

        if y_all :
            create_rich_heatmap (
            df_target ,x_causals ,y_all ,
            os .path .join (save_dir ,"03b_Causal Analysis"),"Causal_CDC_Impact.png",
            f"{folder_name }: Impact of CDC & adj-CDC Scores (Slope & Signed R2)",
            figsize =(8 ,10 )# Vertical
            )

def main ():
    setup_japanese_font ()

    df_org ,df_dept =load_and_prep_data ()

    # 1. Overall Trend
    run_analysis_01_overall (df_org )

    # 2. Community Analysis
    run_analysis_02_community (df_org )

    # 3. Department Analysis (Overall)
    run_analysis_dept_generic (df_dept ,"03_部署分析_全体")# Folder/File: 03_Department Analysis_Overall | Folder/File: 03_Department Analysis_Overall | Folder/File: 03_部署分析_全体

    # 4. Department Analysis (Dept0)
    run_analysis_dept_generic (df_dept [df_dept ['dept_id']==0 ],"04_部署0分析")# Folder/File: 04_Department0 Analysis | Folder/File: 04_Department0 Analysis | Folder/File: 04_部署0分析

    # 5. Department Analysis (Dept1)
    run_analysis_dept_generic (df_dept [df_dept ['dept_id']==1 ],"05_部署1分析")# Folder/File: 05_Department1 Analysis | Folder/File: 05_Department1 Analysis | Folder/File: 05_部署1分析

    print (f"\n>>> All Heatmaps Generated Successfully in: {HEATMAP_ROOT }")

if __name__ =="__main__":
    main ()
