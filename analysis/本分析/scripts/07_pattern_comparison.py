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

# Path to the parent directory
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

    # Integration of DGT and CDU into organizational data (just in case)
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
    # Box plot creation function
    # =========================================================
def create_boxplot (df ,y_col ,output_dir ):
    """
    Draw box plots for each pattern and display the results of the ANOVA test
    """
    # Data preparation (for missing value removal: for adj-CDC, etc.)
    # --- CDU/CDC response: N=200 -> N=100 (average of 2 departments) ---
    if ("CDU"in y_col or "CDC"in y_col )and 'dept_id'in df .columns and df ['dept_id'].nunique ()>1 :
        current_df =df .groupby (['run_id','pattern_id'],as_index =False ).mean (numeric_only =True )
    else :
        current_df =df 

    data =current_df [['pattern_id',y_col ]].dropna ()

    if len (data ['pattern_id'].unique ())<2 :
        return # Skip if there is no comparison target

        # ANOVA (one-way analysis of variance)
    groups =[]
    patterns =sorted (data ['pattern_id'].unique ())
    for p in patterns :
        groups .append (data [data ['pattern_id']==p ][y_col ].values )

        # Statistics Certification
    try :
        f_stat ,p_val =stats .f_oneway (*groups )
        significance =""
        if p_val <0.001 :significance ="***"
        elif p_val <0.01 :significance ="**"
        elif p_val <0.05 :significance ="*"
        else :significance ="(n.s.)"
        title_text =f"{y_col }\nANOVA: p={p_val :.2e} {significance }"
    except :
        title_text =f"{y_col }\n(Stats Error)"

        # Drawing
    plt .figure (figsize =(6 ,5 ))

    # box plot
    # --- [Modification] Obtain ax and set the label to display the sample size (n) on the X-axis ---
    ax =sns .boxplot (
    data =data ,x ='pattern_id',y =y_col ,
    palette ="Set2",showfliers =False 
    )
    sns .stripplot (
    data =data ,x ='pattern_id',y =y_col ,
    color ='black',alpha =0.3 ,jitter =True ,size =3 
    )

    # Count the number of samples for each pattern and add it to the label (e.g., "1\n(n=25)")
    counts =data ['pattern_id'].value_counts ().sort_index ()
    new_labels =[f"{p }\n(n={counts .get (p ,0 )})"for p in sorted (data ['pattern_id'].unique ())]
    ax .set_xticklabels (new_labels )
    # ---------------------------------------------------------------------------------

    plt .title (title_text ,fontsize =12 )
    plt .xlabel ("Pattern ID")
    plt .ylabel (y_col )
    plt .grid (axis ='y',linestyle ='--',alpha =0.5 )

    # Save
    os .makedirs (output_dir ,exist_ok =True )
    fname =y_col .replace (' ','_').replace ('/','_')+"_boxplot.png"
    plt .savefig (os .path .join (output_dir ,fname ),dpi =100 ,bbox_inches ='tight')
    plt .close ()

    # =========================================================
    # Main processing
    # =========================================================
SAVE_ROOT =os .path .join (conf .ANALYSIS_DIR ,"06_パターン間比較")# Folder/File: 06_Pattern Comparison | Folder/File: 06_Pattern Comparison | Folder/File: 06_パターン間比較

def main ():
    setup_japanese_font ()
    df_org ,df_dept =load_data ()

    print (f"Output Directory: {SAVE_ROOT }")

    # --- 1. Organizational (Community) Level Analysis ---
    print (">>> Processing Organization Level Metrics...")

    # Extraction of target columns
    # CCC, Ave, CC-Ave, CCH, CCTF
    org_targets =["CCC Score"]

    # Indicator List
    vars_base =['P','CT','RD','CGT','CCU','DGT','CDU']

    # Search columns with various prefixes
    prefixes =["Ave ","CC-Ave ","CCH-","CCTF-"]
    for pre in prefixes :
        for v in vars_base :
            col =f"{pre }{v }"
            if col in df_org .columns :
                org_targets .append (col )

                # Save by organizing into folders
    for col in org_targets :
    # Subfolder Decision
        if "CCC"in col :sub ="00_Structure"
        elif "CC-Ave"in col :sub ="02_Community_Mean"
        elif "CCH"in col :sub ="03_Community_Hetero"
        elif "CCTF"in col :sub ="04_Temporal"
        elif "Ave"in col :sub ="01_Overall_Mean"
        else :sub ="99_Other"

        create_boxplot (df_org ,col ,os .path .join (SAVE_ROOT ,"01_Organization_Level",sub ))


        # --- 2. Department-level Analysis ---
    print (">>> Processing Department Level Metrics...")

    dept_targets =[]
    if "CDC Score"in df_dept .columns :dept_targets .append ("CDC Score")
    if "adj-CDC Score"in df_dept .columns :dept_targets .append ("adj-CDC Score")

    prefixes_dept =["ID-Ave ","IDIH-","IDTF-"]
    for pre in prefixes_dept :
        for v in vars_base :
            col =f"{pre }{v }"
            if col in df_dept .columns :
                dept_targets .append (col )

    for col in dept_targets :
    # Subfolder Decision
        if "CDC"in col :sub ="00_Structure"
        elif "ID-Ave"in col :sub ="01_Dept_Mean"
        elif "IDIH"in col :sub ="02_Individual_Hetero"
        elif "IDTF"in col :sub ="03_Temporal"
        else :sub ="99_Other"

        create_boxplot (df_dept ,col ,os .path .join (SAVE_ROOT ,"02_Department_Level",sub ))

    print (">>> Completed.")

if __name__ =="__main__":
    main ()