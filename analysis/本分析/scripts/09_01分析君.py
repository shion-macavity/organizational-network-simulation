import os 
import sys 
import pandas as pd 
import numpy as np 
import scipy .stats as stats 
import matplotlib .pyplot as plt 
import seaborn as sns 
import warnings 
import matplotlib .font_manager as fm 
import math 

# Warning suppression
warnings .filterwarnings ("ignore")

# =========================================================
# 1. Definition of paths and settings
# =========================================================

current_script_path =os .path .abspath (__file__ )
scripts_dir =os .path .dirname (current_script_path )
base_analysis_dir =os .path .dirname (scripts_dir )

sys .path .append (base_analysis_dir )

try :
    import config_analysis as conf 
    DATASET_DIR =conf .OUTPUT_DATASETS 
    ANALYSIS_RESULTS_DIR =conf .ANALYSIS_DIR 
except ImportError :
    DATASET_DIR =os .path .join (base_analysis_dir ,"03_datasets")
    ANALYSIS_RESULTS_DIR =os .path .join (base_analysis_dir ,"analysis_results")

DATA_FILE_NAME ="dataset_department_N200.csv"
OUTPUT_DIR =os .path .join (ANALYSIS_RESULTS_DIR ,"08_部署間会話有無比較_CDC")# Folder/File: 08_Inter-department Conversation Presence Comparison_CDC | Folder/File: 08_Inter-department Conversation Presence Comparison_CDC | Folder/File: 08_部署間会話有無比較_CDC
os .makedirs (OUTPUT_DIR ,exist_ok =True )

sns .set (style ="whitegrid",context ="talk")

# =========================================================
# Font Settings
# =========================================================
def setup_japanese_font ():
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
        # 2. Sorting Logic (New!)
        # =========================================================
def get_sort_key (metric_name ):
    """
    Receive the indicator name and return a tuple that represents the sort order
Order:
  1. Statistic type (Ave -> IH -> TF)
  2. Variable type (P -> CT -> RD -> DGT -> CDU)
    """
    # 1. Priority of the statistic type
    if metric_name .startswith ("ID-Ave"):
        stat_rank =1 
        core_name =metric_name .replace ("ID-Ave ","")
    elif metric_name .startswith ("IDIH-"):
        stat_rank =2 
        core_name =metric_name .replace ("IDIH-","")
    elif metric_name .startswith ("IDTF-"):
        stat_rank =3 
        core_name =metric_name .replace ("IDTF-","")
    else :
        stat_rank =99 
        core_name =metric_name 

        # 2. Priority of Variable Types
        # Determine by exact match
    var_order ={
    'P':1 ,
    'CT':2 ,
    'RD':3 ,
    'DGT':4 ,
    'CDU':5 
    }

    var_rank =var_order .get (core_name ,99 )

    return (stat_rank ,var_rank )

    # =========================================================
    # 3. Functions for Analysis
    # =========================================================

def cohens_d (x ,y ):
    nx ,ny =len (x ),len (y )
    dof =nx +ny -2 
    if dof <=0 :return np .nan 
    s_pooled =np .sqrt (((nx -1 )*np .var (x ,ddof =1 )+(ny -1 )*np .var (y ,ddof =1 ))/dof )
    if s_pooled ==0 :return 0 
    return (np .mean (x )-np .mean (y ))/s_pooled 

def load_data ():
    file_path =os .path .join (DATASET_DIR ,DATA_FILE_NAME )
    if not os .path .exists (file_path ):
        raise FileNotFoundError (f"[ERROR] Could not find '{DATA_FILE_NAME }' in {DATASET_DIR }")

    print (f">>> Loading: {file_path }")
    try :
        df =pd .read_csv (file_path ,encoding ='utf-8-sig')
    except UnicodeDecodeError :
        df =pd .read_csv (file_path ,encoding ='cp932')
    return df 

def perform_statistical_test (df ):
    """Comparison between groups with and without CDC (U-related processed with N=100)"""

    # Group Definition
    df ['CDC_Fill']=df ['CDC Score'].fillna (0 )
    df ['Group']=np .where (df ['CDC_Fill']>0 ,'Has Conv (CDC>0)','No Conv (CDC=0)')

    target_cols =[c for c in df .columns if c .startswith (('ID-Ave','IDIH-','IDTF-'))]

    # --- Execute sort here ---
    target_cols .sort (key =get_sort_key )
    # -----------------------

    stats_results =[]

    print (">>> Running Welch's t-test...")

    for col in target_cols :
    # [Important] U-related processing: CDU or CDC related is N=100 (per run)
    # Since the values are symmetrical across departments, duplicate counting is prevented.
        is_symmetric_metric =('CDU'in col )or ('CDC'in col )

        if is_symmetric_metric :
        # --- CDU/CDC response: N=200 -> N=100 (average of 2 departments) ---
            df_subset =df .groupby (['run_id','pattern_id'],as_index =False ).mean (numeric_only =True )
            # Redefinition of the Group column (because string columns disappear when using mean)
            df_subset ['CDC_Fill']=df_subset ['CDC Score'].fillna (0 )
            df_subset ['Group']=np .where (df_subset ['CDC_Fill']>0 ,'Has Conv (CDC>0)','No Conv (CDC=0)')
            n_type ="Run(N=100)"
        else :
            df_subset =df .copy ()# N=200
            n_type ="Dept(N=200)"

        g0 =df_subset [df_subset ['Group']=='No Conv (CDC=0)'][col ].dropna ()
        g1 =df_subset [df_subset ['Group']=='Has Conv (CDC>0)'][col ].dropna ()

        n0 ,n1 =len (g0 ),len (g1 )
        if n0 <2 or n1 <2 :continue 

        # Welch's t-test
        t_stat ,p_val =stats .ttest_ind (g1 ,g0 ,equal_var =False )

        # Calculation of degrees of freedom (Welch-Satterthwaite formula)
        v1 ,v0 =np .var (g1 ,ddof =1 ),np .var (g0 ,ddof =1 )
        denom_dof =((v1 /n1 )**2 /(n1 -1 ))+((v0 /n0 )**2 /(n0 -1 ))
        dof_val =((v1 /n1 +v0 /n0 )**2 )/denom_dof if denom_dof !=0 else np .nan 

        d_val =cohens_d (g1 ,g0 )

        # Judgment Mark (Using Dagger)
        if p_val <0.001 :sig ='***'
        elif p_val <0.01 :sig ='**'
        elif p_val <0.05 :sig ='*'
        elif p_val <0.1 :sig ='†'
        else :sig ='ns'

        stats_results .append ({
        'Metric':col ,
        'Sample_Type':n_type ,
        'Mean_No':g0 .mean (),'Mean_Yes':g1 .mean (),
        'Diff':g1 .mean ()-g0 .mean (),
        't_stat':t_stat ,'dof':dof_val ,
        'p_value':p_val ,'Cohens_d':d_val ,'Significance':sig ,
        'N_No':n0 ,'N_Yes':n1 
        })

    df_stats =pd .DataFrame (stats_results )

    # Also sort df_stats based on the Metric column (just to be safe)
    df_stats ['sort_key']=df_stats ['Metric'].apply (get_sort_key )
    df_stats =df_stats .sort_values ('sort_key').drop (columns =['sort_key'])

    df_stats .to_csv (os .path .join (OUTPUT_DIR ,"summary_stats_welch_corrected.csv"),index =False ,encoding ='utf-8-sig')
    return df ,df_stats ,target_cols 

    # =========================================================
    # 3. Visualization Function
    # =========================================================

def plot_distributions (df ,df_stats ,target_cols ):
    """Create a box plot"""
    print (">>> Generating Plots (Ordered)...")

    # The assumption is that target_cols is already sorted when passed, but just to be safe.
    target_cols .sort (key =get_sort_key )

    # Set the number of columns to 5 (It might look neat when the five—P, CT, RD, DGT, CDU—are lined up horizontally?)
    # Alternatively, should we stick to the traditional three columns? Here, we prioritize readability and maintain three columns.
    n_cols =3 
    n_rows =math .ceil (len (target_cols )/n_cols )
    fig ,axes =plt .subplots (n_rows ,n_cols ,figsize =(15 ,4 *n_rows ))
    axes =axes .flatten ()

    for i ,col in enumerate (target_cols ):
        ax =axes [i ]

        # [Important] Data Switching for U-Around Plot
        is_symmetric_metric =('CDU'in col )or ('CDC'in col )

        if is_symmetric_metric :
        # --- CDU/CDC response: N=200 -> N=100 (average of 2 departments) ---
            plot_df =df .groupby (['run_id','pattern_id'],as_index =False ).mean (numeric_only =True )
            # Redefinition of the Group column
            plot_df ['CDC_Fill']=plot_df ['CDC Score'].fillna (0 )
            plot_df ['Group']=np .where (plot_df ['CDC_Fill']>0 ,'Has Conv (CDC>0)','No Conv (CDC=0)')
        else :
            plot_df =df .copy ()

        row =df_stats [df_stats ['Metric']==col ]
        title_stats =""
        if not row .empty :
            p_val =row .iloc [0 ]['p_value']
            sig =row .iloc [0 ]['Significance']
            d_val =row .iloc [0 ]['Cohens_d']
            title_stats =f"\n(p={p_val :.3f} {sig }, d={d_val :.2f})"

        n_no =plot_df [plot_df ['Group']=='No Conv (CDC=0)'].shape [0 ]
        n_yes =plot_df [plot_df ['Group']=='Has Conv (CDC>0)'].shape [0 ]

        sns .boxplot (x ='Group',y =col ,data =plot_df ,ax =ax ,
        palette ="bright",showfliers =False ,width =0.5 )

        sns .stripplot (x ='Group',y =col ,data =plot_df ,ax =ax ,
        color ="black",alpha =0.3 ,jitter =True ,size =4 )

        ax .set_title (f"{col }{title_stats }",fontweight ='bold',fontsize =11 )
        ax .set_xlabel ('')
        ax .set_ylabel ('')

        labels =[item .get_text ()for item in ax .get_xticklabels ()]
        new_labels =[]
        for l in labels :
            count =n_no if 'No'in l else n_yes 
            new_labels .append (f"{l }\n(n={count })")
        ax .set_xticklabels (new_labels )

    for j in range (i +1 ,len (axes )):
        fig .delaxes (axes [j ])

    plt .tight_layout ()
    plt .savefig (os .path .join (OUTPUT_DIR ,"dist_boxplots_ordered.png"),dpi =150 )
    plt .close ()

def plot_heatmap (df_stats ):
    """Create Heatmap (Sorted)"""
    print (">>> Generating Heatmap (Ordered)...")

    # Reapply the sort key and rearrange the DataFrame
    df_stats ['sort_key']=df_stats ['Metric'].apply (get_sort_key )
    df_vis =df_stats .sort_values ('sort_key').drop (columns =['sort_key']).set_index ('Metric')

    # Change the column names and make the labels on the horizontal axis (bottom) of the heat map properly in English
    heatmap_data =df_vis [['Cohens_d']].rename (columns ={'Cohens_d':'Effect of Inter-Dept Conv'})

    # Annotation of numbers + symbols (change to t-values)
    annot_data =df_vis .apply (lambda row :f"{row ['t_stat']:.2f}{row ['Significance']}",axis =1 )
    annot_data =pd .DataFrame (annot_data ,columns =['Annotation'])

    plt .figure (figsize =(8 ,0.6 *len (heatmap_data )))

    sns .heatmap (heatmap_data ,annot =annot_data ,fmt ='',cmap ='coolwarm',center =0 ,
    cbar_kws ={'label':"Effect Size (Cohen's d)\nRed: Conv > No Conv"},
    linewidths =.5 ,annot_kws ={"size":10 })

    plt .title ("Impact of Inter-Dept Conversation (CDC>0)\nNote: † p<0.1",fontsize =13 )
    plt .ylabel ('')
    plt .tight_layout ()

    plt .savefig (os .path .join (OUTPUT_DIR ,"summary_heatmap_ordered.png"),dpi =150 )
    plt .close ()

    # =========================================================
    # Main processing
    # =========================================================
def main ():
    setup_japanese_font ()

    try :
        df =load_data ()
        df_labeled ,df_stats ,target_cols =perform_statistical_test (df )

        plot_distributions (df_labeled ,df_stats ,target_cols )
        plot_heatmap (df_stats )

        print (f"\n>>> Analysis Complete. Output: {OUTPUT_DIR }")

    except Exception as e :
        print (f"\n[CRITICAL ERROR] {e }")
        import traceback 
        traceback .print_exc ()

if __name__ =="__main__":
    main ()