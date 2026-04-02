import os 
import sys 
import pandas as pd 
import numpy as np 
import matplotlib 
matplotlib .use ('Agg')# Do not display the GUI window and set it to save-only mode
import matplotlib .pyplot as plt 
import seaborn as sns 
from scipy import stats 
import matplotlib .font_manager as fm 
import warnings 

# Warning suppression
warnings .filterwarnings ("ignore")

# Path setting to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

try :
    import config_analysis as conf 
except ImportError :
    class Config :
        OUTPUT_DATASETS ="results/05_amos_input"
        ANALYSIS_DIR ="results/06_main_analysis"
        ENCODING ="utf-8-sig"
    conf =Config ()

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
            # Simple Main Effect Test and Visualization Class
            # =========================================================
class SimpleMainEffectAnalyzer :
    def __init__ (self ,df_org ,df_dept ,output_dir ):
        self .df_org =df_org 
        self .df_dept =df_dept 
        self .output_dir =output_dir 

        os .makedirs (self .output_dir ,exist_ok =True )

    def _get_factors (self ,pid ):
        """
        Pattern 1 (Department Disconnection): Sup=0, Sub=0
Pattern 2 (Superior Connection): Sup=1, Sub=0
Pattern 3 (Subordinate Connection): Sup=0, Sub=1
Pattern 4 (Both Connections): Sup=1, Sub=1
        """
        sup =1 if pid in [2 ,4 ]else 0 
        sub =1 if pid in [3 ,4 ]else 0 
        return sup ,sub 

    def _prepare_data (self ,df ,metric ):
        data =df .copy ()

        # Special processing of the CDU/CDC series: Averaging per run (N=100)
        is_dept_metric ='dept_id'in data .columns 
        needs_averaging =("CDU"in metric or "CDC"in metric )and is_dept_metric 

        if needs_averaging :
            data =data .groupby (['run_id','pattern_id'],as_index =False ).mean (numeric_only =True )

        factors =data ['pattern_id'].apply (lambda x :self ._get_factors (x ))
        data ['Sup']=factors .apply (lambda x :x [0 ])
        data ['Sub']=factors .apply (lambda x :x [1 ])

        target_df =data [['Sup','Sub',metric ]].dropna ()
        return target_df 

    def run_analysis (self ,df ,metric_name ,category_name ):
        data =self ._prepare_data (df ,metric_name )
        if data .empty :return 

        # Create folders by category
        save_dir =os .path .join (self .output_dir ,category_name )
        os .makedirs (save_dir ,exist_ok =True )

        # List for saving statistical results
        stats_logs =[]

        # Drawing (two graphs side by side in one figure)
        fig ,axes =plt .subplots (1 ,2 ,figsize =(14 ,6 ))

        # --- Left graph: Effect of subordinate connections (Sub) by superior connections (Sup) ---
        self ._plot_simple_effect (
        data ,metric_name ,
        x_col ='Sup',hue_col ='Sub',
        ax =axes [0 ],
        title ="Effect of Subordinate Conn.\n(by Supervisor Conn.)",
        stats_logs =stats_logs 
        )

        # --- Right graph: Effect of supervisor connections (Sup) for each subordinate connection (Sub) ---
        self ._plot_simple_effect (
        data ,metric_name ,
        x_col ='Sub',hue_col ='Sup',
        ax =axes [1 ],
        title ="Effect of Supervisor Conn.\n(by Subordinate Conn.)",
        stats_logs =stats_logs 
        )

        plt .suptitle (f"Simple Main Effects: {metric_name }",fontsize =16 )
        plt .tight_layout (rect =[0 ,0.03 ,1 ,0.95 ])

        # Save image
        fname =metric_name .replace (" ","_").replace ("/","-")+"_SME.png"
        plt .savefig (os .path .join (save_dir ,fname ),dpi =100 ,bbox_inches ='tight')
        plt .close ()

        # Save as CSV
        if stats_logs :
            csv_name =metric_name .replace (" ","_").replace ("/","-")+"_stats.csv"
            pd .DataFrame (stats_logs ).to_csv (os .path .join (save_dir ,csv_name ),index =False ,encoding ='utf-8-sig')

    def _plot_simple_effect (self ,data ,metric ,x_col ,hue_col ,ax ,title ,stats_logs =None ):
        """
        Draw a bar graph of the simple main effects and the test results
        """
        # Dictionary for labels
        labels ={0 :'No',1 :'Yes'}

        # Data formatting
        data_plot =data .copy ()
        data_plot [x_col ]=data_plot [x_col ].map (labels )
        data_plot [hue_col ]=data_plot [hue_col ].map (labels )

        # Fix the order of the X-axis
        order =['No','Yes']
        hue_order =['No','Yes']

        # Bar graph (Barplot)
        sns .barplot (
        data =data_plot ,x =x_col ,y =metric ,hue =hue_col ,
        order =order ,hue_order =hue_order ,
        palette ='muted',errorbar =('ci',95 ),capsize =0.1 ,ax =ax ,
        edgecolor ='black',alpha =0.8 
        )

        # --- Statistical Test (Welch's t-test) and Star Plot ---
        # Test the difference in Hue (No vs Yes) for each level (No, Yes) of the X-axis

        y_max =data [metric ].max ()
        y_min =data [metric ].min ()
        y_range =y_max -y_min 

        # Height adjustment for drawing a line on top of the bar
        # Get the maximum value in the graph (including error bars, approximate)
        current_ylim =ax .get_ylim ()
        top_margin =y_range *0.1 
        line_height =current_ylim [1 ]# Temporary initial value

        for i ,x_label in enumerate (order ):# x_label = 'No', 'Yes'
        # Return to the original data and filter
        # The value of x_col is the inverse mapping of labels
            x_val =0 if x_label =='No'else 1 

            group_data =data [data [x_col ]==x_val ]

            # Group with hue_col = 0 and group with 1
            g0 =group_data [group_data [hue_col ]==0 ][metric ]
            g1 =group_data [group_data [hue_col ]==1 ][metric ]

            if len (g0 )>1 and len (g1 )>1 :
                stat ,p_val =stats .ttest_ind (g0 ,g1 ,equal_var =False )# Welch's t-test

                # Save Result (F = t^2)
                if stats_logs is not None :
                # Calculation of Welch-Satterthwaite degrees of freedom approximation and partial eta squared
                    v0 ,v1 =g0 .var (ddof =1 ),g1 .var (ddof =1 )
                    n0 ,n1 =len (g0 ),len (g1 )
                    denom =((v0 /n0 )**2 /(n0 -1 ))+((v1 /n1 )**2 /(n1 -1 ))

                    df_welch =((v0 /n0 +v1 /n1 )**2 )/denom if denom >0 else np .nan 
                    f_val =stat **2 
                    p_eta_sq =f_val /(f_val +df_welch )if (df_welch is not np .nan and f_val +df_welch !=0 )else np .nan 

                    stats_logs .append ({
                    "Metric":metric ,
                    "Condition_Factor":x_col ,# Factors used for categorization (e.g., Sup)
                    "Condition_Level":x_label ,# That level (e.g., No)
                    "Test_Factor":hue_col ,# Tested factor (e.g., Sub)
                    "t_value":stat ,
                    "F_value":f_val ,# F value for the simple main effect (df1=1)
                    "df":df_welch ,
                    "p_value":p_val ,
                    "partial_eta_sq":p_eta_sq ,
                    "N0":n0 ,
                    "N1":n1 
                    })

                    # Significance test
                sig_symbol =""
                if p_val <0.001 :sig_symbol ="***"
                elif p_val <0.01 :sig_symbol ="**"
                elif p_val <0.05 :sig_symbol ="*"
                elif p_val <0.1 :sig_symbol ="†"
                else :sig_symbol ="n.s."

                # Calculation of drawing position
                # I want to obtain the upper end of the mean + CI for each Hue at that X level.
                # Draw using the maximum data value + margin in a simplified manner
                local_max =group_data [metric ].max ()

                # Height at which to draw the line
                draw_h =local_max +(y_range *0.05 )

                # X coordinate to draw the line
                # In Seaborn's barplot, the bars are arranged with a width of 0.8 centered on x = 0, 1, ...
                # In the case of two hues, the left bar is around x - 0.2, and the right bar is around x + 0.2
                x1 =i -0.2 
                x2 =i +0.2 

                # Draw lines and stars
                color ='black'

                # If not significant (n.s.), should it not be drawn or should it be faded?
                # This time, n.s. will also be shown for 'confirmation' (for papers, only significant ones are fine, though).
                # The request is about 'whether it is statistically significant,' so only the significant ones should be highlighted.

                if sig_symbol !="n.s.":
                    ax .plot ([x1 ,x1 ,x2 ,x2 ],[draw_h ,draw_h +top_margin *0.2 ,draw_h +top_margin *0.2 ,draw_h ],lw =1 ,c =color )
                    ax .text ((x1 +x2 )*0.5 ,draw_h +top_margin *0.2 ,sig_symbol ,ha ='center',va ='bottom',color =color ,fontsize =12 ,fontweight ='bold')
                else :
                # It looks cleaner in the figure not to display n.s. (for papers)
                    pass 

        ax .set_title (title )
        ax .set_xlabel (f"{x_col } Connection")
        ax .grid (axis ='y',linestyle ='--',alpha =0.5 )

        # =========================================================
        # Data Load / Main Execution
        # =========================================================
def load_data ():
    print (">>> Loading Datasets...")
    path_org =os .path .join (conf .OUTPUT_DATASETS ,"dataset_organization_N100.csv")
    path_dept =os .path .join (conf .OUTPUT_DATASETS ,"dataset_department_N200.csv")

    if not os .path .exists (path_org )or not os .path .exists (path_dept ):
        print (f"[ERROR] Dataset not found in {conf .OUTPUT_DATASETS }")
        return None ,None 

    df_org =pd .read_csv (path_org ,encoding =conf .ENCODING )
    df_dept =pd .read_csv (path_dept ,encoding =conf .ENCODING )

    # To analyze CCC scores and the like, no merging will be done here.
    return df_org ,df_dept 

def main ():
    setup_japanese_font ()
    OUTPUT_DIR =os .path .join (conf .ANALYSIS_DIR ,"11_Simple_Main_Effects")

    df_org ,df_dept =load_data ()
    if df_org is None :return 

    analyzer =SimpleMainEffectAnalyzer (df_org ,df_dept ,OUTPUT_DIR )

    # 1. Organization Level
    print (">>> Analyzing Organization Level Metrics...")
    org_metrics =[]
    if "CCC Score"in df_org .columns :org_metrics .append ("CCC Score")
    vars_base =['P','CT','RD','CGT','CCU','DGT','CDU']
    prefixes =["Ave ","CCH-","CCTF-","CC-Ave "]
    for pre in prefixes :
        for v in vars_base :
            col =f"{pre }{v }"
            if col in df_org .columns :org_metrics .append (col )

    org_metrics =sorted (list (set (org_metrics )))
    for metric in org_metrics :
        analyzer .run_analysis (df_org ,metric ,"Organization_Level")

        # 2. Department Level
    print (">>> Analyzing Department Level Metrics...")
    dept_metrics =[]
    if "CDC Score"in df_dept .columns :dept_metrics .append ("CDC Score")
    if "adj-CDC Score"in df_dept .columns :dept_metrics .append ("adj-CDC Score")
    prefixes_dept =["ID-Ave ","IDIH-","IDTF-"]
    for pre in prefixes_dept :
        for v in vars_base :
            col =f"{pre }{v }"
            if col in df_dept .columns :dept_metrics .append (col )

    dept_metrics =sorted (list (set (dept_metrics )))
    for metric in dept_metrics :
        analyzer .run_analysis (df_dept ,metric ,"Department_Level")

    print (f">>> Analysis Completed. Results saved to: {OUTPUT_DIR }")

if __name__ =="__main__":
    main ()