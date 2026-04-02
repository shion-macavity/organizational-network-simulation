import os 
import sys 
import pandas as pd 
import numpy as np 
import matplotlib 
matplotlib .use ('Agg')# Addition: Make it save-only mode without displaying the GUI window
import matplotlib .pyplot as plt 
import seaborn as sns 
import statsmodels .api as sm 
from statsmodels .formula .api import ols 
import matplotlib .font_manager as fm 
import matplotlib .gridspec as gridspec 
import warnings 

# Warning suppression
warnings .filterwarnings ("ignore")

# Path setting to parent directory (for loading config_analysis)
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
            # Data Processing and Analysis Class
            # =========================================================
class TwoWayAnovaAnalyzer :
    def __init__ (self ,df_org ,df_dept ,output_dir_plots ,output_dir_summary ):
        self .df_org =df_org 
        self .df_dept =df_dept 
        self .output_dir_plots =output_dir_plots 
        self .output_dir_summary =output_dir_summary 

        os .makedirs (self .output_dir_plots ,exist_ok =True )
        os .makedirs (self .output_dir_summary ,exist_ok =True )

        self .results =[]

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

        if data .empty :
            return 

        try :
        # 1. Two-way ANOVA (Type II)
            clean_metric ="TargetVal"
            data_for_model =data .rename (columns ={metric_name :clean_metric })

            model =ols (f'{clean_metric } ~ C(Sup) * C(Sub)',data =data_for_model ).fit ()
            anova_table =sm .stats .anova_lm (model ,typ =2 )

            # p-value
            p_sup =anova_table .loc ['C(Sup)','PR(>F)']
            p_sub =anova_table .loc ['C(Sub)','PR(>F)']
            p_int =anova_table .loc ['C(Sup):C(Sub)','PR(>F)']

            # F-value
            f_sup =anova_table .loc ['C(Sup)','F']
            f_sub =anova_table .loc ['C(Sub)','F']
            f_int =anova_table .loc ['C(Sup):C(Sub)','F']

            # Degrees of freedom (df)
            df_sup =anova_table .loc ['C(Sup)','df']
            df_sub =anova_table .loc ['C(Sub)','df']
            df_int =anova_table .loc ['C(Sup):C(Sub)','df']
            df_err =anova_table .loc ['Residual','df']

            # Effect Size (Partial Eta Squared)
            ss_sup =anova_table .loc ['C(Sup)','sum_sq']
            ss_sub =anova_table .loc ['C(Sub)','sum_sq']
            ss_int =anova_table .loc ['C(Sup):C(Sub)','sum_sq']
            ss_res =anova_table .loc ['Residual','sum_sq']

            def calc_eta (ss ,ss_r ):return ss /(ss +ss_r )if (ss +ss_r )>0 else 0 
            eta_sup =calc_eta (ss_sup ,ss_res )
            eta_sub =calc_eta (ss_sub ,ss_res )
            eta_int =calc_eta (ss_int ,ss_res )

            # --- Direction of Effect ---
            mean_sup1 =data [data ['Sup']==1 ][metric_name ].mean ()
            mean_sup0 =data [data ['Sup']==0 ][metric_name ].mean ()
            dir_sup =1 if mean_sup1 >mean_sup0 else -1 

            mean_sub1 =data [data ['Sub']==1 ][metric_name ].mean ()
            mean_sub0 =data [data ['Sub']==0 ][metric_name ].mean ()
            dir_sub =1 if mean_sub1 >mean_sub0 else -1 

            dir_int =0 

            self .results .append ({
            "Category":category_name ,
            "Metric":metric_name ,
            "p_Sup":p_sup ,"p_Sub":p_sub ,"p_Int":p_int ,
            "eta_Sup":eta_sup ,"eta_Sub":eta_sub ,"eta_Int":eta_int ,
            "F_Sup":f_sup ,"F_Sub":f_sub ,"F_Int":f_int ,
            "df_Sup":df_sup ,"df_Sub":df_sub ,"df_Int":df_int ,"df_Err":df_err ,
            "dir_Sup":dir_sup ,"dir_Sub":dir_sub ,"dir_Int":dir_int 
            })

            # 2. Interaction plot (created for all indicators)
            self ._plot_interaction (data ,metric_name ,category_name ,p_sup ,p_sub ,p_int )

        except Exception as e :
            print (f"  [Error] Failed to analyze {metric_name }: {e }")

    def _plot_interaction (self ,data ,metric ,category ,p_sup ,p_sub ,p_int ):
        plt .figure (figsize =(7 ,6 ))
        data ['Sub_Label']=data ['Sub'].map ({0 :'Sub: No',1 :'Sub: Yes'})
        data ['Sup_Label']=data ['Sup'].map ({0 :'Sup: No',1 :'Sup: Yes'})

        sns .pointplot (
        data =data ,x ='Sup_Label',y =metric ,hue ='Sub_Label',
        markers =['o','s'],linestyles =['-','--'],capsize =0.1 ,
        palette ='deep',dodge =True ,errorbar =('ci',95 )
        )

        def sig_mark (p ):return "***"if p <.001 else "**"if p <.01 else "*"if p <.05 else "n.s."
        title =(f"{metric }\n"
        f"Sup: {sig_mark (p_sup )} (p={p_sup :.3f}), "
        f"Sub: {sig_mark (p_sub )} (p={p_sub :.3f})\n"
        f"Interaction: {sig_mark (p_int )} (p={p_int :.3f})")

        plt .title (title ,fontsize =12 )
        plt .xlabel ("Supervisor Connection")
        plt .ylabel (metric )
        plt .grid (True ,linestyle =':',alpha =0.6 )
        plt .legend (title ="Subordinate Connection")

        fname =metric .replace (" ","_").replace ("/","-")+"_interaction.png"
        save_path =os .path .join (self .output_dir_plots ,category )
        os .makedirs (save_path ,exist_ok =True )
        plt .savefig (os .path .join (save_path ,fname ),dpi =100 ,bbox_inches ='tight')
        plt .close ()

    def create_summary_heatmap (self ):
        """
        Revised Heatmap:
        - Exclude CCC and CDC-related indicators
        - Left side (Main Effects): Red (increase) vs Blue (decrease). Shade intensity by significance.
        - Right side (Interaction): Purple (complexity). Shade intensity by significance.
        - Display all results, including non-significant ones (p >= 0.1)
        """
        if not self .results :
            return 

        df_res =pd .DataFrame (self .results )
        csv_path =os .path .join (self .output_dir_summary ,"anova_summary_stats.csv")
        df_res .to_csv (csv_path ,index =False ,encoding ='utf-8-sig')

        # --- Items to be corrected: Indicators to be excluded from the heatmap ---
        exclude_metrics =["CCC Score","CDC Score","adj-CDC Score"]

        # Definition of a sorting function (either redefine it so that the same logic as the main function can be used within the class, or add a column for sorting to the df)
        base_order =['P','CT','RD','CGT','CCU','DGT','CDU']

        def get_sort_key (name ):
            score =9999 
            sub_score =99 

            # Organization-related
            if name .startswith ("Ave "):score =100 ;suffix =name [4 :]
            elif name .startswith ("CC-Ave "):score =200 ;suffix =name [7 :]
            elif name .startswith ("CCH-"):score =300 ;suffix =name [4 :]
            elif name .startswith ("CCTF-"):score =400 ;suffix =name [5 :]
            # Department-related
            elif name .startswith ("ID-Ave "):score =100 ;suffix =name [7 :]
            elif name .startswith ("IDIH-"):score =200 ;suffix =name [5 :]
            elif name .startswith ("IDTF-"):score =300 ;suffix =name [5 :]
            else :score =900 ;suffix =name 

            if suffix in base_order :
                sub_score =base_order .index (suffix )

            return score +sub_score 

        categories =df_res ['Category'].unique ()

        for cat in categories :
            df_cat =df_res [df_res ['Category']==cat ].copy ()

            # Exclusion filter
            df_cat =df_cat [~df_cat ['Metric'].isin (exclude_metrics )]

            if df_cat .empty :continue 

            # --- Sort Applied ---
            df_cat ['sort_key']=df_cat ['Metric'].apply (get_sort_key )
            df_cat =df_cat .sort_values ('sort_key')

            # --- Data Preparation ---
            df_cat =df_cat .set_index ('Metric')

            # --- 1. Main Effects (Sup, Sub) ---
            main_data =pd .DataFrame (index =df_cat .index )
            main_annot =pd .DataFrame (index =df_cat .index )

            for factor in ['Sup','Sub']:
                col_name ="Supervisor"if factor =="Sup"else "Subordinate"

                # Data for coloring: eta_p2 * direction (keep the color as is)
                p_vals =df_cat [f'p_{factor }']
                eta_vals =df_cat [f'eta_{factor }']
                dirs =df_cat [f'dir_{factor }']

                # For display: F value
                f_vals =df_cat [f'F_{factor }']

                # No mask processing: calculate eta * dir with all data
                main_data [col_name ]=eta_vals *dirs 

                # Annotation: F-value + mark (***, **, *, †)
                annot_vals =[]
                for p ,f_val in zip (p_vals ,f_vals ):
                    mark =""
                    if p <0.001 :mark ="***"
                    elif p <0.01 :mark ="**"
                    elif p <0.05 :mark ="*"
                    elif p <0.1 :mark ="†"

                    annot_vals .append (f"{f_val :.2f}{mark }")
                main_annot [col_name ]=annot_vals 

                # --- 2. Interaction ---
            int_data =pd .DataFrame (index =df_cat .index )
            int_annot =pd .DataFrame (index =df_cat .index )

            p_int =df_cat ['p_Int']
            eta_int =df_cat ['eta_Int']
            f_int =df_cat ['F_Int']

            # For coloring: eta (without mask)
            int_data ['Interaction']=eta_int 

            # Annotation: F-value
            int_annot_vals =[]
            for p ,f_val in zip (p_int ,f_int ):
                mark =""
                if p <0.001 :mark ="***"
                elif p <0.01 :mark ="**"
                elif p <0.05 :mark ="*"
                elif p <0.1 :mark ="†"
                int_annot_vals .append (f"{f_val :.2f}{mark }")
            int_annot ['Interaction']=int_annot_vals 

            # --- Drawing ---
            fig =plt .figure (figsize =(12 ,len (df_cat )*0.6 +2 ))
            gs =gridspec .GridSpec (1 ,2 ,width_ratios =[2 ,1 ],wspace =0.1 )

            # Left: Main Effects (Blue-White-Red)
            ax0 =plt .subplot (gs [0 ])
            # Estimate the maximum effect size and fix the color range (0.25: near the upper limit of a generally strong effect size)
            limit =0.25 

            sns .heatmap (
            main_data ,annot =main_annot .values ,fmt ="",
            cmap ='coolwarm',center =0 ,vmin =-limit ,vmax =limit ,
            cbar_kws ={'label':r'Effect Size ($\eta_p^2$) * Direction'},
            linewidths =1 ,linecolor ='white',ax =ax0 
            )
            ax0 .set_title ("Main Effects (Red=Increase, Blue=Decrease)",fontsize =12 )
            ax0 .set_ylabel ("")

            # Right: Interaction (White-Purple)
            ax1 =plt .subplot (gs [1 ])
            sns .heatmap (
            int_data ,annot =int_annot .values ,fmt ="",
            cmap ='Purples',vmin =0 ,vmax =limit ,
            cbar_kws ={'label':r'Effect Size ($\eta_p^2$)'},
            linewidths =1 ,linecolor ='white',ax =ax1 ,
            yticklabels =False 
            )
            ax1 .set_title ("Interaction (Complexity)",fontsize =12 )
            ax1 .set_ylabel ("")

            # Use r-string (rf"...") to avoid warnings
            plt .suptitle (rf"ANOVA Summary: {cat }"+"\n"+r"(Color: Effect Size $\eta_p^2$, Text: F-value)",fontsize =14 )
            plt .subplots_adjust (top =0.9 )

            fname =f"heatmap_summary_{cat }.png"
            plt .savefig (os .path .join (self .output_dir_summary ,fname ),dpi =100 ,bbox_inches ='tight')
            plt .close ()

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

    # DGT and CDU integration into organizational data is unnecessary, so it has been removed.
    # CCC scores and the like can be analyzed using organizational data alone
    # Delete the dept_cols = [] ... block

    return df_org ,df_dept 

def main ():
    setup_japanese_font ()
    OUTPUT_PLOTS =os .path .join (conf .ANALYSIS_DIR ,"09_Interaction_Plots")
    OUTPUT_SUMMARY =os .path .join (conf .ANALYSIS_DIR ,"10_Analysis_Summary")

    df_org ,df_dept =load_data ()
    if df_org is None :return 

    analyzer =TwoWayAnovaAnalyzer (df_org ,df_dept ,OUTPUT_PLOTS ,OUTPUT_SUMMARY )

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

            # Deduplication
    org_metrics =list (set (org_metrics ))

    # Definition of order (priority list)
    # 1. Average system (Ave)
    # 2. Community Average (CC-Ave)
    # 3. Heterogeneity (CCH)
    # 4. Time Variation (CCTF)
    # 5. Others (such as CCC Score)

    # The order of the five basic indicators
    base_order =['P','CT','RD','CGT','CCU','DGT','CDU']# DGT and CDU may not exist in the organization, but I will include them just in case.

    def get_sort_key_org (name ):
    # Priority score (smaller is higher)
    # Ave P -> 100 + 0 = 100
    # CCH-P -> 300 + 0 = 300
        score =9999 
        sub_score =99 

        # Major classification by prefix
        if name .startswith ("Ave "):
            score =100 
            suffix =name .replace ("Ave ","")
        elif name .startswith ("CC-Ave "):
            score =200 
            suffix =name .replace ("CC-Ave ","")
        elif name .startswith ("CCH-"):
            score =300 
            suffix =name .replace ("CCH-","")
        elif name .startswith ("CCTF-"):
            score =400 
            suffix =name .replace ("CCTF-","")
        else :
            score =900 
            suffix =name # CCC Score, etc.

            # Subclassification by indicators
        if suffix in base_order :
            sub_score =base_order .index (suffix )

        return score +sub_score 

    org_metrics .sort (key =get_sort_key_org )

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

            # Deduplication
    dept_metrics =list (set (dept_metrics ))

    # Definition of order
    def get_sort_key_dept (name ):
        score =9999 
        sub_score =99 

        # Major classification by prefix
        if name .startswith ("ID-Ave "):
            score =100 
            suffix =name .replace ("ID-Ave ","")
        elif name .startswith ("IDIH-"):
            score =200 
            suffix =name .replace ("IDIH-","")
        elif name .startswith ("IDTF-"):
            score =300 
            suffix =name .replace ("IDTF-","")
        else :
            score =900 # CDC Score, etc.
            suffix =name 

            # Subclassification by indicators
        if suffix in base_order :
            sub_score =base_order .index (suffix )

        return score +sub_score 

    dept_metrics .sort (key =get_sort_key_dept )

    for metric in dept_metrics :
        analyzer .run_analysis (df_dept ,metric ,"Department_Level")

        # 3. Summary Heatmap
    print (">>> Creating Summary Heatmaps...")
    analyzer .create_summary_heatmap ()
    print (">>> All Analysis Completed.")

if __name__ =="__main__":
    main ()