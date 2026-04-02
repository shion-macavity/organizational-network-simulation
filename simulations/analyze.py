import os 
import glob 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import seaborn as sns 

# ==========================================
# Settings: path and font
# ==========================================
RESULTS_DIR ="results"
QUESTIONNAIRE_DIR =os .path .join (RESULTS_DIR ,"03_questionnaire_ssr")
AMOS_INPUT_DIR =os .path .join (RESULTS_DIR ,"05_amos_input")

# Output folder
PLOT_DIR_RUN =os .path .join (AMOS_INPUT_DIR ,"Progress chart for each run")
PLOT_DIR_PATTERN =os .path .join (AMOS_INPUT_DIR ,"Transition chart for each pattern")
PLOT_DIR_SCATTER =os .path .join (AMOS_INPUT_DIR ,"Preliminary analysis scatter plot")

# Japanese font settings (may need to be adjusted to suit your environment)
# For Linux/Ubuntu, specify IPAexGothic etc.
plt .rcParams ['font.family']='sans-serif'
# plt.rcParams['font.sans-serif'] = ['IPAexGothic', 'Noto Sans CJK JP', 'Hiragino Maru Gothic Pro'] 
# *If the characters are garbled, please specify the font according to your environment. This time, proceed with the default, but if you want tofu (□), use only alphanumeric characters.

# group definition
LEADER_IDS =[6 ,13 ]
BOUNDARY_IDS =[0 ,7 ]
SUBORDINATE_IDS =[i for i in range (14 )if i not in LEADER_IDS ]

# ==========================================
# function definition
# ==========================================

def plot_trajectory (df ,title ,filename ,output_dir ):
    """Common functions to draw and save transition graphs"""
    plt .figure (figsize =(10 ,6 ))

    # group by turns
    turns =sorted (df ['turn'].unique ())

    # Calculate data for 4 lines
    line_data ={
    "All Agents":df .groupby ('turn')['ssr_score'].mean (),
    "Subordinates (Excl. Leaders)":df [df ['agent_id'].isin (SUBORDINATE_IDS )].groupby ('turn')['ssr_score'].mean (),
    "Leaders (ID 6, 13)":df [df ['agent_id'].isin (LEADER_IDS )].groupby ('turn')['ssr_score'].mean (),
    "Boundary Spanners (ID 0, 7)":df [df ['agent_id'].isin (BOUNDARY_IDS )].groupby ('turn')['ssr_score'].mean ()
    }

    # plot
    markers =['o','s','^','D']
    colors =['blue','green','red','orange']

    for (label ,series ),marker ,color in zip (line_data .items (),markers ,colors ):
        plt .plot (series .index ,series .values ,label =label ,marker =marker ,color =color ,alpha =0.7 )

    plt .title (title )
    plt .xlabel ("Turn")
    plt .ylabel ("Score")
    plt .legend ()
    plt .grid (True ,linestyle ='--',alpha =0.6 )

    os .makedirs (output_dir ,exist_ok =True )
    plt .savefig (os .path .join (output_dir ,filename ))
    plt .close ()

def plot_pattern_comparison (all_runs_df ,metric_name ,q_filter_func ,output_dir ):
    """Graph of comparison between patterns (comparing the trends in the average of all employees for each pattern)"""
    plt .figure (figsize =(10 ,6 ))

    # Filtering by metric (psychological or agglomerative)
    # Here, we will briefly explain whether to use the average of all questions or to use a specific set of questions.
    # Assuming that data including all questions is received once.

    patterns =sorted (all_runs_df ['pattern_id'].unique ())

    for pid in patterns :
    # Data extraction of that pattern
        pat_df =all_runs_df [all_runs_df ['pattern_id']==pid ]

        # Filtering by question ID (psychological or agglomerative)
        pat_df =pat_df [pat_df ['question_id'].apply (q_filter_func )]

        # Average per turn (including all Runs and all Agents)
        series =pat_df .groupby ('turn')['ssr_score'].mean ()

        plt .plot (series .index ,series .values ,label =f"Pattern {pid }",marker ='o',alpha =0.8 )

    plt .title (f"Comparison by Pattern: {metric_name }")
    plt .xlabel ("Turn")
    plt .ylabel (f"Average {metric_name } Score")
    plt .legend ()
    plt .grid (True )

    os .makedirs (output_dir ,exist_ok =True )
    plt .savefig (os .path .join (output_dir ,f"comparison_{metric_name }.png"))
    plt .close ()

def plot_group_pattern_comparison (all_runs_df ,group_ids ,group_label ,metric_name ,q_filter_func ,output_dir ):
    """Compare the trends of patterns 1 to 4 for a specific group (leaders, etc.) in one graph"""
    plt .figure (figsize =(10 ,6 ))

    # Filter by question ID (psychological safety or group cohesion)
    df_filtered =all_runs_df [all_runs_df ['question_id'].apply (q_filter_func )]
    # Filter by specified employee group (leader, etc.)
    df_filtered =df_filtered [df_filtered ['agent_id'].isin (group_ids )]

    patterns =sorted (df_filtered ['pattern_id'].unique ())
    # Loop patterns 1 to 4 and overlap the graphs
    for pid in patterns :
        series =df_filtered [df_filtered ['pattern_id']==pid ].groupby ('turn')['ssr_score'].mean ()
        plt .plot (series .index ,series .values ,label =f"Pattern {pid }",marker ='o',alpha =0.8 )

    plt .title (f"Comparison by Pattern ({group_label }): {metric_name }")
    plt .xlabel ("Turn")
    plt .ylabel (f"Average {metric_name } Score")
    plt .legend ()
    plt .grid (True ,linestyle ='--',alpha =0.6 )

    os .makedirs (output_dir ,exist_ok =True )
    # Include group name and index name in file name
    filename =f"comparison_pattern_{group_label .replace (' ','_')}_{metric_name }.png"
    plt .savefig (os .path .join (output_dir ,filename ))
    plt .close ()

def main ():
# Loading data
    quest_files =glob .glob (os .path .join (QUESTIONNAIRE_DIR ,"questionnaire_details_run_*.csv"))

    if not quest_files :
        print ("No questionnaire files found.")
        return 

        # DataFrame that combines all data (for pattern analysis)
    all_data =[]

    print ("Generating plots...")

    for q_file in quest_files :
        df =pd .read_csv (q_file )

        # Basic information such as Run ID
        run_id =int (os .path .basename (q_file ).split ('_run_')[1 ].split ('.')[0 ])
        pattern_id =df ['pattern_id'].iloc [0 ]

        all_data .append (df )

        # Classify question ID (assume Q01-Q19: psychological, Q20-23: agglomeration)
        # *Actually, please adjust according to your settings.
        def is_psych (qid ):return int (qid .replace ("Q",""))<=19 
        def is_cohesion (qid ):return int (qid .replace ("Q",""))>19 

        df_psych =df [df ['question_id'].apply (is_psych )]
        df_cohesion =df [df ['question_id'].apply (is_cohesion )]

        # 1. Progress chart for each run
        if not df_psych .empty :
            plot_trajectory (df_psych ,f"Run {run_id } (Pattern {pattern_id }) - Psych Safety",
            f"run_{run_id :03d}_psych.png",os .path .join (PLOT_DIR_RUN ))

        if not df_cohesion .empty :
            plot_trajectory (df_cohesion ,f"Run {run_id } (Pattern {pattern_id }) - Cohesion",
            f"run_{run_id :03d}_cohesion.png",os .path .join (PLOT_DIR_RUN ))

            # Combine all data
    df_all =pd .concat (all_data ,ignore_index =True )

    # 2. Trend chart for each pattern (average)
    for pid in sorted (df_all ['pattern_id'].unique ()):
        pat_df =df_all [df_all ['pattern_id']==pid ]

        # psychology
        pat_psych =pat_df [pat_df ['question_id'].apply (lambda x :int (x .replace ("Q",""))<=19 )]
        plot_trajectory (pat_psych ,f"Pattern {pid } Average - Psych Safety",
        f"pattern_{pid }_psych.png",PLOT_DIR_PATTERN )

        # agglomeration
        pat_cohesion =pat_df [pat_df ['question_id'].apply (lambda x :int (x .replace ("Q",""))>19 )]
        plot_trajectory (pat_cohesion ,f"Pattern {pid } Average - Cohesion",
        f"pattern_{pid }_cohesion.png",PLOT_DIR_PATTERN )

        # 3. Comparison between patterns (superposition)
    plot_pattern_comparison (df_all ,"PsychSafety",lambda x :int (x .replace ("Q",""))<=19 ,PLOT_DIR_PATTERN )
    plot_pattern_comparison (df_all ,"Cohesion",lambda x :int (x .replace ("Q",""))>19 ,PLOT_DIR_PATTERN )

    # 4. [Additional] Comparison between patterns for each specific group (created 6 graphs)
    groups =[
    (LEADER_IDS ,"Leaders"),
    (BOUNDARY_IDS ,"Boundary Spanners"),
    (SUBORDINATE_IDS ,"Subordinates")
    ]

    metrics =[
    ("PsychSafety",lambda x :int (x .replace ("Q",""))<=19 ),
    ("Cohesion",lambda x :int (x .replace ("Q",""))>19 )
    ]

    for g_ids ,g_label in groups :
        for m_name ,m_filter in metrics :
            plot_group_pattern_comparison (df_all ,g_ids ,g_label ,m_name ,m_filter ,PLOT_DIR_PATTERN )

            # 4. Scatter plot (using AMOS data)
            # Assuming that amos_dataset_run_level.csv has been created
    amos_file =os .path .join (AMOS_INPUT_DIR ,"amos_dataset_run_level.csv")
    if os .path .exists (amos_file ):
        df_amos =pd .read_csv (amos_file )

        # Scatter plot settings (plot the correspondence by department)
        # Combination: (X axis, Y axis, title)
        scatter_pairs =[
        # --- Analysis of the entire organization (All) ---
        ('psych_ave_all','output_score_ave','All: Psych vs Output'),
        ('cohesion_ave_all','output_score_ave','All: Cohesion vs Output'),
        # -- By department ---
        ('psych_dept0','output_score_1','Dept 0: Psych vs Output'),
        ('psych_dept1','output_score_2','Dept 1: Psych vs Output'),
        ('cohesion_dept0','output_score_1','Dept 0: Cohesion vs Output'),
        ('cohesion_dept1','output_score_2','Dept 1: Cohesion vs Output')
        ]

        os .makedirs (PLOT_DIR_SCATTER ,exist_ok =True )

        for x_var ,y_var ,title in scatter_pairs :
            if x_var in df_amos .columns and y_var in df_amos .columns :
                plt .figure (figsize =(8 ,6 ))
                sns .scatterplot (
                data =df_amos ,x =x_var ,y =y_var ,
                hue ='pattern_id',style ='pattern_id',
                s =100 
                )
                plt .title (title )
                plt .grid (True )
                plt .savefig (os .path .join (PLOT_DIR_SCATTER ,f"scatter_{x_var }_vs_{y_var }.png"))
                plt .close ()


    """"
        # Scatter plot settings
        x_vars = ['psych_ave_all', 'cohesion_ave_all']
        y_vars = ['output_score_ave', 'output_score_1', 'output_score_2']
        
        os.makedirs(PLOT_DIR_SCATTER, exist_ok=True)
        
        for x_var in x_vars:
            for y_var in y_vars:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df_amos, x=x_var, y=y_var, hue='pattern_id', style='pattern_id',
                                palette='viridis', s=100)
                plt.title(f"{x_var} vs {y_var}")
                plt.grid(True)
                plt.savefig(os.path.join(PLOT_DIR_SCATTER, f"scatter_{x_var}_vs_{y_var}.png"))
                plt.close()

    """

    print ("All plots generated successfully.")

if __name__ =="__main__":
    main ()