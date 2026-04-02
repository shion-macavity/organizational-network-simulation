import os 
import glob 
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib .pyplot as plt 
import seaborn as sns 
import shutil 
from community import community_louvain 

# ==========================================
# 1. Setting the folder structure
# ==========================================
BASE_DIR =os .path .dirname (os .path .abspath (__file__ ))
RESULTS_DIR =os .path .normpath (os .path .join (BASE_DIR ,"../results"))
CONV_DIR =os .path .join (RESULTS_DIR ,"02_conversation")
AMOS_INPUT_DIR =os .path .join (RESULTS_DIR ,"05_amos_input")
# Specify Persona Sheet.xlsx in the simulation folder as instructed
PERSONA_PATH =os .path .join (BASE_DIR ,"Persona Sheet.xlsx")

OUTPUT_BASE =os .path .join (RESULTS_DIR ,"07_psychological_safety_analysis/仮分析/グラフ分析")# Folder/File: 07_psychological_safety_analysis/temporary analysis/graph analysis | Folder/File: 07_psychological_safety_analysis/temporary analysis/graph analysis | Folder/File: 07_psychological_safety_analysis/temporary analysis/graph analysis | Folder/File: 07_psychological_safety_analysis/仮分析/グラフ分析
EACH_RUN_DIR =os .path .join (OUTPUT_BASE ,"Each run graph")
SUMMARY_DIR =os .path .join (OUTPUT_BASE ,"summary")
MATRIX_DIR =os .path .join (SUMMARY_DIR ,"adjacency matrix")

# Delete old files (if opened, show error and stop)
if os .path .exists (OUTPUT_BASE ):
    try :
        shutil .rmtree (OUTPUT_BASE )
    except PermissionError :
        print ("!! Error: The file cannot be deleted because it is opened in Excel, etc. Please close it and try again.")
        exit ()

for d in [EACH_RUN_DIR ,SUMMARY_DIR ,MATRIX_DIR ]:
    os .makedirs (d ,exist_ok =True )

    # ==========================================
    # 2. Definition of constants and arrangement (mirror copy/line symmetry)
    # ==========================================
DEPT0_NODES =list (range (0 ,7 ))
DEPT1_NODES =list (range (7 ,14 ))
ALL_NODES =DEPT0_NODES +DEPT1_NODES 

fixed_pos ={}
def set_dept_layout (nodes ,leader_id ,offset_x ,mirror =False ):
    sub_nodes =[n for n in nodes if n !=leader_id ]
    for i ,node in enumerate (sub_nodes ):
        theta =(2 *np .pi /6 )*i 
        x ,y =np .cos (theta ),np .sin (theta )
        if mirror :x =-x 
        fixed_pos [node ]=np .array ([offset_x +x ,y ])
    fixed_pos [leader_id ]=np .array ([offset_x ,1.5 ])

set_dept_layout (DEPT0_NODES ,6 ,offset_x =-2 ,mirror =False )
set_dept_layout (DEPT1_NODES ,13 ,offset_x =2 ,mirror =True )

# --- Get agent name (1st column: ID, 2nd column: Last name) ---
agent_name_map ={}
if os .path .exists (PERSONA_PATH ):
    try :
        df_p =pd .read_excel (PERSONA_PATH ,usecols =[0 ,1 ])
        for _ ,row in df_p .iterrows ():
            agent_name_map [int (row .iloc [0 ])]=str (row .iloc [1 ])
    except Exception as e :
        print (f"Warning: Failed to load Persona Sheet: {e }")

def calculate_dept_metrics (partition ,dept_nodes ):
    dept_communities =[partition .get (n ,-1 )for n in dept_nodes ]
    counts =pd .Series (dept_communities ).value_counts ()
    split_count =len (counts )
    max_ratio =counts .max ()/len (dept_nodes )if not counts .empty else 0 
    return split_count ,max_ratio 

    # ==========================================
    # 3. Main processing: All Runs × All periods (36 ways)
    # ==========================================
all_run_metrics =[]
community_assignments =[]
run_folders =sorted (glob .glob (os .path .join (CONV_DIR ,"run_*")))

# For linking pattern ID
summary_path =os .path .join (AMOS_INPUT_DIR ,"questionnaire_summary_maxturn_08.csv")
run_pattern_map ={}
if os .path .exists (summary_path ):
    try :
        temp_df =pd .read_csv (summary_path )
        run_pattern_map =temp_df .groupby ('run_id')['pattern_id'].first ().to_dict ()
    except :pass 

print (f"Start analysis: {len (run_folders )}Processing runs...")

for run_path in run_folders :
    run_id_str =os .path .basename (run_path ).replace ("run_","")
    try :run_id =int (run_id_str )
    except :continue 

    pattern_id =run_pattern_map .get (run_id ,"unknown")

    # --- Obtain department name (obtain from results/02_conversation/run_XXX/output.csv as instructed) ---
    dept_name_map ={0 :"Dept_0",1 :"Dept_1"}
    output_csv =os .path .join (run_path ,"output.csv")
    if os .path .exists (output_csv ):
        try :
            df_out =pd .read_csv (output_csv )
            if 'dept_id'in df_out .columns and 'dept'in df_out .columns :
                for _ ,r in df_out [['dept_id','dept']].drop_duplicates ().iterrows ():
                    dept_name_map [int (r ['dept_id'])]=str (r ['dept'])
            elif 'dept'in df_out .columns :
            # If dept_id is missing, assign it to ID 0, 1
                u_depts =df_out ['dept'].unique ()
                for i ,d_name in enumerate (u_depts ):
                    if i <2 :dept_name_map [i ]=str (d_name )
        except :pass 

    conv_csv =os .path .join (run_path ,"conversation_history.csv")
    if not os .path .exists (conv_csv ):continue 

    df_all_conv =pd .read_csv (conv_csv )

    # Total 36 pattern period loops
    for start_t in range (1 ,9 ):
        for end_t in range (start_t ,9 ):
            range_name =f"turn{start_t }to{end_t }"
            df_conv =df_all_conv [(df_all_conv ['Turn']>=start_t )&(df_all_conv ['Turn']<=end_t )]

            G =nx .Graph ()
            G .add_nodes_from (ALL_NODES )
            for _ ,row in df_conv .iterrows ():
                if row .get ('Status')=='Success':
                    u ,v =int (row ['Speaker_ID']),int (row ['Target_ID'])
                    if G .has_edge (u ,v ):G [u ][v ]['weight']+=1 
                    else :G .add_edge (u ,v ,weight =1 )

                    # Storing adjacency matrices
            matrix_range_dir =os .path .join (MATRIX_DIR ,range_name )
            os .makedirs (matrix_range_dir ,exist_ok =True )
            nx .to_pandas_adjacency (G ,nodelist =ALL_NODES ).to_csv (os .path .join (matrix_range_dir ,f"run_{run_id_str }_matrix.csv"))

            # community detection
            if G .number_of_edges ()>0 :
                partition =community_louvain .best_partition (G ,weight ='weight')
                modularity =community_louvain .modularity (partition ,G ,weight ='weight')
            else :
                partition ={n :0 for n in ALL_NODES };modularity =0 

                # Detailed data for each agent is recorded only when 1to8
            if range_name =="turn1to8":
                for aid in ALL_NODES :
                    d_id =0 if aid <7 else 1 
                    community_assignments .append ({
                    'run_id':int (run_id ),
                    'agent_id':int (aid ),
                    'agent_name':agent_name_map .get (aid ,f"Agent_{aid }"),
                    'dept_id':int (d_id ),
                    'dept_name':dept_name_map .get (d_id ),
                    'community_id':int (partition .get (aid ,0 ))
                    })

            d0_split ,d0_purity =calculate_dept_metrics (partition ,DEPT0_NODES )
            d1_split ,d1_purity =calculate_dept_metrics (partition ,DEPT1_NODES )

            all_run_metrics .append ({
            'run_id':int (run_id ),'pattern_id':pattern_id ,'range':range_name ,
            'modularity_q':modularity ,'dept0_split':d0_split ,'dept1_split':d1_split 
            })

            # Drawing and saving graph images
            range_dir =os .path .join (EACH_RUN_DIR ,range_name )
            os .makedirs (range_dir ,exist_ok =True )
            plt .figure (figsize =(10 ,7 ))
            weights =[G [u ][v ]['weight']*0.7 for u ,v in G .edges ()]
            node_colors =[plt .get_cmap ('tab20')(partition .get (n ,0 ))for n in ALL_NODES ]
            nx .draw_networkx_nodes (G ,fixed_pos ,node_color =node_colors ,node_size =600 ,edgecolors ='black')
            nx .draw_networkx_edges (G ,fixed_pos ,width =weights ,edge_color ='gray',alpha =0.3 )
            nx .draw_networkx_labels (G ,fixed_pos ,font_size =8 ,font_weight ='bold')
            plt .title (f"Run {run_id_str } [{range_name }] - Q: {modularity :.3f}")
            plt .axis ('off')
            plt .savefig (os .path .join (range_dir ,f"run_{run_id_str }_network.png"),bbox_inches ='tight')
            plt .close ()

    if run_id %10 ==0 :print (f"Progress: Run {run_id } completion...")

    # ==========================================
    # 4. Database output
    # ==========================================
pd .DataFrame (all_run_metrics ).to_csv (os .path .join (SUMMARY_DIR ,"silo_summary_timeseries.csv"),index =False )
pd .DataFrame (community_assignments ).to_csv (os .path .join (SUMMARY_DIR ,"community_assignments_final_1to8.csv"),index =False )

# Final violin plot
df_final =pd .DataFrame (all_run_metrics )
df_final =df_final [df_final ['range']=='turn1to8']
if not df_final .empty :
    plt .figure (figsize =(10 ,6 ))
    sns .violinplot (x ='pattern_id',y ='modularity_q',data =df_final ,hue ='pattern_id',palette ="Set3",legend =False )
    plt .title ("Modularity Q (Turn 1 to 8 Cumulative)")
    plt .savefig (os .path .join (SUMMARY_DIR ,"modularity_violin_final.png"))
    plt .close ()

print (f"All processes completed. The result is {OUTPUT_BASE } Saved in.")