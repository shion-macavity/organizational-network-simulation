import os 
import glob 
import pandas as pd 
import numpy as np 
import sys 
from dataclasses import dataclass 

# ==========================================
# Settings: Config (from config_simulation.py)
# ==========================================
@dataclass 
class SimulationConfig :
# --- Basic Network Settings ---
    n_runs :int =100 
    network_results_dir :str ="results/01_network"

    # --- Random Number Seed Control Settings ---
    PATTERN_CYCLE =4 
    CUSTOMER_SEED_OFFSET =10000 

    # --- Conversation Simulation Settings ---
    n_turns :int =8 
    max_talk_history :int =8 
    conversation_results_dir :str ="results/02_conversation"
    conversation_max_parallel :int =7 
    report_max_parallel :int =2 

    # --- AI Model Settings ---
    gemini_model :str ="gemini-2.5-flash-lite"
    gemini_advanced_model :str ="gemini-2.5-flash"
    temperature :float =1.0 
    fixed_pattern_id :int =None 

    # Settings for each phase
    tokens_phase1_talk :int =1024 
    budget_phase1_talk :int =524 
    verbosity_phase1_talk :str ="low"

    tokens_phase2_reflect :int =2048 
    budget_phase2_reflect :int =1024 
    verbosity_phase2_reflect :str ="medium"

    tokens_phase3_report :int =4096 
    budget_phase3_report :int =2048 
    verbosity_phase3_report :str ="medium"

    tokens_survey :int =50 
    budget_survey :int =0 
    verbosity_survey :str ="low"

    tokens_select_target :int =50 
    budget_select_target :int =0 
    verbosity_select_target :str ="low"

    tokens_customer_feedback :int =574 
    budget_customer_feedback :int =524 
    verbosity_customer_feedback :str ="low"

    product_evaluation_dir :str ="results/04_product_eval_ssr"

config =SimulationConfig ()

# ==========================================
# Directory Settings
# ==========================================
RESULTS_DIR ="results"
CONVERSATION_DIR =os .path .join (RESULTS_DIR ,"02_conversation")
QUESTIONNAIRE_DIR =os .path .join (RESULTS_DIR ,"03_questionnaire_ssr")
PRODUCT_EVAL_DIR =os .path .join (RESULTS_DIR ,"04_product_eval_ssr")

OUTPUT_BASE_DIR =os .path .join (RESULTS_DIR ,"05_amos_input")
TURN_ANALYSIS_DIR =os .path .join (OUTPUT_BASE_DIR ,"turn_analysis")

# Agent Definition
ALL_AGENTS =list (range (14 ))
DEPT_0_AGENTS =[0 ,1 ,2 ,3 ,4 ,5 ,6 ]# 0-6
DEPT_1_AGENTS =[7 ,8 ,9 ,10 ,11 ,12 ,13 ]# 7-13

# ==========================================
# Helper function
# ==========================================

def get_pattern_attributes (pattern_id ):
    """
    Return dummy variables based on the pattern ID
Instruction: If the pattern is 1, then 00; if 2, then 10; if 3, then 01; if 4, then 11
    """
    try :
        pid =int (pattern_id )
        if pid ==1 :
            return 0 ,0 # is_proximal=0, has_informal=0
        elif pid ==2 :
            return 1 ,0 # is_proximal=1, has_informal=0
        elif pid ==3 :
            return 0 ,1 # is_proximal=0, has_informal=1
        elif pid ==4 :
            return 1 ,1 # is_proximal=1, has_informal=1
        else :
            return np .nan ,np .nan 
    except :
        return np .nan ,np .nan 

def calculate_interactions (run_id ,n_turns ):
    """
    From the conversation log of the specified Run, calculate the "inter-department interaction count (active/passive)" for each agent for each turn. Condition: Speaker_Dept_ID != Target_Dept_ID
    """
    run_folder =f"run_{int (run_id ):03d}"
    hist_path =os .path .join (CONVERSATION_DIR ,run_folder ,"conversation_history.csv")

    # Create a frame for all agents × all turns (for zero-padding)
    records =[]
    for t in range (1 ,n_turns +1 ):
        for ag in ALL_AGENTS :
            records .append ({'turn':t ,'agent_id':ag ,'active_interaction':0 ,'passive_interaction':0 })
    df_base =pd .DataFrame (records )

    if not os .path .exists (hist_path ):
        return df_base 

    try :
        df_hist =pd .read_csv (hist_path )
    except Exception as e :
        print (f"Error reading {hist_path }: {e }")
        return df_base 

    if df_hist .empty :
        return df_base 

    cols =df_hist .columns 
    if 'Speaker_Dept_ID'not in cols or 'Target_Dept_ID'not in cols :
        print (f"Warning: Dept ID columns missing in {hist_path }")
        return df_base 

        # Extract only inter-department interactions
    df_inter =df_hist [df_hist ['Speaker_Dept_ID']!=df_hist ['Target_Dept_ID']].copy ()

    # --- Active (Number of times I spoke) ---
    active_counts =df_inter .groupby (['Turn','Speaker_ID']).size ().reset_index (name ='active_count')
    active_counts =active_counts .rename (columns ={'Turn':'turn','Speaker_ID':'agent_id','active_count':'active_interaction'})

    # --- Passive (Number of times spoken to) ---
    passive_counts =df_inter .groupby (['Turn','Target_ID']).size ().reset_index (name ='passive_count')
    passive_counts =passive_counts .rename (columns ={'Turn':'turn','Target_ID':'agent_id','passive_count':'passive_interaction'})

    # Merge into Base (Active)
    df_merged =pd .merge (df_base ,active_counts ,on =['turn','agent_id'],how ='left')
    df_merged ['active_interaction']=df_merged ['active_interaction_y'].fillna (0 ).astype (int )

    # Merge into base (Passive)
    df_merged =pd .merge (df_merged ,passive_counts ,on =['turn','agent_id'],how ='left')
    df_merged ['passive_interaction']=df_merged ['passive_interaction_y'].fillna (0 ).astype (int )

    # Delete unnecessary columns
    cols_to_drop =[c for c in df_merged .columns if '_x'in c or '_y'in c ]
    df_merged =df_merged .drop (columns =cols_to_drop )

    # Total
    df_merged ['interaction']=df_merged ['active_interaction']+df_merged ['passive_interaction']

    return df_merged 

def main ():
    print ("--- Start Pre-Data Shaping ---")

    # Create directory
    os .makedirs (OUTPUT_BASE_DIR ,exist_ok =True )
    os .makedirs (TURN_ANALYSIS_DIR ,exist_ok =True )

    # 1. Loading customer scores (handling variations in file names)
    possible_files =[
    "customer_scores.csv",
    "customer_scores_summary.csv"
    ]

    score_file_path =None 
    for fname in possible_files :
        path =os .path .join (PRODUCT_EVAL_DIR ,fname )
        if os .path .exists (path ):
            score_file_path =path 
            print (f"Found customer score file: {path }")
            break 

    if score_file_path is None :
        print (f"Error: Customer scores file not found in {PRODUCT_EVAL_DIR }.")
        print (f"Tried: {possible_files }")
        sys .exit (1 )

    df_cust =pd .read_csv (score_file_path )

    # Numeric conversion and missing value handling
    if 'avg_purchase_score'in df_cust .columns :
        df_cust ['avg_purchase_score']=pd .to_numeric (df_cust ['avg_purchase_score'],errors ='coerce')
    else :
        print ("Error: 'avg_purchase_score' column not found.")
        sys .exit (1 )

        # Create a score dictionary for each run
    customer_score_map ={}

    if 'run_id'in df_cust .columns and 'proposal_dept_id'in df_cust .columns :
        run_ids_cust =df_cust ['run_id'].unique ()
        for rid in run_ids_cust :
            df_r =df_cust [df_cust ['run_id']==rid ]

            s0 =df_r [df_r ['proposal_dept_id']==0 ]['avg_purchase_score'].mean ()
            s1 =df_r [df_r ['proposal_dept_id']==1 ]['avg_purchase_score'].mean ()
            s_ave =df_r ['avg_purchase_score'].mean ()

            customer_score_map [int (rid )]={
            'output_score_dept0':s0 ,
            'output_score_dept1':s1 ,
            'output_score_ave':s_ave 
            }
        print (f"Loaded scores for {len (customer_score_map )} runs.")
    else :
        print ("Error: Required columns ('run_id', 'proposal_dept_id') not found.")
        sys .exit (1 )

        # 2. Processing all data and creating a consolidated file
    all_summary_data =[]

    # Obtain survey file
    q_files =glob .glob (os .path .join (QUESTIONNAIRE_DIR ,"questionnaire_details_run_*.csv"))
    print (f"Found {len (q_files )} questionnaire files.")

    for qf in q_files :
        filename =os .path .basename (qf )
        try :
            run_id =int (filename .replace ("questionnaire_details_run_","").replace (".csv",""))
        except :
            continue 

        if run_id %10 ==0 :
            print (f"Processing Run {run_id }...")

            # Loading questionnaire
        df_q =pd .read_csv (qf )

        # --- Safety Device: Turn Count Check ---
        actual_max_turn =df_q ['turn'].max ()
        if actual_max_turn !=config .n_turns :
            print (f"CRITICAL ERROR: Run {run_id } has max turn {actual_max_turn }, but config specifies {config .n_turns }.")
            print ("Stopping analysis to prevent data corruption.")
            sys .exit (1 )

            # Cleaning of ssr_score
        if 'ssr_score'in df_q .columns and df_q ['ssr_score'].dtype ==object :
             df_q ['ssr_score']=df_q ['ssr_score'].astype (str ).str .replace ("'","").astype (float )

             # Get Pattern ID
        pattern_id =df_q ['pattern_id'].iloc [0 ]if 'pattern_id'in df_q .columns else np .nan 

        # Interaction Data Acquisition
        df_inter =calculate_interactions (run_id ,config .n_turns )

        # Create a sequence of question ID numbers
        def get_q_num (qid_str ):
            try :
                return int (str (qid_str ).replace ("Q",""))
            except :
                return 999 
        df_q ['q_num']=df_q ['question_id'].apply (get_q_num )

        # Aggregated per agent × turn
        for turn in range (1 ,config .n_turns +1 ):
            for agent_id in ALL_AGENTS :
            # Extraction of relevant data
                sub_df =df_q [(df_q ['turn']==turn )&(df_q ['agent_id']==agent_id )]

                dept_id =0 if agent_id in DEPT_0_AGENTS else 1 
                position ="General Member"# Default

                if not sub_df .empty :
                    dept_id =sub_df ['dept_id'].iloc [0 ]
                    position =sub_df ['position'].iloc [0 ]

                    # Score calculation for each category
                    s_psych =sub_df [(sub_df ['q_num']>=1 )&(sub_df ['q_num']<=19 )]['ssr_score'].mean ()
                    s_lead =sub_df [(sub_df ['q_num']>=1 )&(sub_df ['q_num']<=9 )]['ssr_score'].mean ()
                    s_coll =sub_df [(sub_df ['q_num']>=10 )&(sub_df ['q_num']<=16 )]['ssr_score'].mean ()
                    s_team =sub_df [(sub_df ['q_num']>=17 )&(sub_df ['q_num']<=19 )]['ssr_score'].mean ()
                    s_coh =sub_df [(sub_df ['q_num']>=20 )&(sub_df ['q_num']<=23 )]['ssr_score'].mean ()
                else :
                # No data (NaN)
                    s_psych ,s_lead ,s_coll ,s_team ,s_coh =np .nan ,np .nan ,np .nan ,np .nan ,np .nan 

                record ={
                'run_id':run_id ,
                'pattern_id':pattern_id ,
                'turn':turn ,
                'agent_id':agent_id ,
                'dept_id':dept_id ,
                'position':position ,
                'psychological_safety':s_psych ,
                'psych_leader':s_lead ,
                'psych_colleague':s_coll ,
                'psych_team':s_team ,
                'task_cohesion':s_coh 
                }

                # Granting of interaction information
                inter_row =df_inter [(df_inter ['turn']==turn )&(df_inter ['agent_id']==agent_id )]
                if not inter_row .empty :
                    record ['active_interaction']=inter_row ['active_interaction'].iloc [0 ]
                    record ['passive_interaction']=inter_row ['passive_interaction'].iloc [0 ]
                    record ['interaction']=inter_row ['interaction'].iloc [0 ]
                else :
                    record ['active_interaction']=0 
                    record ['passive_interaction']=0 
                    record ['interaction']=0 

                all_summary_data .append (record )

                # 3. Saving the integrated file
    df_summary_all =pd .DataFrame (all_summary_data )
    summary_filename =f"questionnaire_summary_maxturn_{config .n_turns :02d}.csv"
    summary_path =os .path .join (OUTPUT_BASE_DIR ,summary_filename )
    df_summary_all .to_csv (summary_path ,index =False )
    print (f"Created Summary File: {summary_path } (Rows: {len (df_summary_all )})")

    # 4. Turn Analysis
    print ("--- Starting Turn Analysis (Creating 36 folders) ---")

    for start_t in range (1 ,config .n_turns +1 ):
        for end_t in range (start_t ,config .n_turns +1 ):

        # Create folder
            folder_name =f"turn_{start_t }to{end_t }"
            folder_path =os .path .join (TURN_ANALYSIS_DIR ,folder_name )
            os .makedirs (folder_path ,exist_ok =True )

            # Period Filter
            df_subset =df_summary_all [(df_summary_all ['turn']>=start_t )&(df_summary_all ['turn']<=end_t )]

            # === basedata (Agent Level) ===
            agg_funcs ={
            'pattern_id':'first',
            'dept_id':'first',
            'position':'first',
            'psychological_safety':'mean',
            'psych_leader':'mean',
            'psych_colleague':'mean',
            'psych_team':'mean',
            'task_cohesion':'mean',
            'active_interaction':'sum',
            'passive_interaction':'sum',
            'interaction':'sum'
            }

            # To prevent NaN from being excluded during groupby, dropna is not used, but runs with no data may disappear.
            df_base =df_subset .groupby (['run_id','agent_id'],as_index =False ).agg (agg_funcs )

            # Change column name
            df_base =df_base .rename (columns ={
            'active_interaction':'sum_active_interaction',
            'passive_interaction':'sum_passive_interaction',
            'interaction':'sum_interaction'
            })

            # Attribute Assignment
            attrs =df_base ['pattern_id'].apply (get_pattern_attributes )
            df_base ['is_proximal']=[x [0 ]for x in attrs ]
            df_base ['has_informal']=[x [1 ]for x in attrs ]

            # Grant Output Score
            out_ave_list ,out_d0_list ,out_d1_list =[],[],[]
            for rid in df_base ['run_id']:
                scores =customer_score_map .get (rid ,{'output_score_ave':np .nan ,'output_score_dept0':np .nan ,'output_score_dept1':np .nan })
                out_ave_list .append (scores ['output_score_ave'])
                out_d0_list .append (scores ['output_score_dept0'])
                out_d1_list .append (scores ['output_score_dept1'])

            df_base ['output_score_ave']=out_ave_list 
            df_base ['output_score_dept0']=out_d0_list 
            df_base ['output_score_dept1']=out_d1_list 

            # Format and save
            df_base =df_base .sort_values (['run_id','agent_id'])
            # Organize column order for visibility
            cols_order =['run_id','pattern_id','agent_id','dept_id','is_proximal','has_informal',
            'sum_interaction','psychological_safety','psych_leader','psych_colleague','psych_team','task_cohesion',
            'output_score_ave','output_score_dept0','output_score_dept1']
            # Add the remaining columns
            remaining =[c for c in df_base .columns if c not in cols_order ]
            df_base =df_base [cols_order +remaining ]

            base_file =os .path .join (folder_path ,f"basedata_{start_t }_to_{end_t }.csv")
            df_base .to_csv (base_file ,index =False )

            # === summary (Run Level) ===
            agg_funcs_summary ={
            'pattern_id':'first',
            'is_proximal':'first',
            'has_informal':'first',
            'psychological_safety':'mean',
            'psych_leader':'mean',
            'psych_colleague':'mean',
            'psych_team':'mean',
            'task_cohesion':'mean',
            'sum_interaction':'sum',
            'sum_active_interaction':'sum',
            'sum_passive_interaction':'sum',
            'output_score_ave':'first',
            'output_score_dept0':'first',
            'output_score_dept1':'first'
            }

            df_run_summary =df_base .groupby ('run_id',as_index =False ).agg (agg_funcs_summary )

            df_run_summary =df_run_summary .rename (columns ={
            'sum_interaction':'sum_sum_interaction',
            'sum_active_interaction':'sum_sum_active_interaction',
            'sum_passive_interaction':'sum_sum_passive_interaction'
            })

            summary_file =os .path .join (folder_path ,f"summary_{start_t }to{end_t }.csv")
            df_run_summary .to_csv (summary_file ,index =False )

    print ("--- All processes completed successfully ---")

if __name__ =="__main__":
    main ()