import os 
import sys 
import pandas as pd 
import numpy as np 
import glob 
from tqdm import tqdm 

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Common Logic Definition (Common Utility Functions)
# =========================================================

def get_pattern_id (run_id ):
    """Determine the pattern ID (1-4) from the RunID (classification based on the remainder when divided by 4)"""
    remainder =run_id %4 
    if remainder ==0 :
        return 4 
    else :
        return remainder 

def calc_silo_metrics (matrix_df ,member_ids ):
    """Calculate the EI index from the adjacency matrix and the member ID list"""
    # Type conversion and column identification
    member_ids_int =[int (m )for m in member_ids ]
    member_ids_str =[str (m )for m in member_ids ]
    all_cols =[c for c in matrix_df .columns if c !='']
    external_cols =[c for c in all_cols if c not in member_ids_str ]

    # Number of internal conversations (halve it since diagonal elements are ignored and assuming a symmetric matrix)
    internal_val =0 
    for i in member_ids_int :
        if i in matrix_df .index :
            for j_str in member_ids_str :
                if j_str in matrix_df .columns :
                    internal_val +=matrix_df .loc [i ,j_str ]
    internal_convs =internal_val /2 

    # Number of external conversations (utterances from members to outside)
    external_val =0 
    for i in member_ids_int :
        if i in matrix_df .index :
            for j_str in external_cols :
                if j_str in matrix_df .columns :
                    external_val +=matrix_df .loc [i ,j_str ]

    total =internal_convs +external_val 
    # EI calculation: (External - Internal) / (External + Internal)
    ei_index =(external_val -internal_convs )/total if total >0 else np .nan 
    has_external =1 if external_val >0 else 0 
    return ei_index ,has_external 

def safe_stats (series ):
    """Safely calculate the mean and unbiased standard deviation from a Series"""
    if series .empty or len (series )<2 :
        return np .nan ,np .nan 
    return series .mean (),series .std (ddof =1 )

    # =========================================================
    # Main process: Data integration and formatting
    # =========================================================

def load_and_merge_ssr_data ():
    """
    Load the SSR scores (P, CT, RD, Consistency) and integrate them into a single large DataFrame (all_metrics_raw.csv).
    """
    print (">>> Loading SSR Data...")
    all_records =[]

    # Processing for each run
    for run_id in tqdm (conf .RUN_IDS ,desc ="Processing SSR Runs"):
    # --- 1. Progress (P) ---
    # Include rallies from both the Speaker and the Listener
        p_files =[
        ("speaker",conf .FILE_PATTERN_P .format (run_id )),
        ("listener",conf .FILE_PATTERN_P .format (run_id ).replace ("speaker","listener"))
        ]

        for role_type ,fname in p_files :
            fpath_p =os .path .join (conf .SSR_RESULTS_DIR ,"Analysis 1_Raw Data",fname )

            if os .path .exists (fpath_p ):
                try :
                    df =pd .read_csv (fpath_p ,encoding =conf .ENCODING )
                    for _ ,row in df .iterrows ():
                    # Attempt to obtain the other party's ID
                        partner_id =row .get ('partner_id')
                        if pd .isna (partner_id ):partner_id =row .get ('listener_id')
                        if pd .isna (partner_id ):partner_id =row .get ('speaker_id')

                        all_records .append ({
                        'run_id':run_id ,
                        'turn':row .get ('turn'),
                        'agent_id':row .get ('agent_id'),
                        'metric_type':'P',# Progress
                        'ssr_score':row .get ('ssr_score'),
                        'target_id':partner_id if pd .notna (partner_id )else np .nan ,
                        'extra_flag':role_type 
                        })
                except Exception as e :
                    print (f"Warning: Failed to read {fname }: {e }")
            else :
                pass # Skip if the file is not present

                # --- 2. Critical Review (CT) ---
        ct_files =[
        ("listener",conf .FILE_PATTERN_CT .format (run_id )),
        ("speaker",conf .FILE_PATTERN_CT .format (run_id ).replace ("listener","speaker"))
        ]

        for role_type ,fname in ct_files :
            fpath_ct =os .path .join (conf .SSR_RESULTS_DIR ,"Analysis 4_Raw Data",fname )

            # Fallback when there is no _listener
            if role_type =="listener"and not os .path .exists (fpath_ct ):
                fallback =os .path .join (conf .SSR_RESULTS_DIR ,"Analysis 4_Raw Data",f"run_{run_id }_4_interaction.csv")
                if os .path .exists (fallback ):
                    fpath_ct =fallback 

            if os .path .exists (fpath_ct ):
                try :
                    df =pd .read_csv (fpath_ct ,encoding =conf .ENCODING )
                    for _ ,row in df .iterrows ():
                    # Identify the party performing the CT and the counterpart
                        agent_id =None 
                        target_id =None 

                        if role_type =="listener":
                            agent_id =row .get ('listener_id',row .get ('agent_id'))
                            target_id =row .get ('speaker_id')
                        else :# speaker
                            agent_id =row .get ('speaker_id',row .get ('agent_id'))
                            target_id =row .get ('listener_id')

                        if pd .notna (agent_id ):
                            all_records .append ({
                            'run_id':run_id ,
                            'turn':row .get ('turn'),
                            'agent_id':agent_id ,
                            'metric_type':'CT',# Critical Thinking
                            'ssr_score':row .get ('ssr_score'),
                            'target_id':target_id ,
                            'extra_flag':role_type 
                            })
                except Exception as e :
                    print (f"Warning: Failed to read {fname }: {e }")

                    # --- 3. Self-Disclosure (RD) ---
        rd_files =[
        ("speaker",conf .FILE_PATTERN_RD .format (run_id )),
        ("listener",conf .FILE_PATTERN_RD .format (run_id ).replace ("speaker","listener"))
        ]

        for role_type ,fname in rd_files :
            fpath_rd =os .path .join (conf .SSR_RESULTS_DIR ,"Analysis 3_Raw Data",fname )

            if os .path .exists (fpath_rd ):
                try :
                    df =pd .read_csv (fpath_rd ,encoding =conf .ENCODING )
                    for _ ,row in df .iterrows ():
                        is_risk =False 
                        val =None 
                        if 'has_risk'in row :val =row ['has_risk']
                        elif 'risk_content_detected'in row :val =row ['risk_content_detected']

                        if str (val ).lower ()in ['true','1','yes']:
                            is_risk =True 

                        if is_risk :
                            target_id =None 
                            if role_type =='speaker':
                                target_id =row .get ('listener_id',row .get ('partner_id'))
                            else :# listener
                                target_id =row .get ('speaker_id',row .get ('partner_id'))

                            all_records .append ({
                            'run_id':run_id ,
                            'turn':row .get ('turn'),
                            'agent_id':row .get ('agent_id'),
                            'metric_type':'RD',# Risk Disclosure
                            'ssr_score':row .get ('ssr_score'),
                            'target_id':target_id if pd .notna (target_id )else np .nan ,
                            'extra_flag':role_type 
                            })
                except Exception as e :
                    print (f"Warning: Failed to read {fname }: {e }")

                    # --- 4. Similarity (PTS) ---
        fpath_pts =os .path .join (conf .SSR_RESULTS_DIR ,"Analysis 2_Raw Data",conf .FILE_PATTERN_CONSISTENCY .format (run_id ))
        if os .path .exists (fpath_pts ):
            df =pd .read_csv (fpath_pts ,encoding =conf .ENCODING )
            for _ ,row in df .iterrows ():
            # A -> B (Original Data)
                all_records .append ({
                'run_id':run_id ,
                'turn':row .get ('turn'),
                'agent_id':row .get ('agent_a'),
                'metric_type':'PTS',# Pairwise Thought Similarity
                'ssr_score':row .get ('ssr_score'),
                'target_id':row .get ('agent_b'),
                'extra_flag':np .nan 
                })
                # B -> A (Symmetrization: reverse direction also registered for calculations in Dept 1)
                all_records .append ({
                'run_id':run_id ,
                'turn':row .get ('turn'),
                'agent_id':row .get ('agent_b'),# Swap
                'metric_type':'PTS',
                'ssr_score':row .get ('ssr_score'),
                'target_id':row .get ('agent_a'),# Swap
                'extra_flag':'symmetrized'
                })

                # --- Save the data (Do it outside the loop!) ---
    df_all =pd .DataFrame (all_records )

    os .makedirs (conf .OUTPUT_PREPROCESS ,exist_ok =True )
    output_path =os .path .join (conf .OUTPUT_PREPROCESS ,"all_metrics_raw.csv")
    df_all .to_csv (output_path ,index =False ,encoding =conf .ENCODING )
    print (f"Saved merged SSR data to: {output_path }")

    if not df_all .empty :
        print ("Data Summary by Metric Type:")
        print (df_all ['metric_type'].value_counts ())
        print (f"Total rows: {len (df_all )}")
        print (f"Unique Runs: {df_all ['run_id'].nunique ()}")# It's OK if it reaches 100
    else :
        print ("Warning: Created DataFrame is empty!")

    return df_all 


def process_conversation_structure ():
    """
    From the adjacency matrix and community definitions, calculate the Internal Conversations (I) and External Conversations (E) for each Run, each Turn, each Agent (and Community).
    """
    print (">>> Processing Structure Data (Matrices & Communities)...")

    if not os .path .exists (conf .PATH_COMMUNITY ):
        print (f"Error: Community file not found at {conf .PATH_COMMUNITY }")
        return 

    df_comm =pd .read_csv (conf .PATH_COMMUNITY ,encoding =conf .ENCODING )
    comm_map ={}
    comm_members_map ={}

    for _ ,row in df_comm .iterrows ():
        rid =int (row ['run_id'])
        aid =int (row ['agent_id'])
        cid =int (row ['community_id'])
        comm_map [(rid ,aid )]=cid 
        if (rid ,cid )not in comm_members_map :
            comm_members_map [(rid ,cid )]=[]
        comm_members_map [(rid ,cid )].append (aid )

    structure_records =[]

    for run_id in tqdm (conf .RUN_IDS ,desc ="Processing Matrices"):
        for turn in conf .TURNS :
            turn_dir_name =f"turn{turn }to{turn }"
            matrix_dir =os .path .join (conf .STRUCTURE_DIR ,"adjacency matrix",turn_dir_name )
            # Make this support three-digit zero padding as well
            matrix_file =os .path .join (matrix_dir ,f"run_{run_id :03}_matrix.csv")

            if not os .path .exists (matrix_file ):
                continue 

            df_mat =pd .read_csv (matrix_file ,index_col =0 ,encoding =conf .ENCODING )

            # --- Community Perspective (for CCC Calculation) ---
            current_run_comms =[k [1 ]for k in comm_members_map .keys ()if k [0 ]==run_id ]
            unique_comms =sorted (list (set (current_run_comms )))

            for comm_id in unique_comms :
                members =comm_members_map .get ((run_id ,comm_id ),[])
                if not members :continue 

                I_count =0 
                E_count =0 
                members_str =[str (m )for m in members ]
                all_agents =[str (a )for a in conf .AGENTS ]
                external_agents =[a for a in all_agents if int (a )not in members ]

                # Utterance (Member -> Internal/External)
                for m_str in members_str :
                    if int (m_str )in df_mat .index :
                        for target in df_mat .columns :
                            weight =df_mat .loc [int (m_str ),target ]
                            if weight >0 :
                                if target in members_str :
                                    I_count +=weight 
                                else :
                                    E_count +=weight 

                                    # Reception (External -> Member)
                for ext_str in external_agents :
                    if int (ext_str )in df_mat .index :
                        for m_str in members_str :
                            if m_str in df_mat .columns :
                                weight =df_mat .loc [int (ext_str ),m_str ]
                                if weight >0 :
                                    E_count +=weight 

                structure_records .append ({
                'run_id':run_id ,
                'turn':turn ,
                'unit_type':'community',
                'unit_id':comm_id ,
                'I_count':I_count ,
                'E_count':E_count 
                })

                # --- Department Perspective (for CDC calculation) ---
            for dept_id ,members in conf .DEPT_DEFINITIONS .items ():
                I_count =0 
                E_count =0 
                members_str =[str (m )for m in members ]
                all_agents =[str (a )for a in conf .AGENTS ]
                external_agents =[a for a in all_agents if int (a )not in members ]

                for m_str in members_str :
                    if int (m_str )in df_mat .index :
                        for target in df_mat .columns :
                            weight =df_mat .loc [int (m_str ),target ]
                            if weight >0 :
                                if target in members_str :
                                    I_count +=weight 
                                else :
                                    E_count +=weight 

                for ext_str in external_agents :
                    if int (ext_str )in df_mat .index :
                        for m_str in members_str :
                            if m_str in df_mat .columns :
                                weight =df_mat .loc [int (ext_str ),m_str ]
                                if weight >0 :
                                    E_count +=weight 

                structure_records .append ({
                'run_id':run_id ,
                'turn':turn ,
                'unit_type':'department',
                'unit_id':dept_id ,
                'I_count':I_count ,
                'E_count':E_count 
                })

    df_struct =pd .DataFrame (structure_records )
    output_path =os .path .join (conf .OUTPUT_PREPROCESS ,"conversation_counts.csv")
    df_struct .to_csv (output_path ,index =False ,encoding =conf .ENCODING )
    print (f"Saved structure data to: {output_path }")


if __name__ =="__main__":
    os .makedirs (conf .OUTPUT_PREPROCESS ,exist_ok =True )
    load_and_merge_ssr_data ()
    process_conversation_structure ()
    print (">>> Preprocessing Completed.")