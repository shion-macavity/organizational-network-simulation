import os 
import glob 
import pandas as pd 
import numpy as np 

# ==========================================
# Settings: Path and Definitions
# ==========================================
RESULTS_DIR ="results"
QUESTIONNAIRE_DIR =os .path .join (RESULTS_DIR ,"03_questionnaire_ssr")
SCORE_DIR =os .path .join (RESULTS_DIR ,"04_product_eval_ssr")
CONVERSATION_DIR =os .path .join (RESULTS_DIR ,"02_conversation")
OUTPUT_DIR =os .path .join (RESULTS_DIR ,"05_amos_input")

# ID Definition (Based on Experimental Design)
LEADER_IDS =[6 ,13 ]
BOUNDARY_IDS =[0 ,7 ]
# Subordinate ID = All except the leader
ALL_IDS =list (range (14 ))
SUBORDINATE_IDS =[i for i in ALL_IDS if i not in LEADER_IDS ]

# ==========================================
# Function definition
# ==========================================

def get_pattern_attributes (pattern_id ):
    """Generate dummy variables from pattern ID"""
    try :
        pid =int (pattern_id )
        # Proximity of the organization (is_proximal): Pattern 2, 4 -> 1 (close), Pattern 1, 3 -> 0 (far)
        is_proximal =1 if pid in [2 ,4 ]else 0 

        # Casual interaction (has_informal): Pattern 3, 4 -> 1 (exists), Pattern 1, 2 -> 0 (does not exist)
        has_informal =1 if pid in [3 ,4 ]else 0 

        return is_proximal ,has_informal 
    except :
        return np .nan ,np .nan 

def load_output_text (run_id ):
    """Read the deliverable text and department name from the output.csv of the specified Run"""
    run_folder =f"run_{int (run_id ):03d}"
    file_path =os .path .join (CONVERSATION_DIR ,run_folder ,"output.csv")

    text_1 ,text_2 ,dept_0 ,dept_1 ="","","",""

    if os .path .exists (file_path ):
        try :
            df =pd .read_csv (file_path )
            # Filter with the current Run ID just in case
            df =df [df ['run_id']==run_id ]

            if 'proposal_text'in df .columns and 'dept'in df .columns :
            # Information of Department 0
                r0 =df [df ['dept_id']==0 ]
                if not r0 .empty :
                    dept_0 =r0 ['dept'].iloc [0 ]
                    text_1 =" ".join (r0 ['proposal_text'].dropna ().astype (str ).tolist ())

                    # Information of Department 1
                r1 =df [df ['dept_id']==1 ]
                if not r1 .empty :
                    dept_1 =r1 ['dept'].iloc [0 ]
                    text_2 =" ".join (r1 ['proposal_text'].dropna ().astype (str ).tolist ())
        except Exception as e :
            print (f"Warning: Failed to read output for Run {run_id }. Error: {e }")

    return text_1 ,text_2 ,dept_0 ,dept_1 

def main ():
# Create output directory
    os .makedirs (OUTPUT_DIR ,exist_ok =True )

    # A list that stores data from all runs
    run_data_list =[]
    agent_data_list =[]

    # Retrieve survey file (process for each run)
    quest_files =glob .glob (os .path .join (QUESTIONNAIRE_DIR ,"questionnaire_details_run_*.csv"))

    print (f"Found {len (quest_files )} runs to process.")

    for q_file in quest_files :
    # Extract Run ID from file name (questionnaire_details_run_001.csv -> 1)
        filename =os .path .basename (q_file )
        try :
            run_id_str =filename .replace ("questionnaire_details_run_","").replace (".csv","")
            run_id =int (run_id_str )
        except ValueError :
            continue 

        print (f"Processing Run {run_id }...")

        # 1. Loading the survey data
        df_q =pd .read_csv (q_file )
        # ★Addition: Remove apostrophes and convert to numeric
        if 'ssr_score'in df_q .columns :
            df_q ['ssr_score']=df_q ['ssr_score'].astype (str ).str .replace ("'","").astype (float )

            # Obtain Pattern ID (Retrieve from the first line of the file)
        pattern_id =df_q ['pattern_id'].iloc [0 ]if 'pattern_id'in df_q .columns else np .nan 
        is_proximal ,has_informal =get_pattern_attributes (pattern_id )

        # -------------------------------------------------------
        # [For Preservation] Creation of Basic Summary Data (Agent Level)
        # -------------------------------------------------------
        # Calculate the average score for all turns for each agent
        # First, aggregate with the image of pivoting for each question item.

        # Distinction between psychological safety (Q01-Qxx) and group cohesion (Qxx...)
        # * Here, determine by the question ID string or list all questions
        question_ids =df_q ['question_id'].unique ()

        for agent_id in ALL_IDS :
            agent_df =df_q [df_q ['agent_id']==agent_id ]

            if agent_df .empty :
                continue 

                # Data dictionary for each agent
            agent_record ={
            'run_id':run_id ,
            'agent_id':agent_id ,
            'dept_id':agent_df ['dept_id'].iloc [0 ],# Obtain ID (0 or 1)
            'dept_name':agent_df ['dept'].iloc [0 ],# Get genre name (e.g., mystery)
            'pattern_id':pattern_id 
            }

            # Calculate the average value for each question
            scores_all =[]
            for q_id in question_ids :
                score =agent_df [agent_df ['question_id']==q_id ]['ssr_score'].mean ()
                agent_record [f'score_{q_id }']=score 
                scores_all .append (score )

                # Average of all questions
            agent_record ['score_ave_all']=np .mean (scores_all )if scores_all else np .nan 

            agent_data_list .append (agent_record )

            # -------------------------------------------------------
            # [Main] Creating Data for AMOS (Run Level)
            # -------------------------------------------------------
        run_record ={
        'run_id':run_id ,
        'pattern_id':pattern_id ,
        'is_proximal':is_proximal ,
        'has_informal':has_informal 
        }

        # Aggregation of psychological data (average for the entire run)
        # For each question item, the average across all agents and all turns
        psych_scores =[]# For psychological safety (tentatively Q01-Q07)
        cohesion_scores =[]# For group cohesiveness (tentatively Q08-Q14)

        # --- ★Addition from here: Prepare a list for each department ---
        psych_dept0 =[]
        psych_dept1 =[]
        cohesion_dept0 =[]
        cohesion_dept1 =[]
        # ------------------------------------------

        # Note: Conditional branching is required to match the actual question ID.
        # For convenience, we will tentatively implement Q01-Q19 as psychological and Q20 onward as cohesion.
        # Please adjust as necessary.
        for q_id in question_ids :
        # The overall average of a specific question in that run
            q_mean =df_q [df_q ['question_id']==q_id ]['ssr_score'].mean ()

            # Add column name as a variable
            # Example: psych_safety_Q01, cohesion_Q08
            # Here, we simply use q_id as it is.
            run_record [f'score_{q_id }']=q_mean 

            # --- ★ Addition from here: Calculate the average for each department ---
            # Average of Department 0 (dept_id=0)
            m0 =df_q [(df_q ['question_id']==q_id )&(df_q ['dept_id']==0 )]['ssr_score'].mean ()
            # Average of Department 1 (dept_id=1)
            m1 =df_q [(df_q ['question_id']==q_id )&(df_q ['dept_id']==1 )]['ssr_score'].mean ()
            # ------------------------------------------

            # Addition to the list for overall average calculation (simple logic)
            # Assume that items with smaller Q numbers correspond to psychology, and items with larger Q numbers correspond to aggregation.
            # ※ Ideally, it is best to read the list of question IDs from a definition file.
            try :
                q_num =int (q_id .replace ("Q",""))
                if q_num <=19 :# Temporary threshold
                    psych_scores .append (q_mean )
                    # ★Add to department list
                    psych_dept0 .append (m0 )
                    psych_dept1 .append (m1 )
                else :
                    cohesion_scores .append (q_mean )
                    # ★Add to department list
                    cohesion_dept0 .append (m0 )
                    cohesion_dept1 .append (m1 )
            except :
                psych_scores .append (q_mean )# If it's an error, put it all into psychology.

        run_record ['psych_ave_all']=np .mean (psych_scores )if psych_scores else np .nan 
        run_record ['cohesion_ave_all']=np .mean (cohesion_scores )if cohesion_scores else np .nan 

        # --- ★Addition from here: Record the averages by department ---
        run_record ['psych_dept0']=np .mean (psych_dept0 )if psych_dept0 else np .nan 
        run_record ['psych_dept1']=np .mean (psych_dept1 )if psych_dept1 else np .nan 
        run_record ['cohesion_dept0']=np .mean (cohesion_dept0 )if cohesion_dept0 else np .nan 
        run_record ['cohesion_dept1']=np .mean (cohesion_dept1 )if cohesion_dept1 else np .nan 
        # ------------------------------------------

        # 2. Loading the performance score
        score_file =os .path .join (SCORE_DIR ,f"customer_scores_detail_run_{run_id :03d}.csv")
        output_score_1 =np .nan 
        output_score_2 =np .nan 
        output_score_ave =np .nan 

        if os .path .exists (score_file ):
            df_s =pd .read_csv (score_file )

            # ★Addition: Remove apostrophes and convert to numeric
            if 'ssr_score'in df_s .columns :
                df_s ['ssr_score']=df_s ['ssr_score'].astype (str ).str .replace ("'","").astype (float )

                # Differentiate 0 and 1 with a column name called proposal_dept_id
            if 'ssr_score'in df_s .columns and 'proposal_dept_id'in df_s .columns :
            # Calculate the average score of Department 0
                output_score_1 =df_s [df_s ['proposal_dept_id']==0 ]['ssr_score'].mean ()

                # Calculate the average score of Department 1
                output_score_2 =df_s [df_s ['proposal_dept_id']==1 ]['ssr_score'].mean ()

                # Overall average score
                output_score_ave =df_s ['ssr_score'].mean ()

            else :
            # For debugging: If this still doesn't work, display
                print (f"Warning: Columns mismatch in {score_file }. Found: {df_s .columns .tolist ()}")

        run_record ['output_score_1']=output_score_1 
        run_record ['output_score_2']=output_score_2 
        run_record ['output_score_ave']=output_score_ave 

        # 3. Loading the deliverable text
        text_1 ,text_2 ,d_name_0 ,d_name_1 =load_output_text (run_id )


        run_record ['output_text_1']=text_1 
        run_record ['output_text_2']=text_2 
        # ★Record: Genre names of each department in this run
        run_record ['dept_name_0']=d_name_0 
        run_record ['dept_name_1']=d_name_1 

        run_data_list .append (run_record )

        # ==========================================
        # Save as CSV
        # ==========================================

        # 1. Data for AMOS (Run Level)
    if run_data_list :
        df_run =pd .DataFrame (run_data_list )
        # Adjust the order of the columns (for readability)
        cols =['run_id','pattern_id','is_proximal','has_informal']
        # Add the remaining columns in alphabetical order or similar
        remaining_cols =[c for c in df_run .columns if c not in cols ]
        df_run =df_run [cols +sorted (remaining_cols )]

        output_path_run =os .path .join (OUTPUT_DIR ,"amos_dataset_run_level.csv")
        df_run .to_csv (output_path_run ,index =False ,encoding ='utf-8-sig')
        print (f"Created: {output_path_run }")

        # 2. Basic Summary Data (Agent Level)
    if agent_data_list :
        df_agent =pd .DataFrame (agent_data_list )
        cols_agent =['run_id','agent_id','pattern_id']
        remaining_cols_agent =[c for c in df_agent .columns if c not in cols_agent ]
        df_agent =df_agent [cols_agent +sorted (remaining_cols_agent )]

        output_path_agent =os .path .join (OUTPUT_DIR ,"summary_agent_level.csv")
        df_agent .to_csv (output_path_agent ,index =False ,encoding ='utf-8-sig')
        print (f"Created: {output_path_agent }")

if __name__ =="__main__":
    main ()