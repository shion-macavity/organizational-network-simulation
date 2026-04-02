"""
product_evaluation_simulation.py

Role:
- Read survey response texts (survey_turn_xxx.csv)
- Vectorize (embed) using the OpenAI API
- Call the existing `semantic_similarity_rating` library to score
- Aggregate and save the results
"""

import sys 
import pandas as pd 
import numpy as np 
import polars as po 
import re # ★ Addition: Necessary to extract numbers from the file name
from pathlib import Path 
from openai import OpenAI 
from dotenv import load_dotenv 

# --- Loading existing libraries ---
# Add the root directory (experiment folder) to the path
current_dir =Path (__file__ ).resolve ().parent 
root_dir =current_dir .parent 
sys .path .append (str (root_dir ))

try :
    from semantic_similarity_rating .response_rater import ResponseRater 
    from simulations .config_simulation import config 
    from simulations import config_prompts # ★Additional: Anchor Definition File
except ImportError as e :
    print (f"Import Error: {e }")
    sys .exit (1 )

load_dotenv ()
try :
    client =OpenAI ()
except :
    client =None 

    # ==========================================
    # Settings & Constants
    # ==========================================
OPENAI_EMBEDDING_MODEL ="text-embedding-3-small"


# Reverse-scored item
REVERSE_ITEMS =["Q21","Q22","Q23"]

# Question ID definitions for each scale
SCALES ={
"psychological_safety":[f"Q{i :02d}"for i in range (1 ,20 )],# Q01-Q19
"task_cohesion":[f"Q{i :02d}"for i in range (20 ,24 )]# Q20-Q23
}

# ==========================================
# Helper function
# ==========================================
def get_openai_embeddings (texts :list [str ],batch_size =100 )->np .ndarray :
    """Bulk retrieval of text list embeddings"""
    if not texts :
        return np .array ([])

    all_embeddings =[]
    print (f"  [SSR] Computing embeddings for {len (texts )} texts...")

    for i in range (0 ,len (texts ),batch_size ):
        batch =texts [i :i +batch_size ]
        try :
            response =client .embeddings .create (
            model =OPENAI_EMBEDDING_MODEL ,
            input =batch 
            )
            embeddings =[item .embedding for item in response .data ]
            all_embeddings .extend (embeddings )
        except Exception as e :
            print (f"Error in embedding batch {i }: {e }")
            all_embeddings .extend ([[0.0 ]*1536 ]*len (batch ))

    return np .array (all_embeddings )

def prepare_reference_rater ()->ResponseRater :
    """
    Calculate the embedding of the anchor text and initialize the ResponseRater
    """
    print ("  [SSR] Preparing reference anchors...")

    # Creation of anchor data
    anchor_data ={
    "id":[],
    "sentence":[],
    "int_response":[]
    }

    anchor_texts =[]

    # Temporarily create a text list (for embedding calculations)
    # ★Fix: Read anchors for each question from config_prompts
    # q_id: "Q01", anchors: {1: "...", 7: "..."}
    for q_id ,anchors in config_prompts .SSR_SPECIFIC_ANCHORS .items ():
        for val ,text in anchors .items ():
            anchor_data ["id"].append (q_id )# Set the ID to 'Q01', etc., instead of 'default'
            anchor_data ["sentence"].append (text )
            anchor_data ["int_response"].append (val )
            anchor_texts .append (text )

            # Anchor embedding calculation
    embeddings =get_openai_embeddings (anchor_texts )

    # Convert to Polars DataFrame
    df_ref =po .DataFrame ({
    "id":anchor_data ["id"],
    "sentence":anchor_data ["sentence"],
    "int_response":anchor_data ["int_response"],
    "embedding":embeddings 
    })

    # Rater Initialization (Embedded Mode)
    rater =ResponseRater (
    df_reference_sentences =df_ref ,
    embeddings_column ="embedding"
    )
    return rater 

    # ==========================================
    # Main processing
    # ==========================================
def run_product_evaluation (run_id :int =1 ):
    print (f"--- Start Product Evaluation (SSR) for Run {run_id } ---")

    # 1. Data Loading
    input_dir =root_dir /config .conversation_results_dir /f"run_{run_id :03d}"

    if not input_dir .exists ():
        print (f"Error: Run directory not found: {input_dir }")
        return 

        # Retrieve all files that match the pattern "survey_turn_*.csv"
    all_files =list (input_dir .glob ("*survey_turn*.csv"))

    if not all_files :
        print (f"Error: No survey files found in {input_dir }")
        return 

    print (f"  Found {len (all_files )} survey files. Merging...")

    # Combine multiple files into one
    df_list =[]
    for f in all_files :
        try :
            temp_df =pd .read_csv (f )

            # ★Correction: Extract the number of turns from the file name and fill it in
            # Example: survey_turn_001.csv -> 1
            match =re .search (r"turn_(\d+)",f .name )
            if match :
                extracted_turn =int (match .group (1 ))
                # If there is no 'turn' column in the CSV (or even if there is), overwrite/complement it with information from the file name
                temp_df ["turn"]=extracted_turn 

            df_list .append (temp_df )
        except Exception as e :
            print (f"  Warning: Failed to read {f .name }: {e }")

    if not df_list :
        print ("No valid data found.")
        return 

    df =pd .concat (df_list ,ignore_index =True )
    print (f"  Loaded {len (df )} responses total.")

    # Removal of error data
    if "answer_text"in df .columns :
        df =df .dropna (subset =["answer_text"])
        df =df [~df ["answer_text"].astype (str ).str .contains ("Error|Failed to retrieve response|No API key")]
    else :
        print ("Error: 'answer_text' column missing in CSV.")
        return 

    if len (df )==0 :
        print ("No valid responses to process.")
        return 

        # 2. Embedded Calculation (Answer Text)
    texts =df ["answer_text"].tolist ()
    response_embeddings =get_openai_embeddings (texts )

    # 3. SSR Score Calculation
    rater =prepare_reference_rater ()

    print ("  [SSR] Calculating similarity scores using ResponseRater...")

    # An array to store the results (initial value set to NaN)
    df ["ssr_score"]=np .nan 

    # ★Correction: Loop and calculate for each question ID
    # Get the list of question IDs in df (e.g., Q01, Q02...)
    unique_q_ids =df ["question_id"].unique ()

    for q_id in unique_q_ids :
    # Check whether an anchor corresponding to that question has been defined
        if q_id not in config_prompts .SSR_SPECIFIC_ANCHORS :
            continue 

            # An index (True/False) to extract only the data for that question ID
        mask =df ["question_id"]==q_id 

        # Extract only the embeddings of the target data
        # Since it's a numpy array, boolean indexing is possible
        target_embeddings =response_embeddings [mask ]

        if len (target_embeddings )==0 :
            continue 

            # Calculate the PMF using the anchor dedicated to that question (reference_set_id=q_id)
        pmfs =rater .get_response_pmfs (
        reference_set_id =q_id ,
        llm_responses =target_embeddings 
        )

        # Expected Value Calculation (1~7)
        # Since the key for any anchor is 1~7, [1, 2, 3, 4, 5, 6, 7]
        values =np .array ([1 ,2 ,3 ,4 ,5 ,6 ,7 ])
        scores =np .sum (pmfs *values ,axis =1 )

        # Assign the calculation results to the corresponding places in the original DataFrame
        df .loc [mask ,"ssr_score"]=scores 

        # ★Addition: Save the probability distribution in the format [0.0, 0.0, ..., 0.8, 0.1]
        # Rounding process to the second decimal place
        #For consistency judgment of determination
        dist_list =[str ([round (float (p ),2 )for p in row ])for row in pmfs ]
        df .loc [mask ,"ssr_distribution"]=dist_list 

        # 4. Handling of reverse-coded items
    for idx ,row in df .iterrows ():
        if row ["question_id"]in REVERSE_ITEMS :
            df .at [idx ,"ssr_score_raw"]=row ["ssr_score"]
            df .at [idx ,"ssr_score"]=8.0 -row ["ssr_score"]
        else :
            df .at [idx ,"ssr_score_raw"]=row ["ssr_score"]

            # 5. Aggregation (by Turn, by Agent)
    final_rows =[]

    # ★ Addition: Calculation of Pattern ID
    if config .fixed_pattern_id is not None :
        current_pattern_id =config .fixed_pattern_id 
    else :
        current_pattern_id =(run_id -1 )%4 +1 

        # Since the 'turn' column was included when merging the files, grouping is definitely possible.
    if "turn"not in df .columns :
        df ["turn"]=0 # Backup for emergencies

    grouped =df .groupby (["turn","agent_id"])

    for (turn ,agent_id ),group in grouped :
    # Acquire representative value
        agent_name =group .iloc [0 ].get ("agent_name","Unknown")
        dept =group .iloc [0 ].get ("dept","Unknown")
        position =group .iloc [0 ].get ("position","Unknown")

        # Psychological Safety (Average)
        psy_items =group [group ["question_id"].isin (SCALES ["psychological_safety"])]
        psy_score =psy_items ["ssr_score"].mean ()if not psy_items .empty else None 

        # Task Cohesion (Average)
        task_items =group [group ["question_id"].isin (SCALES ["task_cohesion"])]
        task_score =task_items ["ssr_score"].mean ()if not task_items .empty else None 

        final_rows .append ({
        "run_id":run_id ,
        "pattern_id":current_pattern_id ,# ★Add: Add to dictionary
        "turn":turn ,
        "agent_id":agent_id ,
        "agent_name":agent_name ,
        "dept":dept ,
        "position":position ,
        "psychological_safety":psy_score ,
        "task_cohesion":task_score 
        })

        # 6. Save
    output_df =pd .DataFrame (final_rows )

    # Create output destination
    output_dir =root_dir /"results/03_questionnaire_ssr"
    output_dir .mkdir (parents =True ,exist_ok =True )

    output_path =output_dir /"questionnaire_scores.csv"

    if output_path .exists ():
        existing =pd .read_csv (output_path )
        if "run_id"in existing .columns :
            existing =existing [existing ["run_id"]!=run_id ]
        output_df =pd .concat ([existing ,output_df ],ignore_index =True )

    output_df .to_csv (output_path ,index =False ,encoding ="utf-8-sig")

    # Also save detailed data
    detail_path =output_dir /f"questionnaire_details_run_{run_id :03d}.csv"
    df .to_csv (detail_path ,index =False ,encoding ="utf-8-sig")

    print (f"  [SSR] Finished. Scores saved to {output_path }")

if __name__ =="__main__":
    run_product_evaluation ()