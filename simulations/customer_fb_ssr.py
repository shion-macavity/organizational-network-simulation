"""  # Folder/File: 
customer_fb_ssr.py

Role:
- Read the consumer feedback (04_product_eval_ssr/customer_feedback_run_xxx.csv)
Determine 'which option (1-7) was chosen' using SSR (ResponseRater)
- Simulate the 'discrete choices' selected by individual consumers, rather than the average values

customer_fb_ssr.py

Role:
- Read the consumer feedback (04_product_eval_ssr/customer_feedback_run_xxx.csv)
Determine 'which option (1-7) was chosen' using SSR (ResponseRater)
- Simulate the 'discrete choices' selected by individual consumers, rather than the average values
"""

import sys 
import pandas as pd 
import numpy as np 
import polars as po 
from pathlib import Path 
from openai import OpenAI 
from dotenv import load_dotenv 

# --- Loading existing libraries ---
current_dir =Path (__file__ ).resolve ().parent 
root_dir =current_dir .parent 
sys .path .append (str (root_dir ))

try :
# ★★★ Using SSR package: Loading the library here ★★★
# Do not create your own calculation logic, and use this ResponseRater class.
    from semantic_similarity_rating .response_rater import ResponseRater 
    from simulations .config_simulation import config 
except ImportError as e :
    print (f"Import Error: {e }")
    sys .exit (1 )

load_dotenv ()
try :
    client =OpenAI ()
except :
    client =None 

    # ==========================================
    # Settings: Seven-Stage Anchor of Purchase Intention
    # ==========================================
OPENAI_EMBEDDING_MODEL ="text-embedding-3-small"

PURCHASE_ANCHORS ={
1 :"I will never buy this product.",
2 :"I will probably not buy this product.",
3 :"I think I probably won't buy this product.",
4 :"I can't say either way.",
5 :"I think I will probably buy this product.",
6 :"I will buy this product.",
7 :"I will definitely buy this product."
}

# ==========================================
# Helper function
# ==========================================
def get_openai_embeddings (texts :list [str ],batch_size =50 )->np .ndarray :
    if not texts :return np .array ([])
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

def prepare_purchase_rater ()->ResponseRater :
    print ("  [SSR] Preparing purchase intent anchors...")

    anchor_ids =[]
    anchor_sentences =[]
    anchor_values =[]
    anchor_texts_for_embed =[]

    for val ,text in PURCHASE_ANCHORS .items ():
        anchor_ids .append ("default")
        anchor_sentences .append (text )
        anchor_values .append (val )
        anchor_texts_for_embed .append (text )

        # Vectorization of anchor text
    embeddings =get_openai_embeddings (anchor_texts_for_embed )

    # Creating a DataFrame for the SSR Library
    df_ref =po .DataFrame ({
    "id":anchor_ids ,
    "sentence":anchor_sentences ,
    "int_response":anchor_values ,
    "embedding":embeddings 
    })

    # ★★★ Using SSR Package: Library Initialization ★★★
    # I won't do any independent calculations and will leave everything to this rater object.
    return ResponseRater (
    df_reference_sentences =df_ref ,
    embeddings_column ="embedding"
    )

    # ==========================================
    # Main processing
    # ==========================================
def run_ssr_analysis (run_id :int =1 ):
    print (f"--- Start Customer Evaluation SSR (Run {run_id }) ---")

    # 1. Data Loading
    input_dir =root_dir /config .product_evaluation_dir 
    input_file =input_dir /f"customer_feedback_run_{run_id :03d}.csv"

    if not input_file .exists ():
        print (f"Error: Input file not found: {input_file }")
        return 

    df =pd .read_csv (input_file )
    print (f"  Loaded {len (df )} feedbacks.")
    if len (df )==0 :return 

    # 2. Embedded Calculation
    feedback_texts =df ["feedback_text"].fillna ("").tolist ()
    feedback_embeddings =get_openai_embeddings (feedback_texts )

    # 3. SSR Calculation
    # ★★★ Using the SSR Package: Preparation ★★★
    rater =prepare_purchase_rater ()

    print ("  [SSR] Calculating Purchase Intent choices (Argmax)...")

    # ★★★ Using SSR Package: Execute Calculation ★★★
    # Here, get_response_pmfs of semantic_similarity_rating/response_rater.py is called,
    # Cosine similarity and the like are calculated.
    # As instructed, I have removed the temperature specification (it will default to 1.0).
    pmfs =rater .get_response_pmfs (
    reference_set_id ="default",
    llm_responses =feedback_embeddings 
    )

    # 4. Confirmation of Selection Result (Argmax)
    values =np .array (list (PURCHASE_ANCHORS .keys ()))# [1, 2, 3, 4, 5, 6, 7]

    # Choose the option with the highest probability
    selected_indices =np .argmax (pmfs ,axis =1 )
    selected_choices =values [selected_indices ]

    df ["selected_choice"]=selected_choices 
    df ["confidence"]=np .max (pmfs ,axis =1 )

    # 5. Aggregation (Average for Each Proposal)
    summary_rows =[]

    # ★ Addition: Calculation of Pattern ID
    if config .fixed_pattern_id is not None :
        current_pattern_id =config .fixed_pattern_id 
    else :
        current_pattern_id =(run_id -1 )%4 +1 

    grouped =df .groupby (["proposal_agent_id","proposal_dept"])

    for (agent_id ,dept ),group in grouped :
        avg_choice =group ["selected_choice"].mean ()

        dist =group ["selected_choice"].value_counts ().to_dict ()
        dist_str =", ".join ([f"{k }:{v }"for k ,v in sorted (dist .items ())])

        summary_rows .append ({
        "run_id":run_id ,
        "pattern_id":current_pattern_id ,
        "proposal_agent_id":agent_id ,
        "proposal_dept":dept ,
        "avg_purchase_score":avg_choice ,
        "sample_count":len (group ),
        "distribution":dist_str 
        })

        # 6. Save
    output_dir =root_dir /config .product_evaluation_dir 
    output_dir .mkdir (parents =True ,exist_ok =True )

    detail_path =output_dir /f"customer_scores_detail_run_{run_id :03d}.csv"
    df .to_csv (detail_path ,index =False ,encoding ="utf-8-sig")

    summary_path =output_dir /"customer_scores_summary.csv"
    df_summary =pd .DataFrame (summary_rows )

    if summary_path .exists ():
        existing =pd .read_csv (summary_path )
        if "run_id"in existing .columns :
            existing =existing [existing ["run_id"]!=run_id ]
        df_summary =pd .concat ([existing ,df_summary ],ignore_index =True )

    df_summary .to_csv (summary_path ,index =False ,encoding ="utf-8-sig")

    print (f"Done. Scores saved to {output_dir }")

if __name__ =="__main__":
    run_ssr_analysis ()