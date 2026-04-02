"""
customer_fb_simulation.py (Strict Version)

Role:
- Read the leader's proposal (output.csv)
- Randomly select the specified number of people from the consumer sheet
- ★Fix: If the file does not exist, produce an error and exit (do not use dummy data)
- ★Fix: Apply the verbosity setting (low) from the config
"""

import csv 
import random 
import pandas as pd 
from pathlib import Path 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from openai import OpenAI 
from dotenv import load_dotenv 

try :
    from .config_simulation import config 
except ImportError :
    import config_simulation as config 

load_dotenv ()
try :
    client =OpenAI ()
except :
    client =None 

def run_customer_simulation (run_id :int =1 ):
# ★Addition: Fix the random number seed based on the Run ID (for customer feedback)
# While using a separate random number series (+10000) for employee generation, control it so that the same customer appears in Runs 1 to 4.
    seed_val =((run_id -1 )//config .PATTERN_CYCLE )+config .CUSTOMER_SEED_OFFSET 
    random .seed (seed_val )

    print (f"--- Start Customer Feedback Simulation (Run {run_id }) ---")

    root_dir =Path (__file__ ).resolve ().parents [1 ]

    # 1. Loading the project plan
    # Correction: Changed to search inside the run_xxx folder
    proposal_file =root_dir /config .conversation_results_dir /f"run_{run_id :03d}"/"output.csv"
    if not proposal_file .exists ():
        print (f"Error: Proposal file not found: {proposal_file }")
        print ("First, please run conversation_simulation.py to generate a project proposal.")
        return 

    df_proposals =pd .read_csv (proposal_file )
    if "run_id"in df_proposals .columns :
        df_proposals =df_proposals [df_proposals ["run_id"]==run_id ]

    if df_proposals .empty :
        print (f"No proposals found for Run ID {run_id }.")
        return 

        # 2. Loading Consumer Persona (Strict Mode)
    customer_file =root_dir /"simulations"/"customer_persona_sheet.xlsx"

    if not customer_file .exists ():
    # ★Fix here: Abolish dummy data generation and make it stop on error
        print (f"!!! Error: Customer sheet not found at {customer_file }")
        print ("!!! Stopping the experiment. Please place the correct 'customer_persona_sheet.xlsx'.")
        return 

    try :
    # Read the first sheet without specifying the sheet name, or you may explicitly specify it.
        df_customers =pd .read_excel (customer_file )
        customers =df_customers .to_dict (orient ="records")
        print (f"Loaded {len (customers )} customers from sheet.")
    except Exception as e :
        print (f"!!! Error reading customer sheet: {e }")
        return 

        # 3. Execute Evaluation
    results =[]

    def evaluate_proposal (proposal_row ,customer_row ):
    # Corresponds to variations in whether column names are in Japanese or English
        age =customer_row .get ("Age",customer_row .get ("age",30 ))
        income =customer_row .get ("Annual income",customer_row .get ("income",4000000 ))

        system_prompt =(
        f"You are a general consumer.\n"
        f"Age: {age } years old\n"
        f"Annual Income: Approximately {income } million yen\n"
        f"Please answer the questionnaire regarding the presented product idea."
        )

        user_prompt =(
        f"【Product Idea】\n{proposal_row ['proposal_text']}\n\n"
        f"How likely would you be to purchase this product, or how unlikely would you be?\n"
        f"Please provide only the conclusion. Answer honestly based on your true feelings.\n"
        f"However, regarding the response format, the following are prohibited:\n"
        f"1. When expressing a negative opinion, do not use negative sentence structures. Instead, use affirmative sentences containing negative vocabulary.\n"
        f"2. Similarly, when expressing a positive opinion, do not use double negatives or the negation of negative words.\n"
        f"3. Answering with proverbs or idioms is also prohibited.\n"
        )

        if client is None :
            return None 

        try :
            params ={
            "model":config .openai_model ,
            "messages":[
            {"role":"system","content":system_prompt },
            {"role":"user","content":user_prompt }
            ],
            "max_completion_tokens":config .tokens_customer_feedback ,
            }

            # Parameters for inference model
            is_advanced_model =any (x in config .openai_model for x in ["o1","o3","nano","gpt-5"])
            if is_advanced_model :
                 params ["reasoning_effort"]=config .reasoning_customer_feedback 
                 params ["verbosity"]=config .verbosity_customer_feedback 

            res =client .chat .completions .create (**params )
            feedback =res .choices [0 ].message .content .strip ()

            return {
            "proposal_agent_id":proposal_row ["agent_id"],
            "proposal_dept":proposal_row ["dept"],
            "customer_age":age ,
            "customer_income":income ,
            "feedback_text":feedback 
            }
        except Exception as e :
            print (f"Feedback Error: {e }")
            return None 

            # Sampling execution
    SAMPLE_SIZE =20 
    with ThreadPoolExecutor (max_workers =20 )as executor :
        futures =[]
        for _ ,p_row in df_proposals .iterrows ():
        # Randomly extracted from the consumer list
            sampled_customers =random .sample (customers ,min (len (customers ),SAMPLE_SIZE ))
            for c_row in sampled_customers :
                futures .append (executor .submit (evaluate_proposal ,p_row ,c_row ))

        for f in as_completed (futures ):
            res =f .result ()
            if res :
                results .append (res )

                # 4. Save
    if not results :
        print ("No feedback collected.")
        return 

    output_dir =root_dir /config .product_evaluation_dir 
    output_dir .mkdir (parents =True ,exist_ok =True )

    out_path =output_dir /f"customer_feedback_run_{run_id :03d}.csv"

    # ★Correction: Add columns after converting to DataFrame
    df_results =pd .DataFrame (results )

    if config .fixed_pattern_id is not None :
        current_pattern_id =config .fixed_pattern_id 
    else :
        current_pattern_id =(run_id -1 )%4 +1 

    df_results ["pattern_id"]=current_pattern_id 

    df_results .to_csv (out_path ,index =False ,encoding ="utf-8-sig")

    print (f"Customer feedback saved to {out_path }")

if __name__ =="__main__":
    run_customer_simulation (1 )