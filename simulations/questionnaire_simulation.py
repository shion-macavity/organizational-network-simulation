"""
questionnaire_simulation.py
Role:
- Conduct a questionnaire with the agent based on its current state of thought
- ★Correction: Apply the verbosity setting (low) in the config to prevent unnecessary output
"""
import pandas as pd 
from typing import List ,Any 
from concurrent .futures import ThreadPoolExecutor 
from pathlib import Path 
from openai import OpenAI 
from dotenv import load_dotenv 

try :
    from .config_simulation import config 
    from .import config_prompts 
except ImportError :
    import config_simulation as config 
    import simulations .config_prompts as config_prompts 

load_dotenv ()
try :
    client =OpenAI ()
except :
    client =None 

MAX_WORKERS =50 
def _ask_single_question (agent :Any ,question_id :str ,question_text :str )->dict :
    profile_text =agent .get_profile_text ()
    user_prompt =(
    f"【Your Current Thought】\n{agent .current_thought }\n\n"
    f"【Question】Answer the following question honestly.\n"
    f"\n'{question_text }'"
    f"No sugarcoating. Answer in one sentence."
    )

    # ★Correction: Reflect the profile in detail in the system prompt
    system_prompt =(
    f"【Your Prifile】\n"
    f"{profile_text }\n"
    f"Answer the questionnaire reflecting your attributes and personality.(e.g. age, gender, role, performance) "
    f"However, regarding the response format, the following are prohibited:\n"
    f"1. When expressing a negative opinion, do not use negative sentence structures. Instead, use affirmative sentences containing negative vocabulary.\n"
    f"2. Similarly, when expressing a positive opinion, do not use double negatives or the negation of negative words.\n"
    f"3. Answering with proverbs or idioms is also prohibited.\n"
    )

    if agent .node_id ==0 and question_id =="Q01":
        print (f"\n#======== [DEBUG] Survey Prompt (ID: {agent .node_id }, Q: {question_id }) ========")
        print ("#--- System Prompt ---")
        print (system_prompt )
        print ("#--- User Prompt ---")
        print (user_prompt )
        print ("#=========================================================\n")

    answer =""
    if client :
        try :
            params ={
            "model":config .openai_model ,
            "messages":[
            {"role":"system","content":system_prompt },
            {"role":"user","content":user_prompt }
            ],
            "max_completion_tokens":config .tokens_survey ,
            }

            # Parameters for inference model
            is_advanced_model =any (x in config .openai_model for x in ["o1","o3","nano","gpt-5"])
            if is_advanced_model :
                 params ["reasoning_effort"]=config .reasoning_survey # "high"
                 params ["verbosity"]=config .verbosity_survey # "low"

            res =client .chat .completions .create (**params )
            answer =res .choices [0 ].message .content .strip ()


            # ▼▼▼▼▼▼▼▼▼▼ ★Addition: Display answers (ID=0, Q01 only) ▼▼▼▼▼▼▼▼▼▼
            if agent .node_id ==0 and question_id =="Q01":
                print (f"\n#======== [DEBUG] Survey Answer (ID: {agent .node_id }, Q: {question_id }) ========")
                print (f"Agent Name: {agent .name }")
                print (f"Raw Answer: {answer }")
                print ("#=========================================================\n")
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲



        except Exception as e :
            answer ="Error"
    else :
        answer ="No API Key"

    return {
    "agent_id":agent .node_id ,
    "question_id":question_id ,
    "answer_text":answer 
    }

def run_questionnaire_for_agents (turn :int ,agents :List [Any ],run_id :int ):
    """
    Execute all agents x all questions in parallel
    """
    results =[]
    with ThreadPoolExecutor (max_workers =MAX_WORKERS )as executor :
        futures =[]
        for agent in agents :
            for q_id ,q_text in config_prompts .SURVEY_QUESTIONS .items ():
                futures .append (executor .submit (_ask_single_question ,agent ,q_id ,q_text ))

        for f in futures :
            results .append (f .result ())

    df =pd .DataFrame (results )
    agent_map ={a .node_id :{"name":a .name ,"dept":a .dept_name ,"position":a .position }for a in agents }

    df ["agent_name"]=df ["agent_id"].map (lambda x :agent_map [x ]["name"])
    df ["dept"]=df ["agent_id"].map (lambda x :agent_map [x ]["dept"])
    df ["position"]=df ["agent_id"].map (lambda x :agent_map [x ]["position"])

    # ★Assigning a Pattern ID
    if config .fixed_pattern_id is not None :
        current_pattern_id =config .fixed_pattern_id 
    else :
        current_pattern_id =(run_id -1 )%4 +1 

    df ["pattern_id"]=current_pattern_id 

    out_dir =Path (__file__ ).resolve ().parents [1 ]/config .conversation_results_dir /f"run_{run_id :03d}"
    out_dir .mkdir (parents =True ,exist_ok =True )

    df .to_csv (out_dir /f"survey_turn_{turn :02d}.csv",index =False ,encoding ="utf-8-sig")