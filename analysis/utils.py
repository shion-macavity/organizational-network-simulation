# analysis/utils.py
import sys 
import os 
import time # Added for time measurement
import numpy as np 
import pandas as pd 
import polars as po 
from pathlib import Path 
from dotenv import load_dotenv 

# Set the path to the root directory
current_dir =Path (__file__ ).resolve ().parent 
root_dir =current_dir .parent 
sys .path .append (str (root_dir ))

from google import genai 
from google .genai import types 
from openai import OpenAI 
from analysis .config import Config 

# Custom Module (SSR)
try :
    from semantic_similarity_rating .response_rater import ResponseRater 
except ImportError :
    pass 

load_dotenv ()

# ==========================================
# 1. Initialization of the API client
# ==========================================
def get_clients ():
    try :
        gemini_key =os .getenv ("GEMINI_API_KEY")
        openai_key =os .getenv ("OPENAI_API_KEY")

        if not gemini_key or not openai_key :
            return None ,None 

        client_gemini =genai .Client (api_key =gemini_key )
        client_openai =OpenAI (api_key =openai_key )
        return client_gemini ,client_openai 
    except Exception as e :
        print (f"API Client Init Error: {e }")
        return None ,None 

CLIENT_GEMINI ,CLIENT_OPENAI =get_clients ()

# ==========================================
# 2. Basic Functions
# ==========================================
def get_embeddings (texts ):
    if not texts or CLIENT_OPENAI is None :
        return np .array ([])
    try :
        t0 =time .time ()
        response =CLIENT_OPENAI .embeddings .create (
        model =Config .EMBEDDING_MODEL ,
        input =texts 
        )
        dt =time .time ()-t0 
        # Adjustable to reduce noise, such as when it takes more than 0.5 seconds or to always display.
        # This time, display everything and look for bottlenecks.
        print (f"    [Time] OpenAI: {dt :.2f}s")

        return np .array ([item .embedding for item in response .data ])
    except Exception as e :
        print (f"Embedding Error: {e }")
        return np .array ([])

def call_llm (prompt ,check_format =True ):
    """A general-purpose function that calls Gemini or OpenAI based on the config"""
    max_retries =3 
    last_error =None 

    for attempt in range (max_retries ):
        try :
            text_response =""

            # --- 1. Gemini ---
            if Config .LLM_PROVIDER =="gemini":
                if CLIENT_GEMINI is None :
                    return "ERROR: Gemini Client not initialized"

                t0 =time .time ()
                response =CLIENT_GEMINI .models .generate_content (
                model =Config .LLM_MODEL ,
                contents =prompt ,
                config =types .GenerateContentConfig (
                max_output_tokens =Config .MAX_OUTPUT_TOKENS ,
                temperature =Config .TEMPERATURE 
                )
                )
                dt =time .time ()-t0 
                print (f"    [Time] Gemini ({Config .LLM_MODEL }): {dt :.2f}s")
                text_response =response .text 

                # --- 2. OpenAI ---
            elif Config .LLM_PROVIDER =="openai":
                if CLIENT_OPENAI is None :
                    return "ERROR: OpenAI Client not initialized"

                t0 =time .time ()

                if Config .IS_REASONING_MODEL :
                # For inference models: Use the Responses API (GPT-5 series)
                    response =CLIENT_OPENAI .responses .create (
                    model =Config .LLM_MODEL ,
                    input =prompt ,
                    reasoning ={"effort":Config .REASONING_EFFORT },
                    text ={"verbosity":Config .OUTPUT_VERBOSITY },
                    max_output_tokens =Config .MAX_OUTPUT_TOKENS 
                    )
                    dt =time .time ()-t0 
                    print (f"    [Time] OpenAI ({Config .LLM_MODEL }): {dt :.2f}s")

                    # Search for Message from the Output list and extract the Text within it
                    found_text =False 
                    if hasattr (response ,'output'):
                        for item in response .output :
                            if getattr (item ,'type',None )=='message':
                                for content in getattr (item ,'content',[]):
                                    if getattr (content ,'type',None )=='output_text':
                                        text_response =content .text 
                                        found_text =True 
                                        break 
                            if found_text :break 

                    if not found_text :
                        text_response =str (response )

                else :
                # For standard models: Use the Chat Completions API
                    response =CLIENT_OPENAI .chat .completions .create (
                    model =Config .LLM_MODEL ,
                    messages =[{"role":"user","content":prompt }],
                    max_tokens =Config .MAX_OUTPUT_TOKENS ,
                    temperature =Config .TEMPERATURE 
                    )
                    dt =time .time ()-t0 
                    print (f"    [Time] OpenAI ({Config .LLM_MODEL }): {dt :.2f}s")
                    text_response =response .choices [0 ].message .content 

            else :
                return "ERROR: Unknown LLM_PROVIDER"

                # --- Format Check ---
                # As a rule, check whether [REASON] and [ANSWER] are included, and if not, make it retry as an error.
            if check_format :
                if "[REASON]"not in text_response or "[ANSWER]"not in text_response :
                    raise ValueError (f"Response format error: Missing [REASON] or [ANSWER]. Response sample: {text_response [:50 ]}...")

            return text_response 

        except Exception as e :
            last_error =e 
            print (f"    [Retry {attempt +1 }/{max_retries }] Error: {e }")
            time .sleep (1 )# Waiting to retry

    return f"ERROR ({Config .LLM_PROVIDER }): {str (last_error )}"

    # Set an alias for compatibility with existing analysis modules
call_gemini =call_llm 

def parse_response (text ):
    reason ="ERROR"
    answer ="ERROR"
    try :
        if "[REASON]"in text and "[ANSWER]"in text :
            parts =text .split ("[ANSWER]")
            reason =parts [0 ].split ("[REASON]")[1 ].strip ()
            answer =parts [1 ].strip ()
    except Exception :
        pass 
    return reason ,answer 

    # ==========================================
    # 3. SSR Related
    # ==========================================
def create_rater (anchors ):
    embeddings =get_embeddings (anchors )
    if len (embeddings )==0 :
        return None 

    df_ref =po .DataFrame ({
    "id":["default"]*len (anchors ),
    "sentence":anchors ,
    "int_response":list (range (1 ,len (anchors )+1 )),
    "embedding":embeddings 
    })

    return ResponseRater (
    df_reference_sentences =df_ref ,
    embeddings_column ="embedding"
    )

def calculate_ssr_metrics (rater ,answer_text ,n_anchors =5 ):
    """Calculate the expected value and probability distribution of a 5-point rating"""
    if answer_text =="ERROR"or answer_text is None or rater is None :
        return None ,[]

    answer_vec =get_embeddings ([answer_text ])
    if len (answer_vec )==0 :return None ,[]

    pmfs =rater .get_response_pmfs (
    reference_set_id ="default",
    llm_responses =answer_vec 
    )
    probs =pmfs [0 ]

    values =np .arange (1 ,n_anchors +1 )
    expected_value =np .sum (values *probs )

    return expected_value ,probs .tolist ()
    # ==========================================
    # 4. Data Loading
    # ==========================================
def load_agent_name_map ():
    """Return a dictionary mapping ID to name (surname) {0: 'Sato', ...}"""
    try :
    # Correction: Config.BASE_DIR is the root, so .parent is unnecessary
    # Priority 1: inside the simulations folder
        sheet_path =Config .BASE_DIR /"simulations"/"Persona Sheet.xlsx"

        if not sheet_path .exists ():
        # Priority 2: inside the analysis folder (Config.BASE_DIR/analysis/...)
            sheet_path =Config .BASE_DIR /"analysis"/"Persona Sheet.xlsx"

        if not sheet_path .exists ():
        # Priority 3: Directly under root
            sheet_path =Config .BASE_DIR /"Persona Sheet.xlsx"

        if sheet_path .exists ():
            print (f"  [Info] Loading Persona Sheet from: {sheet_path .name }")# Debug log
            df =pd .read_excel (sheet_path )

            # Handling Fluctuations in Column Names
            name_col =None 
            if 'Surname'in df .columns :name_col ='Surname'
            elif 'Name'in df .columns :name_col ='Name'

            id_col =None 
            if 'agent_id'in df .columns :id_col ='agent_id'
            elif 'ID'in df .columns :id_col ='ID'
            elif 'id'in df .columns :id_col ='id'

            if name_col and id_col :
                return dict (zip (df [id_col ],df [name_col ]))
            else :
                print (f"  [Warning] Persona Sheet loaded but columns not found. Available: {df .columns .tolist ()}")
        else :
        # If it could not be found anywhere
            pass 

    except Exception as e :
        print (f"  [Warning] Failed to load Persona Sheet: {e }")
    return {}

def load_run_data (run_id ):
    run_dir =Config .OUTPUT_DIR .parent /"02_conversation"/f"run_{run_id :03d}"
    history_path =run_dir /"conversation_history.csv"
    initial_path =run_dir /"initial_thoughts.csv"

    if not history_path .exists ()or not initial_path .exists ():
        return None ,None 

    try :
        df_history =pd .read_csv (history_path )
        df_initial =pd .read_csv (initial_path )
        return df_history ,df_initial 
    except Exception :
        return None ,None 