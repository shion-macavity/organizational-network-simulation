"""  # Folder/File: 
conversation_simulation.py (Leader Proposal Added)

Role:
- Read the CSV files (edges, nodes) saved during the network generation phase
- Restore the graph (G)
- Initialize the agent and run the conversation simulation
- ★Fix: Restored the run_turn_0 method that had disappeared
- ★Function: Reflects the verbosity / reasoning_effort settings of the config
- ★Function: Loading Persona Sheet.xlsx (Excel)
- ★Function: The leader creates a project proposal after the simulation ends

conversation_simulation.py (Leader Proposal Added)

Role:
- Read the CSV files (edges, nodes) saved during the network generation phase
- Restore the graph (G)
- Initialize the agent and run the conversation simulation
- ★Fix: Restored the run_turn_0 method that had disappeared
- ★Function: Reflects the verbosity / reasoning_effort settings of the config
- ★Function: Loading Persona Sheet.xlsx (Excel)
- ★Function: The leader creates a project proposal after the simulation ends
"""

import csv 
import random 
import re 
import math 
import pandas as pd 
import networkx as nx 
from pathlib import Path 
import time 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from pathlib import Path 
from typing import Dict ,List ,Any ,Optional 
from concurrent .futures import ThreadPoolExecutor 
from openai import OpenAI 
from dotenv import load_dotenv 

try :
    from .config_simulation import config 
    from .import config_prompts 
    from .import questionnaire_simulation 
except ImportError :
    import config_simulation as config 
    import simulations .config_prompts as config_prompts 
    import questionnaire_simulation 

load_dotenv ()
try :
    client =OpenAI ()
except :
    client =None 
    print ("Warning: Failed to initialize the OpenAI client.")

    # ==========================================
    # Agent class
    # ==========================================
class Agent :
    def __init__ (self ,node_id :int ,community_g1_formal :int ,community_id :int ,neighbors :List [int ],
    name :str ,gender :str ,age :int ,proficiency :str ,is_leader :bool ):
        self .node_id =node_id 
        self .community_g1_formal =community_g1_formal # G1 (official organization) ID
        self .community_id =community_id # ID of G2 (actual state)
        self .neighbors =neighbors 
        self .name =name 
        self .gender =gender 
        self .age =age 
        self .proficiency =proficiency 
        self .is_leader =is_leader 

        dept_info =config_prompts .DEPARTMENTS [community_g1_formal %3 ]
        self .dept_name =dept_info ["name"]
        self .position ="Department Leader"if is_leader else "General Member"

        self .current_thought :str ="I haven't thought about anything yet."
        self .next_target_id :int =-1 
        self .conversation_log :List [str ]=[]

    def get_profile_text (self )->str :
        return (
        f"Name: {self .name } ({self .age }years old, {self .gender })\n"
        f"Department: {self .dept_name }\n"
        f"Position: {self .position }\n"
        f"Performance: {self .proficiency }\n"#Since 'performance' is a better translation than 'proficiency,' changed it.

        )

        # ==========================================
        # Helper function
        # ==========================================
def parse_thought_and_target (text :str ,default_target :int )->tuple [str ,int ]:
    thought =""
    target_id =default_target 
    match_thought =re .search (r'\[THOUGHT\]\s*(.*?)\s*(\[NEXT_TARGET_ID\]|$)',text ,re .DOTALL )
    if match_thought :
        thought =match_thought .group (1 ).strip ()
    else :
        thought =text 
    match_id =re .search (r'\[NEXT_TARGET_ID\]\s*(-?\d+)',text )
    if match_id :
        try :
            target_id =int (match_id .group (1 ))
        except ValueError :
            pass 
    return thought ,target_id 

def parse_target_id_only (text :str ,default_target :int )->int :
    """Extract IDs from text that contains only numbers"""
    match =re .search (r'(\d+)',text )
    if match :
        try :
            return int (match .group (1 ))
        except ValueError :
            pass 
    return default_target 

def get_agent_info_text (agent :Agent )->str :
# ★Correction: Added age, gender, and skill level for more detail
    return (
    f"[ID:{agent .node_id }] {agent .name } "
    f"({agent .age }years old, {agent .gender }, Performance:{agent .proficiency })\n"
    f"    Department: {agent .dept_name } / Position: {agent .position }"
    )
def load_graph_from_csv (run_id :int )->Optional [nx .Graph ]:
    root_dir =Path (__file__ ).resolve ().parents [1 ]
    net_dir =root_dir /config .network_results_dir 
    edges_path =net_dir /f"run_{run_id :03d}_G2_edges.csv"
    nodes_path =net_dir /"nodes_all_runs.csv"

    if not edges_path .exists ():
        print (f"Error: Edge file not found: {edges_path }")
        return None 
    try :
        G =nx .read_edgelist (edges_path ,delimiter =",",nodetype =int )
    except Exception as e :
        print (f"  ! Failed to load edge list: {e }")
        return None 

    if nodes_path .exists ():
        try :
            df_nodes =pd .read_csv (nodes_path )
            df_run =df_nodes [df_nodes ["run_id"]==run_id ]
            for _ ,row in df_run .iterrows ():
                node_id =int (row ["node_id"])
                if node_id in G .nodes :
                    G .nodes [node_id ]["community_G1_formal"]=int (row ["community_G1_formal"])
                    G .nodes [node_id ]["community_G2_actual"]=int (row ["community_G2_actual"])
                    G .nodes [node_id ]["is_leader_G1_formal"]=int (row ["is_leader_G1_formal"])
        except Exception as e :
            print (f"  ! Node attribute loading error: {e }")
    return G 

    # ==========================================
    # Main processing class ef _load_graph_from_csv(self) -> nx.Graph:
    # ==========================================
class ConversationSimulation :
    def __init__ (self ,run_id :int ,pattern_id :int ):
        self .run_id =run_id 
        self .pattern_id =pattern_id # ★Addition

        #Restore graphs and role information from CSV
        self .G =self ._load_graph_from_csv ()

        # Directory for log storage
        self .logs_dir =Path (config .conversation_results_dir )/f"run_{run_id :03d}"
        self .logs_dir .mkdir (parents =True ,exist_ok =True )

        # Agent Generation (Strict Application of Persona Sheet)
        self .agents =self ._initialize_agents ()
        self .finished_speakers =set ()

    def _load_graph_from_csv (self )->nx .Graph :
        """  # Folder/File:  | Folder/File:   # Folder/File: 
        results/01_network フォルダから、
        nodes_patterns.csv と pattern_{id}_edges.csv を読み込む
        
        results/01_network フォルダから、
        nodes_patterns.csv と pattern_{id}_edges.csv を読み込む
        
        results/01_network フォルダから、
        nodes_patterns.csv と pattern_{id}_edges.csv を読み込む
        
        results/01_network フォルダから、
        nodes_patterns.csv と pattern_{id}_edges.csv を読み込む
        """
        net_dir =Path (config .network_results_dir )

        # I will match the current file naming conventions.
        edges_path =net_dir /f"pattern_{self .pattern_id }_edges.csv"
        nodes_path =net_dir /"nodes_patterns.csv"# ★Change

        if not edges_path .exists ():
        # If the file cannot be found, display an error message that makes it easy to check the pattern_id and the actual file name.
            raise FileNotFoundError (f"Edge file not found: {edges_path } (Pattern ID: {self .pattern_id })")
            # 1. Edge loading
        try :
            G =nx .read_edgelist (edges_path ,delimiter =",",nodetype =int )
        except Exception as e :
            raise ValueError (f"Edge list loading failed: {e }")

            # 2. Loading Node Information (Community Leader)
        if nodes_path .exists ():
            df_nodes =pd .read_csv (nodes_path )
            # ★Change: Filtering by pattern_id
            df_target =df_nodes [df_nodes ["pattern_id"]==self .pattern_id ]

            if df_target .empty :
                print (f"Warning: Pattern ID {self .pattern_id } The data could not be found in nodes_patterns.csv.")

            for _ ,row in df_target .iterrows ():
                node_id =int (row ["node_id"])
                # Attach information if it is a node included in the graph
                if node_id in G .nodes :
                    G .nodes [node_id ]["community_G1_formal"]=int (row ["community_G1_formal"])
                    G .nodes [node_id ]["community_G2_actual"]=int (row ["community_G2_actual"])#This area is a remnant of the old graph creation file
                    G .nodes [node_id ]["is_leader_G1_formal"]=int (row ["is_leader_G1_formal"])#The same.
        else :
            print (f"Warning: Node attribute file not found: {nodes_path }")

        return G 

    def _initialize_agents (self )->Dict [int ,Agent ]:
        """
        Read the persona sheet and generate an Agent object. Do not fill in information that is not written on the sheet on your own.
        """
        agents ={}

        # ★Addition: Fix the random seed based on Run ID (for employee generation)
        # Run 1–4 -> seed 0, Run 5–8 -> seed 1, and so on.
        seed_val =(self .run_id -1 )//config .PATTERN_CYCLE 
        random .seed (seed_val )

        # 1. Identification of the Persona Sheet
        sim_dir =Path (__file__ ).parent 
        excel_path =sim_dir /"Persona Sheet.xlsx"

        if not excel_path .exists ():
            raise FileNotFoundError (f"'Persona Sheet.xlsx' cannot be found: {excel_path }")

        try :
            df_persona =pd .read_excel (excel_path )
        except Exception as e :
            raise ValueError (f"Failed to load the persona sheet: {e }")

            # 2. Allocation per node
        sorted_nodes =sorted (list (self .G .nodes ()))

        # Line count check
        if len (df_persona )<len (sorted_nodes ):
             raise ValueError (f"Error: Number of rows in the persona sheet ({len (df_persona )}) with the number of employees ({len (sorted_nodes )}) is insufficient.")

             # ★Addition: Extract the types of values (unique values) and list them
             # Use dropna() to remove empty fields and unique() to eliminate duplicates. This way, you can select them with equal probability as 'types'.
        gender_opts =df_persona ['Gender'].dropna ().unique ().tolist ()
        age_opts =df_persona ['Age'].dropna ().unique ().tolist ()
        prof_opts =df_persona ['Skillfulness'].dropna ().unique ().tolist ()

        for i ,node_id in enumerate (sorted_nodes ):
            row =df_persona .iloc [i ]

            # Get the values of the sheet as they are (leave blanks as blanks)
            name =str (row ['Surname'])if not pd .isna (row ['Surname'])else f"Employee_{node_id }"

            # Randomly select gender from a list (equal probability)
            gender =str (random .choice (gender_opts ))

            # Randomly select from an age list (equal probability)
            try :
                age =int (random .choice (age_opts ))
            except :
                age =-1 

                # Select randomly from the list of skills (equal probability)
            proficiency =str (random .choice (prof_opts ))

            # Acquire attributes from the graph
            # Set the default value if the key does not exist
            c_g1 =self .G .nodes [node_id ].get ("community_G1_formal",0 )
            c_g2 =self .G .nodes [node_id ].get ("community_G2_actual",c_g1 )
            is_leader_val =self .G .nodes [node_id ].get ("is_leader_G1_formal",0 )
            is_leader =(is_leader_val ==1 )

            neighbors =list (self .G .neighbors (node_id ))

            # Agent generation (arguments should match existing class definitions)
            agent =Agent (
            node_id =node_id ,
            community_g1_formal =c_g1 ,
            community_id =c_g2 ,
            neighbors =neighbors ,
            name =name ,
            gender =gender ,
            age =age ,
            proficiency =proficiency ,
            is_leader =is_leader 
            )
            agents [node_id ]=agent 

        print (f"  [Run {self .run_id }] Agent {len (agents )} Name initialization completed (Persona sheet applied)")
        return agents 

        # ------------------------------------------------------------------
        # Below this, def _call_llm ... continues (please do not delete)
        # ------------------------------------------------------------------

    def _call_llm (self ,system_prompt :str ,user_prompt :str ,max_tokens :int ,reasoning_effort :str ,verbosity :str )->str :
        if not client :
            return ""
        try :
            params ={
            "model":config .openai_model ,
            "messages":[
            {"role":"system","content":system_prompt },
            {"role":"user","content":user_prompt }
            ],
            "max_completion_tokens":max_tokens ,
            }

            is_advanced_model =any (x in config .openai_model for x in ["o1","o3","nano","gpt-5"])#Determine whether it is an inference-type AI.

            if is_advanced_model :#Settings related to inference tokens and the like.
                 params ["reasoning_effort"]=reasoning_effort 
                 params ["verbosity"]=verbosity 

            res =client .chat .completions .create (**params )
            return res .choices [0 ].message .content .strip ()
        except Exception as e :
            print (f"  ! API Error: {e }")
            return ""

            # [Revised] run_turn_0: Execute thinking and ID selection in two separate steps
    def run_turn_0 (self ):
        print (f"\n[Turn 0] 初期思考を生成中 ({len (self .agents )}Person)...")  # Folder/File: 
        #[Turn 0] Generating initial thoughts 
        results =[]# Folder/File: 


        def process_agent_turn0 (agent :Agent ):
        # ★Correction: Remove try-except and let it stop as is when an error occurs
            neighbors_info ="\n".join ([get_agent_info_text (self .agents [n_id ])for n_id in agent .neighbors ])

            # --- 1. Generation of Initial Thoughts (Think) ---
            prompt_0 =(
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{agent .get_profile_text ()}\n\n"
            f"[Connected Colleagues List]\n{neighbors_info }\n\n"
            f"You do not need to decide who to talk to yet. First, please think about your strategy."
            )

            thought_text =self ._call_llm (
            config_prompts .TURN0_SYSTEM_PROMPT ,
            prompt_0 ,
            max_tokens =config .tokens_phase2_reflect ,
            reasoning_effort =config .reasoning_phase2_reflect ,
            verbosity =config .verbosity_phase2_reflect 
            )

            # If it is empty due to an API error or similar, raise an exception and stop.
            if not thought_text :
                raise RuntimeError (f"Agent {agent .node_id }: Thought generation failed (Empty response)")

                # --- 2. Initial Target Selection (Select) ---
            prompt_2 =(
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{agent .get_profile_text ()}\n\n"
            f"[Your Current Thought]\n{thought_text }\n\n"
            f"【[Connected Colleagues List]\n{neighbors_info }\n\n"
            f"Based on the settings above, please select the ID of the person you should speak to first, as a single number."
            )

            resp_target =self ._call_llm (
            config_prompts .SELECT_TARGET_SYSTEM_PROMPT ,
            prompt_2 ,
            max_tokens =config .tokens_select_target ,
            reasoning_effort =config .reasoning_select_target ,
            verbosity =config .verbosity_select_target 
            )

            target_id =parse_target_id_only (resp_target ,default_target =-1 )

            # Relief: If the ID is invalid, choose randomly from nearby
            if target_id not in agent .neighbors :
                if agent .neighbors :
                    target_id =random .choice (agent .neighbors )
                else :
                    target_id =-1 

            agent .current_thought =thought_text 
            agent .next_target_id =target_id 
            return {"Agent_ID":agent .node_id ,"Initial_Thought":thought_text ,"First_Target_ID":target_id }

        with ThreadPoolExecutor (max_workers =config .conversation_max_parallel )as executor :
            futures =[executor .submit (process_agent_turn0 ,agent )for agent in self .agents .values ()]
            for i ,f in enumerate (futures ):
            # ★Correction: Removed try-except. In case of an error, an exception is immediately raised and the program stops.
                res =f .result ()
                results .append (res )
                if i %10 ==0 :print (".",end ="",flush =True )
        print (" Completed")

        out_path =self .logs_dir /"initial_thoughts.csv"
        pd .DataFrame (results ).to_csv (out_path ,index =False ,encoding ='utf-8-sig')
        print (f"  - Turn 0 Save Results: {out_path }")

    def _choose_target (self ,agent :Agent )->Optional [int ]:
        target_id =agent .next_target_id 
        if target_id not in self .agents :
            if agent .neighbors :
                target_id =random .choice (agent .neighbors )
            else :
                return None 
        return target_id 


    def run_conversation_pair (self ,speaker_id :int ,turn :int ,target_id :int =None )->Dict [str ,Any ]:
        speaker =self .agents [speaker_id ]

        # Initialization of return value
        row_data ={
        "Turn":turn ,"Speaker_ID":speaker_id ,"Target_ID":-1 ,
        "Speaker_Thought_Before":speaker .current_thought ,"Speaker_Speech":"",
        "Target_Thought_After":"","Target_Next_ID":-1 ,
        "Target_Reply":"","Speaker_Thought_After":"","Speaker_Next_ID":-1 ,
        "Status":"Success"
        }

        # Debug display flag for ID:0
        is_debug_target =(speaker_id ==0 )

        # Helper function to shorten logs for display
        def _get_debug_prompt (full_text :str )->str :
            lines =full_text .split ('\n')
            new_lines =[]
            is_log_section =False 
            for line in lines :
                if "[Log]"in line or "[Previous Conversation Log]"in line :
                    is_log_section =True 
                    new_lines .append (line )
                    continue 
                if is_log_section and line .startswith ("【"):
                    is_log_section =False 

                if is_log_section and line .strip ():
                    if len (line )>20 :
                        new_lines .append (line [:20 ]+"...")
                    else :
                        new_lines .append (line )
                else :
                    new_lines .append (line )
            return "\n".join (new_lines )

        remaining_turns =config .n_turns -turn 
        try :
            time_info_text =config_prompts .TIME_INSTRUCTION .format (remaining_turn =remaining_turns )
        except AttributeError :
            time_info_text =f"Current turn: {turn } / {config .n_turns }"

            # Choosing a partner
        if target_id is None :
            target_id =self ._choose_target (speaker )

            # Error if there is no opponent
        if target_id is None or target_id ==-1 :
            row_data ["Status"]="Error: No Valid Target (Isolated node or Empty neighbors)"
            return row_data 

        row_data ["Target_ID"]=target_id 
        listener =self .agents [target_id ]

        # Normal Log
        if speaker_id %5 ==0 :
            print (f"    Conversation: {speaker .name }({speaker_id }) -> {listener .name }({target_id })")

            # ★Debug Output: Start
        if is_debug_target :
            print (f"\n# ID=0 Turn Start (Turn {turn }) -----------------") 
            print (f"現在の思考：\n{speaker .current_thought }")
            print (f"#---")


            # --- 1. Speaker Talk (Speech Generation) ---
            # ★Correction: Because there is thinking involved, do not let it read the 'conversation log' here (cost reduction)
        try :
            p1_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{speaker .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_i }\n"
            f"[Your Current Thought]\n{speaker .current_thought }\n"
            f"[Person you are about to speak to]\n{get_agent_info_text (listener )}\n"
            f"Now, please speak to him/her."
            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [1] The speech prompt given to the speaker (Log Removed)")
                print (f"[System Prompt]\n{config_prompts .PHASE1_SPEAKER_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p1_prompt ))
                print (f"#------------------------------------------")

            speech_text =self ._call_llm (
            config_prompts .PHASE1_SPEAKER_PROMPT ,p1_prompt ,
            max_tokens =config .tokens_phase1_talk ,
            reasoning_effort =config .reasoning_phase1_talk ,
            verbosity =config .verbosity_phase1_talk 
            )
            if not speech_text :raise ValueError ("Empty speech generated")

            row_data ["Speaker_Speech"]=speech_text 
            speaker .conversation_log .append (f"Myself (To {target_id }): {speech_text }")
            listener .conversation_log .append (f"Opponent({speaker_id }): {speech_text }")

            if is_debug_target :  # Folder/File: ▶実際の発言：\n
                print (f"▶実際の発言：\n{speech_text }")
                print (f"#---")



        except Exception as e :
            row_data ["Status"]=f"Error in Speaker Talk: {e }"
            row_data ["Speaker_Speech"]="[ERROR] Speech generation failed"
            return row_data 

            # --- 2. Listener Reflect (Thought Update) ---
            # ★Important: Maintain the "conversation log" here for context understanding
        try :
            neighbors_info_j ="\n".join ([get_agent_info_text (self .agents [nid ])for nid in listener .neighbors ])
            p2_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{listener .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_j }\n"
            f"[Your Current Thought]\n{listener .current_thought }\n"
            f"[The person who just spoke to you]\n{get_agent_info_text (speaker )}\n"# ★Add this line
            f"[His/Her Speech]\n{speech_text }\n"
            f"[Conversation Memory]\n"+"\n".join (listener .conversation_log [-config .max_talk_history :])+"\n"
            f"Now, please update your thought."

            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [2] Listener({target_id }Thought prompt given to ) (Full Context)")
                print (f"[System Prompt]\n{config_prompts .PHASE2_LISTENER_UPDATE_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p2_prompt ))
                print (f"#------------------------------------------")

            resp_p2 =self ._call_llm (
            config_prompts .PHASE2_LISTENER_UPDATE_PROMPT ,p2_prompt ,
            max_tokens =config .tokens_phase2_reflect ,
            reasoning_effort =config .reasoning_phase2_reflect ,
            verbosity =config .verbosity_phase2_reflect 
            )
            new_thought_j ,_ =parse_thought_and_target (resp_p2 ,default_target =-1 )
            listener .current_thought =new_thought_j 
            row_data ["Target_Thought_After"]=new_thought_j 

            if is_debug_target :  # Folder/File: ▶実際の思考：\n
                print (f"▶実際の思考：\n{new_thought_j }")
                print (f"#---")



        except Exception as e :
            row_data ["Status"]=f"Error in Listener Reflect: {e }"
            return row_data 

            # --- 3. Listener Select Target ---
            # ★Correction: No log (judged only by thoughts and list)
        try :
            p2_5_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{listener .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_j }\n"
            f"[Your Current Thought]\n{listener .current_thought }\n"
            f"Please select the ID of the person you will speak to next, as a single number."# ★Add this line

            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [3] Listener({target_id }) Prompt for selecting ID given to (Log Removed)")
                print (f"[System Prompt]\n{config_prompts .SELECT_TARGET_SYSTEM_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p2_5_prompt ))
                print (f"#------------------------------------------")

            resp_target_j =self ._call_llm (
            config_prompts .SELECT_TARGET_SYSTEM_PROMPT ,p2_5_prompt ,
            max_tokens =config .tokens_select_target ,
            reasoning_effort =config .reasoning_select_target ,
            verbosity =config .verbosity_select_target 
            )
            next_target_j =parse_target_id_only (resp_target_j ,default_target =listener .next_target_id )
            listener .next_target_id =next_target_j 
            row_data ["Target_Next_ID"]=next_target_j 

            if is_debug_target :
                print (f"▶Actual Selection:{next_target_j }")
                print (f"#---")

        except Exception as e :
            row_data ["Status"]=f"Error in Listener Select: {e }"

            # --- 4. Listener Reply (Response Generation) ---
            # ★Correction: Because there is thinking involved, do not let it read the 'conversation log' here (cost reduction)
        try :
            p3_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{listener .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_j }\n"# ★Addition
            f"[Your Current Thought]\n{listener .current_thought }\n"
            f"[The person who just spoke to you]\n{get_agent_info_text (speaker )}\n"# ★Add this line
            f"[His/Her Speech]\n{speech_text }\n"
            f"Now, please reply to him/her."
            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [4] Listener({target_id }FB prompt given to ) (Log Removed)")
                print (f"[System Prompt]\n{config_prompts .PHASE3_LISTENER_REPLY_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p3_prompt ))
                print (f"#------------------------------------------")

            reply_text =self ._call_llm (
            config_prompts .PHASE3_LISTENER_REPLY_PROMPT ,p3_prompt ,
            max_tokens =config .tokens_phase1_talk ,
            reasoning_effort =config .reasoning_phase1_talk ,
            verbosity =config .verbosity_phase1_talk 
            )
            if not reply_text :reply_text ="..."
            row_data ["Target_Reply"]=reply_text 
            listener .conversation_log .append (f"Myself (To {speaker_id }): {reply_text }")
            speaker .conversation_log .append (f"Opponent({target_id }): {reply_text }")

            if is_debug_target :  # Folder/File: ▶実際のFB：\n
                print (f"▶実際のFB：\n{reply_text }")
                print (f"#---")



        except Exception as e :
            row_data ["Status"]=f"Error in Listener Reply: {e }"
            row_data ["Target_Reply"]="[ERROR] Response generation failed"
            return row_data 

            # --- 5. Speaker Reflect (Updating Thoughts) ---
            # ★Important: Maintain the "conversation log" here for context understanding
        try :
            neighbors_info_i ="\n".join ([get_agent_info_text (self .agents [nid ])for nid in speaker .neighbors ])
            p4_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{speaker .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_i }\n"
            f"[Your Current Thought]\n{speaker .current_thought }\n"
            f"[The person you just spoke to]\n{get_agent_info_text (listener )}\n"# ★Add this line
            f"[His/Her Reply]\n{reply_text }\n"
            f"[Conversation Memory]\n"+"\n".join (speaker .conversation_log [-config .max_talk_history :])+"\n"
            f"Now, please update your thought."
            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [5] Thought organization prompt given to the speaker (Full Context)")
                print (f"[System Prompt]\n{config_prompts .PHASE4_SPEAKER_UPDATE_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p4_prompt ))
                print (f"#------------------------------------------")

            resp_p4 =self ._call_llm (
            config_prompts .PHASE4_SPEAKER_UPDATE_PROMPT ,p4_prompt ,
            max_tokens =config .tokens_phase2_reflect ,
            reasoning_effort =config .reasoning_phase2_reflect ,
            verbosity =config .verbosity_phase2_reflect 
            )
            new_thought_i ,_ =parse_thought_and_target (resp_p4 ,default_target =-1 )
            speaker .current_thought =new_thought_i 
            row_data ["Speaker_Thought_After"]=new_thought_i 

            if is_debug_target :  # Folder/File: ▶実際の思考：\n
                print (f"▶実際の思考：\n{new_thought_i }")
                print (f"#---")



        except Exception as e :
            row_data ["Status"]=f"Error in Speaker Reflect: {e }"
            return row_data 

            # --- 6. Speaker Select Target (Target Selection) ---
            # ★Correction: No log (judged only by thoughts and list)
        try :
            p4_5_prompt =(
            f"[About Time Concept]\n{time_info_text }\n"
            f"[Project Context]\n{config_prompts .PROJECT_CONTEXT }\n\n"
            f"[Your Profile]\n{speaker .get_profile_text ()}\n"
            f"[Connected Colleagues List]\n{neighbors_info_i }\n"
            f"[Your Current Thought]\n{speaker .current_thought }\n"
            f"Please select the ID of the person you will speak to next, as a single number."
            )

            if is_debug_target :
                print (f"\n#------------------------------------------")
                print (f"# [6] ID Selection Prompt Given to the Speaker (Log Removed)")
                print (f"[System Prompt]\n{config_prompts .SELECT_TARGET_SYSTEM_PROMPT }\n")# ★Addition
                print (_get_debug_prompt (p4_5_prompt ))
                print (f"#------------------------------------------")

            resp_target_i =self ._call_llm (
            config_prompts .SELECT_TARGET_SYSTEM_PROMPT ,p4_5_prompt ,
            max_tokens =config .tokens_select_target ,
            reasoning_effort =config .reasoning_select_target ,
            verbosity =config .verbosity_select_target 
            )
            next_target_i =parse_target_id_only (resp_target_i ,default_target =speaker .next_target_id )
            speaker .next_target_id =next_target_i 
            row_data ["Speaker_Next_ID"]=next_target_i 

            if is_debug_target :
                print (f"▶Actual Selection:{next_target_i }")
                print (f"#---")

        except Exception as e :
            row_data ["Status"]=f"Error in Speaker Select: {e }"

        return row_data 

        # ▼▼▼ Please add this method inside the ConversationSimulation class ▼▼▼
        # ▼▼▼ Modification: Replace the entire run_leader_proposals method ▼▼▼
    def run_leader_proposals (self ):
        """
        After the simulation ends, the leader (or representative) of each department will compile the project proposals.
        """
        print (f"  [Run {self .run_id }] Generating project proposals by the leader...")

        # --- Helper function for log formatting (only abbreviates the conversation log part) ---
        def _get_debug_prompt (full_text :str )->str :
            lines =full_text .split ('\n')
            new_lines =[]
            is_log_section =False 
            for line in lines :
            # Detect the start section of the log
                if "[Discussion log so far]"in line or "[Conversation Log]"in line or "[Log]"in line :
                    is_log_section =True 
                    new_lines .append (line )
                    continue 
                    # End the log when the next section (such as [Instructions]) comes.
                if is_log_section and line .startswith ("【"):
                    is_log_section =False 

                    # Display in shortened form within the log section, otherwise display as is
                if is_log_section and line .strip ():
                    if len (line )>20 :
                        new_lines .append (line [:20 ]+"...")
                    else :
                        new_lines .append (line )
                else :
                    new_lines .append (line )
            return "\n".join (new_lines )

            # 1. Identification of the leader
        leaders =[]
        try :
        # Search for results/01_network/nodes_all_runs.csv
            nodes_path =self .logs_dir .parent .parent /"01_network"/"nodes_patterns.csv"

            if nodes_path .exists ():
                df_nodes =pd .read_csv (nodes_path )
                current_run_leaders =df_nodes [
                (df_nodes ["pattern_id"]==self .pattern_id )&
                (df_nodes ["is_leader_G1_formal"]==1 )
                ]["node_id"].tolist ()

                for aid in current_run_leaders :
                    if aid in self .agents :
                        leaders .append (self .agents [aid ])
        except Exception as e :
            print (f"Warning: Failed to load leader information ({e })")

            # Remedial measures in the absence of a leader
        if not leaders :
            print ("  (Since leader information cannot be found, representatives for each department will be selected)")
            dept_seen =set ()
            for agent in self .agents .values ():
                if agent .dept_name not in dept_seen :
                    leaders .append (agent )
                    dept_seen .add (agent .dept_name )

        if not leaders :
             print ("Error: There is no agent available to create a project proposal.")
             return 

             # 2. Generation of project proposals
        proposal_rows =[]

        # Identify the reader with the smallest ID for debug display
        min_leader_id =min ([l .node_id for l in leaders ])if leaders else -1 

        for leader in leaders :
        # Retrieving conversation logs (only a set number, such as the most recent 20)
            log_text ="\n".join (leader .conversation_log [-20 :])

            # System Prompt
            system_prompt =config_prompts .PROPOSAL_SYSTEM_PROMPT 

            # User prompt (embed profile, thoughts, logs)
            user_prompt =config_prompts .PROPOSAL_USER_PROMPT_TEMPLATE .format (
            profile_text =leader .get_profile_text (),# ★Addition: Profile
            current_thought =leader .current_thought ,# ★Addition: Current Thoughts
            log_text =log_text ,# Discussion log
            dept_name =leader .dept_name ,
            agent_name =leader .name 
            )

            # ★Debug Display: Only the leader with the smallest ID is formatted and displayed in the prompt
            if leader .node_id ==min_leader_id :
                print (f"\n#======== [DEBUG] Leader Prompt (ID: {leader .node_id }) ========")
                print ("#--- System Prompt ---")
                print (system_prompt )
                print ("#--- User Prompt (Formatted for Display) ---")
                print (_get_debug_prompt (user_prompt ))# ★Use the formatting function here
                print ("#=========================================================\n")

            try :
                resp =client .chat .completions .create (
                model =config .openai_model ,
                messages =[
                {"role":"system","content":system_prompt },
                {"role":"user","content":user_prompt }
                ],
                )
                proposal_text =resp .choices [0 ].message .content .strip ()

                # ★Debug Display: Show Answers Too
                if leader .node_id ==min_leader_id :
                    print (f"\n#======== [DEBUG] Leader Response (ID: {leader .node_id }) ========")
                    print (proposal_text )
                    print ("#===========================================================\n")

            except Exception as e :
                print (f"Error generating proposal for {leader .name }: {e }")
                proposal_text ="Generation failed"

            proposal_rows .append ({
            "run_id":self .run_id ,
            "pattern_id":self .pattern_id ,# ★Add here
            "agent_id":leader .node_id ,
            "agent_name":leader .name ,
            "dept":leader .dept_name ,
            "proposal_text":proposal_text 
            })

            # 3. Save
        out_path =self .logs_dir /"output.csv"
        header =not out_path .exists ()
        df_out =pd .DataFrame (proposal_rows )
        df_out .to_csv (out_path ,mode ='a',header =header ,index =False ,encoding ='utf-8-sig')
        print (f"  The project plan has been saved: {out_path }")

        # (Everything below here is the original code)
        print (f"--- Start Conversation Simulation (Run {self .run_id }) ---")
        # ...


        # -----------------------------------------------------
        # [Revised] run (with batch parallel processing and agent generation)
        # -----------------------------------------------------
    def run (self ):
        print (f"--- Start Conversation Simulation (Run {self .run_id }) ---")



        # Initialization such as Persona loading (existing)
        self .run_turn_0 ()

        history_rows =[]

        for turn in range (1 ,config .n_turns +1 ):
            print (f"\n#=================================")
            print (f"\n[Turn {turn }/{config .n_turns }] Running conversation batch...")

            # Everyone starts from a state of 'has not spoken yet'
            pending_speakers =list (self .agents .keys ())
            random .shuffle (pending_speakers )# Random order

            # List of people who finished processing this turn
            finished_speakers =set ()

            # Repeat the batch until everyone has finished speaking
            while len (finished_speakers )<len (self .agents ):

            # --- 1. Search for pairs that can be executed simultaneously (matching) ---
                batch_pairs =[]# List of (speaker_id, target_id)
                busy_agents =set ()# Person who will be in conversation in this batch (speaker or listener)

                # Check the people you haven't talked to in order
                for s_id in pending_speakers :
                    if s_id in finished_speakers :
                        continue 
                    if s_id in busy_agents :
                        continue # I have already been designated as a listener in this batch.

                        # Decide on a conversation partner (Since it's a momentary calculation, it's okay to do this serially)
                    s_agent =self .agents [s_id ]
                    t_id =self ._choose_target (s_agent )

                    if t_id is None :
                    # Skip if there is no partner (treat as completed)
                        finished_speakers .add (s_id )
                        continue 

                        # If the other person is still free in this batch, the pair is formed.
                    if t_id not in busy_agents :
                        batch_pairs .append ((s_id ,t_id ))
                        busy_agents .add (s_id )
                        busy_agents .add (t_id )

                if not batch_pairs :
                # If pairs can no longer be made (for example, when all remaining members are included in busy_agents),
                # The remaining people do not need to exit the loop in order to pass to the next batch, but
                # Prevention of an unlikely infinite loop
                    if len (finished_speakers )+len (batch_pairs )==len (self .agents ):
                        break 

                        # --- 2. Batch Parallel Execution ---
                print (f"  Batch execution: {len (batch_pairs )} pairs running in parallel...")

                # ThreadPoolExecutor: A mechanism that delegates multiple tasks to separate threads (workers)
                with ThreadPoolExecutor (max_workers =config .conversation_max_parallel )as executor :
                    future_to_speaker ={
                    executor .submit (self .run_conversation_pair ,s ,turn ,t ):s 
                    for s ,t in batch_pairs 
                    }

                    for future in as_completed (future_to_speaker ):
                        s_id =future_to_speaker [future ]
                        try :
                            result =future .result ()
                            if result :
                                history_rows .append (result )
                        except Exception as e :
                            print (f"Error in conversation pair (Speaker {s_id }): {e }")

                            # After execution is finished, add to the 'List of People Who Have Finished Talking'
                        finished_speakers .add (s_id )

                        # Batch ended. In the next loop, we will match the remaining people who have not spoken yet.
                        # (Those who were skipped while being the listener get another chance here)

                        # --- End-of-Turn Processing ---
            print (f"  [Turn {turn }] Survey in progress...")
            questionnaire_simulation .run_questionnaire_for_agents (turn ,list (self .agents .values ()),self .run_id )

            # Save process after all turns have ended
        out_path =self .logs_dir /"conversation_history.csv"
        df_hist =pd .DataFrame (history_rows )
        df_hist .to_csv (out_path ,index =False ,encoding ='utf-8-sig')

        # Report Generation
        self .run_leader_proposals ()

        print (f"\n[Run {self .run_id }] Entire process completed.")

        # ★Change: Change the second argument to pattern_id (int)
def run_simulation (run_id :int ,pattern_id :int ):
    sim =ConversationSimulation (run_id ,pattern_id )
    sim .run ()

if __name__ =="__main__":  # Folder/File: 
#!!! Debug Mode: Executing with specified pattern !!!
    print ("\n!!! デバッグモード: 指定パターンで実行します !!!")
    # Folder/File: 

    # Give priority to the config settings, and if not available, use 1 as the default.
    if config .fixed_pattern_id is not None :
        target_pattern_id =config .fixed_pattern_id 
    else :
        target_pattern_id =1 

    target_run_id =1 

    print (f"Run {target_run_id } (Pattern {target_pattern_id }) Starting...")
    # Since the loading of the graph is done within the class, only the ID is passed.
    run_simulation (target_run_id ,target_pattern_id )