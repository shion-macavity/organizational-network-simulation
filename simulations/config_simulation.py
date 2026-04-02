"""
config_simulation.py
A file that summarizes "setting values" that are shared throughout the simulation.
★Fixed: Created instance variable 'config' and made it visible from outside.
"""

from dataclasses import dataclass 



@dataclass 
class SimulationConfig :
# --- Basic network settings for hk_network ---
#n_nodes: int = 15 Only compatible with #3 and above
#m_edges: int = 3
#hk_p: float = 1
#rewire_q: float = 0.1
#　-----------------------------------------

    n_runs :int =100 
    network_results_dir :str ="results/01_network"

    # ==========================================
    # Random number seed control settings Do not touch!
    # ==========================================
    PATTERN_CYCLE =4 # How many patterns do you run once? (Use the same random number for Runs 1 to 4)
    CUSTOMER_SEED_OFFSET =10000 # Random number value to be shifted when generating customer FB

    # --- Conversation simulation settings ---
    n_turns :int =8 
    max_talk_history :int =8 
    conversation_results_dir :str ="results/02_conversation"
    conversation_max_parallel :int =7 

    # ★Added: Parallel number of report generation (set to a small number because the amount of tokens is large)
    report_max_parallel :int =2 

    # --- AI model settings ---
    # --- AI model settings (for Gemini API) ---
    # Basic model (light version: used for nomination, utterance, reply, customer FB, questionnaire, etc.)
    gemini_model :str ="gemini-2.5-flash-lite"
    # Upper model (thought version: used for initial thinking, reflection, and plan generation)
    gemini_advanced_model :str ="gemini-2.5-flash"
    temperature :float =1.0 # Diversity of answers (1.0 is maximum)


    # ★Added: If you want to experiment with a specific pattern ID, specify a number (e.g. 4)
    # If you want to loop 1,2,3,4... as usual, set it to None.
    fixed_pattern_id :int =None 

    # 1. Conversation phase (Talk)
    tokens_phase1_talk :int =1024 
    #reasoning_phase1_talk: str = "MINIMAL"
    budget_phase1_talk :int =524 
    verbosity_phase1_talk :str ="low"

    # 2. Thought update/reflection phase (Reflect)
    tokens_phase2_reflect :int =2048 
    #reasoning_phase2_reflect: str = "MINIMAL"
    budget_phase2_reflect :int =1024 
    verbosity_phase2_reflect :str ="medium"

    # 3. Final report creation (Report)
    tokens_phase3_report :int =4096 
    #reasoning_phase3_report: str = "MINIMAL"
    budget_phase3_report :int =2048 
    verbosity_phase3_report :str ="medium"

    # 4. Psychological Safety Survey
    tokens_survey :int =50 
    #reasoning_survey: str = "MINIMAL"
    budget_survey :int =0 
    verbosity_survey :str ="low"

    # ★Added: Target selection phase (Select Target)
    tokens_select_target :int =50 # Increase it a little (10 may be difficult)
    #reasoning_select_target: str = "MINIMAL"
    budget_select_target :int =0 
    verbosity_select_target :str ="low"

    # 5.Customer Feedback
    tokens_customer_feedback :int =574 # Increase it a little
    #reasoning_customer_feedback: str = "MINIMAL" # MINIMAL is also recommended for speedup here
    budget_customer_feedback :int =524 
    verbosity_customer_feedback :str ="low"

    product_evaluation_dir :str ="results/04_product_eval_ssr"

    # ★★★ This is important: materialize the class and put it in the config variable ★★★
config =SimulationConfig ()