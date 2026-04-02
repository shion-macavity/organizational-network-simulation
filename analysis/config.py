# analysis/config.py
import os 
from pathlib import Path 

# ==========================================
# 1. Execution Environment Settings
# ==========================================
class Config :
# --- Path Settings ---
# The root directory as seen from this file (config.py)
    BASE_DIR =Path (__file__ ).resolve ().parent .parent 
    OUTPUT_DIR =BASE_DIR /"results"/"08_analysis_by_llm"



    # --- Execution Mode Setting ---
    # "light": Execute only 01 (chronological), 03 (psychological), 04 (interaction) (high speed)
    # "heavy": Only execute 02 (exact match) and 05 (community) (slow, high API usage)
    # "speaker_only": Execute only 04-Speaker (Speaker's Agreeableness) [Addition]
    # "psych_only" : Execute only 03 (psychological safety) [Addition]
    # "all": Execute all
    ANALYSIS_MODE ="heavy"

    # --- Execution Range (Run ID) ---
    START_RUN =1 
    END_RUN =100 

    # --- Parallel Processing Settings ---
    MAX_WORKERS =50 

    # --- Parallel Processing Settings ---
    MAX_WORKERS =50 

    # --- Model Settings ---
    #LLM_PROVIDER = "gemini"  # "gemini" or "openai"
    LLM_PROVIDER ="openai"# "gemini" or "openai"

    #LLM_MODEL = "gemini-2.5-flash-lite"  # "gpt-4o-mini", "o1-mini", "gpt-5-mini", etc.
    LLM_MODEL ="gpt-5-mini"# "gpt-4o-mini", "o1-mini", "gpt-5-mini", etc.
    EMBEDDING_MODEL ="text-embedding-3-small"

    # Parameter Settings
    TEMPERATURE =1.0 

    # Reasoning Model (Reasoning Models / o1, o3, etc.) Flag
    # If True: temperature control is disabled (fixed at 1), and parameters for thought tokens (such as max_completion_tokens) are applied.
    # Please set to True if models like GPT-5-mini have thought tokens. Normal models are False.
    IS_REASONING_MODEL =True # Change to True for GPT-5-mini

    # Options for reasoning model (only effective when IS_REASONING_MODEL = True)
    REASONING_EFFORT ="low"# "minimal", "low", "medium", "high"
    OUTPUT_VERBOSITY ="low"# "low", "medium", "high"


    # Thought Budget / Output Token Limit
    MAX_OUTPUT_TOKENS =500 

    # ==========================================
    # 2. Common Prompt Settings
    # ==========================================
    SITUATION_CONTEXT ="""
Situation: You are an expert in Organizational Behavior and Team Dynamics analysis.
You are observing the internal discussion and private thoughts of a game company's team.
The team consists of 14 members working on a project across 8 turns.
"""

    OUTPUT_RULE ="""
Rule: You must output your response in the following format:

Rule: You must output your response in the following format:
[REASON] Briefly provide essential support (logic, reason, or context) for [ANSWER] in a single, concise sentence; you may quote one to three phrases to support it, but do not list more than necessary.\n.
[ANSWER] State only the direct conclusion in one short sentence. Strictly exclude any reasoning, background details or verbatim quotes from the text.

However, regarding the response format, the following are prohibited:
1. When expressing a negative opinion, do not use negative sentence structures (e.g., "do not"). Instead, use affirmative sentences containing negative vocabulary (e.g., "disagree", "reject", "doubt").
2. Similarly, when expressing a positive opinion, do not use double negatives or the negation of negative words.
3. Answering with proverbs or idioms is also prohibited.
"""
    Risk_Check_Rule ="""
    Rule: You must output your response in the following format:
    [REASON] (If you judge that the individual does not recognize the concern, briefly state the reason in English. \n
    If you judge that the individual does recognize the concern, briefly present your view in one sentence in English, 
    citing a relevant portion of the original text verbatim as the basis for your judgment that “the individual recognizes the risk"; you may identify and quote one to three concerns if applicable, but do not list more than necessary.\n)
    [ANSWER] (Answer only Yes or No. You must not output reason and supplement here.)

    However, regarding the response format, the following are prohibited:
    1. When expressing a negative opinion, do not use negative sentence structures (e.g., "do not"). Instead, use affirmative sentences containing negative vocabulary (e.g., "disagree", "reject", "doubt").
    2. Similarly, when expressing a positive opinion, do not use double negatives or the negation of negative words.
    3. Answering with proverbs or idioms is also prohibited.

    """

    # ==========================================
    # 3. Parameters by Analysis & 5-Level Anchor
    # ==========================================
ANALYSIS_PARAMS ={
# ---------------------------------------------------------
# Analysis 01: Time Series Progress (Timeseries)
# ---------------------------------------------------------
"analysis_1_timeseries":{
# Let [someone] consider the influence of 'speech' that exists between thoughts
#"question": "Considering the speech event (input/output) that occurred between these thoughts, has the agent's understanding **stagnated** or **progressed**?",
"question":"How much has {agent_name}'s understanding stagnated or progressed, considering the speech event (input/output) that occurred between these thoughts?"
"In this context, 'progress' refers to practical and functional advancement. It excludes simple repetition of thoughts, aimless abstract ideation, or mere expressions of praise and emotion.\n "
"The focus is to determine whether there has been actual movement toward the next idea or task through intentional concretization, abstraction, or the introduction of new perspectives.\n",
"anchors":[
"The thought has stagnated or is repeating.",# 1: Stagnation
"The thought has slightly stagnated.",# 2: Slight stagnation
"The thought has neither progressed nor stagnated.",# 3: Neutral
"The thought has slightly progressed.",# 4: Slight progress
"The thought has progressed with new details."# 5: Progress
]
},

# ---------------------------------------------------------
# Analysis 02: Full Match (Consistency)
# ---------------------------------------------------------
"analysis_2_consistency":{
#"question": "Comparing the thoughts of Agent A and Agent B, are they speaking conflictingly or consistently regarding genre, direction, and intensity?",
"question":"How conflictingly or consistently are {agent_a} and {agent_b} speaking regarding logic, genre, direction, and intensity, when comparing their thoughts?",
"anchors":[
"They are speaking conflictingly.",# 1: Conflict/Disagreement
"They are speaking slightly conflictingly.",# 2: Slight mismatch
"They are speaking neither consistently nor conflictingly.",# 3: Neutral
"They are speaking slightly consistently.",# 4: Slight match
"They are speaking consistently."# 5: Match
]
},

# ---------------------------------------------------------
# Analysis 03: Psychological Safety
# ---------------------------------------------------------
"analysis_3_disclosure":{
# Phase 1: Recognition of Negative Elements (Yes/No)

"risk_check_question":"Does {agent_name} consciously harbor specific risks, doubts, or objections in their private thought? Answer 'No' if the concern is only an external observation and not recognized by {agent_name} themselves. Answer only Yes or No.\n"
"The 'concern' in this context refers to any risk, doubt, or objection that {agent_name} is consciously aware of, regardless of their own assessment of its significance or severity. This includes issues they may deem minor or acceptable. Do not include objective risks that you have identified if {agent_name} does not personally recognize them.",
# Phase 2: Determination of disclosure level (only if there is a risk)
# Correction:
# 1. Make clear that {detected_risk} is an 'internal concern' that includes 'risks, doubts, and objections'.
# 2. Use "to what extent" to ask about the degree.
# 3. By questioning not the content of the statements themselves, but whether the concerns were 'communicated',
#    Even if it is in the form of a question or an indirect expression, it should be appreciated as long as the intention is conveyed.
"disclosure_question":"Given the PREVIOUS DETECTION of the internal concern \"{detected_risk}\" (including risks, doubts, or objections), to what extent is this concern communicated or witheld by {agent_name} in the speech?\n"
"The 'concern' in this context refers to any risk, doubt, or objection that {agent_name} is consciously aware of, regardless of their own assessment of its significance or severity. This includes issues they may deem minor or acceptable. Do not include objective risks that you have identified if {agent_name} does not personally recognize them.\n"
"Do not mention whether or not information unrelated to the question is disclosed in [ANSWER].",



"anchors":[
"The concern is withheld.",# 1: Completely confidential (not shared at all)
"The concern is slightly withheld.",# 2: Mostly kept secret (not shared much)
"The concern is partially communicated.",# 3: Partially communicated/shared
"The concern is slightly communicated.",# 4: Approximately communicated/shared
"The concern is communicated."# 5: Fully communicated/shared
]
},

# ---------------------------------------------------------
# Analysis 04: Interaction
# ---------------------------------------------------------

"analysis_4_interaction":{
# Question: Did the listener examine the issue from multiple perspectives and update their thinking, or did they just believe what they wanted to believe (blindly follow/conform)?
#"question": "Considering the listener's original thought and the speaker's input, did the listener update their thought by **critically examining** perspectives, or did they just **blindly conform**?",
"question":"How critically did {listener_name} examine perspectives to update their thought, versus blindly conforming, considering {listener_name}'s original thought and the speaker's input?",
"anchors":[
"The listener accepted the opinion blindly.",# 1: Blind obedience/conformity
"The listener accepted the opinion slightly blindly.",# 2: Slightly blind obedience
"The listener accepted the opinion neutrally.",# 3: Neutral
"The listener accepted the opinion slightly critically.",# 4: Somewhat multi-faceted/critical
"The listener accepted the opinion critically."# 5: Examine from multiple perspectives and critically
]
},

# ---------------------------------------------------------
# Analysis 04-Speaker: Interaction of the Speaker
# ---------------------------------------------------------
"analysis_4_interaction_speaker":{
# Question: Did the speaker consider it from multiple perspectives in response to the other person's reply, or did they simply acquiesce?
"question":"How critically did {speaker_name} examine perspectives to update their thought, versus blindly conforming, considering {speaker_name}'s original thought and the feedback reply from the partner?",


"anchors":[
"The speaker accepted the feedback blindly.",# 1: Blind obedience/conformity
"The speaker accepted the feedback slightly blindly.",# 2: Slightly blind obedience
"The speaker accepted the feedback neutrally.",# 3: Neutral
"The speaker accepted the feedback slightly critically.",# 4: Somewhat multi-faceted/critical
"The speaker accepted the feedback critically."# 5: Examine from multiple perspectives and critically
]
}

# ---------------------------------------------------------
# Analysis 04-Speaker: Interaction of the Speaker
# ---------------------------------------------------------

}

