import os 
import sys 

# --- Obtaining the Project Root Path ---
# This file is the assumption in analysis/config_analysis.py for this analysis
current_dir =os .path .dirname (os .path .abspath (__file__ ))
PROJECT_ROOT =os .path .dirname (os .path .dirname (current_dir ))

# --- Input Directory (Input) ---
RESULTS_DIR =os .path .join (PROJECT_ROOT ,"results")
SSR_RESULTS_DIR =os .path .join (RESULTS_DIR ,"08_analysis_by_llm")
STRUCTURE_DIR =os .path .join (RESULTS_DIR ,"07_psychological_safety_analysis","仮分析","グラフ分析","Summary")# Folder/File: Graph Analysis | Folder/File: Graph Analysis | Folder/File: グラフ分析

# Specific input file path/pattern
PATH_COMMUNITY =os .path .join (STRUCTURE_DIR ,"community_assignments_final_1to8.csv")
# Adjacency matrix: turn{t}to{t}/run_{rrr}_matrix.csv format
PATH_MATRIX_DIR_FMT =os .path .join (STRUCTURE_DIR ,"adjacency matrix","turn{}to{}")

# File name pattern of SSR data (for format, not for glob)
# Replace and use {run_id}
# ※Since the file name is zero-padded to three digits like run_001_..., {:03} is used
FILE_PATTERN_P ="run_{:03}_1_timeseries_speaker.csv"# Progress (P)
FILE_PATTERN_CONSISTENCY ="run_{:03}_2_consistency.csv"# Similarity (PTS)
FILE_PATTERN_RD ="run_{:03}_3_disclosure_speaker.csv"# Self-disclosure (RD)
FILE_PATTERN_CT ="run_{:03}_4_interaction_listener.csv"# Critical Thinking (CT)
# ※ Since CT may not have a _listener, check for its existence in the script

# --- Output Directory (Output) ---
ANALYSIS_DIR =os .path .join (PROJECT_ROOT ,"analysis","本分析")# Folder/File: This Analysis | Folder/File: This Analysis | Folder/File: 本分析
OUTPUT_PREPROCESS =os .path .join (ANALYSIS_DIR ,"01_preprocessing")
OUTPUT_INDICES =os .path .join (ANALYSIS_DIR ,"02_indices")
OUTPUT_DATASETS =os .path .join (ANALYSIS_DIR ,"03_datasets")

# --- Analysis Parameters ---
RUN_IDS =list (range (1 ,101 ))# Run 1-100
TURNS =list (range (1 ,9 ))# Turn 1-8
AGENTS =list (range (14 ))# Agent 0-13
LIKERT_SCALE_MAX =5 # Maximum SSR Score

# Department Definition
DEPT_DEFINITIONS ={
0 :[0 ,1 ,2 ,3 ,4 ,5 ,6 ],
1 :[7 ,8 ,9 ,10 ,11 ,12 ,13 ]
}

# --- Common Settings ---
ENCODING ="utf-8-sig"