# analysis/run_full_analysis.py
import sys 
import os 
import traceback 
from pathlib import Path 
from concurrent .futures import ThreadPoolExecutor ,as_completed 

# --- Path Settings ---
current_dir =Path (__file__ ).resolve ().parent 
root_dir =current_dir .parent 
sys .path .append (str (root_dir ))

from analysis .config import Config 
from analysis .utils import load_run_data 

# Module Import
try :
    from analysis .modules .timeseries import run_timeseries_analysis 
    from analysis .modules .psych_safety import run_psych_safety_analysis 
    from analysis .modules .interaction import run_interaction_analysis 
    from analysis .modules .interaction_speaker import run_interaction_speaker_analysis 
    from analysis .modules .global_consistency import run_consistency_analysis 
    from analysis .modules .community import run_community_analysis 
except ImportError as e :
    print (f"Import Error: {e }")
    sys .exit (1 )

def process_single_run (run_id ):
    print (f"--- [START] Processing Run {run_id } ---")
    try :
    # Data Load
        df_history ,df_initial =load_run_data (run_id )
        if df_history is None or df_initial is None :
            print (f"[ERROR] Run {run_id }: Data load failed.")
            return 

        mode =Config .ANALYSIS_MODE .lower ()
        print (f"  [Info] Run {run_id }: Loaded {len (df_history )} logs. Mode={mode }")

        # --- Light Mode (01, 03, 04) ---
        if mode in ["light","all"]:
            run_timeseries_analysis (run_id ,df_history ,df_initial )
            run_psych_safety_analysis (run_id ,df_history ,df_initial )
            run_interaction_analysis (run_id ,df_history ,df_initial )

            # --- Psych Only Mode (03) ---
        if mode =="psych_only":
            run_psych_safety_analysis (run_id ,df_history ,df_initial )

            # --- Speaker Analysis (04-Speaker) ---
        if mode in ["speaker_only","all"]:
            run_interaction_speaker_analysis (run_id ,df_history ,df_initial )

            # --- Heavy Mode (02, 05) ---
        if mode in ["heavy","all"]:
            run_consistency_analysis (run_id ,df_history ,df_initial )
            run_community_analysis (run_id ,df_history ,df_initial )

        print (f"--- [DONE] Finished Run {run_id } ---")

    except Exception :
        print (f"\n[CRITICAL ERROR] Run {run_id } Failed!")
        print (traceback .format_exc ())
        raise 

def main ():
    print ("==================================================")
    print (f"Analysis Manager")
    print (f"Mode: {Config .ANALYSIS_MODE }")
    print (f"LLM: {Config .LLM_PROVIDER } ({Config .LLM_MODEL })")
    print (f"Range: Run {Config .START_RUN } to {Config .END_RUN }")
    print (f"Workers: {Config .MAX_WORKERS }")
    print ("==================================================")

    run_ids =list (range (Config .START_RUN ,Config .END_RUN +1 ))
    os .makedirs (Config .OUTPUT_DIR ,exist_ok =True )

    try :
        for run_id in run_ids :
            process_single_run (run_id )

    except KeyboardInterrupt :
        print ("\n\n!!! KeyboardInterrupt Detected !!!")
        sys .exit (1 )
    except Exception as e :
        print (f"Unexpected Error: {e }")

    print ("\nAll analyses completed.")

if __name__ =="__main__":
    main ()