# analysis/modules/timeseries.py
import pandas as pd 
import traceback 
from concurrent .futures import ThreadPoolExecutor ,as_completed # Addition
from analysis .config import Config ,ANALYSIS_PARAMS 

from analysis .utils import call_gemini ,parse_response ,calculate_ssr_metrics ,create_rater ,load_agent_name_map 
def run_timeseries_analysis (run_id ,df_history ,df_initial ):
    """
    Analysis 1: Time Series Progress (Timeseries)
    Analyze the "change in thinking (stagnation vs progress)" for each conversation.
    Parallelized & CSV split output version
    """
    try :
    # Divide the output path into Speaker/Listener
        path_s =Config .OUTPUT_DIR /f"run_{run_id :03d}_1_timeseries_speaker.csv"
        path_l =Config .OUTPUT_DIR /f"run_{run_id :03d}_1_timeseries_listener.csv"

        if path_s .exists ()and path_l .exists ():
            print (f"  [Skip] Run {run_id }: Timeseries files already exist.")
            return 

        print (f"  [START] Analysis 1: Timeseries for Run {run_id } (Parallel)")

        params =ANALYSIS_PARAMS ["analysis_1_timeseries"]
        rater =create_rater (params ["anchors"])
        if rater is None :
            print (f"    [ERROR] Failed to create Rater for Run {run_id }")
            return 

        name_map =load_agent_name_map ()

        # --- 1. Initialize memory with initial thinking ---
        # { agent_id: {"text": "...", "after_whom": "..."} }
        agent_tracker ={}

        init_id_col ='Agent_ID'if 'Agent_ID'in df_initial .columns else 'agent_id'
        init_thought_col ='Initial_Thought'if 'Initial_Thought'in df_initial .columns else 'thought'

        for _ ,row in df_initial .iterrows ():
            aid =row .get (init_id_col )
            th =str (row .get (init_thought_col ,"")).strip ()
            if aid is not None :
                agent_tracker [aid ]={"text":th ,"after_whom":"the start of the project"}

                # --- 2. Task Preparation (Carry out up to prompt creation sequentially) ---
        tasks =[]
        total_rows =len (df_history )

        for idx ,row in df_history .iterrows ():
            turn =row ['Turn']

            # (A) Speaker
            speaker_id =row ['Speaker_ID']
            s_before =str (row .get ('Speaker_Thought_Before',"")).strip ()
            s_after =str (row .get ('Speaker_Thought_After',"")).strip ()

            if s_before and s_after :
                my_speech =str (row .get ('Speaker_Speech',"")).strip ()
                partner_reply =str (row .get ('Target_Reply',"")).strip ()

                # Get Name
                s_name =name_map .get (speaker_id ,"Speaker")
                t_id =row .get ('Target_ID')
                t_name =name_map .get (t_id ,"Partner")if pd .notna (t_id )else "Partner"

                # Double check: Whether the text matches exactly
                cached =agent_tracker .get (speaker_id ,{})
                context_info =""
                if cached .get ("text")==s_before :
                    context_info =f" (formed immediately after interacting with {cached ['after_whom']})"

                prompt_s =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                prompt_s +=f"Previous Thought of {s_name }: {s_before }{context_info }\n"
                prompt_s +=f"Conversation Exchange:\n"
                prompt_s +=f"  {s_name }: \"{my_speech }\"\n"
                prompt_s +=f"  {t_name }: \"{partner_reply }\"\n"
                prompt_s +=f"Current Thought of {s_name }: {s_after }\n"
                # Replace {agent_name} in the question text
                question_s =params ['question'].replace ("{agent_name}",s_name )
                prompt_s +=f"Question: {question_s }"

                # Add to task list (API not called yet)
                tasks .append ({
                "prompt":prompt_s ,
                "meta":{
                "run_id":run_id ,"turn":turn ,"step_in_turn":idx ,
                "agent_id":speaker_id ,"agent_name":name_map .get (speaker_id ,""),
                "role":"Speaker",
                "prev_thought":s_before [:50 ]+"...",
                "curr_thought":s_after [:50 ]+"..."
                }
                })
                # Memory update
                agent_tracker [speaker_id ]={"text":s_after ,"after_whom":t_name }

                # (B) Listener (Target)
            target_id =row .get ('Target_ID')
            t_after =str (row .get ('Target_Thought_After',"")).strip ()

            if pd .notna (target_id )and t_after :
            # Retrieve from agent_tracker and perform a match check
                cached_l =agent_tracker .get (target_id ,{})
                t_before =str (row .get ('Target_Thought_Before',"")).strip ()# Explicitly obtain the previous thought
                if not t_before :t_before =cached_l .get ("text","")# If not, from the cache

                context_info_l =""
                if cached_l .get ("text")==t_before :
                    context_info_l =f" (formed immediately after interacting with {cached_l ['after_whom']})"

                if t_before :
                    partner_speech =str (row .get ('Speaker_Speech',"")).strip ()

                    # Get Name
                    l_name =name_map .get (target_id ,"Listener")
                    s_id =row ['Speaker_ID']
                    s_name =name_map .get (s_id ,"Speaker")

                    prompt_t =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                    prompt_t +=f"Previous Thought of {l_name }: {t_before }{context_info_l }\n"
                    prompt_t +=f"Event (Speech by {s_name }): \"{partner_speech }\"\n"
                    prompt_t +=f"Current Thought of {l_name }: {t_after }\n"
                    # Replace {agent_name} in the question text
                    question_t =params ['question'].replace ("{agent_name}",l_name )
                    prompt_t +=f"Question: {question_t }"

                    tasks .append ({
                    "prompt":prompt_t ,
                    "meta":{
                    "run_id":run_id ,"turn":turn ,"step_in_turn":idx ,
                    "agent_id":target_id ,"agent_name":name_map .get (target_id ,""),
                    "role":"Listener",
                    "prev_thought":t_before [:50 ]+"...",
                    "curr_thought":t_after [:50 ]+"..."
                    }
                    })

                    # Memory update: The other party is Speaker(s_name)
                agent_tracker [target_id ]={"text":t_after ,"after_whom":s_name }

                # --- 3. Parallel Execution (Worker) ---
        print (f"    [Run {run_id }] Executing {len (tasks )} tasks with {Config .MAX_WORKERS } workers...")

        # Internal function that processes one item
        def process_item (item ):
            raw =call_gemini (item ['prompt'])
            res_reason ,res_answer =parse_response (raw )
            res_score ,res_probs =calculate_ssr_metrics (rater ,res_answer )

            # Result Integration
            out =item ['meta'].copy ()
            out .update ({
            "llm_reason":res_reason ,"llm_answer":res_answer ,
            "ssr_score":res_score ,"probs":res_probs ,"prompt":item ['prompt']
            })
            return out 

        results =[]
        with ThreadPoolExecutor (max_workers =Config .MAX_WORKERS )as executor :
            future_map ={executor .submit (process_item ,t ):t for t in tasks }
            for i ,future in enumerate (as_completed (future_map )):
                try :
                    results .append (future .result ())
                    if (i +1 )%50 ==0 :
                        print (f"      Progress: {i +1 }/{len (tasks )} done...")
                except Exception as e :
                    print (f"      [Error in worker] {e }")

                    # --- 4. Save Results (Split & Sort) ---
        if results :
        # The order is mixed up, so sort it and return it.
            results .sort (key =lambda x :(x ['turn'],x ['step_in_turn']))

            df_res =pd .DataFrame (results )

            # Save Speaker
            df_s =df_res [df_res ['role']=='Speaker']
            if not df_s .empty :
                df_s .to_csv (path_s ,index =False ,encoding ='utf-8-sig')
                print (f"  [SAVED] {path_s .name }")

                # Save Listener
            df_l =df_res [df_res ['role']=='Listener']
            if not df_l .empty :
                df_l .to_csv (path_l ,index =False ,encoding ='utf-8-sig')
                print (f"  [SAVED] {path_l .name }")
        else :
            print (f"  [WARNING] No results for Run {run_id }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Timeseries] Run {run_id }")
        print (traceback .format_exc ())
        raise 