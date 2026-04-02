# analysis/modules/psych_safety.py
import pandas as pd 
import traceback 
from concurrent .futures import ThreadPoolExecutor ,as_completed # Addition
from analysis .config import Config ,ANALYSIS_PARAMS 
from analysis .utils import call_gemini ,parse_response ,calculate_ssr_metrics ,create_rater ,load_agent_name_map 
def run_psych_safety_analysis (run_id ,df_history ,df_initial ):
    """
    Analysis 3: Psychological Safety
    Parallelized Version
    """
    try :
        path_s =Config .OUTPUT_DIR /f"run_{run_id :03d}_3_disclosure_speaker.csv"
        path_l =Config .OUTPUT_DIR /f"run_{run_id :03d}_3_disclosure_listener.csv"

        if path_s .exists ()and path_l .exists ():
            print (f"  [Skip] Run {run_id }: Psych Safety files already exist.")
            return 

        print (f"  [START] Analysis 3: Psych Safety for Run {run_id } (Parallel)")

        params =ANALYSIS_PARAMS ["analysis_3_disclosure"]
        rater =create_rater (params ["anchors"])

        name_map =load_agent_name_map ()

        # 1. List the data to be processed
        agent_tracker ={}
        init_id_col ='Agent_ID'if 'Agent_ID'in df_initial .columns else 'agent_id'
        init_thought_col ='Initial_Thought'if 'Initial_Thought'in df_initial .columns else 'thought'
        for _ ,row in df_initial .iterrows ():
            aid =row .get (init_id_col )
            th =str (row .get (init_thought_col ,"")).strip ()
            if aid is not None :
                agent_tracker [aid ]={"text":th ,"after_whom":"the start of the project"}

        tasks =[]
        for idx ,row in df_history .iterrows ():
            s_id =row ['Speaker_ID']
            t_id =row .get ('Target_ID')
            s_name =name_map .get (s_id ,"Speaker")
            t_name =name_map .get (t_id ,"Partner")if pd .notna (t_id )else "Partner"

            # (A) Speaker
            s_thought =str (row .get ('Speaker_Thought_Before',"")).strip ()
            s_speech =str (row .get ('Speaker_Speech',"")).strip ()

            if s_thought and s_speech :
                cached_s =agent_tracker .get (s_id ,{})
                ctx_s =f" (formed immediately after interacting with {cached_s ['after_whom']})"if cached_s .get ("text")==s_thought else ""

                tasks .append ({
                "idx":idx ,
                "turn":row ['Turn'],
                "agent_id":s_id ,
                "thought":f"{s_thought }{ctx_s }",
                "speech":s_speech ,
                "name":s_name ,
                "role":"Speaker"
                })

                # (B) Listener (Target)
            t_reply =row .get ('Target_Reply')
            t_thought =str (row .get ('Target_Thought_After',"")).strip ()# Thoughts immediately after hearing

            if pd .notna (t_id )and pd .notna (t_reply )and t_thought and str (t_reply ).strip ():
            # For the listener, this thought is immediately after talking with Speaker (s_name)
                ctx_l =f" (formed immediately after interacting with {s_name })"

                tasks .append ({
                "idx":idx ,
                "turn":row ['Turn'],
                "agent_id":t_id ,
                "thought":f"{t_thought }{ctx_l }",
                "speech":str (t_reply ).strip (),
                "name":t_name ,
                "role":"Listener"
                })

                # --- Tracker Update (For Comparison in the Next Turn) ---
            s_after =str (row .get ('Speaker_Thought_After',"")).strip ()
            if s_after :
                agent_tracker [s_id ]={"text":s_after ,"after_whom":t_name }
            if pd .notna (t_id )and t_thought :
                agent_tracker [t_id ]={"text":t_thought ,"after_whom":s_name }

        print (f"    [Run {run_id }] Executing {len (tasks )} tasks with {Config .MAX_WORKERS } workers...")

        # 2. Function for parallel execution (internally performs Phase 1 -> Phase 2 sequentially)
        def process_item (item ):
            agent_name =item .get ('name','The Agent')

            # --- Phase 1: Risk Awareness ---
            check_prompt =f"{Config .SITUATION_CONTEXT }\n{Config .Risk_Check_Rule }\n"
            check_prompt +=f"Private Thought of {agent_name }: {item ['thought']}\n"
            # Replace {agent_name} in the question text
            q_phase1 =params ['risk_check_question'].replace ("{agent_name}",agent_name )
            check_prompt +=f"Question: {q_phase1 }"

            check_res_raw =call_gemini (check_prompt )
            risk_reason ,risk_answer =parse_response (check_res_raw )

            has_risk ="Yes"in risk_answer or "yes"in risk_answer or "Yes"in check_res_raw 

            ssr_score =None 
            ssr_probs =[]
            disclosure_reason =""
            disclosure_answer =""
            prompt_phase2 =""

            # --- Phase 2: Disclosure Evaluation ---
            if has_risk and rater :
            # Replace both {detected_risk} and {agent_name}
                formatted_question =params ['disclosure_question'].replace ("{detected_risk}",risk_reason ).replace ("{agent_name}",agent_name )

                prompt_phase2 =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                prompt_phase2 +=f"Context: {agent_name } has a private thought: \"{item ['thought']}\".\n"
                prompt_phase2 +=f"PREVIOUS DETECTION: You (the expert observer) have already identified that this thought contains the following negative element: \"{risk_reason }\".\n"
                prompt_phase2 +=f"Actual Speech by {agent_name }: \"{item ['speech']}\"\n"
                prompt_phase2 +=f"Question: {formatted_question }"

                res_phase2 =call_gemini (prompt_phase2 )
                disclosure_reason ,disclosure_answer =parse_response (res_phase2 )
                ssr_score ,ssr_probs =calculate_ssr_metrics (rater ,disclosure_answer )

            return {
            "run_id":run_id ,"turn":item ['turn'],"agent_id":item ['agent_id'],
            "agent_name":item ['name'],
            "role":item .get ('role','Unknown'),# Inherit role
            "has_risk":1 if has_risk else 0 ,
            "risk_content_detected":risk_reason ,
            "risk_check_raw":check_res_raw ,
            "prompt_phase1":check_prompt ,# Add the Phase 1 prompt
            "ssr_score":ssr_score ,
            "probs":ssr_probs ,
            "disclosure_reason":disclosure_reason ,
            "disclosure_answer":disclosure_answer ,
            "prompt_phase2":prompt_phase2 ,
            "original_idx":item ['idx']# Sort Key
            }

            # 3. Parallel Execution
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

                    # 4. Save
        if results :
            results .sort (key =lambda x :x ['original_idx'])
            df_res =pd .DataFrame (results )

            # Save Speaker
            df_s =df_res [df_res ['role']=='Speaker'].drop (columns =['original_idx','role'])
            if not df_s .empty :
                df_s .to_csv (path_s ,index =False ,encoding ='utf-8-sig')
                print (f"  [SAVED] {path_s .name }")

                # Save Listener
            df_l =df_res [df_res ['role']=='Listener'].drop (columns =['original_idx','role'])
            if not df_l .empty :
                df_l .to_csv (path_l ,index =False ,encoding ='utf-8-sig')
                print (f"  [SAVED] {path_l .name }")
        else :
            print (f"  [WARNING] No results for Run {run_id }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Psych Safety] Run {run_id }")
        print (traceback .format_exc ())
        raise 