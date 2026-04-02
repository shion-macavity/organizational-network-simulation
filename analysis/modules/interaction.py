# analysis/modules/interaction.py
import pandas as pd 
import traceback 
from concurrent .futures import ThreadPoolExecutor ,as_completed # Addition
from analysis .config import Config ,ANALYSIS_PARAMS 
from analysis .utils import call_gemini ,parse_response ,calculate_ssr_metrics ,create_rater ,load_agent_name_map 

def run_interaction_analysis (run_id ,df_history ,df_initial ):
    """
    Analysis 4: Interaction
    Parallelization-compatible version
    """
    try :
        output_path =Config .OUTPUT_DIR /f"run_{run_id :03d}_4_interaction.csv"
        if output_path .exists ():
            print (f"  [Skip] Run {run_id }: {output_path .name } already exists.")
            return 

        print (f"  [START] Analysis 4: Interaction for Run {run_id } (Parallel)")

        params =ANALYSIS_PARAMS ["analysis_4_interaction"]
        rater =create_rater (params ["anchors"])
        if rater is None :
            return 

        name_map =load_agent_name_map ()

        # 1. Memory Initialization (Unification)
        agent_tracker ={}
        init_id_col ='Agent_ID'if 'Agent_ID'in df_initial .columns else 'agent_id'
        init_thought_col ='Initial_Thought'if 'Initial_Thought'in df_initial .columns else 'thought'

        for _ ,row in df_initial .iterrows ():
            aid =row .get (init_id_col )
            th =str (row .get (init_thought_col ,"")).strip ()
            if aid is not None :
            # The initial state is 'immediately after the project starts'.
                agent_tracker [aid ]={"text":th ,"after_whom":"the start of the project"}

                # 2. Create a task list (track memory in serial processing)
        tasks =[]
        for idx ,row in df_history .iterrows ():
            turn =row ['Turn']

            speaker_id =row ['Speaker_ID']
            s_name =name_map .get (speaker_id ,"Speaker")
            s_after =str (row .get ('Speaker_Thought_After',"")).strip ()

            target_id =row .get ('Target_ID')
            l_name =name_map .get (target_id ,"Listener")
            target_reply =row .get ('Target_Reply')
            t_after =str (row .get ('Target_Thought_After',"")).strip ()

            if pd .notna (target_id )and pd .notna (target_reply )and t_after :
            # Obtain the immediate prior thoughts of the Listener from the tracker
                cached_l =agent_tracker .get (target_id ,{})
                t_before =str (row .get ('Target_Thought_Before',"")).strip ()
                if not t_before :t_before =cached_l .get ("text","")

                # Exact match check
                ctx_l =""
                if cached_l .get ("text")==t_before :
                    ctx_l =f" (formed immediately after interacting with {cached_l ['after_whom']})"

                speaker_speech =str (row .get ('Speaker_Speech',"")).strip ()

                if t_before and speaker_speech :
                    prompt =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                    prompt +=f"Original Thought of Listener {l_name }: \"{t_before }\"{ctx_l }\n"
                    prompt +=f"Input Speech from {s_name }: \"{speaker_speech }\"\n"
                    prompt +=f"Updated Thought of Listener {l_name }: \"{t_after }\"\n"
                    # Replace {listener_name} in the question text
                    question =params ['question'].replace ("{listener_name}",l_name )
                    prompt +=f"Question: {question }"

                    tasks .append ({
                    "prompt":prompt ,
                    "meta":{
                    "run_id":run_id ,"turn":turn ,"step_in_turn":idx ,
                    "speaker_id":speaker_id ,"speaker_name":name_map .get (speaker_id ,""),
                    "listener_id":target_id ,"listener_name":name_map .get (target_id ,""),
                    "listener_prev_thought":t_before [:50 ]+"...",
                    "speaker_speech":speaker_speech [:50 ]+"...",
                    "listener_curr_thought":t_after [:50 ]+"..."
                    }
                    })

                    # Memory update (Listener is saved immediately after the conversation with the Speaker, Speaker is saved immediately after the conversation with the Listener)
                agent_tracker [target_id ]={"text":t_after ,"after_whom":s_name }
                if s_after :
                    agent_tracker [speaker_id ]={"text":s_after ,"after_whom":l_name }

                    # 3. Parallel Execution
        print (f"    [Run {run_id }] Executing {len (tasks )} tasks with {Config .MAX_WORKERS } workers...")

        def process_item (item ):
            raw =call_gemini (item ['prompt'])
            res_reason ,res_answer =parse_response (raw )
            res_score ,res_probs =calculate_ssr_metrics (rater ,res_answer )

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

                    # 4. Save
        if results :
            results .sort (key =lambda x :(x ['turn'],x ['step_in_turn']))
            pd .DataFrame (results ).to_csv (output_path ,index =False ,encoding ='utf-8-sig')
            print (f"  [SAVED] {output_path .name }")
        else :
            print (f"  [WARNING] No interactions analyzed for Run {run_id }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Interaction] Run {run_id }")
        print (traceback .format_exc ())
        raise 