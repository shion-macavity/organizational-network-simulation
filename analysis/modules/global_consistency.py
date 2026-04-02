# analysis/modules/global_consistency.py
import pandas as pd 
import itertools 
import traceback 
from concurrent .futures import ThreadPoolExecutor ,as_completed # Addition
from analysis .config import Config ,ANALYSIS_PARAMS 
from analysis .utils import call_gemini ,parse_response ,calculate_ssr_metrics ,create_rater ,load_agent_name_map 
def run_consistency_analysis (run_id ,df_history ,df_initial ):
    """
    Analysis 2: Global Consistency
    Parallelization-Supported Version (Speedup of Heavy Processing)
    """
    try :
        output_path =Config .OUTPUT_DIR /f"run_{run_id :03d}_2_consistency.csv"
        if output_path .exists ():
            print (f"  [Skip] Run {run_id }: {output_path .name } already exists.")
            return 

        print (f"  [START] Analysis 2: Global Consistency (Heavy) for Run {run_id } (Parallel)")

        params =ANALYSIS_PARAMS ["analysis_2_consistency"]
        rater =create_rater (params ["anchors"])

        name_map =load_agent_name_map ()

        # Retrieve Agent List
        init_id_col ='Agent_ID'if 'Agent_ID'in df_initial .columns else 'agent_id'
        all_agents =sorted (df_initial [init_id_col ].unique ())

        # Initialization of the thought cache (unification)
        agent_tracker ={}# { id: {"text": "...", "after_whom": "..."} }
        init_thought_col ='Initial_Thought'if 'Initial_Thought'in df_initial .columns else 'thought'
        for _ ,row in df_initial .iterrows ():
            aid =row [init_id_col ]
            th =str (row .get (init_thought_col ,"")).strip ()
            if aid is not None :
                agent_tracker [aid ]={"text":th ,"after_whom":"the start of the project"}

                # Create task list (make for all turns at once)
        tasks =[]
        turns =sorted (df_history ['Turn'].unique ())

        for turn in turns :
        # Update thoughts in the log for this turn
            df_turn =df_history [df_history ['Turn']==turn ]
            for _ ,row in df_turn .iterrows ():
                s_id =row ['Speaker_ID']
                t_id =row .get ('Target_ID')

                # Identify the name
                s_name =name_map .get (s_id ,"Speaker")
                t_name =name_map .get (t_id ,"Partner")if pd .notna (t_id )else "Partner"

                # Speaker update (the other party is Target)
                s_after =str (row .get ('Speaker_Thought_After',"")).strip ()
                if s_after :
                    agent_tracker [s_id ]={"text":s_after ,"after_whom":t_name }

                    # Target update (the other party is the Speaker)
                t_after =str (row .get ('Target_Thought_After',"")).strip ()
                if pd .notna (t_id )and t_after :
                    agent_tracker [t_id ]={"text":t_after ,"after_whom":s_name }

                    # All pair comparisons (14C2 = 91 pairs)
            pairs =list (itertools .combinations (all_agents ,2 ))

            for agent_a ,agent_b in pairs :
            # Get text from tracker
                th_a =agent_tracker .get (agent_a ,{}).get ("text","")
                th_b =agent_tracker .get (agent_b ,{}).get ("text","")

                if not th_a or not th_b :
                    continue 

                name_a =name_map .get (agent_a ,"Agent A")
                name_b =name_map .get (agent_b ,"Agent B")

                ctx_a =f" (formed immediately after talking to {agent_tracker [agent_a ]['after_whom']})"if agent_a in agent_tracker else ""
                ctx_b =f" (formed immediately after talking to {agent_tracker [agent_b ]['after_whom']})"if agent_b in agent_tracker else ""

                prompt =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                prompt +=f"Thought of Agent {name_a }: \"{th_a }\"{ctx_a }\n"
                prompt +=f"Thought of Agent {name_b }: \"{th_b }\"{ctx_b }\n"
                # Replace the placeholder in the question text
                question =params ['question'].replace ("{agent_a}",name_a ).replace ("{agent_b}",name_b )
                prompt +=f"Question: {question }"

                tasks .append ({
                "prompt":prompt ,
                "meta":{
                "run_id":run_id ,"turn":turn ,
                "agent_a":agent_a ,"agent_a_name":name_map .get (agent_a ,""),
                "agent_b":agent_b ,"agent_b_name":name_map .get (agent_b ,"")
                }
                })

                # Parallel execution
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
                    if (i +1 )%100 ==0 :
                        print (f"      Progress: {i +1 }/{len (tasks )} done...")
                except Exception as e :
                    print (f"      [Error in worker] {e }")

        if results :
            results .sort (key =lambda x :(x ['turn'],x ['agent_a'],x ['agent_b']))
            pd .DataFrame (results ).to_csv (output_path ,index =False ,encoding ='utf-8-sig')
            print (f"  [SAVED] {output_path .name }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Consistency] Run {run_id }")
        print (traceback .format_exc ())
        raise 