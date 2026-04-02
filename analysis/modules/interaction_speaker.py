import pandas as pd 
import traceback 
from concurrent .futures import ThreadPoolExecutor ,as_completed 
from analysis .config import Config ,ANALYSIS_PARAMS 
from analysis .utils import call_gemini ,parse_response ,calculate_ssr_metrics ,create_rater ,load_agent_name_map 

def run_interaction_speaker_analysis (run_id ,df_history ,df_initial ):
    """
    Analysis 4 - Speaker: Speaker Interaction (Interaction Speaker)
    Analyze how one's own thoughts (Speaker_Thought_After) are updated in response to the reply from the other person (Target_Reply).
    """
    try :
    # Attach _speaker to the file name as instructed
        output_path =Config .OUTPUT_DIR /f"run_{run_id :03d}_4_interaction_speaker.csv"
        if output_path .exists ():
            print (f"  [Skip] Run {run_id }: {output_path .name } already exists.")
            return 

        print (f"  [START] Analysis 4-Speaker: Interaction for Run {run_id } (Parallel)")

        params =ANALYSIS_PARAMS ["analysis_4_interaction_speaker"]
        rater =create_rater (params ["anchors"])
        if rater is None :
            return 

        name_map =load_agent_name_map ()

        tasks =[]
        for idx ,row in df_history .iterrows ():
            turn =row ['Turn']

            # Speaker's information
            speaker_id =row ['Speaker_ID']
            s_name =name_map .get (speaker_id ,f"Agent {speaker_id }")
            s_before =str (row .get ('Speaker_Thought_Before',"")).strip ()
            s_after =str (row .get ('Speaker_Thought_After',"")).strip ()

            # Response from the other party (Listener)
            partner_reply =str (row .get ('Target_Reply',"")).strip ()
            partner_id =row .get ('Target_ID')
            p_name =name_map .get (partner_id ,"Partner")

            # Analysis only when the necessary data (before thinking, response, after thinking) is complete
            if s_before and partner_reply and s_after :
                prompt =f"{Config .SITUATION_CONTEXT }\n{Config .OUTPUT_RULE }\n"
                prompt +=f"Original Thought of Speaker {s_name }: \"{s_before }\"\n"
                prompt +=f"Feedback Reply from {p_name }: \"{partner_reply }\"\n"
                prompt +=f"Updated Thought of Speaker {s_name }: \"{s_after }\"\n"

                # Replace {speaker_name} in the question text
                question =params ['question'].replace ("{speaker_name}",s_name )
                prompt +=f"Question: {question }"

                tasks .append ({
                "prompt":prompt ,
                "meta":{
                "run_id":run_id ,"turn":turn ,"step_in_turn":idx ,
                "speaker_id":speaker_id ,"speaker_name":s_name ,
                "partner_id":partner_id ,"partner_name":p_name ,
                "speaker_prev_thought":s_before [:50 ]+"...",
                "partner_reply":partner_reply [:50 ]+"...",
                "speaker_curr_thought":s_after [:50 ]+"..."
                }
                })

                # Parallel execution
        print (f"    [Run {run_id }] Executing {len (tasks )} tasks for Speaker analysis...")

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
                except Exception as e :
                    print (f"      [Error in worker] {e }")

                    # Save
        if results :
            results .sort (key =lambda x :(x ['turn'],x ['step_in_turn']))
            pd .DataFrame (results ).to_csv (output_path ,index =False ,encoding ='utf-8-sig')
            print (f"  [SAVED] {output_path .name }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Interaction Speaker] Run {run_id }")
        print (traceback .format_exc ())
        raise 