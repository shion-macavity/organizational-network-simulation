# analysis/modules/community.py
import pandas as pd 
import traceback 
from analysis .config import Config 

def run_community_analysis (run_id ,df_history =None ,df_initial =None ):
    """  # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File: 
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    
    分析5: コミュニティ分析 (Community)
    Analysis 2の結果とコミュニティ正解データを用いて、
    コミュニティ内/外の思考一致度の差を集計する。
    """
    try :
        input_path =Config .OUTPUT_DIR /f"run_{run_id :03d}_2_consistency.csv"
        # Path to the community definition file (match the user-specified location)
        comm_path =Config .BASE_DIR /"results"/"07_psychological_safety_analysis"/"仮分析"/"グラフ分析"/"Summary"/"community_assignments_final_1to8.csv"# Folder/File: Graph Analysis | Folder/File: Graph Analysis | Folder/File: グラフ分析
        output_path =Config .OUTPUT_DIR /f"run_{run_id :03d}_5_community.csv"

        if output_path .exists ():
            print (f"  [Skip] Run {run_id }: {output_path .name } already exists.")
            return 

        if not input_path .exists ():
            print (f"  [Skip] Run {run_id }: Analysis 2 result not found. Cannot run Analysis 5.")
            return 

        if not comm_path .exists ():
            print (f"  [Error] Community file not found: {comm_path }")
            return 

        print (f"  [START] Analysis 5: Community Analysis for Run {run_id }")

        df_scores =pd .read_csv (input_path )
        df_comm =pd .read_csv (comm_path )

        # This Run's community definition
        df_comm_run =df_comm [df_comm ['run_id']==run_id ].copy ()
        if df_comm_run .empty :
            print (f"  [Error] No community definitions found for Run {run_id }")
            return 

            # Agent ID -> Community ID Map
            # Column name agent_id or Agent_ID support
        aid_col ='agent_id'if 'agent_id'in df_comm_run .columns else 'Agent_ID'
        cid_col ='community_id'if 'community_id'in df_comm_run .columns else 'Community_ID'
        agent_to_comm =dict (zip (df_comm_run [aid_col ],df_comm_run [cid_col ]))

        results =[]

        # Totaled per turn
        for turn ,group in df_scores .groupby ('turn'):
            within_scores =[]
            between_scores =[]

            for _ ,row in group .iterrows ():
                a =row ['agent_a']
                b =row ['agent_b']
                score =row ['ssr_score']

                if pd .isna (score ):continue 

                ca =agent_to_comm .get (a )
                cb =agent_to_comm .get (b )

                if ca is not None and cb is not None :
                    if ca ==cb :
                        within_scores .append (score )
                    else :
                        between_scores .append (score )

            avg_within =sum (within_scores )/len (within_scores )if within_scores else None 
            avg_between =sum (between_scores )/len (between_scores )if between_scores else None 

            results .append ({
            "run_id":run_id ,"turn":turn ,
            "avg_within":avg_within ,
            "avg_between":avg_between ,
            "count_within":len (within_scores ),
            "count_between":len (between_scores )
            })

        if results :
            pd .DataFrame (results ).to_csv (output_path ,index =False )
            print (f"  [SAVED] {output_path .name }")

    except Exception :
        print (f"\n[CRITICAL ERROR in Community] Run {run_id }")
        print (traceback .format_exc ())
        raise 