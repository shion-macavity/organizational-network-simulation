import pandas as pd 
import os 
import sys 
from pathlib import Path 

# --- Settings: Folder Path ---
# This file is assumed to be in the analysis/ folder.
BASE_DIR =Path (__file__ ).resolve ().parent 
ROOT_DIR =BASE_DIR .parent 
CONV_DIR =ROOT_DIR /"results"/"02_conversation"

def prepare_agent_datasets ():
    """  # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File: 
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    
    会話ログと初期思考を統合し、エージェントごとの時系列分析用CSVを作成する。
    列名を「自分(My)」と「相手(Partner)」に統一することで、役割(Speaker/Listener)に関わらず
    思考や発言の変化を追いやすくする。
    """

    if not CONV_DIR .exists ():
        print (f"Error: Conversation log folder not found: {CONV_DIR }")
        return 

        # Explore the run_XXX folder
    run_folders =sorted ([f for f in CONV_DIR .iterdir ()if f .is_dir ()and f .name .startswith ("run_")])

    if not run_folders :
        print ("The run_XXX folder to be processed could not be found.")
        return 

    for run_path in run_folders :
        print (f"Creating dataset: {run_path .name }...")

        # Create destination folder
        output_dir =run_path /"agent_timeseries_csv"
        output_dir .mkdir (exist_ok =True )

        history_path =run_path /"conversation_history.csv"
        initial_path =run_path /"initial_thoughts.csv"

        if not history_path .exists ():
            print (f"  Skip: {history_path .name } There is none")
            continue 

            # CSV Import
        try :
            df_hist =pd .read_csv (history_path )
            df_init =pd .read_csv (initial_path )if initial_path .exists ()else pd .DataFrame ()
        except Exception as e :
            print (f"  Error: Failed to load CSV ({e })")
            continue 

            # Identification of all agent IDs
        agent_ids =set (df_hist ['Speaker_ID'].unique ())if 'Speaker_ID'in df_hist .columns else set ()
        if 'Target_ID'in df_hist .columns :
            agent_ids =agent_ids |set (df_hist ['Target_ID'].dropna ().unique ())

            # Obtain ID from initial thinking as well
        init_id_col =None 
        if not df_init .empty :
            for col in ['Agent_ID','agent_id','id']:
                if col in df_init .columns :
                    init_id_col =col 
                    break 
            if init_id_col :
                agent_ids =agent_ids |set (df_init [init_id_col ].unique ())

        agent_ids =sorted (list (agent_ids ))

        # --- Extract and format data for each agent ---
        for agent_id in agent_ids :
            records =[]

            # 1. Initial State (Turn 0)
            if not df_init .empty and init_id_col :
                thought_col ='Initial_Thought'if 'Initial_Thought'in df_init .columns else 'thought'
                target_col ='First_Target_ID'if 'First_Target_ID'in df_init .columns else 'target_id'

                row_init =df_init [df_init [init_id_col ]==agent_id ]
                if not row_init .empty :
                    initial_thought =row_init .iloc [0 ].get (thought_col ,"")
                    records .append ({
                    'Turn':0 ,
                    'Role':'Initial',
                    'Partner_ID':None ,
                    'My_Thought_Before':initial_thought ,
                    'My_Speech':None ,
                    'Partner_Speech':None ,
                    'My_Thought_After':initial_thought ,# Include the initial thoughts in After as well
                    'Next_Target_ID':row_init .iloc [0 ].get (target_col ,None )
                    })

                    # 2. Conversation History (Turn 1~)
                    # Extract the lines that this agent was involved in
            if 'Speaker_ID'in df_hist .columns and 'Target_ID'in df_hist .columns :
                df_agent =df_hist [(df_hist ['Speaker_ID']==agent_id )|(df_hist ['Target_ID']==agent_id )].copy ()

                for _ ,row in df_agent .iterrows ():
                    is_speaker =(row ['Speaker_ID']==agent_id )

                    if is_speaker :
                    # When I am the speaker
                    # Sequence: Thought_Before -> Speech -> Partner_Reply -> Thought_After
                        records .append ({
                        'Turn':row ['Turn'],
                        'Role':'Speaker',
                        'Partner_ID':row ['Target_ID'],
                        'My_Thought_Before':row .get ('Speaker_Thought_Before'),
                        'My_Speech':row .get ('Speaker_Speech'),
                        'Partner_Speech':row .get ('Target_Reply'),# The other person's response
                        'My_Thought_After':row .get ('Speaker_Thought_After'),
                        'Next_Target_ID':row .get ('Speaker_Next_ID')
                        })
                    else :
                    # When I am the listener
                    # Order: (Prev_Thought) -> Partner_Speech -> My_Reply -> Thought_After
                        records .append ({
                        'Turn':row ['Turn'],
                        'Role':'Listener',
                        'Partner_ID':row ['Speaker_ID'],
                        'My_Thought_Before':None ,# In the case of the listener, the immediate previous thought is not in the log (it is necessary to refer to the After of the previous turn).
                        'My_Speech':row .get ('Target_Reply'),# My response
                        'Partner_Speech':row .get ('Speaker_Speech'),# The other person's remark
                        'My_Thought_After':row .get ('Target_Thought_After'),
                        'Next_Target_ID':row .get ('Target_Next_ID')
                        })

                        # Convert to DataFrame and Save
            if records :
                df_out =pd .DataFrame (records )

                # Sort by turn order
                if 'Turn'in df_out .columns :
                    df_out =df_out .sort_values (by =['Turn']).reset_index (drop =True )

                    # [Important] Supplement My_Thought_Before during Listener
                    # Copy the My_Thought_After from the previous line
                df_out ['My_Thought_Before']=df_out ['My_Thought_Before'].fillna (df_out ['My_Thought_After'].shift (1 ))

                # In cases where there is no Turn 0 and it suddenly starts with Turn 1 as the Listener, it may remain NaN, so fill it with an empty string.
                df_out ['My_Thought_Before']=df_out ['My_Thought_Before'].fillna ("")

                # Organize the column order to make it easier to see
                cols =['Turn','Role','Partner_ID','My_Thought_Before','Partner_Speech','My_Speech','My_Thought_After','Next_Target_ID']
                # Select only the existing columns
                cols =[c for c in cols if c in df_out .columns ]
                df_out =df_out [cols ]

                save_path =output_dir /f"agent_{agent_id :02d}.csv"
                df_out .to_csv (save_path ,index =False ,encoding ='utf-8-sig')

        print (f"  -> {output_dir .name } to {len (agent_ids )} I have created a CSV for the number of people.")

    print ("\n全処理完了。")# Folder/File: | Folder/File: | Folder/File: \n全処理完了。
All processing completed .
All processing completed .

if __name__ =="__main__":
    prepare_agent_datasets ()