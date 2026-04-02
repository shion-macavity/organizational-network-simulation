import pandas as pd 
from pathlib import Path 
import glob 

def check_ssr_outliers ():
# --- Settings ---
# 【Threshold】 (the boundary value used as the criterion for judgment) for finding data exceeding 5.0
    THRESHOLD =5.0 

    # Location of raw data (Analysis 2: Similarity)
    base_dir =Path (__file__ ).parent .parent .parent /"results"/"08_analysis_by_llm"/"Analysis 2_Raw Data"

    print (f"Starting exploration (reference value: {THRESHOLD }Super) ...")
    print (f"Target folder: {base_dir }\n")

    files =glob .glob (str (base_dir /"*.csv"))
    outliers_list =[]

    if not files :
        print ("The CSV file could not be found. Please check the path.")
        return 

    for f in files :
    # Load the CSV
        df =pd .read_csv (f ,encoding ='utf-8-sig')

        # Extract only the rows where ssr_score exceeds 5.0
        outliers =df [df ['ssr_score']>THRESHOLD ]

        if not outliers .empty :
            for _ ,row in outliers .iterrows ():
                outliers_list .append ({
                'File name':Path (f ).name ,
                'Run_ID':row ['run_id'],
                'Turn':row ['turn'],
                'Pair':f"{row ['agent_a']} & {row ['agent_b']}",
                'Actual score':row ['ssr_score']
                })

                # --- Display of Results ---
    if outliers_list :
        print (f"[Warning]{len (outliers_list )}The abnormal value in question has been found:")
        # The detected abnormal values will be displayed in a table.
        error_df =pd .DataFrame (outliers_list )
        print (error_df .to_string (index =False ))

        # For later reference, I will save the results in the miscellaneous tasks folder.
        output_path =Path (__file__ ).parent /"detected_outliers.csv"
        error_df .to_csv (output_path ,index =False ,encoding ='utf-8-sig')
        print (f"\n詳細を以下のファイルに保存しました：\n{output_path }")# Folder/File:  | Folder/File:  | Folder/File: \n詳細を以下のファイルに保存しました：\n
The details have been saved in the following file :

The details have been saved in the following file :

    else :
        print ("No data exceeding 5.0 was found.")
        print ("The reason it appeared to exceed 5 on the graph is likely due to 'overflow' caused by the specifications of the drawing software (Seaborn).")

if __name__ =="__main__":
    check_ssr_outliers ()