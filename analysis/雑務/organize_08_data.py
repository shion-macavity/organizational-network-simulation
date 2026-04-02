import os 
import shutil 
from pathlib import Path 

def organize_files ():
# --- Settings ---
# Specify the analysis number you want to organize.
# If you want to include analysis 3, please rewrite it as [1, 2, 3, 4].
    TARGET_NUMBERS =[1 ,2 ,3 ,4 ]

    # Location of the data (relative location from this script)
    # Correction: Changed to look at the results located outside the analysis folder
    base_dir =Path (__file__ ).parent .parent .parent /"results"/"08_analysis_by_llm"

    print (f"Starting the organization: {base_dir }")

    # First, check if the folder exists.
    if not base_dir .exists ():
        print (f"Error: Folder not found. Please check the location: {base_dir }")
        return 

        # I will check the files in the folder one by one.
    for file_path in base_dir .iterdir ():
    # Only files (.csv) are targeted, and folders are ignored.
        if file_path .is_file ()and file_path .suffix ==".csv":
            file_name =file_path .name 

            # Check whether the file name starts with 'run_XXX_Y_'
            parts =file_name .split ("_")
            if len (parts )>=3 and parts [0 ]=="run":
                try :
                    analysis_num =int (parts [2 ])# Acquire analysis number

                    # Process only if it matches the specified numbers (1, 2, 4)
                    if analysis_num in TARGET_NUMBERS :
                    # Create the name of the destination folder
                        new_folder_name =f"分析{analysis_num }Raw data"# Folder/File: Analysis | Folder/File: Analysis | Folder/File: 分析
                        dest_folder =base_dir /new_folder_name 

                        # Create the folder if it does not exist
                        if not dest_folder .exists ():
                            dest_folder .mkdir (parents =True )
                            print (f"Folder created: {new_folder_name }")

                            # Decide on the name (attach _listener only during Analysis 4)
                        dest_file_name =file_name 
                        if analysis_num ==4 and file_name .endswith ("_4_interaction.csv"):
                            dest_file_name =file_name .replace ("_4_interaction.csv","_4_interaction_listener.csv")

                            # Move the file (it will be renamed at the same time if the name changes)
                        shutil .move (str (file_path ),str (dest_folder /dest_file_name ))
                        print (f"Move complete: {file_name } -> {new_folder_name }/{dest_file_name }")

                except ValueError :
                    continue 

                    # --- [Addition] Rename the old files that are already in 'Analysis4_RawData' ---
    folder_4 =base_dir /"Analysis 4_Raw Data"
    if folder_4 .exists ():
        for f in folder_4 .iterdir ():
            if f .is_file ()and f .name .endswith ("_4_interaction.csv"):
                new_name =f .name .replace ("_4_interaction.csv","_4_interaction_listener.csv")
                f .rename (f .parent /new_name )
                print (f"Renamed the existing file: {f .name } -> {new_name }")

    print ("\nすべての整理が完了しました。")# Folder/File:  | Folder/File:  | Folder/File: \nすべての整理が完了しました。
All organization has been completed .
All organization has been completed .

if __name__ =="__main__":
    organize_files ()