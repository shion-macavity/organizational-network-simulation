import os 
import sys 
import pandas as pd 
from tqdm import tqdm 

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

def get_formal_pattern (run_id ):
    """Formal pattern (the remainder of run_id divided by 4, with 0 considered as 4)"""
    rem =run_id %4 
    return 4 if rem ==0 else rem 

def get_actual_pattern (w_6_13 ,w_0_7 ):
    """
    Definition of actual patterns:
    - If the nodes between 6-13 are 1 or more and the nodes between 0-7 are 0, then Pattern 2
    - If the nodes between 0-7 are 1 or more and the nodes between 6-13 are 0, then Pattern 3
    - If both are 1 or more, then Pattern 4
    - All others are Pattern 1
    """
    cond_6_13 =w_6_13 >=1.0 
    cond_0_7 =w_0_7 >=1.0 
    zero_6_13 =w_6_13 ==0.0 
    zero_0_7 =w_0_7 ==0.0 

    if cond_6_13 and zero_0_7 :
        return 2 
    elif cond_0_7 and zero_6_13 :
        return 3 
    elif cond_6_13 and cond_0_7 :
        return 4 
    else :
        return 1 

def main ():
    print (">>> Starting Actual Pattern Analysis...")

    # Output directory
    output_dir =os .path .join (parent_dir ,'16_実パターン分析')# Folder/File: 16_Actual Pattern Analysis | Folder/File: 16_Actual Pattern Analysis | Folder/File: 16_実パターン分析
    os .makedirs (output_dir ,exist_ok =True )
    output_path =os .path .join (output_dir ,'actual_patterns.csv')

    target_dir =os .path .join (conf .STRUCTURE_DIR ,"adjacency matrix","turn1to8")

    records =[]

    # conf.RUN_IDS is a list from 1 to 100
    for run_id in tqdm (conf .RUN_IDS ,desc ="Processing Matrix data"):
    # Zero-padding support for file names (such as run_001_matrix.csv)
        matrix_file =os .path .join (target_dir ,f"run_{run_id :03}_matrix.csv")

        formal_pattern =get_formal_pattern (run_id )

        w_6_13 =0.0 
        w_0_7 =0.0 

        if os .path .exists (matrix_file ):
        # Read the first row and first column as header and index with index_col=0
            df_mat =pd .read_csv (matrix_file ,index_col =0 ,encoding =conf .ENCODING )

            # CSV column names are often strings, so adjustments may be needed ('13' or 13)
            col_13 ='13'if '13'in df_mat .columns else 13 
            col_7 ='7'if '7'in df_mat .columns else 7 

            # Node 6 and Node 0 exist on the index side
            # try-except to prepare for the unlikely case that it does not exist
            try :
                w_6_13 =float (df_mat .loc [6 ,col_13 ])
            except KeyError :
                pass 

            try :
                w_0_7 =float (df_mat .loc [0 ,col_7 ])
            except KeyError :
                pass 

        actual_pattern =get_actual_pattern (w_6_13 ,w_0_7 )

        records .append ({
        'run_id':run_id ,
        'formal_pattern':formal_pattern ,
        'weight_6_13':w_6_13 ,
        'weight_0_7':w_0_7 ,
        'actual_pattern':actual_pattern 
        })

        # Convert to DataFrame
    df_out =pd .DataFrame (records )
    # Ensure user-specified column order
    df_out =df_out [['run_id','formal_pattern','weight_6_13','weight_0_7','actual_pattern']]

    df_out .to_csv (output_path ,index =False ,encoding =conf .ENCODING )
    print (f"Saved actual patterns data to: {output_path }")

if __name__ =="__main__":
    main ()
