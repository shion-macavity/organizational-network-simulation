"""
run_full_simulation.py
Script to execute the entire simulation
"""
import networkx as nx 
import random # ← Add
import numpy as np # ← Add
from .config_simulation import config 

from .import conversation_simulation 
from .import questionnaire_ssr # For survey aggregation
from .import customer_fb_simulation # For generating consumer comments
from .import customer_fb_ssr # For aggregating consumer evaluations
#from . import aggregate_for_amos      # For final CSV creation


# ▼Addition: Fixing the seed value (done only once outside the loop)
# As a result, Run1, Run2, Run3... each move differently, but
# When this entire program is run again, exactly the same results will be reproduced.


def main ()->None :
    print ("=== Graduation Thesis Simulation Start ===")

    SEED_VALUE =42 
    random .seed (SEED_VALUE )
    np .random .seed (SEED_VALUE )

    # Loop the number of times Run is executed
    for run_id in range (1 ,config .n_runs +1 ):
        print (f"\n--- [Run {run_id }/{config .n_runs }] Start ---")

        # 1. Determination of the Pattern ID
        # Use the fixed setting in the config if it exists, otherwise go through them in order.
        if config .fixed_pattern_id is not None :
            pattern_id =config .fixed_pattern_id #For test run
        else :
            pattern_id =(run_id -1 )%4 +1 

            # 2. Conversation Simulation (Including Survey)
        conversation_simulation .run_simulation (run_id ,pattern_id )

        # 3. Aggregation of Employee Survey (SSR Scoring)
        # We calculate scores such as psychological safety.
        questionnaire_ssr .run_product_evaluation (run_id )

        # 4. Generation of Consumer Feedback
        # Consumers will comment on the leader's project proposal.
        customer_fb_simulation .run_customer_simulation (run_id )

        # 5. Aggregation of Consumer Evaluations (SSR Scoring)
        # Convert consumer comments into numerical values (such as 1 to 5)
        customer_fb_ssr .run_ssr_analysis (run_id )

        # 6. Integration of data for analysis (Creating CSV for AMOS)
        # We will compile the results of this run into a single line of data.
        #aggregate_for_amos.run_aggregation(run_id)

    print ("\n=== 全シミュレーション終了 ===")# Folder/File:  | Folder/File:  | Folder/File: \n=== 全シミュレーション終了 ===
===All simulations finished ===
===All simulations finished ===

if __name__ =="__main__":
    main ()