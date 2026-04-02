import os 
import sys 
from pathlib import Path 
from dotenv import load_dotenv 
from openai import OpenAI 

# Path setting: go back from analysis/雑務/ to the root (where the .env is located)
# Structure: root/analysis/misc_tasks/debug_api_structure.py, so .parent.parent gives root
current_dir =Path (__file__ ).resolve ().parent 
root_dir =current_dir .parent .parent 
sys .path .append (str (root_dir ))

# Load .env
env_path =root_dir /".env"
load_dotenv (env_path )

api_key =os .getenv ("OPENAI_API_KEY")
if not api_key :
    print (f"[ERROR] OPENAI_API_KEY not found. Checked: {env_path }")
    sys .exit (1 )

client =OpenAI (api_key =api_key )

# The expected parameters specified in the config
MODEL ="gpt-5-mini"# Match the config
PROMPT ="Say 'Test Successful' and nothing else."

print (f"--- [DEBUG] Testing Responses API with {MODEL } ---")

try :
# Request with settings similar to the user's Config
# Using the Responses API (client.responses.create)
    response =client .responses .create (
    model =MODEL ,
    input =PROMPT ,
    reasoning ={"effort":"low"},
    text ={"verbosity":"low"},
    max_output_tokens =100 
    )

    print ("\n=== 1. Raw Response Object (print) ===")
    print (response )

    print ("\n=== 2. Type of Response ===")
    print (type (response ))

    print ("\n=== 3. Attributes (dir) ===")
    # Display the list of available attributes
    print ([d for d in dir (response )if not d .startswith ('_')])

    # Checking whether it is iterable (such as a list)
    try :
        iterator =iter (response )
        print ("\n=== 4. Iterating over response items ===")
        for i ,item in enumerate (iterator ):
            print (f"-- Item {i } --")
            print (f"  Type: {type (item )}")
            print (f"  Raw: {item }")
            print (f"  Attrs: {[d for d in dir (item )if not d .startswith ('_')]}")

            # Search for and display attributes that are likely to contain text
            if hasattr (item ,'content'):
                print (f"  -> item.content: {item .content }")
            if hasattr (item ,'output'):
                print (f"  -> item.output: {item .output }")
            if hasattr (item ,'message'):
                print (f"  -> item.message: {item .message }")
            if hasattr (item ,'type'):
                print (f"  -> item.type: {item .type }")

    except TypeError :
        print ("\n=== 4. Response is not iterable ===")
        # If it is not iterable, check the immediate attributes
        if hasattr (response ,'output'):
             print (f"-> response.output: {response .output }")
        if hasattr (response ,'content'):
             print (f"-> response.content: {response .content }")

except Exception as e :
    print ("\n[ERROR] Request Failed")
    print (e )
    import traceback 
    traceback .print_exc ()

print ("\n--- [DEBUG] Finished ---")