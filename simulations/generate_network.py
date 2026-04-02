# simulations/generate_fixed_patterns.py
"""
generate_fixed_patterns.py

For undergraduate thesis experiments: A script to generate and save four fixed network patterns.

Roles:
- Generate four fixed patterns (pattern_id: 1~4).
- Regardless of the number of simulation runs, only output these four “types.”
- All output file names and IDs are based on 'pattern_id'.

Generated patterns:
1. Fragmented (No connection)
2. Leader cooperation (Leader-Leader connection)
3. Field cooperation (Member-Member connection, m=1)
4. Tight coupling (Both connections)
"""

import networkx as nx 
import numpy as np 
import matplotlib .pyplot as plt 
import csv 
from pathlib import Path 

# --- Settings ---
OUTPUT_DIR =Path ("results/01_network")
OUTPUT_DIR .mkdir (parents =True ,exist_ok =True )

# Settings for fixed color
CMAP =plt .get_cmap ("tab20")
V_MIN ,V_MAX =0 ,19 

def create_prism_team (start_node_id ):
    """
    Create a 7-person team consisting of a triangular prism graph (6 people) + 1 leader  
    Leader node ID: start_node_id + 6  
    Member node ID: start_node_id + 0~5
    """
    G =nx .Graph ()
    members =[start_node_id +i for i in range (6 )]
    leader =start_node_id +6 

    # 1. Construction of a triangular prism
    # The upper triangle
    G .add_edge (members [0 ],members [1 ])
    G .add_edge (members [1 ],members [2 ])
    G .add_edge (members [2 ],members [0 ])
    # The triangle below
    G .add_edge (members [3 ],members [4 ])
    G .add_edge (members [4 ],members [5 ])
    G .add_edge (members [5 ],members [3 ])
    # Vertical connection
    G .add_edge (members [0 ],members [3 ])
    G .add_edge (members [1 ],members [4 ])
    G .add_edge (members [2 ],members [5 ])

    # 2. Leader connection (connect with everyone)
    for m in members :
        G .add_edge (leader ,m )

    return G ,leader ,members 

def get_fixed_pos ():
    """Fix node positions for drawing to make them easier to see"""
    pos ={}
    # Team A (left side)
    pos [0 ]=(-2 ,1 );pos [1 ]=(-1 ,0.5 );pos [2 ]=(-3 ,0.5 )
    pos [3 ]=(-2 ,-1 );pos [4 ]=(-1 ,-1.5 );pos [5 ]=(-3 ,-1.5 )
    pos [6 ]=(-2 ,0 )# Leader A (center)

    # Team B (right side) - symmetrical left and right
    offset_x =5 
    pos [7 ]=(offset_x +2 ,1 );pos [8 ]=(offset_x +1 ,0.5 );pos [9 ]=(offset_x +3 ,0.5 )
    pos [10 ]=(offset_x +2 ,-1 );pos [11 ]=(offset_x +1 ,-1.5 );pos [12 ]=(offset_x +3 ,-1.5 )
    pos [13 ]=(offset_x +2 ,0 )# Leader B (Center)

    return pos 

def save_graph_image (G ,pos ,pattern_id ,title_suffix ,filename ):
    """Save the graph as an image"""
    plt .figure (figsize =(10 ,6 ))

    # Color coding of nodes (assuming Team A = Com0, Team B = Com2)
    node_colors =[]
    for n in sorted (G .nodes ()):
        if n <=6 :node_colors .append (0 )# Team A color
        else :node_colors .append (2 )# Team B color

    nx .draw_networkx_edges (G ,pos ,alpha =0.4 ,edge_color ="gray")
    nx .draw_networkx_nodes (G ,pos ,node_size =300 ,cmap =CMAP ,node_color =node_colors ,vmin =V_MIN ,vmax =V_MAX ,edgecolors ="white")
    nx .draw_networkx_labels (G ,pos ,font_size =10 ,font_color ="black")

    # Indicator calculation (displayed in the title as a reference value)
    try :
        avg_path =nx .average_shortest_path_length (G )
    except :
        avg_path =0 # Cannot be calculated during division
    avg_clust =nx .average_clustering (G )

    plt .title (f"Pattern {pattern_id }: {title_suffix }\nL={avg_path :.2f} | C={avg_clust :.3f}")
    plt .axis ("off")
    plt .savefig (OUTPUT_DIR /filename ,bbox_inches ="tight")
    plt .close ()

def main ():
# A list that accumulates node information for all patterns
    all_nodes_rows =[]

    # Definition of 4 patterns
    patterns =[
    {"pattern_id":1 ,"leader_conn":False ,"member_conn":False ,"name":"Separated"},
    {"pattern_id":2 ,"leader_conn":True ,"member_conn":False ,"name":"Leader_Link_Only"},
    {"pattern_id":3 ,"leader_conn":False ,"member_conn":True ,"name":"Member_Link_Only"},
    {"pattern_id":4 ,"leader_conn":True ,"member_conn":True ,"name":"Dense_Link"},
    ]

    pos =get_fixed_pos ()

    print (f"Generating 4 fixed patterns to: {OUTPUT_DIR }")

    for p in patterns :
        pat_id =p ["pattern_id"]
        print (f"--- Generating Pattern {pat_id }: {p ['name']} ---")

        # 1. Create base
        G_A ,leader_A ,members_A =create_prism_team (0 )
        G_B ,leader_B ,members_B =create_prism_team (7 )
        G =nx .compose (G_A ,G_B )

        # 2. Add Connection
        if p ["leader_conn"]:
            G .add_edge (leader_A ,leader_B )
            print (f"  Added Leader Edge: {leader_A } - {leader_B }")

        if p ["member_conn"]:
        # Fixed pair: No. 0 of Team A and No. 7 of Team B
            m_a =members_A [0 ]# Node 0
            m_b =members_B [0 ]# Node 7
            G .add_edge (m_a ,m_b )
            print (f"  Added Member Edge: {m_a } - {m_b }")

            # 3. Save Image
        image_filename =f"pattern_{pat_id }_network.png"
        save_graph_image (G ,pos ,pat_id ,p ["name"],image_filename )

        # 4. Save edge list (change the file name to pattern_X_edges.csv)
        edge_filename =f"pattern_{pat_id }_edges.csv"
        nx .write_edgelist (G ,OUTPUT_DIR /edge_filename ,delimiter =",",data =False )

        # 5. Creating Node Information (for nodes_patterns.csv)
        deg_cent =nx .degree_centrality (G )
        bet_cent =nx .betweenness_centrality (G )

        for n in sorted (G .nodes ()):
            neighbors =sorted (G .neighbors (n ))
            neighbors_str =",".join (str (v )for v in neighbors )

            # Community Definition (Team A=0, Team B=1)
            comm_id =0 if n <=6 else 1 

            # Leader Judgment
            is_leader =1 if n in [leader_A ,leader_B ]else 0 

            row ={
            "pattern_id":pat_id ,# pattern_id instead of run_id
            "node_id":n ,
            "community_G1_formal":comm_id ,
            "community_G2_actual":comm_id ,
            "degree_centrality":deg_cent [n ],
            "betweenness_centrality":bet_cent [n ],
            "neighbors_G2":neighbors_str ,
            "is_leader_G1_formal":is_leader ,
            }
            all_nodes_rows .append (row )

            # 6. Saving the Integrated Node List (nodes_patterns.csv)
    header =[
    "pattern_id","node_id","community_G1_formal","community_G2_actual",
    "degree_centrality","betweenness_centrality","neighbors_G2","is_leader_G1_formal"
    ]

    csv_path =OUTPUT_DIR /"nodes_patterns.csv"
    with open (csv_path ,mode ="w",newline ="",encoding ="utf-8-sig")as f :
        writer =csv .DictWriter (f ,fieldnames =header )
        writer .writeheader ()
        writer .writerows (all_nodes_rows )

    print (f"\nCompleted! Saved all data to {OUTPUT_DIR }")

if __name__ =="__main__":
    main ()