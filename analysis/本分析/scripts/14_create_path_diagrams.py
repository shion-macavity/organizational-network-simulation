import os 
import sys 
import glob 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
import matplotlib .colors as mcolors 
import matplotlib .font_manager as fm 
import networkx as nx 
import warnings 

# Warning suppression
warnings .filterwarnings ("ignore")

# Make config_analysis importable by adding the path to the parent directory
current_dir =os .path .dirname (os .path .abspath (__file__ ))
parent_dir =os .path .dirname (current_dir )
sys .path .append (parent_dir )

import config_analysis as conf 

# =========================================================
# Settings & Constants
# =========================================================
INPUT_ROOT =os .path .join (conf .ANALYSIS_DIR ,"14_単回帰分析結果")# Folder/File: 14_Simple Regression Analysis Results | Folder/File: 14_Simple Regression Analysis Results | Folder/File: 14_単回帰分析結果
OUTPUT_ROOT =os .path .join (conf .ANALYSIS_DIR ,"15_パス図")# Folder/File: 15_Path Diagram | Folder/File: 15_Path Diagram | Folder/File: 15_パス図

# Filtering criteria
P_VALUE_STRICT =0.001 # Solid line
P_VALUE_WEAK =0.01 # Dashed line (do not draw beyond this)

# =========================================================
# Font Settings (Japanese Supported)
# =========================================================
def setup_japanese_font ():
    """Automatic setting of Japanese fonts"""
    try :
        import japanize_matplotlib 
        return 
    except ImportError :
        pass 

    candidate_fonts =[
    'MS Gothic','MS Mincho','Meiryo','Yu Gothic',
    'Hiragino Sans','AppleGothic','Noto Sans CJK JP','IPAGothic'
    ]
    found_font =None 
    for font_name in candidate_fonts :
        try :
            fm .findfont (font_name ,fallback_to_default =False )
            found_font =font_name 
            break 
        except :
            continue 

    if found_font :
        plt .rcParams ['font.family']=found_font 
    else :
        plt .rcParams ['font.family']='sans-serif'

        # =========================================================
        # Data Loading and Integration
        # =========================================================
def load_all_regression_results ():
    """14_単回帰分析結果 以下の全CSVを読み込んで結合する"""# Folder/File: 14_Simple Regression Analysis Results Read and combine all the following CSVs | Folder/File: 14_Simple Regression Analysis Results Read and combine all the following CSVs | Folder/File: 14_単回帰分析結果 以下の全CSVを読み込んで結合する
    print (">>> Loading Regression Results...")

    all_files =glob .glob (os .path .join (INPUT_ROOT ,"**","regression_summary.csv"),recursive =True )

    if not all_files :
        print (f"[ERROR] No regression_summary.csv found in {INPUT_ROOT }")
        sys .exit (1 )

    df_list =[]
    for f in all_files :
        try :
            df =pd .read_csv (f ,encoding =conf .ENCODING )
            df_list .append (df )
        except Exception as e :
            print (f"[WARNING] Failed to read {f }: {e }")

    if not df_list :
        return pd .DataFrame ()

    df_total =pd .concat (df_list ,ignore_index =True )
    print (f"[INFO] Loaded {len (df_total )} regression results.")
    return df_total 

    # =========================================================
    # Helper function for drawing
    # =========================================================
def get_edge_color (slope ,r_squared ):
    """  # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File:  | Folder/File:   # Folder/File: 
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    
    傾きの正負で赤/青を分岐。
    R^2の大きさで色の濃さ（彩度・明度）を決定する。
    """
    min_intensity =0.2 
    intensity =min_intensity +(1.0 -min_intensity )*r_squared 
    intensity =max (0.0 ,min (1.0 ,intensity ))

    if slope >0 :
    # Positive correlation: Red
        base_grey =np .array ([0.7 ,0.7 ,0.7 ])
        target_red =np .array ([1.0 ,0.0 ,0.0 ])
        color =base_grey +(target_red -base_grey )*(r_squared **0.5 )
    else :
    # Negative correlation: Blue
        base_grey =np .array ([0.7 ,0.7 ,0.7 ])
        target_blue =np .array ([0.0 ,0.0 ,1.0 ])
        color =base_grey +(target_blue -base_grey )*(r_squared **0.5 )

    return tuple (color )

def get_edge_width (slope ):
    """
    Determine the thickness according to the absolute value of the regression coefficient. Widen the range to make the differences easier to understand.
    """
    abs_beta =abs (slope )
    # 0.5 (thin) to 8.0 (extra thick)
    # Scale using log1p, but increase the coefficient to emphasize the differences
    width =0.5 +4.0 *np .log1p (abs_beta )
    return min (8.0 ,max (0.5 ,width ))

    # =========================================================
    # Modifications: Added argument fixed_pos, added return value pos
    # =========================================================
def build_and_draw_graph (df_target ,title ,filename_prefix ,p_threshold =P_VALUE_WEAK ,fixed_pos =None ):
    """
    Receive a DataFrame and draw and save a graph according to the conditions
    """
    # p-value filtering
    df_sig =df_target [df_target ['p_value']<p_threshold ].copy ()

    if df_sig .empty :
        print (f"[WARNING] No significant relationships found for {title }.")
        return None # Return None if the graph was not generated

        # Although using a DirectedGraph, it is practically treated as undirected (with arrows in both directions).
    G =nx .DiGraph ()

    # --- Deduplication and Direct Attribute Saving ---
    seen_pairs =set ()

    print (f"--- Building Graph: {title } (Source rows: {len (df_sig )}) ---")

    for _ ,row in df_sig .iterrows ():
        u =row ['X_Variable']
        v =row ['Y_Variable']

        # Sort variable names and convert to a tuple, check for duplicates (consider A->B and B->A as the same)
        pair_key =tuple (sorted ([u ,v ]))
        if pair_key in seen_pairs :
            continue 
        seen_pairs .add (pair_key )

        slope =row ['Slope(Beta)']
        r2 =row ['R_squared']
        pval =row ['p_value']

        # Calculate color, thickness, and style
        col =get_edge_color (slope ,r2 )
        wid =get_edge_width (slope )
        sty ='solid'if pval <P_VALUE_STRICT else 'dashed'

        # Save as edge attribute
        G .add_edge (u ,v ,color =col ,width =wid ,style =sty )

    isolates =list (nx .isolates (G ))
    G .remove_nodes_from (isolates )

    if len (G .nodes )==0 :
        return None 

        # --- Drawing Settings ---
    plt .figure (figsize =(20 ,18 ))# Increase canvas size

    # Modification points: Layout adjustment (use fixed placement if available)
    if fixed_pos is None :
        pos =nx .spring_layout (G ,k =2.5 ,iterations =100 ,seed =42 )

        # --- Layout Adjustment: Pull outer nodes toward the center ---
        # Center of gravity calculation
        coords =np .array (list (pos .values ()))
        center =np .mean (coords ,axis =0 )

        # Calculate maximum distance (scaling criterion)
        max_dist =0 
        for p in pos .values ():
            dist =np .linalg .norm (np .array (p )-center )
            if dist >max_dist :
                max_dist =dist 

        if max_dist >0 :
            for node ,p in pos .items ():
                p_arr =np .array (p )
                vec =p_arr -center 
                dist =np .linalg .norm (vec )

                # Penalty function: the farther the distance, the stronger the attraction
                # Example: 0.7 times (30% compression) at the point of maximum distance, almost 1.0 times near the center
                # scale = 1.0 - (dist / max_dist) * 0.3 
                # To make it look more natural, compress the whole slightly while strengthening the outer parts.
                scale =1.0 /(1.0 +(dist /max_dist )*1.5 )

                pos [node ]=center +vec *scale 
    else :
    # Even when using fixed_pos, only extract the nodes included in the current graph G.
    # This prevents isolated nodes from being rendered in Strict mode
        pos ={n :fixed_pos [n ]for n in G .nodes ()if n in fixed_pos }

        # Corrections: Node rendering (custom-made ellipse)
        # Place matplotlib.patches.Ellipse without using nx.draw_networkx_nodes
    from matplotlib .patches import Ellipse 
    ax =plt .gca ()

    # Parameters related to node size
    # width and height are the sizes in the data coordinate system
    # Adjustment is necessary according to the layout scale (typically in the range of -1.0 to 1.0)
    # Here, variables are defined to make adjustments easier
    ellipse_width =0.12 # Width (0.25 -> 0.12)
    ellipse_height =0.08 # Vertical dimension (0.15 -> 0.08)

    for node ,(x ,y )in pos .items ():
    # Draw an ellipse
        e =Ellipse (xy =(x ,y ),
        width =ellipse_width ,
        height =ellipse_height ,
        edgecolor ='black',
        facecolor ='white',
        linewidth =2.0 ,
        zorder =2 )# In front of the edge, behind the label
        ax .add_patch (e )

        # Note: When using an ellipse, the node_size setting in nx.draw_networkx_edges will not take effect.
        # There is a possibility that adjusting the position of the arrow (such as with connectionstyle) could become difficult.
        # This time, I will try drawing it as it is for now (it's fine as long as it doesn't overlap with the labels).

        # Label rendering
    nx .draw_networkx_labels (G ,pos ,
    font_family =plt .rcParams ['font.family'],
    font_size =25 ,# Increase the font size
    font_weight ='bold')

    # Edge drawing
    edges_data =list (G .edges (data =True ))

    import math 

    # Parameters for calculating the radius of an ellipse (a = horizontal radius, b = vertical radius)
    a =ellipse_width /2.0 
    b =ellipse_height /2.0 

    # Base node_size (corresponding to the size for the vertical radius b)
    # Based on the rule of thumb, when b=0.04, a node_size of around 8000 is just right.
    base_node_size =8000 
    base_radius =b 

    for u ,v ,data in edges_data :
    # Starting and ending coordinates
        x1 ,y1 =pos [u ]
        x2 ,y2 =pos [v ]

        # Calculate the angle of the edge (u -> v)
        # Note: The actual edge is curved with arc3, but the curvature (rad = 0.1) is slight.
        # Assume that sufficient accuracy can be achieved with a linear approximation. Strictly speaking, the tangent angle is required, but it is omitted because it would be complicated.
        dx =x2 -x1 
        dy =y2 -y1 
        theta =math .atan2 (dy ,dx )

        # Calculate the radius r(theta) on the circumference of an ellipse from the polar coordinate equation
        # r = ab / sqrt((b*cos(theta))^2 + (a*sin(theta))^2)
        term_cos =(b *math .cos (theta ))**2 
        term_sin =(a *math .sin (theta ))**2 
        r_theta =(a *b )/math .sqrt (term_cos +term_sin )

        # Calculate the node_size corresponding to radius r
        # Assume that node_size is proportional to the area (square of the radius)
        # size = base_size * (r / base_radius)^2
        # The coefficient of 1.0 can be adjusted to allow a little margin.
        dynamic_node_size =base_node_size *((r_theta /base_radius )**2 )

        nx .draw_networkx_edges (
        G ,pos ,
        edgelist =[(u ,v )],
        width =data ['width'],# Use saved attributes
        edge_color =[data ['color']],# Use saved attributes
        style =data ['style'],# Use saved attributes
        arrowstyle ='<|-|>',# Double arrow
        arrowsize =25 ,# Not mutation_scale, but arrowsize
        node_size =dynamic_node_size ,# Apply the calculated dynamic size
        connectionstyle ="arc3,rad=0.1"
        )

        # Legend
    legend_elements =[
    plt .Line2D ([0 ],[0 ],color ='red',lw =4 ,label ='Positive correlation (Slope > 0)'),
    plt .Line2D ([0 ],[0 ],color ='blue',lw =4 ,label ='Negative correlation (Slope < 0)'),
    plt .Line2D ([0 ],[0 ],color ='grey',lw =3 ,ls ='-',label ='p < 0.1% (solid line)'),
    plt .Line2D ([0 ],[0 ],color ='grey',lw =3 ,ls ='--',label ='p < 1% (dashed line)'),
    # Also increase the marker size
    plt .Line2D ([0 ],[0 ],marker ='o',color ='w',markeredgecolor ='black',markerfacecolor ='white',label ='Observed variable',markersize =20 ),
    ]
    plt .legend (handles =legend_elements ,loc ='upper left',fontsize =20 ,bbox_to_anchor =(1 ,1 ),handlelength =4.0 ,handleheight =3.0 )

    sub_title ="p < 1%"if p_threshold >=0.01 else "p < 0.1% (Strict)"
    plt .title (f"{title }\n[{sub_title }] (Square Nodes, Wide spacing, Edge thickness scaled)",fontsize =18 )
    plt .axis ('off')

    os .makedirs (OUTPUT_ROOT ,exist_ok =True )
    suffix ="_p01"if p_threshold >=0.01 else "_p001_strict"
    save_path =os .path .join (OUTPUT_ROOT ,f"{filename_prefix }{suffix }.png")
    plt .savefig (save_path ,dpi =150 ,bbox_inches ='tight')
    plt .close ()
    print (f"[OK] Saved: {save_path }")

    return pos # Return the layout

    # =========================================================
    # Main processing
    # =========================================================
def main ():
    setup_japanese_font ()

    df_all =load_all_regression_results ()
    if df_all .empty :
        return 

        # Correction points: Instead of a loop, create a reference (0.01) -> get the layout -> apply strictly (0.001) in that order

        # 1. Community Analysis
    def is_community_var (v ):
        return any (x in v for x in ['CCC','CC-Ave','CCH-','CCTF-'])and 'CDC'not in v 
    mask_com =df_all ['X_Variable'].apply (is_community_var )&df_all ['Y_Variable'].apply (is_community_var )

    # Step 1: Confirm placement at 0.01 (Weak)
    pos_com =build_and_draw_graph (df_all [mask_com ],"Community Analysis Path Diagram","01_community_path_diagram",p_threshold =P_VALUE_WEAK ,fixed_pos =None )
    # Step 2: Use the same arrangement with 0.001 (Strict)
    if pos_com is not None :
        build_and_draw_graph (df_all [mask_com ],"Community Analysis Path Diagram","01_community_path_diagram",p_threshold =P_VALUE_STRICT ,fixed_pos =pos_com )


        # 2. Department Analysis (CDC)
    def is_dept_cdc_var (v ):
        if 'adj-CDC'in v :return False 
        return ('CDC'in v )or any (x in v for x in ['ID-Ave','IDIH-','IDTF-'])
    mask_dept_cdc =df_all ['X_Variable'].apply (is_dept_cdc_var )&df_all ['Y_Variable'].apply (is_dept_cdc_var )

    # Step 1
    pos_cdc =build_and_draw_graph (df_all [mask_dept_cdc ],"Department Analysis (CDC Score)","02_department_cdc_path_diagram",p_threshold =P_VALUE_WEAK ,fixed_pos =None )
    # Step 2
    if pos_cdc is not None :
        build_and_draw_graph (df_all [mask_dept_cdc ],"Department Analysis (CDC Score)","02_department_cdc_path_diagram",p_threshold =P_VALUE_STRICT ,fixed_pos =pos_cdc )


        # 3. Department Analysis (adj-CDC)
    def is_dept_adj_var (v ):
        if v =='CDC Score':return False 
        return ('adj-CDC'in v )or any (x in v for x in ['ID-Ave','IDIH-','IDTF-'])
    mask_dept_adj =df_all ['X_Variable'].apply (is_dept_adj_var )&df_all ['Y_Variable'].apply (is_dept_adj_var )

    # Step 1
    pos_adj =build_and_draw_graph (df_all [mask_dept_adj ],"Department Analysis (adj-CDC Score)","03_department_adj_cdc_path_diagram",p_threshold =P_VALUE_WEAK ,fixed_pos =None )
    # Step 2
    if pos_adj is not None :
        build_and_draw_graph (df_all [mask_dept_adj ],"Department Analysis (adj-CDC Score)","03_department_adj_cdc_path_diagram",p_threshold =P_VALUE_STRICT ,fixed_pos =pos_adj )

    print ("\n>>> All Path Diagrams Generated Successfully.")

if __name__ =="__main__":
    main ()