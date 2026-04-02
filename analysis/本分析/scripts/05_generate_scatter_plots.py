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
P_VALUE_WEAK =0.01 # dashed line

# =========================================================
# Font Settings (Japanese Supported)
# =========================================================
def setup_japanese_font ():
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
    print (">>> Loading Regression Results...")
    all_files =glob .glob (os .path .join (INPUT_ROOT ,"**","regression_summary.csv"),recursive =True )
    if not all_files :
        print (f"[ERROR] No regression_summary.csv found.")
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

    return pd .concat (df_list ,ignore_index =True )

    # =========================================================
    # Helper function for drawing
    # =========================================================
def get_edge_color (slope ,r_squared ):
    """Color determination: Shades with R2, Red and blue with Slope"""
    min_intensity =0.3 # So that it is not too thin
    intensity =min_intensity +(1.0 -min_intensity )*r_squared 
    intensity =max (0.0 ,min (1.0 ,intensity ))

    if slope >0 :
        base =np .array ([0.8 ,0.8 ,0.8 ])# Make the base a lighter gray
        target =np .array ([0.8 ,0.0 ,0.0 ])# Vivid red
        color =base +(target -base )*(r_squared **0.5 )
    else :
        base =np .array ([0.8 ,0.8 ,0.8 ])
        target =np .array ([0.0 ,0.0 ,0.8 ])# Vivid blue
        color =base +(target -base )*(r_squared **0.5 )

    return tuple (color )

def get_edge_width (slope ):
    """Thickness determination: Emphasis based on the absolute value of the coefficient"""
    abs_beta =abs (slope )
    width =1.0 +5.0 *np .log1p (abs_beta )
    return min (10.0 ,max (1.0 ,width ))

def build_and_draw_graph (df_target ,title ,filename_prefix ):
# p-value filter
    df_sig =df_target [df_target ['p_value']<P_VALUE_WEAK ].copy ()

    if df_sig .empty :
        print (f"[WARNING] No significant edges for {title }.")
        return 

    G =nx .DiGraph ()

    # List for storing edge information
    edge_meta =[]

    print (f"--- Building Graph: {title } ({len (df_sig )} edges) ---")

    for _ ,row in df_sig .iterrows ():
        u =row ['X_Variable']
        v =row ['Y_Variable']

        G .add_edge (u ,v )

        # Metadata for rendering
        slope =row ['Slope(Beta)']
        r2 =row ['R_squared']
        pval =row ['p_value']

        col =get_edge_color (slope ,r2 )
        wid =get_edge_width (slope )
        sty ='solid'if pval <P_VALUE_STRICT else 'dashed'

        edge_meta .append ({
        'u':u ,'v':v ,
        'color':col ,'width':wid ,'style':sty ,
        'slope':slope ,'r2':r2 
        })

    isolates =list (nx .isolates (G ))
    G .remove_nodes_from (isolates )

    if len (G .nodes )==0 :
        return 

        # --- Drawing Settings ---
    plt .figure (figsize =(24 ,20 ))# Further enlargement

    # Layout: The kamada_kawai_layout has good balance
    # Expand the whole with scale
    pos =nx .kamada_kawai_layout (G ,scale =2.0 )

    # -------------------------------------------------
    # 1. Draw edge (back) zorder=1
    # -------------------------------------------------
    # Overlap Avoidance Technique: Change the curvature (rad) according to the index
    rad_steps =[0.1 ,0.2 ,0.3 ,-0.1 ,-0.2 ,-0.3 ]# Curve variations

    for i ,meta in enumerate (edge_meta ):
        u ,v =meta ['u'],meta ['v']

        # Cyclic selection
        rad =rad_steps [i %len (rad_steps )]
        conn_style =f"arc3,rad={rad }"

        nx .draw_networkx_edges (
        G ,pos ,
        edgelist =[(u ,v )],
        width =meta ['width'],
        edge_color =[meta ['color']],
        style =meta ['style'],
        arrowsize =20 ,
        connectionstyle =conn_style ,
        alpha =0.7 ,# Make it slightly transparent so the overlaps are easier to see
        min_source_margin =20 ,# Distance from the node
        min_target_margin =20 ,
        )

        # -------------------------------------------------
        # 2. Drawing nodes (foreground) zorder=2
        # -------------------------------------------------
        # Square, white fill, black border
    nx .draw_networkx_nodes (
    G ,pos ,
    node_shape ='s',
    node_size =6000 ,# Make it large so that the characters do not overflow
    node_color ='white',
    edgecolors ='black',
    linewidths =2.0 ,
    margins =0.1 
    )

    # -------------------------------------------------
    # 3. Drawing the label (topmost) zorder=3
    # -------------------------------------------------
    nx .draw_networkx_labels (
    G ,pos ,
    font_family =plt .rcParams ['font.family'],
    font_size =11 ,
    font_weight ='bold'
    )

    # Legend
    legend_elements =[
    plt .Line2D ([0 ],[0 ],color ='red',lw =4 ,label ='Positive correlation (Slope > 0)'),
    plt .Line2D ([0 ],[0 ],color ='blue',lw =4 ,label ='Negative correlation (Slope < 0)'),
    plt .Line2D ([0 ],[0 ],color ='grey',lw =3 ,ls ='-',label ='p < 0.1% (solid line)'),
    plt .Line2D ([0 ],[0 ],color ='grey',lw =3 ,ls ='--',label ='p < 1% (dashed line)'),
    plt .Line2D ([0 ],[0 ],marker ='s',color ='w',markeredgecolor ='black',label ='Observed variable',markersize =15 ),
    ]
    plt .legend (handles =legend_elements ,loc ='upper left',fontsize =14 ,bbox_to_anchor =(1 ,1 ))

    plt .title (f"{title }\n(Curved edges to avoid overlap)",fontsize =20 )
    plt .axis ('off')

    os .makedirs (OUTPUT_ROOT ,exist_ok =True )
    save_path =os .path .join (OUTPUT_ROOT ,f"{filename_prefix }_v3.png")
    plt .savefig (save_path ,dpi =150 ,bbox_inches ='tight')
    plt .close ()
    print (f"[OK] Saved: {save_path }")


    # =========================================================
    # Main processing
    # =========================================================
def main ():
    setup_japanese_font ()

    df_all =load_all_regression_results ()
    if df_all .empty :return 

    # 1. Community Analysis
    def is_com (v ):return any (x in v for x in ['CCC','CC-Ave','CCH-','CCTF-'])and 'CDC'not in v 
    mask_com =df_all ['X_Variable'].apply (is_com )&df_all ['Y_Variable'].apply (is_com )
    build_and_draw_graph (df_all [mask_com ],"Community Analysis Path Diagram","01_community_path_diagram")

    # 2. Department Analysis (CDC)
    def is_dep_cdc (v ):
        if 'adj-CDC'in v :return False 
        return ('CDC'in v )or any (x in v for x in ['ID-Ave','IDIH-','IDTF-'])
    mask_cdc =df_all ['X_Variable'].apply (is_dep_cdc )&df_all ['Y_Variable'].apply (is_dep_cdc )
    build_and_draw_graph (df_all [mask_cdc ],"Department Analysis (CDC Score)","02_department_cdc_path_diagram")

    # 3. Department Analysis (adj-CDC)
    def is_dep_adj (v ):
        if v =='CDC Score':return False 
        return ('adj-CDC'in v )or any (x in v for x in ['ID-Ave','IDIH-','IDTF-'])
    mask_adj =df_all ['X_Variable'].apply (is_dep_adj )&df_all ['Y_Variable'].apply (is_dep_adj )
    build_and_draw_graph (df_all [mask_adj ],"Department Analysis (adj-CDC Score)","03_department_adj_cdc_path_diagram")

    print ("\n>>> All Path Diagrams Generated Successfully.")

if __name__ =="__main__":
    main ()