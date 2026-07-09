import os
import glob
import pandas as pd
import networkx as nx

def get_data_paths():
    """Resolve paths to events and raw directories dynamically."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations of the data directory
    possible_roots = [
        os.path.abspath(os.path.join(script_dir, '..', '..')), # project/ (from project/src/models/)
        os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'project')), # from repository root
        os.getcwd(), # current working directory
        os.path.join(os.getcwd(), 'project')
    ]
    
    for root in possible_roots:
        events_path = os.path.join(root, 'data', 'processed', 'events')
        raw_path = os.path.join(root, 'data', 'raw')
        processed_path = os.path.join(root, 'data', 'processed')
        if os.path.exists(events_path) and os.path.exists(raw_path):
            # Ensure processed directory exists
            os.makedirs(processed_path, exist_ok=True)
            return events_path, raw_path, processed_path
            
    raise FileNotFoundError("Could not locate F1 dataset directory. Ensure you are running within the F1-data-project repository.")

def build_overtake_graph(race_name, events_path, raw_path):
    """Build directed weighted graph of overtakes for a single race."""
    events_file = os.path.join(events_path, f"{race_name}_events.parquet")
    drivers_file = os.path.join(raw_path, race_name, "drivers.csv")
    
    if not os.path.exists(events_file) or not os.path.exists(drivers_file):
        raise FileNotFoundError(f"Missing files for race {race_name}")
        
    df_events = pd.read_parquet(events_file)
    df_drivers = pd.read_csv(drivers_file)
    
    # Map driver numbers to 3-letter acronyms
    driver_map = dict(zip(df_drivers['driver_number'].astype(str), df_drivers['name_acronym']))
    driver_map.update(dict(zip(df_drivers['driver_number'], df_drivers['name_acronym'])))
    
    # Filter only on-track overtakes
    overtakes = df_events[df_events['event_type'] == 'On_Track_Overtake']
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add all drivers as nodes to avoid ignoring those with 0 interactions
    for driver in df_drivers['name_acronym'].unique():
        G.add_node(driver)
        
    # Process edges (defended -> overtake)
    for _, row in overtakes.iterrows():
        u_num = row['target_driver']      # Overtaken driver (defender)
        v_num = row['initiator_driver']   # Overtaking driver (attacker)
        
        u = driver_map.get(u_num, str(u_num))
        v = driver_map.get(v_num, str(v_num))
        
        if G.has_edge(u, v):
            G[u][v]['weight'] += 1
        else:
            G.add_edge(u, v, weight=1)
            
    return G

def calculate_graph_metrics(G):
    """Calculate and return key graph metrics."""
    # In-degree: Overtakes done (offense)
    in_deg = dict(G.in_degree(weight='weight'))
    # Out-degree: Times overtaken (defense)
    out_deg = dict(G.out_degree(weight='weight'))
    
    # PageRank (tactical dominance)
    pagerank = nx.pagerank(G, weight='weight')
    
    # Betweenness Centrality (DRS trains / bottlenecks)
    betweenness = nx.betweenness_centrality(G, weight='weight')
    
    # Weakly and Strongly Connected Components
    weak_components = list(nx.weakly_connected_components(G))
    strong_components = list(nx.strongly_connected_components(G))
    
    # Consolidate metrics in a DataFrame
    metrics_df = pd.DataFrame({
        'Overtakes Made (Offense)': in_deg,
        'Times Overtaken (Defense)': out_deg,
        'PageRank (Dominance)': pagerank,
        'Betweenness Centrality': betweenness
    }).sort_values(by='PageRank (Dominance)', ascending=False)
    
    return metrics_df, weak_components, strong_components

def build_global_overtake_graph(events_path, raw_path):
    """Build aggregated season-wide overtake graph."""
    event_files = glob.glob(os.path.join(events_path, "*_events.parquet"))
    
    G_global = nx.DiGraph()
    global_driver_map = {}
    
    # First pass: load all driver definitions across all races to ensure complete node mappings
    for event_file in event_files:
        race_name = os.path.basename(event_file).replace('_events.parquet', '')
        drivers_file = os.path.join(raw_path, race_name, "drivers.csv")
        if os.path.exists(drivers_file):
            df_drivers = pd.read_csv(drivers_file)
            d_map = dict(zip(df_drivers['driver_number'].astype(str), df_drivers['name_acronym']))
            d_map.update(dict(zip(df_drivers['driver_number'], df_drivers['name_acronym'])))
            global_driver_map.update(d_map)
            
            for driver in df_drivers['name_acronym'].unique():
                G_global.add_node(driver)
                
    # Second pass: process all events
    for event_file in event_files:
        df_events = pd.read_parquet(event_file)
        if 'event_type' in df_events.columns:
            overtakes = df_events[df_events['event_type'] == 'On_Track_Overtake']
            for _, row in overtakes.iterrows():
                u_num = row['target_driver']
                v_num = row['initiator_driver']
                
                u = global_driver_map.get(u_num, str(u_num))
                v = global_driver_map.get(v_num, str(v_num))
                
                if G_global.has_edge(u, v):
                    G_global[u][v]['weight'] += 1
                else:
                    G_global.add_edge(u, v, weight=1)
                    
    return G_global

def main():
    print("Resolving data paths...")
    events_path, raw_path, processed_path = get_data_paths()
    print(f"Events Path: {events_path}")
    print(f"Raw Path: {raw_path}")
    print(f"Processed Path: {processed_path}\n")
    
    # 1. Process Australia 2026
    print("Processing GP of Australia 2026...")
    G_aus = build_overtake_graph("australia_2026", events_path, raw_path)
    metrics_aus, weak_comps_aus, strong_comps_aus = calculate_graph_metrics(G_aus)
    
    print("\n--- Australia 2026 Top Metrics ---")
    print(metrics_aus.head(10).to_string())
    print(f"\nConnected Components: {len(weak_comps_aus)} groups")
    for idx, comp in enumerate(weak_comps_aus, 1):
        print(f"Group {idx}: {', '.join(comp)}")
        
    # Save Australia graph
    graphs_processed_path = os.path.join(processed_path, "graphs")
    os.makedirs(graphs_processed_path, exist_ok=True)
    nx.write_graphml(G_aus, os.path.join(graphs_processed_path, "australia_2026_overtakes.graphml"))
    print("\nAustralia GraphML saved.")
    
    # 2. Process Global Season Graph
    print("\nProcessing Global Season Graph (all races)...")
    G_global = build_global_overtake_graph(events_path, raw_path)
    metrics_global, weak_comps_global, _ = calculate_graph_metrics(G_global)
    
    print("\n--- Global Season Top Metrics ---")
    print(metrics_global.head(10).to_string())
    print(f"Total Nodes: {G_global.number_of_nodes()}")
    print(f"Total Edges: {G_global.number_of_edges()}")
    
    # Save Global graph
    nx.write_graphml(G_global, os.path.join(graphs_processed_path, "global_overtakes.graphml"))
    print("\nGlobal GraphML saved.")
    
    # Print comparison with popularity (overtakes made)
    print("\n--- Comparison: PageRank (Dominance) vs Overtakes Made (Popularity) ---")
    comparison_df = metrics_global[['Overtakes Made (Offense)', 'PageRank (Dominance)']].copy()
    comparison_df['Popularity Rank'] = comparison_df['Overtakes Made (Offense)'].rank(ascending=False, method='min')
    comparison_df['PageRank Rank'] = comparison_df['PageRank (Dominance)'].rank(ascending=False, method='min')
    comparison_df['Rank Difference'] = comparison_df['Popularity Rank'] - comparison_df['PageRank Rank']
    print(comparison_df.sort_values(by='PageRank (Dominance)', ascending=False).head(10).to_string())

if __name__ == '__main__':
    main()
