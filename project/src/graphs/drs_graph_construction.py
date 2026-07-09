import os
import glob
import pandas as pd
import networkx as nx

def get_data_paths():
    """Resolve paths to events, raw, and processed directories dynamically."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Try different possible locations of the data directory
    possible_roots = [
        os.path.abspath(os.path.join(script_dir, '..', '..')), # project/ (from project/src/graphs/)
        os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'project')), # from repository root
        os.getcwd(), # current working directory
        os.path.join(os.getcwd(), 'project')
    ]
    
    for root in possible_roots:
        events_path = os.path.join(root, 'data', 'events')
        raw_path = os.path.join(root, 'data', 'raw')
        processed_path = os.path.join(root, 'data', 'processed')
        if os.path.exists(raw_path) and os.path.exists(processed_path):
            return raw_path, processed_path
            
    raise FileNotFoundError("Could not locate F1 dataset directory. Ensure you are running within the F1-data-project repository.")

def build_drs_interval_graph(race_folder_name, raw_path, processed_path):
    """Build undirected weighted graph of DRS interactions for a single race."""
    # Find master parquet file in the consolidated master directory
    race_clean = race_folder_name.replace('_2026', '')
    master_dir = os.path.join(processed_path, "master")
    
    # Check if directory exists
    if not os.path.exists(master_dir):
        raise FileNotFoundError(f"Master directory not found at {master_dir}")
        
    master_files = glob.glob(os.path.join(master_dir, f"{race_clean}_*_master.parquet"))
    if not master_files:
        # Try finding v2 or fallback
        master_files = glob.glob(os.path.join(master_dir, f"{race_clean}_*master*.parquet"))
        
    if not master_files:
        raise FileNotFoundError(f"No master parquet file found for race {race_clean} in {master_dir}")
        
    master_file = master_files[0]
    intervals_file = os.path.join(raw_path, race_folder_name, "intervals.csv")
    drivers_file = os.path.join(raw_path, race_folder_name, "drivers.csv")
    
    if not os.path.exists(intervals_file):
        raise FileNotFoundError(f"Missing intervals.csv for race {race_folder_name} at {intervals_file}")
    if not os.path.exists(drivers_file):
        raise FileNotFoundError(f"Missing drivers.csv for race {race_folder_name} at {drivers_file}")
        
    df_master = pd.read_parquet(master_file)
    df_intervals = pd.read_csv(intervals_file)
    df_drivers = pd.read_csv(drivers_file)
    
    # Map driver numbers to 3-letter acronyms
    driver_map = dict(zip(df_drivers['driver_number'].astype(str), df_drivers['name_acronym']))
    driver_map.update(dict(zip(df_drivers['driver_number'], df_drivers['name_acronym'])))
    
    # Process intervals dates
    df_intervals = df_intervals.dropna(subset=['date'])
    df_intervals['interval'] = pd.to_numeric(df_intervals['interval'], errors='coerce')
    df_intervals['date'] = pd.to_datetime(df_intervals['date'], format='ISO8601')
    df_intervals = df_intervals.sort_values('date')
    
    # Process master dates
    df_master['date_start'] = pd.to_datetime(df_master['date_start'], format='ISO8601')
    df_master = df_master.sort_values('date_start')
    
    # Align intervals to laps with merge_asof (backward direction)
    intervals_with_lap = pd.merge_asof(
        df_intervals,
        df_master[['driver_number', 'lap_number', 'date_start', 'position']],
        left_on='date',
        right_on='date_start',
        by='driver_number',
        direction='backward'
    )
    
    # Aggregate to get the last interval in the lap
    lap_intervals = intervals_with_lap.groupby(['driver_number', 'lap_number']).agg(
        gap_ahead=('interval', 'last'),
        position=('position', 'last')
    ).reset_index()
    
    # Merge back to base df
    df_race = pd.merge(
        df_master[['driver_number', 'lap_number', 'position']],
        lap_intervals[['driver_number', 'lap_number', 'gap_ahead']],
        on=['driver_number', 'lap_number'],
        how='left'
    )
    df_race['gap_ahead'] = df_race['gap_ahead'].fillna(30.0)
    
    total_laps = df_race['lap_number'].max()
    
    # Create undirected graph
    G = nx.Graph()
    for driver in df_drivers['name_acronym'].unique():
        G.add_node(driver)
        
    drs_laps_count = {}
    
    # Process lap by lap
    for lap, group in df_race.groupby('lap_number'):
        group = group.sort_values('position')
        drivers_in_group = group.to_dict('records')
        for i in range(1, len(drivers_in_group)):
            curr_car = drivers_in_group[i]
            ahead_car = drivers_in_group[i-1]
            
            gap = curr_car['gap_ahead']
            if gap < 1.0:
                c_acr = driver_map.get(curr_car['driver_number'])
                a_acr = driver_map.get(ahead_car['driver_number'])
                if c_acr and a_acr:
                    edge = tuple(sorted([c_acr, a_acr]))
                    drs_laps_count[edge] = drs_laps_count.get(edge, 0) + 1
                    
    # Add edges with weights and distances
    for (u, v), count in drs_laps_count.items():
        weight = count / total_laps
        distance = 1.0 / weight if weight > 0 else float('inf')
        G.add_edge(u, v, weight=weight, distance=distance, laps_in_drs=count)
        
    return G, total_laps

def calculate_drs_metrics(G):
    """Calculate key graph metrics for the DRS interval graph."""
    # Betweenness Centrality (distance-based shortest paths)
    betweenness = nx.betweenness_centrality(G, weight='distance')
    
    # Connected Components (pelotons)
    components = list(nx.connected_components(G))
    # Sort components by size descending
    components = sorted(components, key=len, reverse=True)
    
    # Consolidate metrics in a DataFrame
    metrics_df = pd.DataFrame({
        'Betweenness Centrality': betweenness
    }).sort_values(by='Betweenness Centrality', ascending=False)
    
    return metrics_df, components

def build_global_drs_interval_graph(raw_path, processed_path, races_with_intervals):
    """Build season-wide aggregated DRS interval graph."""
    global_drs_laps = {}
    total_season_laps = 0
    global_driver_map = {}
    
    for race in races_with_intervals:
        race_clean = race.replace('_2026', '')
        race_processed_dir = os.path.join(processed_path, race_clean)
        master_files = glob.glob(os.path.join(race_processed_dir, "*_master.parquet"))
        if not master_files:
            master_files = glob.glob(os.path.join(race_processed_dir, "*_master*.parquet"))
        master_file = master_files[0]
        intervals_file = os.path.join(raw_path, race, "intervals.csv")
        drivers_file = os.path.join(raw_path, race, "drivers.csv")
        
        df_master = pd.read_parquet(master_file)
        df_intervals = pd.read_csv(intervals_file)
        df_drivers = pd.read_csv(drivers_file)
        
        # Driver map update
        d_map = dict(zip(df_drivers['driver_number'].astype(str), df_drivers['name_acronym']))
        d_map.update(dict(zip(df_drivers['driver_number'], df_drivers['name_acronym'])))
        global_driver_map.update(d_map)
        
        df_intervals = df_intervals.dropna(subset=['date'])
        df_intervals['interval'] = pd.to_numeric(df_intervals['interval'], errors='coerce')
        df_intervals['date'] = pd.to_datetime(df_intervals['date'], format='ISO8601')
        df_intervals = df_intervals.sort_values('date')
        
        df_master['date_start'] = pd.to_datetime(df_master['date_start'], format='ISO8601')
        df_master = df_master.sort_values('date_start')
        
        intervals_with_lap = pd.merge_asof(
            df_intervals,
            df_master[['driver_number', 'lap_number', 'date_start', 'position']],
            left_on='date',
            right_on='date_start',
            by='driver_number',
            direction='backward'
        )
        
        lap_intervals = intervals_with_lap.groupby(['driver_number', 'lap_number']).agg(
            gap_ahead=('interval', 'last'),
            position=('position', 'last')
        ).reset_index()
        
        df_race = pd.merge(
            df_master[['driver_number', 'lap_number', 'position']],
            lap_intervals[['driver_number', 'lap_number', 'gap_ahead']],
            on=['driver_number', 'lap_number'],
            how='left'
        )
        df_race['gap_ahead'] = df_race['gap_ahead'].fillna(30.0)
        
        race_laps = df_race['lap_number'].max()
        total_season_laps += race_laps
        
        for lap, group in df_race.groupby('lap_number'):
            group = group.sort_values('position')
            drivers_in_group = group.to_dict('records')
            for i in range(1, len(drivers_in_group)):
                curr_car = drivers_in_group[i]
                ahead_car = drivers_in_group[i-1]
                gap = curr_car['gap_ahead']
                if gap < 1.0:
                    c_acr = d_map.get(curr_car['driver_number'])
                    a_acr = d_map.get(ahead_car['driver_number'])
                    if c_acr and a_acr:
                        edge = tuple(sorted([c_acr, a_acr]))
                        global_drs_laps[edge] = global_drs_laps.get(edge, 0) + 1
                        
    # Create global graph
    G_global = nx.Graph()
    for acr in set(global_driver_map.values()):
        G_global.add_node(acr)
        
    for (u, v), count in global_drs_laps.items():
        weight = count / total_season_laps
        distance = 1.0 / weight if weight > 0 else float('inf')
        G_global.add_edge(u, v, weight=weight, distance=distance, laps_in_drs=count)
        
    return G_global, total_season_laps

def main():
    print("Resolving data paths...")
    raw_path, processed_path = get_data_paths()
    print(f"Raw Path: {raw_path}")
    print(f"Processed Path: {processed_path}\n")
    
    graphs_processed_path = os.path.join(processed_path, "graphs")
    os.makedirs(graphs_processed_path, exist_ok=True)
    
    # List of races to process
    race_folders = ['australia_2026', 'china_2026', 'japan_2026', 'united_states_2026']
    races_with_intervals = []
    
    for race_folder in race_folders:
        print(f"Processing race folder: {race_folder}...")
        intervals_file = os.path.join(raw_path, race_folder, "intervals.csv")
        
        if not os.path.exists(intervals_file):
            print(f"    [WARN] No intervals.csv found for {race_folder}. Skipping DRS interval graph construction.\n")
            continue
            
        races_with_intervals.append(race_folder)
        
        # Build and calculate metrics
        G, total_laps = build_drs_interval_graph(race_folder, raw_path, processed_path)
        metrics_df, components = calculate_drs_metrics(G)
        
        # Print summaries
        print(f"\n--- {race_folder} DRS Interval Graph Summary ---")
        print(f"Total Laps: {total_laps}")
        print(f"Nodes (Drivers): {G.number_of_nodes()}")
        print(f"Edges (DRS Interactions): {G.number_of_edges()}")
        print("\nTop 10 Betweenness Centrality (Midfield Bottleneck/Tapón):")
        print(metrics_df.head(10).to_string())
        
        print(f"\nConnected Components (Pelotons/DRS Trains): {len(components)} groups")
        for idx, comp in enumerate(components, 1):
            print(f"Group {idx} ({len(comp)} drivers): {', '.join(comp)}")
        print("-" * 50 + "\n")
        
        # Save GraphML file
        output_file = os.path.join(graphs_processed_path, f"{race_folder}_drs_interval.graphml")
        nx.write_graphml(G, output_file)
        print(f"Saved GraphML to {output_file}\n")
        
    # Global Season Graph
    if races_with_intervals:
        print("Processing Global Season DRS Interval Graph...")
        G_global, total_season_laps = build_global_drs_interval_graph(raw_path, processed_path, races_with_intervals)
        metrics_global, components_global = calculate_drs_metrics(G_global)
        
        print("\n--- Global Season DRS Interval Graph Summary ---")
        print(f"Total Season Laps: {total_season_laps}")
        print(f"Nodes (Drivers): {G_global.number_of_nodes()}")
        print(f"Edges (DRS Interactions): {G_global.number_of_edges()}")
        print("\nTop 10 Global Betweenness Centrality (Midfield Bottleneck/Tapón):")
        print(metrics_global.head(10).to_string())
        
        print(f"\nConnected Components (Global Pelotons): {len(components_global)} groups")
        for idx, comp in enumerate(components_global, 1):
            print(f"Group {idx} ({len(comp)} drivers): {', '.join(comp)}")
        print("-" * 50 + "\n")
        
        # Save Global GraphML file
        output_global_file = os.path.join(graphs_processed_path, "global_drs_interval.graphml")
        nx.write_graphml(G_global, output_global_file)
        print(f"Saved Global GraphML to {output_global_file}\n")

if __name__ == '__main__':
    main()
