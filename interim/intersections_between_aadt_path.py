import osmnx as ox
import networkx as nx
import pandas as pd

# 1. Load the Excel file
df = pd.read_csv('pairings2.csv')

# print("Loading NYC street network...")
G = ox.graph_from_place("New York City, New York", network_type="drive")

G = G.to_undirected() # not drive network?? does not take oneway street into account because traffic can be either way

expanded_data = []

# 3. Process with Feedback
for index, row in df.iterrows():
    print(f"\n--- Processing Row {index} ---")
    
    # Check if coords exist
    try:
        latA, lonA = row['latA'], row['lonA']
        latB, lonB = row['latB'], row['lonB']
        
        if pd.isna([latA, lonA, latB, lonB]).any():
            print(f"⚠️ Skipping: Missing coordinates.")
            continue
            
        # Find nearest intersections (nodes)
        node_start = ox.nearest_nodes(G, X=lonA, Y=latA)
        node_end = ox.nearest_nodes(G, X=lonB, Y=latB)
        print(f"Found nodes: Start({node_start}), End({node_end})")

        # Attempt to find the path
        try:
            path = nx.shortest_path(G, node_start, node_end, weight='length')
            print(f"✅ Success! Path found with {len(path)} intersections.")
            
            for node in path:
                expanded_data.append({
                    'node_id': node,
                    'aadt': row['aadt'],
                    'start_street': row.get('StreetA', 'N/A'),
                    'end_street': row.get('StreetB', 'N/A')
                    
                })
        except nx.NetworkXNoPath:
            print(f"❌ No Path: Points are not connected in the road network.")
            
    except Exception as e:
        print(f"❌ Error processing row: {e}")

# 4. Final Save
if expanded_data:
    final_df = pd.DataFrame(expanded_data)
    
    # Get coordinates for the nodes
    nodes_gdf, _ = ox.graph_to_gdfs(G)
    final_df = final_df.merge(nodes_gdf[['y', 'x']], left_on='node_id', right_index=True)
    
    traceable_output = final_df.groupby('node_id').agg({
        'aadt': 'mean',
        'start_street': lambda x: ' | '.join(set(x.astype(str))), # Unique parents only
        'end_street': lambda x: ' | '.join(set(x.astype(str)))    # Unique parents only
    }).reset_index()

    # Join back the Lat/Lon
    traceable_output = traceable_output.merge(nodes_gdf[['y', 'x']], left_on='node_id', right_index=True)
    
    traceable_output.to_csv('output_test.csv', index=False)
    print("✅ Traceable file saved! Check the 'start_street' column for multiple parents.")
else:
    print("\n🛑 ERROR: The script finished but found 0 intersections. Check the console logs above.")
