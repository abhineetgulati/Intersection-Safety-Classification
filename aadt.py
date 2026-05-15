import osmnx as ox
import networkx as nx
import pandas as pd

# 1. Load data and create a unique ID for each segment
df = pd.read_csv('AADT.csv', nrows=10) # Testing with 10 rows
df['segment_id'] = df.index 

# Load NYC graph
G = ox.graph_from_place("New York City, New York", network_type="drive")

# To store the mapping with the original ID
node_mapping_list = []

for index, row in df.iterrows():
    try:
        # Geocode and find path (Same logic as before)
        loc_a, loc_b = f"{row['BeginDescription']}, NYC", f"{row['EndDescription']}, NYC"
        coords_a, coords_b = ox.geocode(loc_a), ox.geocode(loc_b)
        
        node_start = ox.nearest_nodes(G, X=coords_a[1], Y=coords_a[0])
        node_end = ox.nearest_nodes(G, X=coords_b[1], Y=coords_b[0])
        path = nx.shortest_path(G, node_start, node_end, weight='length')
        
        # KEY CHANGE: Store the segment_id from the original CSV
        for node in path:
            node_mapping_list.append({
                'node_id': node, 
                'segment_id': row['segment_id'],
                'assigned_aadt': row['AADT']
            })
            
    except Exception as e:
        print(f"Error on row {index}: {e}")

# 2. Create the mapping DataFrame
mapping_df = pd.DataFrame(node_mapping_list)

# 3. Join back with the original CSV to verify
# This gives you a 'long-form' table where each original segment 
# is repeated for every intersection it contains.
verification_df = mapping_df.merge(df, on='segment_id')

# 4. Final step: Get your clean node-level averages
# This averages AADT for nodes that appeared in multiple segments
final_nodes = mapping_df.groupby('node_id')['assigned_aadt'].mean().reset_index()

# 5. Bring in Lat/Lon for the final nodes
nodes_gdf, _ = ox.graph_to_gdfs(G)
final_nodes = final_nodes.merge(nodes_gdf[['y', 'x']], left_on='node_id', right_index=True)

print("--- Verification Table (Sample) ---")
print(verification_df[['segment_id', 'BeginDescription', 'EndDescription', 'node_id', 'assigned_aadt']].head(10))

print("\n--- Final Cleaned Dataset ---")
print(final_nodes.head())

final_nodes.to_csv("aadtgeocoded.csv")