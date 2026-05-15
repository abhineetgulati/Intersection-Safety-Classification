import pandas as pd
import osmnx as ox
import numpy as np
from crashes import df

city_graph = ox.graph_from_place("New York City, New York, USA", network_type="drive")
city_graph_projected = ox.project_graph(city_graph)
# This creates a data table of all nodes (points where roads meet or end)
#nodes, edges = ox.graph_to_gdfs(city_graph)

consolidated = ox.consolidate_intersections(
        city_graph_projected, tolerance=30, rebuild_graph=True, dead_ends=False
    )

# 2. SNAPPING: Find the closest intersection for every row
# This uses a spatial index for high speed, even with NYC-sized data
# 1. Project your consolidated graph BACK to WGS84 (Lat/Lon)
# This ensures the graph and the accidents are using the same units (degrees)
consolidated_wgs84 = ox.project_graph(consolidated, to_crs="EPSG:4326")

# 2. Now perform the snapping using the WGS84 version of the graph
print("Snapping accidents using matching coordinate systems...")
nearest_node_ids = ox.nearest_nodes(
    consolidated_wgs84, 
    X=df['LONGITUDE'].values, 
    Y=df['LATITUDE'].values
)

# 3. Add the IDs back to your dataframe
df['nearest_intersection_id'] = nearest_node_ids

# 4. Update your lookup table logic (make sure it uses the same node IDs)
nodes_wgs84 = ox.graph_to_gdfs(consolidated_wgs84, edges=False)
intersection_lookup = pd.DataFrame({
    'intersection_lat': nodes_wgs84.geometry.y,
    'intersection_lon': nodes_wgs84.geometry.x
}, index=nodes_wgs84.index)

# 5. Final Merge
df_final = df.merge(intersection_lookup, left_on='nearest_intersection_id', right_index=True, how='left')

df_unique = df_final.drop_duplicates(subset=["intersection_lat", "intersection_lon"], keep="first")

df_unique.to_csv("snappedaccidents.csv")
