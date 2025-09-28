import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd

from src import GraphicalMarkovRandomField
from shapely.ops import nearest_points



with open('./config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
root = config.get('target_path')
gson = os.path.join(root, config['result_files']['vector'])
eval = os.path.join(root, config['result_files']['eval'])


if not gson or not gson.endswith('.geojson'):
    raise ValueError("Invalid or missing geojson path in config")

gdf = gpd.read_file(gson)
df = pd.read_csv(eval)

# Merge the dataframes on the 'id' column
merged_df = df.merge(gdf[['id', 'oid','start_time', 'end_time']], on='id', how='left')

# Save the merged dataframe to a new CSV file
merged_df = merged_df.sort_values(by='start_time', ascending=True)

merged_df = merged_df[merged_df['类型编号'] == 'RoadLineType']
merged_df = merged_df[merged_df['预测'].isin(['准确', '不准确'])]
merged_output_path = os.path.join(root, 'merged_output.csv')
# Save the merged dataframe to a new CSV file
merged_df.to_csv(merged_output_path, index=False)

# Select a middle segment of 100 elements from merged_df
middle_segment = merged_df.iloc[len(merged_df)//2 - 50 : len(merged_df)//2 + 50]
selected_ids = middle_segment['id'].tolist()


selected_gdf = gdf[gdf['id'].isin(selected_ids)]


# Save the selected features to a new GeoJSON file
selected_output_path = './temp/selected_files.geojson'
selected_gdf.to_file(selected_output_path, driver='GeoJSON')

# Convert selected_gdf to EPSG:32648 projection coordinate system
selected_gdf = selected_gdf.to_crs(epsg=32648)
id_to_feature_map = {row['id']: selected_gdf[selected_gdf['id'] == row['id']] for _, row in middle_segment.iterrows()}

# Function to calculate the nearest neighbors
def find_nearest_neighbors(gdf, target_row, k=8):
    distances = gdf.geometry.apply(lambda geom: target_row.geometry.distance(geom))
    nearest_neighbors = distances[distances <= 15].nsmallest(k + 1).iloc[1:]  # Exclude the target_row itself
    return gdf.loc[nearest_neighbors.index, 'id'].tolist()

# Iterate over middle_segment and find the nearest 8 features for each element
nearest_neighbors_map = {}
for _, row in middle_segment.iterrows():
    feature = id_to_feature_map[row['id']]
    nearest_neighbors = find_nearest_neighbors(selected_gdf, feature.iloc[0])
    nearest_neighbors_map[row['id']] = nearest_neighbors

# Print the nearest neighbors map
# print(nearest_neighbors_map)
middle_segment.to_csv('./temp/middle_segment.csv', index=False)
node_eval = {row['id']: row['预测'] for _, row in middle_segment.iterrows()}
nodes = {_k:{} for _k in nearest_neighbors_map.keys()}

mrf = GraphicalMarkovRandomField(nodes, nearest_neighbors_map,node_eval)
min_energy = float('inf')
best_labels = None

# Repeat the process multiple times to find the solution with the lowest total energy
for _ in range(1000):  # Adjust the number of iterations as needed
    mrf.simulate()
    current_energy = mrf.total_energy_mine()
    print(current_energy)
    if current_energy < min_energy:
        min_energy = current_energy
        best_labels = mrf.labels.copy()

print("Lowest Total Energy:", min_energy)
mrf.labels = best_labels  # Set the labels to the best found solution

smooth_label = mrf.labels.values()
smooth_label_sum = sum(smooth_label)
print("Sum of smooth labels:", smooth_label_sum)
# print()


from src.eval_detail import compare_labels_and_node_eval
print(
    compare_labels_and_node_eval(mrf.labels,mrf.node_eval)
    )