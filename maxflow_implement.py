import maxflow
import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd

from src import GraphicalMarkovRandomField
from shapely.ops import nearest_points

temp_folder = './temp'

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
    nearest_neighbors = distances[distances <= 10].nsmallest(k + 1).iloc[1:]  # Exclude the target_row itself
    return gdf.loc[nearest_neighbors.index, 'id'].tolist()

# Iterate over middle_segment and find the nearest 8 features for each element
nearest_neighbors_map = {}
for _, row in middle_segment.iterrows():
    feature = id_to_feature_map[row['id']]
    nearest_neighbors = find_nearest_neighbors(selected_gdf, feature.iloc[0])
    nearest_neighbors_map[row['id']] = nearest_neighbors

accurate_count = middle_segment[middle_segment['预测'] == '准确'].shape[0]
inaccurate_count = middle_segment[middle_segment['预测'] == '不准确'].shape[0]

print(f"Accurate count: {accurate_count}")
print(f"Inaccurate count: {inaccurate_count}")

nearest_neighbors_map

def build_graph(nearest_neighbors_map):
    nodes = {}
    edges = {}
    for node_id, neighbors in nearest_neighbors_map.items():
        nodes[node_id] = 0
        edges[node_id] = neighbors
    return nodes, edges

g = maxflow.Graph[float]()
nodes = list(nearest_neighbors_map.keys())
node_ids = {node: g.add_nodes(1)[0] for node in nodes}

node_index_map = {node: index for index, node in enumerate(nodes)}

# for node, neighbors in nearest_neighbors_map.items():
#     for neighbor in neighbors:
#         g.add_edge(node_ids[node], node_ids[neighbor], 0.1, 0.1)

# 用来赋值的字典
node_eval = {row['id']: row['预测'] for _, row in middle_segment.iterrows()}

PREFER_TO_NEG_IF_ZHUN = 0.1
PREFER_TO_NEG_IF_BUZHUN = 0.9

def add_tedge_for_nodes(node_ids,node_eval,g):

    # ``不准确`` 对应的不确定性更高，设置为0.9
    # ``准确``   对应的不确定性更低，设置为0.1
    
    for node, node_id in node_ids.items():
        # 按照此种方式进行设置，分配为0指的是未变化，为1指的是变化
        if node_eval[node] == '准确':
            # 标签为 准确 时，分配为0（未变化）的偏好为0.9；分配为1（变化）的偏好为0.1 \ 因此倾向于分配为0
            g.add_tedge(node_id, 0.9, 0.1)
        elif node_eval[node] == '不准确':
            # 标签为不准确时，分配为0（未变化）的偏好为0.1；分配为1（变化）的代价为0.9 \ 因此倾向于分配为1
            g.add_tedge(node_id, 0.1, 0.9)

BETA_SET = 0.2

def add_edge_for_nodes(nodes,edges,g,beta=0.2):
    for node_id in nodes:
        for neighbor_id in edges[node_id]:
            g.add_edge(node_ids[node_id], node_ids[neighbor_id], beta*abs(PREFER_TO_NEG_IF_ZHUN-PREFER_TO_NEG_IF_BUZHUN), beta*abs(PREFER_TO_NEG_IF_ZHUN-PREFER_TO_NEG_IF_BUZHUN))
print(g.get_edge_num())
add_tedge_for_nodes(node_ids,node_eval,g)
print(g.get_edge_num())
add_edge_for_nodes(nodes,nearest_neighbors_map,g,BETA_SET)
print(g.get_edge_num())
flow = g.maxflow()
print(f"Maximum flow: {flow}")
# Create a list to store the results
results = []

# Iterate over the nodes and get the segment for each node
for id, node_G_id in node_index_map.items():
    segment = g.get_segment(node_G_id)
    results.append({'id': id, 'segment': segment})

# Convert the results to a DataFrame
results_df = pd.DataFrame(results)

# Count the number of segments for each value (0 and 1)
segment_counts = results_df['segment'].value_counts()
print(f"Segment 0 count: {segment_counts.get(0, 0)}")
print(f"Segment 1 count: {segment_counts.get(1, 0)}")
# Define the output path for the CSV file
file_name = 'segmentation_results.csv'
file_name = file_name.replace('.csv', str(BETA_SET)+'.csv')
output_csv_path = os.path.join(temp_folder, file_name)

# Save the DataFrame to a CSV file
results_df.to_csv(output_csv_path, index=False)