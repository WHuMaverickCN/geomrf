import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
import csv
from src import GraphicalMarkovRandomField
from shapely.ops import nearest_points

def run_mrf_fit2sf(SEGMENT_POSITION):
    with open('./config_sf.yaml', 'r') as f:
        sf_config = yaml.safe_load(f)

    params = sf_config.get('params', {})
    WINDOW_SIZE = params.get('window_size', 100)
    PLAUSIBLE = params.get('plausible', 0.95)
    SUSPICIOUS = params.get('suspicious', 0.05)
    LAMBDA_PARAM = params.get('lambda_param', 0.1)
    BETA = params.get('beta', 0.5)
    ITER = params.get('iterations', 1000)
        
    root = sf_config.get('target_path')

    shapaformer_data_schema = {
        "train_set": {
            "vector": "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_1052_l35_727/train_all_vec.geojson",
            "eval": "temp/sf_result/2025-05-06_00-03_inference/train/combined_predictions.csv"
        },
        "validate_set": {
            "vector": "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_0932_l17_619/train_all_vec.geojson",
            "eval": "temp/sf_result/2025-05-06_00-03_inference/validate/combined_predictions.csv"
        },
        "test_set": {  
            "vector": "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_2125_l35_809/train_all_vec.geojson",
            "eval": "temp/sf_result/2025-05-06_00-03_inference/test/combined_predictions.csv"
        }
    }

    _gson = shapaformer_data_schema['train_set']['vector']
    _eval = shapaformer_data_schema['train_set']['eval']

    if not _gson or not _gson.endswith('.geojson'):
        raise ValueError("Invalid or missing geojson path in config")

    gdf = gpd.read_file(_gson)
    df = pd.read_csv(_eval)

    df['oid'] = df['oid'].astype(str)
    gdf['oid'] = gdf['oid'].astype(str)

    merged_df = df.merge(gdf[['id', 'oid','start_time', 'end_time']], on='oid', how='left')
    merged_df = merged_df.sort_values(by='start_time', ascending=True)
    merged_df = merged_df[merged_df['predictions'].isin([0, 1])]
    merged_output_path = os.path.join(root, 'mrf_merged_output.csv')
    merged_df.to_csv(merged_output_path, index=False)

    # middle_segment = merged_df.iloc[len(merged_df)//3 -int(WINDOW_SIZE/2) : len(merged_df)//3 +int(WINDOW_SIZE/2)]
    # 参数化设置选取数据段在数据框中的位置
    # SEGMENT_POSITION = params.get('segment_position', 0.33)  # 默认为1/3处，可在config中设置0~1之间
    center_idx = int(len(merged_df) * SEGMENT_POSITION)
    start_idx = max(center_idx - int(WINDOW_SIZE / 2), 0)
    end_idx = min(center_idx + int(WINDOW_SIZE / 2), len(merged_df))
    middle_segment = merged_df.iloc[start_idx:end_idx]
    selected_ids = middle_segment['oid'].tolist()
    selected_gdf = gdf[gdf['oid'].isin(selected_ids)]
    selected_output_path = './temp/selected_files.geojson'
    selected_gdf.to_file(selected_output_path, driver='GeoJSON')

    selected_gdf = selected_gdf.to_crs(epsg=32648)
    id_to_feature_map = {row['oid']: selected_gdf[selected_gdf['oid'] == row['oid']] for _, row in middle_segment.iterrows()}

    def find_nearest_neighbors(gdf, target_row, k=8):
        distances = gdf.geometry.apply(lambda geom: target_row.geometry.distance(geom))
        nearest_neighbors = distances[distances <= 15].nsmallest(k + 1).iloc[1:]
        return gdf.loc[nearest_neighbors.index, 'oid'].tolist()

    nearest_neighbors_map = {}
    for _, row in middle_segment.iterrows():
        feature = id_to_feature_map[row['oid']]
        nearest_neighbors = find_nearest_neighbors(selected_gdf, feature.iloc[0])
        nearest_neighbors_map[row['oid']] = nearest_neighbors

    middle_segment.to_csv('./temp/middle_segment.csv', index=False)
    node_eval = {str(row['oid']): row['predictions'] for _, row in middle_segment.iterrows()}
    nodes = {_k:{} for _k in nearest_neighbors_map.keys()}

    mrf = GraphicalMarkovRandomField(nodes, nearest_neighbors_map, node_eval)
    min_energy = float('inf')
    best_labels = None

    for _ in range(ITER):
        mrf.simulate()
        current_energy = mrf.total_energy_mine()
        print(f"{_}/{ITER}: {current_energy}", end='\r')
        if current_energy < min_energy:
            min_energy = current_energy
            best_labels = mrf.labels.copy()

    print("Lowest Total Energy:", min_energy)
    mrf.labels = best_labels

    smooth_label = mrf.labels.values()
    smooth_label_sum = sum(smooth_label)
    print("Sum of smooth labels:", smooth_label_sum)

    gt = {str(row['oid']): row['targets'] for _, row in middle_segment.iterrows() if row['targets'] in [0, 1]}

    from src.eval_detail import compare_labels_and_node_eval, check_mrf_performance

    cross = compare_labels_and_node_eval(mrf.labels, mrf.node_eval)
    real_one = check_mrf_performance(mrf.labels, mrf.node_eval, gt)
    print(
        "\n","cross:", cross,
        "\n","real_one:", real_one
    )

    return {
        "cross": cross,
        "real_one": real_one
    }

results = []
step = 0.01
positions = np.arange(0.05, 0.96, step)

for SEGMENT_POSITION in positions:
    print(f"Processing SEGMENT_POSITION: {SEGMENT_POSITION:.2f} ({int((SEGMENT_POSITION-positions[0])/step)+1}/{len(positions)})")
    result = run_mrf_fit2sf(SEGMENT_POSITION)
    real_acc = result['real_one'].get('labels', {}).get('accuracy', 0)
    node_acc = result['real_one'].get('node_eval', {}).get('accuracy', 0)
    if_better = 1 if real_acc > node_acc else 0
    row = {
        'SEGMENT_POSITION': SEGMENT_POSITION,
        'real_acc': real_acc,
        'node_acc': node_acc,
        'if_better': if_better,
        'cross': result.get('cross', None),
        'real_one': result.get('real_one', None)
    }
    results.append(row)

with open('batch_mrf_result.csv', 'w', newline='') as csvfile:
    fieldnames = ['SEGMENT_POSITION', 'real_acc', 'node_acc', 'if_better', 'cross', 'real_one']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)