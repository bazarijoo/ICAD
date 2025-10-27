import math
from datetime import datetime

import os
import h3
import sys
import numpy as np
import pandas as pd
import pickle
import pytz
import torch
from pyproj import Proj, Transformer
from scipy.spatial import distance_matrix

from utils.constants import *
from utils.dataset import ICADDataset, construct_batched_visit_collate_fn
from torch.utils.data import DataLoader

#####################################################################
# Data Processing
#####################################################################

def process_space_time_numosim(visits_df, poi_df):
    merged_df = visits_df.merge(poi_df, how='left', on='poi_id')
    merged_df = merged_df.sort_values(by=['agent_id', 'start_datetime'], ignore_index=True)
    
    oldest_timestamp = merged_df['start_datetime'].min()

    # discretize locations
    merged_df['h3_index'] = merged_df.apply(lambda x: h3.geo_to_h3(x['latitude'], x['longitude'], resolution=6), axis=1)
    # The first N_SPECIAL_TOKENS indices are reserved for special tokens
    merged_df['region_id'] = merged_df['h3_index'].astype("category").cat.codes + N_SPECIAL_TOKENS

    # normalize lat and lon
    merged_df['latitude'] = (merged_df['latitude'] - merged_df['latitude'].mean()) / merged_df['latitude'].std()
    merged_df['longitude'] = (merged_df['longitude'] - merged_df['longitude'].mean()) / merged_df['longitude'].std()

    # Time (unit: day)
    merged_df['arrival_time'] = (merged_df['start_datetime'] - oldest_timestamp).dt.total_seconds() / (60*60*24)
    merged_df['departure_time'] = (merged_df['end_datetime'] - oldest_timestamp).dt.total_seconds() / (60*60*24)

    # arrival_time_of_day and departure_time_of_day (unit: fraction of hours in a day)
    merged_df['arrival_time_of_day'] =  (merged_df['start_datetime'] - merged_df['start_datetime'].dt.normalize()).dt.total_seconds() / (60*60*24)
    merged_df['departure_time_of_day'] = (merged_df['end_datetime'] - merged_df['end_datetime'].dt.normalize()).dt.total_seconds() / (60*60*24)

    # duration of stay (Unit: hours)
    # merged_df['duration'] = (merged_df['departure_time'] - merged_df['arrival_time'])

    # Calculate travel time (unit: fraction of daily hours)
    travel_time = merged_df.groupby('agent_id').apply(
        lambda group: (group['arrival_time'] - group['departure_time'].shift(1))
        ).reset_index(level=0, drop=True)
    merged_df['travel_time'] = travel_time
    merged_df.loc[merged_df.groupby('agent_id').nth(0).index, 'travel_time'] = 0 # First row of each user does not have travel time

    # Keep only the intersection of FIELDS and df.columns
    fields_to_keep = list(set(['agent_id']) | (set(FIELDS) & set(merged_df.columns))) # keep agent_id
    merged_df = merged_df[fields_to_keep]

    return merged_df

def split_indices_for_next_prediction(df, mode='test'):
    # Identify instances for next visit prediction
    groupby_user = df.groupby('agent_id')

    # Find the first index of each instance
    # 1. First instance of each user
    user_first_indices = groupby_user.nth(0).index
    # 2. Instance with at least RAW_SEQ_LEN visits
    max_num_visits_per_user = groupby_user.size().max()
    first_indices = user_first_indices

    # Find the corresponding last indices
    # 1. Last index of each user
    each_user_last_index = groupby_user.nth(-1).set_index('agent_id')['id']
    # 2. Match it with first_indices
    user_last_indices = df.loc[first_indices]['agent_id'].apply(lambda agent_id: each_user_last_index.loc[agent_id]).values
    # 3. Last index of each rolling window
    rolling_window_last_indices = (first_indices + max_num_visits_per_user - 1).values
    last_indices = np.minimum(rolling_window_last_indices, user_last_indices)

    # Sort instances by arrival_time
    index_df = pd.DataFrame({'first_index': first_indices, 'last_index': last_indices, 'arrival_time': df.loc[first_indices]['arrival_time']})
    index_df.sort_values(by='arrival_time', inplace=True, ignore_index=True)
    index_df.drop(columns=['arrival_time'], inplace=True)

    indices = index_df.reset_index(drop=True)

    return indices

    
def split_index_df_for_infilling(index_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Ensure the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1
    
    # Shuffle the DataFrame to ensure randomness
    index_df = index_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate the number of agent_ids in each split
    agent_ids = index_df['agent_id'].unique()
    total_users = len(agent_ids)
    train_size = int(total_users * train_ratio)
    val_size = int(total_users * val_ratio)
    
    # Split the agent_ids
    train_agent_ids = agent_ids[:train_size]
    val_agent_ids = agent_ids[train_size:train_size + val_size]
    test_agent_ids = agent_ids[train_size + val_size:]
    
    # Create the split DataFrames
    train_indices = index_df[index_df['agent_id'].isin(train_agent_ids)].reset_index(drop=True)
    val_indices = index_df[index_df['agent_id'].isin(val_agent_ids)].reset_index(drop=True)
    test_indices = index_df[index_df['agent_id'].isin(test_agent_ids)].reset_index(drop=True)

    # Drop `agent_id` column
    train_indices.drop(columns=['agent_id'], inplace=True)
    val_indices.drop(columns=['agent_id'], inplace=True)
    test_indices.drop(columns=['agent_id'], inplace=True)
    
    return train_indices, val_indices, test_indices

def load_numosim_dataset(args):
    # Get the base directory of the project
    base_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/NUMOSIM/LA/"

    staypoints_train_df = pd.read_parquet(os.path.join(base_path, 'preprocessed_stay_points_train.parquet'), engine='fastparquet')
    staypoints_test_anomalous_df = pd.read_parquet(os.path.join(base_path, 'preprocessed_stay_points_test_anomalous.parquet'), engine='fastparquet')

    # Concatenate the train and test anomalous DataFrames
    combined_df = pd.concat([staypoints_train_df, staypoints_test_anomalous_df], ignore_index=True)
    num_regions = combined_df['region_id'].nunique()

    staypoints_train_df['id'] = staypoints_train_df.index
    staypoints_test_anomalous_df['id'] = staypoints_test_anomalous_df.index

    train_indices = split_indices_for_next_prediction(staypoints_train_df)
    test_indices = split_indices_for_next_prediction(staypoints_test_anomalous_df)

    train_dataset = ICADDataset(staypoints_train_df, train_indices, include_anomalies=False)
    test_dataset = ICADDataset(staypoints_test_anomalous_df, test_indices, include_anomalies=True)
    train_loader, test_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, collate_fn=construct_batched_visit_collate_fn), DataLoader(test_dataset, args.eval_batch_size, shuffle=False, collate_fn=construct_batched_visit_collate_fn)


    return train_loader, test_loader, num_regions

    

def preprocess_numosim_visits():
    # Get the base directory of the project
    base_path = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}/data/NUMOSIM/LA/"

    # Load the dataset (removing first unnamed column)
    poi_df = pd.read_parquet(os.path.join(base_path, 'poi.parquet'), engine='fastparquet')
    staypoints_train_df = pd.read_parquet(os.path.join(base_path, 'stay_points_train.parquet'), engine='fastparquet')
    staypoints_test_truth_df = pd.read_parquet(os.path.join(base_path, 'stay_points_test_truth.parquet'), engine='fastparquet')
    staypoints_test_anomalous_df = pd.read_parquet(os.path.join(base_path, 'stay_points_test_anomalous.parquet'), engine='fastparquet')

    staypoints_train_df = process_space_time_numosim(staypoints_train_df, poi_df)
    staypoints_test_anomalous_df = process_space_time_numosim(staypoints_test_anomalous_df, poi_df)
    staypoints_test_truth_df = process_space_time_numosim(staypoints_test_truth_df, poi_df)

    # save preprocessed data    
    staypoints_train_df.to_parquet(os.path.join(base_path, 'preprocessed_stay_points_train.parquet'), engine='fastparquet')
    staypoints_test_truth_df.to_parquet(os.path.join(base_path, 'preprocessed_stay_points_test_truth.parquet'), engine='fastparquet')
    staypoints_test_anomalous_df.to_parquet(os.path.join(base_path, 'preprocessed_stay_points_test_anomalous.parquet'), engine='fastparquet')


def split_input_target(batch_dict):
    """Split batch dictionary into input and target dictionaries"""
    # Keeping the last token in input because we do teacher forcing for joint prediction
    input_dict = {key: batch_dict[key] for key in IN_FIELDS}
    target_dict = {key: batch_dict[key][:, 1:] for key in OUT_FIELDS if key in batch_dict.keys()}
    if 'anomaly' in target_dict:
        target_dict['anomaly'] = target_dict['anomaly'].int()
        target_dict['anomaly_type'] = target_dict['anomaly_type'].int()

    return input_dict, target_dict


def convert_batch_to_dict(batch):
    """Convert batch tensor to dictionary"""
    batch = {FIELDS[j]: batch[:, :, j] for j in range(batch.shape[-1])}
    batch['region_id'] = batch['region_id'].long()
    return batch

def convert_batch_to_model_io(batch, device):
    batch = batch.to(device)
    batch_dict = convert_batch_to_dict(batch)
    input_dict, target_dict = split_input_target(batch_dict)
    return input_dict, target_dict

if __name__ == "__main__":
    preprocess_numosim_visits()
