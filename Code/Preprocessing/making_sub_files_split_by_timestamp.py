"""
This script splits a large JSON file into smaller batch-wise files based on unique timestamps, 
and then generates training, validation, and test data with a configurable lookback and forecast horizon.

The 'line_count == 13' line must be changed for training on only the atmopsheric variables (without the vegetation variables). 
In that case it must be changed to 'line_count == 7'. If the train or validation ratio is changed in this file, it must 
be changed correspondingly in the pipeline.

The variables 'output_dir', 'timestamps_file', 'file_directory', and 'file_name' must be changed based on your setup:

- 'file_directory' and 'file_name' should point to the location of the full JSON file containing all variable entries.
- 'output_dir' should point to the directory where the batch-wise split files will be saved.
- 'timestamps_file' refers to the JSON file that will store all extracted timestamps and should match your saved output path.

Ensure that the output directories exist, or they will be created by the script.

In this code, there are three blocks of repeated code. These are not automised to ensure complete control of erroneous files 
in the dataset. There are 3 erronous files (that are not found) and are skipped over. This can be automised with caution.
"""

import numpy as np
import json
import pandas as pd
# from tensorflow.keras.metrics import MeanSquaredError
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from sklearn.model_selection import KFold 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from scipy.stats import kendalltau
import seaborn as sns
from tqdm import tqdm
import os
from datetime import datetime
from collections import defaultdict


#-----------------------------------------------------------------------------------------------------------------------------


#define the directory and filename for the full json dataset
file_directory = 'Perturbed Vegetation Data Constant/'
file_name ='perturbed_accelerating_consistent.json'

#construct the full path to the dataset file
file_path = os.path.join(file_directory, file_name)

#specify the output directory for saving split files
output_dir = 'split_files_cleaned/'  #ensure this directory exists or is created below
timestamps_file = 'alltimestamps_cleaned.json'  #name of the file that will store all timestamps

#initialize a set to store all unique timestamps
alltimestamps = set()

#create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

#initialize counters for batching
line_count = 0  #tracks how many lines have been added to the current batch
file_count = 1  #tracks how many output files have been written
batch = []  #holds the current batch of data

#open the large input file and process it line by line
with open(file_path, 'r') as infile:
    for line in tqdm(infile, desc="Processing lines"):
        #parse the line into a dictionary
        data = json.loads(line.strip())

        #extract the timestamp
        timestamp = data["Time"]

        #add the data to the current batch
        batch.append(data)
        line_count += 1

        #store the timestamp in the set of all timestamps
        alltimestamps.add(timestamp)

        #once 13 lines (i.e. variables for one timestamp) are collected, save to a new file
        if line_count == 13:
            #create the output filename using the file count and timestamp
            output_file = os.path.join(output_dir, f"{file_count}_{timestamp}.json")
            
            #write the current batch to the output file
            with open(output_file, 'w') as outfile:
                json.dump(batch, outfile, indent=2)
            
            #reset the batch and line counter for the next file
            batch = []
            line_count = 0
            file_count += 1

    #if there are any leftover lines that didn't form a complete batch, save them too
    if batch:
        output_file = f"{output_dir}{file_count}.json"
        with open(output_file, 'w') as outfile:
            json.dump(batch, outfile, indent=2)

#write all collected timestamps to a file for later use
timestamp_file = os.path.join(output_dir, timestamps_file)
with open(timestamp_file, 'w') as file:
    json.dump(list(alltimestamps), file)

#print confirmation of completion and the number of files written
print("Files created successfully.")
print(file_count - 1)


#-----------------------------------------------------------------------------------------------------------------------------

#set the directory containing the input json files (split by timestamp)
output_dir = 'split_files_cleaned/'

#set the directory to save the processed .npz files containing input-output training batches
output_dir_saved_files = 'lookback_and_lookahead_files_cleaned/'

#function to load json files and split them into supervised training samples
def load_and_split_data_with_lookback_and_forecast(timestamp_directory, saved_files_directory, train_timestamps, val_timestamps, test_timestamps, lookback, forecast_horizon):
    
    print(lookback)
    print(forecast_horizon)
    
    #initialize containers for different dataset splits
    train_inputs, train_targets = [], []
    val_inputs, val_targets = [], []
    test_inputs, test_targets = [], []

    input_data_combined = []  #holds the final stacked input sequence for one sample
    next_precipitation_data = []  #holds the stacked future target sequence

    #set working directories
    output_dir = timestamp_directory
    output_dir_saved_files = saved_files_directory

    #create directory if it doesn't exist
    os.makedirs(output_dir_saved_files, exist_ok=True)
    
    #load the list of all available timestamps
    timestamps_file = 'alltimestamps_cleaned.json'
    timestamp_file = os.path.join(output_dir, timestamps_file)

    with open(timestamp_file, 'r') as file:
        all_timestamps = json.load(file)

    #sort the timestamps chronologically
    sorted_timestamps = sorted(all_timestamps)

    print("timestamps are now sorted")

    #loop over the available timestamps using a sliding window approach
    for i in tqdm(range(lookback, len(sorted_timestamps) - forecast_horizon), desc="Processing data with lookback and horizon"):
        current_timestamp = sorted_timestamps[i]

        #get timestamps for the lookback window
        previous_timestamps = sorted_timestamps[i - lookback:i]

        #get timestamps for the forecast horizon
        target_timestamps = sorted_timestamps[i:i + forecast_horizon]

        #initialize a list to collect input data from lookback steps
        input_data_sequence = []
        for idx, prev_timestamp in enumerate(previous_timestamps):
            file_index = i - lookback + idx + 1
            file_name = f"{file_index}_{prev_timestamp}.json"
            file_path = os.path.join(output_dir, file_name)
            
            current_data = []
            try:
                #try to load the file for the previous timestamp
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    #exclude precipitation variables from the input
                    for variable in data:
                        if variable["Variable"] not in ('Convective precipitation', 'Large-scale precipitation'):
                            values = variable["Value"]
                            current_data.append(values)
                if current_data != []:
                    input_data_sequence.append(np.stack(current_data).astype(np.float32))
            except Exception:
                #if file not found, try adding 1 to index and retry up to 3 times
                print(f"File not found: {file_path}")
                print("Adding one to the file indices")
                for attempt in range(3):
                    try:
                        current_data = []
                        file_index += 1
                        file_name = f"{file_index}_{prev_timestamp}.json"
                        file_path = os.path.join(output_dir, file_name)
                        with open(file_path, 'r') as infile:
                            data = json.load(infile)
                            for variable in data:
                                if variable["Variable"] not in ('Convective precipitation', 'Large-scale precipitation'):
                                    values = variable["Value"]
                                    current_data.append(values)
                        if current_data != []:
                            input_data_sequence.append(np.stack(current_data).astype(np.float32))
                            break
                    except Exception:
                        print(f"file not found again: {file_path} even with adding {attempt + 1} to the index")

        #combine the lookback input sequence
        input_data_combined = np.stack(input_data_sequence)

        #initialize list to collect precipitation targets
        future_precipitation = []
        for idx, target_timestamp in enumerate(target_timestamps):
            file_index = i + idx + 1
            file_name = f"{file_index}_{target_timestamp}.json"
            file_path = os.path.join(output_dir, file_name)
            
            current_data = []
            try:
                #load precipitation from target timestamps
                with open(file_path, 'r') as infile:
                    data = json.load(infile)
                    for variable in data:
                        if variable["Variable"] == "Total precipitation":
                            print("TRUE")
                            values = variable["Value"]
                            current_data.append(values)
                if current_data != []:
                    future_precipitation.append(np.stack(current_data).astype(np.float32))
            except Exception:
                #if file not found, try adding 1 to index and retry up to 3 times
                print(f"File not found: {file_path}")
                print("Adding one to the file indices")
                for attempt in range(3):
                    try:
                        current_data = []
                        file_index += 1
                        file_name = f"{file_index}_{target_timestamp}.json"
                        file_path = os.path.join(output_dir, file_name)
                        with open(file_path, 'r') as infile:
                            data = json.load(infile)
                            for variable in data:
                                if variable["Variable"] == "Total precipitation":
                                    values = variable["Value"]
                                    current_data.append(values)
                        if current_data != []:
                            future_precipitation.append(np.stack(current_data).astype(np.float32))
                            break
                    except Exception:
                        print(f"file not found again: {file_path} even with adding {attempt + 1} to the index")

        #combine the forecasted precipitation data
        next_precipitation_data = np.stack(future_precipitation)

        #save the processed data based on split type
        if current_timestamp in train_timestamps:
            save_file_index = i
            save_file_timestamp = current_timestamp

            train_inputs = input_data_combined
            train_targets = next_precipitation_data

            save_file_name = os.path.join(output_dir_saved_files, f'{save_file_index}_{save_file_timestamp}.npz')

            np.savez(save_file_name, X_batches=train_inputs, y_batches=train_targets, allow_pickle=True)

        elif current_timestamp in val_timestamps:
            save_file_index = i
            save_file_timestamp = current_timestamp

            val_inputs = input_data_combined
            val_targets = next_precipitation_data

            save_file_name = os.path.join(output_dir_saved_files, f'{save_file_index}_{save_file_timestamp}.npz')

            np.savez(save_file_name, X_batches=val_inputs, y_batches=val_targets, allow_pickle=True)

        elif current_timestamp in test_timestamps:
            save_file_index = i
            save_file_timestamp = current_timestamp

            test_inputs = input_data_combined
            test_targets = next_precipitation_data

            save_file_name = os.path.join(output_dir_saved_files, f'{save_file_index}_{save_file_timestamp}.npz')

            np.savez(save_file_name, X_batches=test_inputs, y_batches=test_targets, allow_pickle=True)

        #clear data to prepare for next iteration
        input_data_combined = []
        next_precipitation_data = []

        train_inputs, train_targets = [], []
        val_inputs, val_targets = [], []
        test_inputs, test_targets = [], []

#-----------------------------------------------------------------------------

#custom function to split timestamps into train, validation, and test sets
def split_timestamps_custom(data, train_ratio=0.6, val_test_ratio=0.5):
    data = sorted(data)  #ensure timestamps are sorted

    total_len = len(data)
    train_size = int(total_len * train_ratio)  #calculate training set size

    val_size = int((total_len - train_size) * val_test_ratio)  #calculate validation size

    print(val_size)
    print(train_size)

    test_size = total_len - train_size - val_size  #remaining timestamps go to test set

    #slice the data into splits
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    return train_data, val_data, test_data

#function to load all timestamps from a json file and split them
def split_timestamps(directory, file_name):
    all_timestamps = []

    file_path = os.path.join(directory, file_name)

    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        all_timestamps = json.loads(first_line)

    all_timestamps = sorted(all_timestamps)  #ensure timestamps are in order

    #split into training, validation, and testing
    train_timestamps, val_timestamps, test_timestamps = split_timestamps_custom(all_timestamps)
    print("Here are the training, validation and test timestamps for my data")

    #save the split timestamps to file
    np.savez('timestamps_splits_cleaned.npz', train=train_timestamps, val=val_timestamps, test=test_timestamps)

    return train_timestamps, val_timestamps, test_timestamps

#run the timestamp split and store the results
train_timestamps, val_timestamps, test_timestamps = split_timestamps(directory=output_dir, file_name='alltimestamps_cleaned.json')

print("timestamps have split")

#set the lookback and forecast horizon used for splitting
lookback = 1
forecast_horizon = 1

#run the loader and split processor
load_and_split_data_with_lookback_and_forecast(output_dir, output_dir_saved_files, train_timestamps, val_timestamps, test_timestamps, lookback, forecast_horizon)

#set paths for later utilities
timestamps_directory = 'split_files_cleaned/'
timestamps_file_path = os.path.join(timestamps_directory, 'alltimestamps_cleaned.json')
saved_files = 'lookback_and_lookahead_files_cleaned/'  #directory where npz files are stored
split_file = 'timestamps_splits_cleaned.npz'  #file containing train, val, test splits

#utility to load and sort all timestamps
def load_all_timestamps():
    with open(timestamps_file_path, 'r') as file:
        timestamps = json.load(file)
        sorted_timestamps = sorted(timestamps)
        return sorted_timestamps

#utility to get lengths of each data split
def load_split_lengths():
    loaded_data = np.load(split_file)
    train_len = len(loaded_data['train'])
    val_len = len(loaded_data['val'])
    return train_len, train_len + val_len  #returns train length and cumulative train+val

#load a single training sample by index
def load_singular_train_data(index, lookback):
    all_timestamps = load_all_timestamps()
    timestamp_name = all_timestamps[lookback + index]
    file_name = f'{index + lookback}_{timestamp_name}.npz'
    file_path = os.path.join(saved_files, file_name)
    
    print(file_path)
    data = np.load(file_path, allow_pickle=True)
    return data['X_batches'], data['y_batches']

#load a single validation sample by index
def load_singular_val_data(index, lookback):
    all_timestamps = load_all_timestamps()
    train_len, _ = load_split_lengths()
    timestamp_name = all_timestamps[lookback + train_len + index]
    file_name = f'{index + train_len + lookback}_{timestamp_name}.npz'
    file_path = os.path.join(saved_files, file_name)
    
    data = np.load(file_path, allow_pickle=True)
    return data['X_batches'], data['y_batches']

#load a single test sample by index
def load_singular_test_data(index, lookback):
    all_timestamps = load_all_timestamps()
    train_len, train_val_len = load_split_lengths()
    timestamp_name = all_timestamps[lookback + train_val_len + index]
    file_name = f'{index + train_val_len + lookback}_{timestamp_name}.npz'
    file_path = os.path.join(saved_files, file_name)
    
    try:
        data = np.load(file_path, allow_pickle=True)
    except:
        print("The last erroneous files don't exist")

    return data['X_batches'], data['y_batches']
