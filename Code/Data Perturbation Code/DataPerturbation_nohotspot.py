

"""This code should not be run alone. This code produces the perturbed data according to deforestation scenarios.
This is simply a snippet from the full pipeline for deforestation scenario analysis. 
The data preprocessing section in the MultiTask ConvLSTM pipeline can be replaced with this, with the correct vegetation 
dataset used, to perform scenario analysis. The model can then be tested on the perturbed data and the results analysed."""

import torch
import numpy as np
from tqdm import tqdm
import json

# Definition of log epsilon
epsilon = 1e-3

#Variable names
variable_names = ['10 metre U wind component', '10 metre V wind component', '2 metre dewpoint temperature', '2 metre temperature', 'UV visible albedo for direct radiation (climatological)', 'Total column rain water', 'Volumetric soil water layer 1', 'Leaf area index, high vegetation', 'Leaf area index, low vegetation', 'Forecast surface roughness', 'Total precipitation', 'Time-integrated surface latent heat net flux', 'Evaporation']

vegetation_variable_indices = [
    variable_names.index('Leaf area index, high vegetation'),
    variable_names.index('Leaf area index, low vegetation'),
    variable_names.index('Forecast surface roughness'),
    variable_names.index('Evaporation'),
    variable_names.index('UV visible albedo for direct radiation (climatological)'),
    variable_names.index('Volumetric soil water layer 1')
]

with open('globaldatastatistics_withVEG.json', 'r') as f:
    global_stats = json.load(f)

import torch
import numpy as np

# Definition of log epsilon
epsilon = 1e-3
batch_size = 16

min_lai_threshold = 0

vegetation_variable_indices = [
    variable_names.index('Leaf area index, high vegetation'),
    variable_names.index('Leaf area index, low vegetation'),
    variable_names.index('Forecast surface roughness'),
    variable_names.index('Evaporation'),
    variable_names.index('UV visible albedo for direct radiation (climatological)'),
    variable_names.index('Volumetric soil water layer 1')
]

def add_accelerating_trend(time_series, mean, std, min_value, max_value, batch_count, rate=0.000003, variable_name=None):
    """Gradually reduce vegetation indices OR increase UV Albedo (small step)."""
    batch_size, time_steps, grid_points = time_series.shape  # Get dimensions
    global_time = np.arange(batch_count, (batch_count) + time_steps).reshape(1, time_steps, 1)

    range_factor = (max_value - min_value)/2
    trend = rate * global_time * range_factor  # Shape becomes (1, time_steps, 1)
    
    # Ensure trend is **broadcast correctly** across batch/grid points
    trend = np.broadcast_to(trend, (batch_size, time_steps, grid_points))

    perturbed_values = time_series + trend

    # If the variable is 'Volumetric soil water layer 1', ensure it does not drop below 0
    if variable_name == "Volumetric soil water layer 1":
        perturbed_values = np.maximum(perturbed_values, 0)

    return perturbed_values

# Function to normalize using z-score
def z_score_normalisation(batch, mean, std, epsilon=1e-6):
    return (batch - mean) / (std + epsilon)

# Function to normalize using min-max scaling
def min_max_scaling(batch, min_value, max_value, epsilon=1e-6):
    return (batch - min_value) / (max_value - min_value + epsilon)

# Function to apply log transformation
def log_normalisation(batch, epsilon=1e-3):
    return torch.log(batch + epsilon)

def handle_nans_and_infs(batch, variable_name, global_stats):
    max_value = global_stats[variable_name].get('max', 1e6)  
    min_value = global_stats[variable_name].get('min', 0)
    batch = torch.nan_to_num(batch, nan=min_value, posinf=max_value, neginf=-max_value)
    return batch

def preprocess_data(data_loader, variable_names, global_stats, name, perturbation_type):
    """
    Preprocess dataset with normalization and optional perturbations.
    """
    log_variables = ['Total precipitation', 'Total column rain water', 'Volumetric soil water layer 1']
    log_stats = {var: {'logs': []} for var in log_variables}

    # ---- First pass: Compute log mean and std for specified variables ----

    first_count = 0
    with torch.no_grad():
        for X_batch, _ in tqdm(data_loader, desc=f"Computing log stats for variables"):

            if first_count == len(data_loader) - 2:
                break 

            first_count += 1

            X_batch = X_batch.clone()

            for var in log_variables:
                var_idx = variable_names.index(var)
                X_var = X_batch[:, :, var_idx]
                if var == 'Total precipitation':  # Convert precipitation to mm
                    X_var = X_var * 1000

                log_X_var = log_normalisation(X_var, epsilon)
                log_stats[var]['logs'].append(log_X_var.flatten().cpu().numpy())

    # Calculate mean and std for log-transformed variables
    for var in log_variables:
        logs_all = np.concatenate(log_stats[var]['logs']) if log_stats[var]['logs'] else np.array([0])
        mean_log = np.mean(logs_all)
        std_log = np.std(logs_all)
        global_stats.setdefault(var, {})
        global_stats[var]['mean_log'] = mean_log
        global_stats[var]['std_log'] = std_log

    second_count = 0
    # ---- Second pass: Normalize and perturb vegetation indices ----
    normalized_batches = []
    for X_batch, y_batch in tqdm(data_loader, desc=f"Preprocessing {name.capitalize()} Data"):

        if second_count == len(data_loader) - 2:
            break 

        second_count +=1

        X_batch = X_batch.clone()
        y_batch = y_batch.clone()

        # Index for total precipitation
        precip_idx = variable_names.index('Total precipitation')
        X_precip_mm = X_batch[:, :, precip_idx] * 1000  # Convert to mm
        zero_indicator = (X_batch[:, :, precip_idx] == 0).float()
        y_zero_indicator = (y_batch == 0).float()

        # Log-normalize precipitation
        X_batch[:, :, precip_idx] = log_normalisation(X_precip_mm, epsilon)

        # Normalize & Perturb Vegetation Variables
        for var_idx in vegetation_variable_indices:
            variable_name = variable_names[var_idx]
            mean = global_stats[variable_name]['mean']
            std = global_stats[variable_name]['std']
            min_val = global_stats[variable_name]['min']
            max_val = global_stats[variable_name]['max']

            # **Apply perturbation only to vegetation variables**
            values = X_batch[:, :, var_idx].cpu().numpy()

            # UV Albedo increases, others decrease
            perturbation_sign = -1 if "UV visible albedo" in variable_name else 1

            if perturbation_type == 'accelerating':
                values = add_accelerating_trend(values, mean, std, min_val, max_val, second_count, rate=perturbation_sign * 0.000003, variable_name=variable_name)

            # Convert back to tensor
            X_batch[:, :, var_idx] = torch.tensor(values, dtype=torch.float32).to(X_batch.device)

                # Count NaNs and infinite values
        num_nans = torch.isnan(X_batch).sum().item()
        num_infs = torch.isinf(X_batch).sum().item()
        
        if (num_nans > 0) or (num_infs > 0):
            print("before")
            print(num_nans, num_infs)

        for var_idx, variable_name in enumerate(variable_names):
            
            if variable_name == 'Total column rain water' or variable_name == 'Volumetric soil water layer 1':
                # Log-normalize without standardizing
                X_batch[:, :, var_idx] = log_normalisation(X_batch[:, :, var_idx], epsilon)

            elif variable_name == 'Leaf area index, high vegetation' or variable_name == 'Leaf area index, low vegetation':
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx], mean, std)

            elif variable_name == 'Forecast surface roughness':
                min_value = global_stats[variable_name]['min']
                max_value = global_stats[variable_name]['max']
                X_batch[:, :, var_idx] = min_max_scaling(X_batch[:, :, var_idx], min_value, max_value)

            elif variable_name == 'Evaporation':
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx] * 1000, mean, std)

            elif variable_name == 'Time-integrated surface latent heat net flux':
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx] / 10800, mean, std)

            elif variable_name == 'UV visible albedo for direct radiation (climatological)':
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx], mean, std)

            elif variable_name in ['2 metre temperature', '2 metre dewpoint temperature']:
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx], mean, std)

            elif variable_name in ['10 metre U wind component', '10 metre V wind component']:
                mean = global_stats[variable_name]['mean']
                std = global_stats[variable_name]['std']
                X_batch[:, :, var_idx] = z_score_normalisation(X_batch[:, :, var_idx], mean, std)

            # Count NaNs and infinite values
            num_nans = torch.isnan(X_batch).sum().item()
            num_infs = torch.isinf(X_batch).sum().item()

            if (num_nans > 0) or (num_infs > 0):
                print(variable_name)
                print(num_nans, num_infs)
                X_batch[:, :, var_idx]  = handle_nans_and_infs(X_batch[:, :, var_idx], variable_name, global_stats)

                # Count NaNs and infinite values
        num_nans = torch.isnan(X_batch).sum().item()
        num_infs = torch.isinf(X_batch).sum().item()

        if (num_nans > 0) or (num_infs > 0):
            print("after")
            print(num_nans, num_infs)

        # Keep the target precipitation in millimeters (no normalization)
        y_batch = y_batch * 1000

        # Add zero indicator to input features
        X_batch = torch.cat((X_batch, zero_indicator.unsqueeze(2)), dim=2)

        # Append normalized batch
        normalized_batches.append((X_batch, y_batch, y_zero_indicator))

    return normalized_batches

# # Precompute perturbed data
perturbation_type = 'accelerating' 

# There are three data loaders in the main pipeline: train_loader, val_loader and test_loader. This is when validation data is used separate to the training data. 
# For 5-fold-cross-validation, the two should be combined. Change the name of the loader in the following to preprocess all data.
normalized_test_data = preprocess_data(test_loader, variable_names, global_stats, "test", perturbation_type)