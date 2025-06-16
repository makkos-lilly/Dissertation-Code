
"""This code should not be run alone. This code produces the perturbed data according to deforestation scenarios.
This is simply a snippet from the full pipeline for deforestation scenario analysis. 
The data preprocessing section in the MultiTask ConvLSTM pipeline can be replaced with this, with the correct vegetation 
dataset used, to perform scenario analysis. The model can then be tested on the perturbed data and the results analysed."""

import torch
import numpy as np
from scipy.spatial.distance import cdist
import json
import tqdm

#Variable names
variable_names = ['10 metre U wind component', '10 metre V wind component', '2 metre dewpoint temperature', '2 metre temperature', 'UV visible albedo for direct radiation (climatological)', 'Total column rain water', 'Volumetric soil water layer 1', 'Leaf area index, high vegetation', 'Leaf area index, low vegetation', 'Forecast surface roughness', 'Total precipitation', 'Time-integrated surface latent heat net flux', 'Evaporation']

with open('globaldatastatistics_withVEG.json', 'r') as f:
    global_stats = json.load(f)

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

# Function to normalize using z-score
def z_score_normalisation(batch, mean, std, epsilon=1e-6):
    return (batch - mean) / (std + epsilon)

# Function to normalize using min-max scaling
def min_max_scaling(batch, min_value, max_value, epsilon=1e-6):
    return (batch - min_value) / (max_value - min_value + epsilon)

# Function to apply log transformation
def log_normalisation(batch, epsilon=1e-3):
    return torch.log(batch + epsilon)

# *Precompute Gaussian Spatial Decay Mask
def precompute_gaussian_decay(grid_shape, hotspot_indices, sigma=2):
    """
    Compute a Gaussian decay mask centered around deforestation hotspots.
    The closer to a hotspot, the stronger the effect.
    """
    lat_lon_grid = np.indices(grid_shape).reshape(2, -1).T
    distances = cdist(lat_lon_grid, hotspot_indices, metric='euclidean')
    min_distances = distances.min(axis=1).reshape(grid_shape)

    # Gaussian decay
    decay_mask = sigma*(np.exp(-min_distances**2 / (2 * sigma**2)))
    decay_mask = np.clip(decay_mask, 0.2, 1.0)  # Prevent complete removal
    return decay_mask

# Load Manual Hotspots
def load_manual_hotspots(hotspot_file):
    """Load deforestation hotspots from a JSON file."""
    with open(hotspot_file, 'r') as f:
        hotspot_data = json.load(f)
    return np.array([(entry["grid_row"], entry["grid_col"]) for entry in hotspot_data])

def add_accelerating_trend(time_series, mean, std, min_value, max_value, batch_count, decay_mask, rate=0.000004, variable_name=None):
    """ 
    Gradually reduce vegetation indices OR increase UV Albedo (scaled by decay mask). 
    The perturbation is applied **proportionally** to how close the grid point is to the deforestation hotspot.
    """
    batch_size, time_steps, grid_points = time_series.shape  # Get dimensions
    global_time = batch_count

    # Scale perturbation by (max-min)/2
    range_factor = (max_value - min_value) / 2

    # Apply decay mask per grid point and broadcast across batch/time
    decay_mask = decay_mask.reshape(1, 1, grid_points)  # Reshape for broadcasting

    # Adjust perturbation rate for volumetric soil water layer
    adjusted_rate = rate if variable_name not in ["Volumetric soil water layer 1"] else rate / 100

    trend = adjusted_rate * global_time * range_factor * decay_mask  # Apply hotspot-dependent scaling

    # Ensure correct broadcasting across batch
    trend = np.broadcast_to(trend, (batch_size, time_steps, grid_points))

    perturbed_values = time_series + trend

    # Ensure volumetric soil water layer does not drop below 0
    if variable_name == "Volumetric soil water layer 1":
        perturbed_values = np.maximum(perturbed_values, 0)

    if variable_name == 'Leaf area index, low vegetation':
        perturbed_values = np.maximum(perturbed_values, 0)

    return perturbed_values

def apply_hotspot_perturbation(time_series, perturbation_type, mean, std, min_value, max_value, batch_count, decay_mask, variable_name):
    """
    Apply perturbations relative to the decay mask from deforestation hotspots.
    - `accelerating`: Gradual decline, scaled by decay mask
    """
    if perturbation_type == 'accelerating':
        return add_accelerating_trend(time_series=time_series, mean=mean, std=std, min_value=min_value, max_value=max_value, batch_count = batch_count, decay_mask=decay_mask, variable_name=variable_name)
    
    return time_series 

# preprocessing Pipeline with Hotspot-Based Perturbation
def preprocess_data(data_loader, variable_names, global_stats, name, perturbation_type='accelerating', hotspot_file=None):
    """
    Updated preprocessing pipeline that:
    1. Loads hotspots and computes a **spatial decay mask**.
    2. Perturbs vegetation indices **proportionally** to distance from deforestation hotspots.
    3. Normalizes all variables before training.
    """
    hotspot_indices = load_manual_hotspots(hotspot_file)
    grid_shape = (81, 97)
    decay_mask = precompute_gaussian_decay(grid_shape, hotspot_indices, sigma=2)

    second_count = 0
    normalized_batches = []

    for X_batch, y_batch in tqdm(data_loader, desc=f"Preprocessing {name.capitalize()} Data"):
        if second_count == len(data_loader) - 2:
            break  
        second_count += 1

        X_batch = X_batch.clone()
        y_batch = y_batch.clone()

        precip_idx = variable_names.index('Total precipitation')
        X_precip_mm = X_batch[:, :, precip_idx] * 1000  # Convert to mm
        zero_indicator = (X_batch[:, :, precip_idx] == 0).float()
        y_zero_indicator = (y_batch == 0).float()
        X_batch[:, :, precip_idx] = log_normalisation(X_precip_mm, epsilon)

        for var_idx in vegetation_variable_indices:
            variable_name = variable_names[var_idx]
            mean = global_stats[variable_name]['mean']
            std = global_stats[variable_name]['std']
            min_value = global_stats[variable_name]['min']
            max_value = global_stats[variable_name]['max']

            values = X_batch[:, :, var_idx].cpu().numpy()

            values = apply_hotspot_perturbation(values, perturbation_type, mean, std, min_value, max_value, second_count, decay_mask, variable_name)
            X_batch[:, :, var_idx] = torch.tensor(values, dtype=torch.float32).to(X_batch.device)

        for var_idx, variable_name in enumerate(variable_names):
            if variable_name in ['Total column rain water', 'Volumetric soil water layer 1']:
                X_batch[:, :, var_idx] = log_normalisation(X_batch[:, :, var_idx], epsilon)
            elif variable_name in ['Leaf area index, high vegetation', 'Leaf area index, low vegetation']:
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

        num_nans = torch.isnan(X_batch).sum().item()
        num_infs = torch.isinf(X_batch).sum().item()

        if (num_nans > 0) or (num_infs > 0):
            print("after")
            print(num_nans, num_infs)

        y_batch = y_batch * 1000  # Keep precipitation in mm
        X_batch = torch.cat((X_batch, zero_indicator.unsqueeze(2)), dim=2)
        normalized_batches.append((X_batch, y_batch, y_zero_indicator))

    return normalized_batches

hotspot_file = "hotspots.json"
perturbation_type = 'accelerating' 

# There are three data loaders in the main pipeline: train_loader, val_loader and test_loader. This is when validation data is used separate to the training data. 
# For 5-fold-cross-validation, the two should be combined. Change the name of the loader in the following to preprocess all data.
normalized_test_data = preprocess_data(test_loader, variable_names, global_stats, "test", perturbation_type, hotspot_file)