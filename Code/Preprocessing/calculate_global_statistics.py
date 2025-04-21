"""
this script calculates global statistics (min, max, mean, and standard deviation) 
for each variable in a large json dataset, and simultaneously outputs a cleaned version 
of the dataset where null values have been handled.

each line in the input json file should be a separate json object with at least two keys:
  - 'Variable': a string indicating the variable name
  - 'Value': a list of numerical values (which may include nulls or missing values)

null values are replaced with 0, and all values are cast to float for consistency.

---------------------
How to use this script
---------------------

1. Make sure your merged json file (with one dictionary per line) is ready and stored in the same directory.
2. Replace the 'merged_output_file' and 'cleaned_output_file' variables below with the actual names of:
    - the file containing all data you want to process (`merged_output_file`)
    - the file you want to save the cleaned output to (`cleaned_output_file`)
3. Run the script. it will:
    - clean nulls from the input data
    - calculate global statistics for each variable
    - save cleaned data into a new json file
    - print min, max, mean, and standard deviation for each variable
"""

import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np

#handle null values in the values list
def handle_null_values(values):
    return [0 if (v == "null" or v == "None" or v is None or np.isnan(v)) else v for v in values]

#calculate global stats 
def calculate_global_stats(merged_file, cleaned_output_file):
    variable_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0, 'sum_of_squares': 0})

    #count total lines for progress bar
    with open(merged_file, 'r') as infile:
        total_lines = sum(1 for _ in infile)

    #open the file again and process with tqdm
    with open(merged_file, 'r') as infile, open(cleaned_output_file, 'w') as outfile, tqdm(total=total_lines, desc="calculating global stats and cleaning data", unit="lines") as pbar:
        for line in infile:
            try:
                data = json.loads(line.strip())
                
                if 'Variable' not in data or 'Value' not in data:
                    print(f"skipping line due to missing keys: {line}")
                    continue

                var_name = data['Variable']
                values = data['Value']

                #handle null values and overwrite values with cleaned values
                cleaned_values = handle_null_values(values)
                data['Value'] = list(map(float, cleaned_values))  #convert all cleaned values to float

                #write cleaned data to the new output file
                outfile.write(json.dumps(data) + '\n')

                #update global stats
                variable_stats[var_name]['min'] = min(variable_stats[var_name]['min'], min(cleaned_values))
                variable_stats[var_name]['max'] = max(variable_stats[var_name]['max'], max(cleaned_values))
                variable_stats[var_name]['sum'] += sum(cleaned_values)
                variable_stats[var_name]['count'] += len(cleaned_values)
                variable_stats[var_name]['sum_of_squares'] += sum([v**2 for v in cleaned_values])

            except json.JSONDecodeError as e:
                print(f"error decoding json: {e}")
            except KeyError as e:
                print(f"key error in json: {e}")
            except Exception as e:
                print(f"unexpected error: {e}")

            #update the progress bar
            pbar.update(1)

    #calculate mean and std for each variable
    for var_name, stats in variable_stats.items():
        if stats['count'] > 0:
            stats['mean'] = stats['sum'] / stats['count']
            stats['std'] = np.sqrt((stats['sum_of_squares'] / stats['count']) - (stats['mean'] ** 2))
        else:
            stats['mean'] = 0
            stats['std'] = 1 

    return variable_stats

#change these file names to your own files
merged_output_file = 'CLEANEDfilteredHOURLYatmosphericdata2021to2023.json'
cleaned_output_file = 'FINAL_CLEANED_HOURLYatmosphericdata2021to2023.json'

#calculate global min and max for each variable and write cleaned data
variable_stats = calculate_global_stats(merged_output_file, cleaned_output_file)

#display the min, max, mean, and std for each variable
for variable, stats in variable_stats.items():
    print(f"variable: {variable}, min: {stats['min']}, max: {stats['max']}, mean: {stats['mean']:.2f}, std: {stats['std']:.2f}")
