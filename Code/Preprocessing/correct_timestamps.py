"""
This script processes multiple JSON files that each contain atmospheric variables 
and their associated timestamps. It should be run after the GRIB files have been 
converted into small JSON files in batches of 10,000 lines.

Each line in the input files must contain a "Time" field in the format 'YYYY-MM-DD HH:MM:SS'. 
This script replaces the human-readable time with a Unix timestamp (in milliseconds) 
and adds additional temporal features: the year, and sine/cosine encodings of the month 
to capture seasonality for machine learning models.

To use this script:
1. Add the list of input JSON files to the 'input_files' variable. These are the small JSON 
   batches produced after GRIB conversion.
2. Set the 'merged_output_file' to the desired output file name. This will store the 
   processed and merged data with corrected timestamps and added features.
"""

import json
import numpy as np
from datetime import datetime

#function to process a list of json files and merge them into one output file
def replace_time_values(input_files, output_file):
    with open(output_file, 'w') as outfile: 
        for input_file in input_files:
            print("filtering...", input_file)
            with open(input_file, 'r') as infile:
                for line in infile:
                    try:
                        #load the dictionary from the current line
                        data = json.loads(line.strip())
                        
                        #check and parse the time field
                        time_str = data.get("Time")
                        if time_str:
                            #convert string to datetime object
                            datetime_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                            year = datetime_obj.year
                            month = datetime_obj.month

                            #calculate sine and cosine encodings for the month
                            month_sin = np.sin(2 * np.pi * month / 12)
                            month_cos = np.cos(2 * np.pi * month / 12)
                            
                            #replace human-readable time with unix timestamp in milliseconds
                            data["Time"] = int(datetime_obj.timestamp() * 1000)

                            #add new fields to the dictionary
                            data["Year"] = year
                            data["Month_Sin"] = month_sin
                            data["Month_Cos"] = month_cos
                        
                        #write the updated dictionary to the output file
                        outfile.write(json.dumps(data) + '\n')

                    except json.JSONDecodeError as e:
                        print(f"error decoding json in file {input_file}: {e}")
                    except KeyError as e:
                        print(f"key error in file {input_file}: {e}")
                    except Exception as e:
                        print(f"unexpected error in file {input_file}: {e}")

#set the name of the output file that will contain all merged and processed data
merged_output_file = 'filteredHOURLYatmosphericdata2021to2024.json'

#list all input json files (each a small batch from the grib-to-json conversion)
input_files = [
    'hourlyatmosphericdata2022JSON_0.json', 'hourlyatmosphericdata2022JSON_1.json', 'hourlyatmosphericdata2022JSON_2.json',
    'hourlyatmosphericdata2022JSON_3.json', 'hourlyatmosphericdata2022JSON_4.json',
    'hourlyatmosphericdata2022JSON_5.json', 'hourlyatmosphericdata2022JSON_6.json',
    'hourlyatmosphericdata2022JSON_last.json',
    'hourlyatmosphericdata2023JSON_0.json', 'hourlyatmosphericdata2023JSON_1.json',
    'hourlyatmosphericdata2023JSON_2.json', 'hourlyatmosphericdata2023JSON_3.json',
    'hourlyatmosphericdata2023JSON_4.json', 'hourlyatmosphericdata2023JSON_5.json',
    'hourlyatmosphericdata2023JSON_6.json', 'hourlyatmosphericdata2023JSON_last.json',
    'hourlyatmosphericdata2024JSON_0.json', 'hourlyatmosphericdata2024JSON_1.json',
    'hourlyatmosphericdata2024JSON_2.json', 'hourlyatmosphericdata2024JSON_3.json',
    'hourlyatmosphericdata2024JSON_4.json', 'hourlyatmosphericdata2024JSON_5.json',
    'hourlyatmosphericdata2024JSON_last.json'
]

#run the function to process all input files and write the cleaned output
replace_time_values(input_files, merged_output_file)
