"""
This script reads hourly atmospheric data from a grib file and converts it into json format.

Each line in the resulting json files corresponds to one variable at one timestamp, and contains:
- the variable name
- the timestamp (as a string)
- the flattened list of values for the entire grid
- the units of the variable

The variables that are shifted backwards, are those variables that are cumulative across the hour, 
not instantaneous. In our case this is only total precipitation and surface latent heat flux. 
Information on whether variables are instantaneous are not can be found on the ECWMF ERA5 Reanalysis data store.

Data is saved in batches of 10,000 lines to prevent excessive memory usage. 
The grib file is assumed to contain 9 variables per timestamp block.

a separate lat/lon json file is also written the first time the coordinates are encountered.
"""

import pygrib
import pandas as pd
import numpy as np
import json

#print file path setup confirmation
print("setting path")
file_path_atmospheric = 'hourlyatmosphericdata2024/data.grib'

#open the grib file using pygrib
print("retrieving data")
grbs = pygrib.open(file_path_atmospheric)

#initialize containers for data and batching
data_list = []
time_block_limit = 9  #number of variables per time block
temp_list = []

count = 0  #counts how many variables are loaded within a block
lat_lon_count = 0  #ensures lat/lon is saved only once
index_count = 0  #total variable entries processed (across all blocks)

print("iteration begins")
for grb in grbs:
    
    #for precipitation variables, shift time by -1 hour
    if grb.name == '10 metre U wind component':
        prev_valid_time = grb.validDate  #reference time for precipitation correction
    
    if grb.name in ("Total precipitation", "Surface latent heat flux"):
        valid_time = prev_valid_time - pd.Timedelta(hours=1)  #adjust time for these variables
    else:
        valid_time = grb.validDate  #use original time for other variables
    
    variable = grb.name
    units = grb.units
    
    data = grb.values
    lats, lons = grb.latlons()
    
    #save latitude and longitude data only once
    if lat_lon_count == 0:
        latlon_dataframe = {
            'Latitude': np.array(lats.flatten(), dtype=np.float16).tolist(),
            'Longitude': np.array(lons.flatten(), dtype=np.float16).tolist(),
        }

        #write lat/lon data to json file
        with open("LATLON.json", "w") as outfile:
            json.dump(latlon_dataframe, outfile)
        lat_lon_count += 1
    
    #construct the json-friendly dictionary for this variable
    grib_dataframe = {
        'Variable': variable,
        'Time': valid_time.strftime('%Y-%m-%d %H:%M:%S'),
        'Value': np.array(data.flatten(), dtype=np.float16).tolist(),
        'Units': units
    }
    
    #decide where to store the variable depending on type and block position
    if count < time_block_limit and not variable in ("Total precipitation", "Surface latent heat flux"):
        temp_list.append(grib_dataframe)
        count += 1
    elif count < time_block_limit and variable in ("Total precipitation", "Surface latent heat flux"):
        data_list.append(grib_dataframe)
        count += 1
    
    #when block is full, flush temp_list into data_list
    if count >= time_block_limit:
        for elem in temp_list:
            data_list.append(elem)
        count = 0
        temp_list = []
    
    #determine index suffix for naming the output file
    current_list_index = (index_count // 9999) % 10
    index_count += 1

    #every 9999 entries, save a json file
    if index_count % 9999 == 0:
        print("saving")
        final_df = pd.DataFrame(data_list)
        final_df.to_json(f'hourlyatmosphericdata2024JSON_{current_list_index}.json', orient='records', lines=True)
        
        #clear data for the next batch
        data_list = []
        temp_list = []

#close the grib file after processing
grbs.close()
