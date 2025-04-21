PROJECT OVERVIEW  
-----------------

This repository contains the code necessary to run experiments for precipitation prediction models, as outlined in the attached Final Report.

REPOSITORY STRUCTURE  
---------------------

- ReadMe.txt  
  This file.

- environment.yml  
  Contains all libraries required to run the code.

- Data/  
  Contains the hourly atmospheric and vegetation data from 2021 to 2023.
    - FINAL_CLEANED_HOURLYatmosphericdata2021to2023.json

- Data Perturbation Code/  
  Contains preprocessing functions for hotspot and non-hotspot deforestation experiments.
    - DataPerturbation_hotspot.py
    - DataPerturbation_nohotspot.py

- Global Statistics/  
  Supporting files for training and inference, including global data statistics and spatial masks.
    - globaldatastatistics.json
    - globaldatastatistics_withVEG.json
    - grid_min_max_full.json
    - hotspots.json
    - LATLON.json

- Metadata/  
  Contains geographic metadata of the Brazilian Legal Amazon boundary.
    - brazilian_amazon_legal_boundary_coordinates.geojson

- Models/  
  Contains model architectures used in this study.
    - CNN.py
    - ConvLSTM.py
    - ConvLSTM_2CELL.py
    - ConvLSTM_PERIODIC.py
    - DConvLSTM_SAC.py
    - LSTM.py
    - MLP.py

- Pipelines/  
  Jupyter notebooks for model training and analysis.
    - DConvLSTM-SAC_Full_Pipeline.ipynb
    - EDA.ipynb
    - MultiTaskConvLSTM_Full_Pipeline.ipynb
    - PGD_Full_Pipeline.ipynb
    - TConvLSTM_FULL_PIPELINE.ipynb

- Preprocessing/  
  Scripts for converting GRIB data into .npz format.
    - calculate_global_statistics.py
    - correct_timestamps.py
    - making_sub_files_split_by_timestamp.py
    - timestamps.npz
    - timestamps_splits.npz
    - transform_from_grib.py

RETRIEVING DATA  
---------------

Raw atmospheric data can be downloaded from the Copernicus ERA5 reanalysis dataset: 
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=documentation
Select "Reanalysis" across the spatial region and variables outlined in the Final Report.

For vegetation and atmospheric data, use the file FINAL_CLEANED_HOURLYatmosphericdata2021to2023.json.  
After downloading, run the script making_sub_files_split_by_timestamp.py, setting the number of variables to 13, and setting lookback and lookahead to 1.

The 2024 data must be downloaded separately.

If running the models without vegetation variables, remove vegetation variables during preprocessing in the pipeline notebooks using the exclude_variables variable. Do not change the all_variables variable.

RUNNING THE CODE  
-----------------

If starting from raw GRIB data:

1. Place all files in the same directory.
2. Convert each GRIB file to JSON using transform_from_grib.py.
3. Correct timestamps and merge JSON files using correct_timestamps.py.
4. Generate .npz files using making_sub_files_split_by_timestamp.py.

If starting from the preprocessed JSON:

- Run making_sub_files_split_by_timestamp.py directly on FINAL_CLEANED_HOURLYatmosphericdata2021to2023.json.

RUNNING THE PIPELINES  
----------------------

- MultiTaskConvLSTM_Full_Pipeline.ipynb  
  Used to train and evaluate the MultiTask ConvLSTM model.  
  Exclude vegetation variables if needed using the exclude_variables setting inside the preprocessing function.

- Deforestation scenarios:  
  Replace the preprocessing function in MultiTaskConvLSTM_Full_Pipeline.ipynb with scripts from Data Perturbation Code.

- PGD_Full_Pipeline.ipynb  
  Contains all necessary code to run Projected Gradient Descent experiments.

- EDA.ipynb  
  For basic exploratory data analysis on the atmospheric and vegetation dataset.

- TConvLSTM_FULL_PIPELINE
  To run this pipeline, a fully functional ConvLSTM should already be saved in the same directory.

WARNING:  
The models are highly sensitive to input data normalization. The DConvLSTM-SAC is most sensitive to changes in data normalization. 
Changes to preprocessing or scaling must be made carefully to avoid severe performance degradation.

REQUIRED LIBRARIES  
------------------

This project uses a Python virtual environment.  
All necessary libraries are listed in environment.yml.

To create the environment:

conda env create -f environment.yml

Then activate the environment:

conda activate [your-environment-name]

NOTES  
-----

- The repository is available online on GitHub: 
- Instructions for experimental setups can be found inside the corresponding notebooks.
- To change the lookback and lookahead of the models, the making_sub_files_split_by_timestamp.py must be re-run on  FINAL_CLEANED_HOURLYatmosphericdata2021to2023.json with the lookback and lookahead variables changed. The input and output dimensions of the data should then be changed within the pipeline, along with any architectural modifications. The pipeline can then be run.
- In all of the preprocessing functions, Surface Latent Heat Flux is divided by 10800. This is required, as outlined on the Copernicus Data Store website for the ECWMF Reanalysis dataset.