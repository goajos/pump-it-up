# Simple setup for the [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/) challenge
Disclaimer: this is not a complete repository and the code is not properly documented and validated.

## Project Requirements
The project is set up with [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Project setup
Clone the repository cd into the root folder and run `uv sync`

## Project stucture
```
|- data/*
|- outputs/*
|- main.py
|- read_data.py
|- transform_data.py
|- model.py
|- prioritize.py
|- utils.py
```

## Code overview
### main.py
The main entrypoint. Run it with `uv run main.py`.
This run the pipeline and call the following sequence: read_data.py -> transform_data.py -> model.py

### read_data.py
To read in the provided csv files located in the data/ folder.
Can also be used for some basic data exploration and cleanup. Run it with `uv run read_data.py`.

### transform_data.py 
To perform some basic transformation on the data to further enhance the data exlopration and prepare it for the model.
It performs fuzzy matching for combining similar strings in high cardinality columns to cap them for encoding. Run it with `uv run transform_data.py`

### model.py
Model setup for a simple (Balanced)RandomForestClassifier. 

### utils.py
Holds various constants and utility functions used over the complete pipeline.

### prioritize.py
Setup for a simple weighted scoring function to compute priority scores for each label: functional, functional needs repair and non functional.
This will also run the main pipeline because it requires the predicted labels to compute the priority scores. Run it with `uv run prioritize.py`.

## Data overview
data/ folder holds the input csv files: training values, training labels and test values.
outputs/ folders holds several .png files used for data exploration and the presentation. 
Also includes some .csv output files: predictions.csv is the output file for the main pipeline. priority_\{label\}.csv is the the output file for the priority scoring.
