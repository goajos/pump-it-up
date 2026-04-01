# Simple setup for the [Pump it Up: Data Mining the Water Table](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/page/25/) challenge

## Project Requirements
The project is set up with [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Project setup
Clone the repository cd into the root folder and run `uv sync`

## Project stucture
|- main.py
|- read_data.py
|- transform_data.py
|- model.py
|- prioritize.py
|- utils.py
|- data/*
|- outputs/*

## Code overview
### main.py
The main entrypoint. Run it with `uv run main.py`.
This will start the pipeline and call the following sequence: read_data.py -> transform_data.py -> model.py

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
