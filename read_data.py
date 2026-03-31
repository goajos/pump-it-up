from pathlib import Path
from utils import print_column_stats, plot_coordinates, plot_histogram

import pandas as pd
import numpy as np

DATA_DIR = Path("data")
TRAIN_VALUES_FILE = "trainval.csv"
TRAIN_LABELS_FILE = "trainlabel.csv"
TEST_VALUES_FILE = "testval.csv"
OUTPUT_DIR = Path("outputs")

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    # manually checking the column values shows a bunch of 'NA' strings
    cleaned = cleaned.replace({"none": np.nan, "not known": np.nan, "Not Known": np.nan, "unknown": np.nan, "Unknown": np.nan})
    # '-' and '0' present in the installer and funder column values
    cleaned = cleaned.replace({'-': np.nan, '0': np.nan})

    plot_coordinates(cleaned, "before", OUTPUT_DIR)
    # longitude of 0 makes no sense on the scatter plot
    cleaned["longitude"] = cleaned["longitude"].replace(0, np.nan)
    plot_coordinates(cleaned, "after", OUTPUT_DIR)

    # construction year and population of 0 makes no sense
    cleaned[["construction_year", "population"]] = cleaned[["construction_year", "population"]].replace(0, np.nan)

    # uniform lowercase string values for str columns
    str_cols = cleaned.select_dtypes(include=["str"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].str.lower()

    # recorded by has only 1 value, so no info
    cleaned = cleaned.drop("recorded_by", axis=1)

    plot_histogram(cleaned, "amount_tsh", "before", OUTPUT_DIR)
    # amount_tsh contains mostly 0s
    cleaned["amount_tsh"] = cleaned["amount_tsh"].replace(0, np.nan)
    plot_histogram(cleaned, "amount_tsh", "after", OUTPUT_DIR)
    # almost 70% nan values
    cleaned = cleaned.drop("amount_tsh", axis=1)

    print_column_stats(cleaned)
    return cleaned

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_values = pd.read_csv(DATA_DIR / TRAIN_VALUES_FILE, parse_dates= ["date_recorded"])
    train_labels = pd.read_csv(DATA_DIR / TRAIN_LABELS_FILE)
    test_values = pd.read_csv(DATA_DIR / TEST_VALUES_FILE, parse_dates= ["date_recorded"])

    # print_column_stats(train_values, ["funder"])
    train_values = _clean_data(train_values)
    test_values = _clean_data(test_values)

    return train_values, train_values

if __name__ == "__main__":
    train_df, test_df = load_data()
