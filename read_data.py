from pathlib import Path
from utils import plot_coordinates, plot_histogram

import pandas as pd
import numpy as np

DATA_DIR = Path("data")
TRAIN_VALUES_FILE = "trainval.csv"
TRAIN_LABELS_FILE = "trainlabel.csv"
TEST_VALUES_FILE = "testval.csv"
OUTPUT_DIR = Path("outputs")

def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    # uniform lowercase string values for str columns
    str_cols = cleaned.select_dtypes(include=["str"]).columns
    for col in str_cols:
        cleaned[col] = cleaned[col].str.lower()

    # manually checking the column values shows a bunch of 'NA' strings
    cleaned = cleaned.replace({"none": np.nan, "not known": np.nan, "not kno": np.nan, "unknown": np.nan})
    # '-' and '0' present in the installer and funder column values
    cleaned = cleaned.replace({'-': np.nan, '0': np.nan})

    plot_coordinates(cleaned, "before", OUTPUT_DIR)
    # longitude of 0 makes no sense on the scatter plot
    cleaned["longitude"] = cleaned["longitude"].replace(0, np.nan)
    plot_coordinates(cleaned, "after", OUTPUT_DIR)

    # construction year and population of 0 makes no sense
    cleaned[["construction_year", "population"]] = cleaned[["construction_year", "population"]].replace(0, np.nan)


    # recorded by has only 1 value, so no info
    cleaned = cleaned.drop("recorded_by", axis=1)

    plot_histogram(cleaned, "amount_tsh", "before", OUTPUT_DIR)
    # amount_tsh contains mostly 0s
    cleaned["amount_tsh"] = cleaned["amount_tsh"].replace(0, np.nan)
    plot_histogram(cleaned, "amount_tsh", "after", OUTPUT_DIR)
    # almost 70% nan values, so drop column instead
    cleaned = cleaned.drop("amount_tsh", axis=1)

    plot_histogram(cleaned, "gps_height", "before", OUTPUT_DIR)
    # gps_height also has a lot of 0s and that doesn't seem to make sense?
    cleaned["gps_height"] = cleaned["gps_height"].replace(0, np.nan)
    plot_histogram(cleaned, "gps_height", "after", OUTPUT_DIR)

    # region and district code look categorical instead of numerical
    cleaned["region_code"] = cleaned["region_code"].astype(str)
    cleaned["district_code"] = cleaned["district_code"].astype(str)

    # wpt_name has 37396 unique values, subvillage has 19287, naively drop them
    cleaned = cleaned.drop(["wpt_name", "subvillage"], axis=1)

    # fuzzy matching results show a lot of nonsensical data, drop them
    cleaned = cleaned.drop(["ward", "scheme_name"], axis=1)

    # print_column_stats(cleaned)
    return cleaned

def read_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_values = pd.read_csv(DATA_DIR / TRAIN_VALUES_FILE, parse_dates= ["date_recorded"])
    train_labels = pd.read_csv(DATA_DIR / TRAIN_LABELS_FILE)
    test_values = pd.read_csv(DATA_DIR / TEST_VALUES_FILE, parse_dates= ["date_recorded"])

    # print_column_stats(train_values, ["funder"])
    train_values = _clean_data(train_values)
    test_values = _clean_data(test_values)

    return train_values, train_labels, test_values

if __name__ == "__main__":
    train_df, train_labels_df, test_df = read_data()
