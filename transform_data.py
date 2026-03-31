from utils import group_fuzzy_matches
from sklearn.preprocessing import OrdinalEncoder

import pandas as pd
import numpy as np

UNIQUE_PRINT_THRESHOLD = 150
FUZZY_THRESHOLD = 0.75
TOP_N_CATEGORIES = 100

def _cap_high_cardinality(series: pd.Series, known_categories: list | None = None) -> tuple[pd.Series, list]:
    """Caps the high cardinality series.

    If else block ensures the test data uses the same TOP_N_CATEGORIES learned from the train data.
    """
    if known_categories is None:
        top_categories= series.value_counts().nlargest(TOP_N_CATEGORIES).index.tolist()
    else:
        top_categories = known_categories
    return series.where(series.isin(top_categories), other="other"), top_categories

def transform_data(df: pd.DataFrame, fit_encoders: bool, encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    transformed = df.copy()

    if encoders is None:
        encoders = {}

    # id holds no value
    transformed = transformed.drop("id", axis = 1)

    # extract simple year since temporal feature from the date recorded
    transformed["years_since_recorded"] = (pd.Timestamp.now() - transformed["date_recorded"]).dt.days / 365.25
    transformed = transformed.drop("date_recorded", axis = 1)

    # use fuzzy matching to group similar strings
    transformed["funder"] = group_fuzzy_matches(transformed, "funder", FUZZY_THRESHOLD)
    transformed["installer"] = group_fuzzy_matches(transformed, "installer", FUZZY_THRESHOLD) 
    # cap high cardinality categories, TOP_N_CATEGORIES are kept, rest is grouped into "other"
    for col in ["funder", "installer"]:
        known_categories = None if fit_encoders else encoders.get(f"{col}_top_categories")
        transformed[col], top_categories = _cap_high_cardinality(transformed[col], known_categories)
        if fit_encoders:
            encoders[f"{col}_top_categories"] = top_categories

    # naive mean strategy for numerical columns
    numerical_cols = transformed.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if transformed[col].isna().any():
            # only calculate the mean from the train data
            if fit_encoders:
                encoders[f"{col}_mean"] = transformed[col].mean()
            transformed[col] = transformed[col].fillna(encoders[f"{col}_mean"])

    # map the boolean columns to 1 and 0
    for col in ["public_meeting", "permit"]:
        transformed[col] = transformed[col].map({"True": 1, "False": 0}).fillna(0)

    categorical_cols = transformed.select_dtypes(include=["str", "object"]).columns
    # set the remaining NA categorical columsn to "no data"
    transformed[categorical_cols] = transformed[categorical_cols].fillna("no data")

    # TODO: also try OneHotEncoder?
    if fit_encoders:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        transformed[categorical_cols] = encoder.fit_transform(transformed[categorical_cols])
        encoders["ordinal_encoder"] = encoder
    else:
        encoder = encoders["ordinal_encoder"]
        transformed[categorical_cols] = encoder.transform(transformed[categorical_cols])

    # print_column_stats_many_unique(transformed, UNIQUE_PRINT_THRESHOLD) 

    return transformed, encoders

if __name__ == "__main__":
    from read_data import read_data

    train_df, train_labels_df,test_df = read_data()

    X_train, encoders = transform_data(train_df, True)
    X_test, _ = transform_data(test_df, False, encoders)
    y_train = train_labels_df["status_group"].map({"functional": 0, "functional needs repair": 1, "non functional": 2})

    print(f"X train shape: {X_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")

 
