from sklearn.model_selection import train_test_split
from read_data import read_data
from transform_data import transform_data
from model import train_rfc, evaluate_rfc
from utils import SEED, LABEL_MAP, LABEL_NAMES, OUTPUT_DIR, plot_priority_scatter

import pandas as pd
import numpy as np

WEIGHT_CONSTRUCTION_YEAR = 0.5
WEIGHT_POPULATION = 0.5

def _normalize(series: pd.Series) -> pd.Series:
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)

def _compute_priority_scores(data: pd.DataFrame) -> pd.DataFrame:
    prioritized = data[["id", "construction_year", "population", "longitude", "latitude", "status_group"]].copy()

    construction_year_norm = _normalize(prioritized["construction_year"])
    population_norm =  _normalize(prioritized["population"])

    functional_mask = prioritized["status_group"] == "functional"
    functional_needs_repair_mask = prioritized["status_group"] == "functional needs repair"
    non_functional_mask = prioritized["status_group"] == "non functional"

    scores = pd.Series(np.nan, index=prioritized.index)
    # higher priority = weight * norm
    # lower piority = (1 - weight) * norm
    # functional: more population = higher priority, newer construction year = lower priority
    scores.loc[functional_mask] = WEIGHT_CONSTRUCTION_YEAR * (1 - construction_year_norm.loc[functional_mask]) + WEIGHT_POPULATION * population_norm.loc[functional_mask]
    # functional need repair: more population = higher piority, newer construction year = lower priority
    scores.loc[functional_needs_repair_mask] = WEIGHT_CONSTRUCTION_YEAR * (1 - construction_year_norm.loc[functional_needs_repair_mask]) + WEIGHT_POPULATION * population_norm.loc[functional_needs_repair_mask]
    # non functional: more population = higher priority, newer construction year = higher priority 
    scores.loc[non_functional_mask] = WEIGHT_CONSTRUCTION_YEAR * construction_year_norm.loc[non_functional_mask] + WEIGHT_POPULATION * population_norm.loc[non_functional_mask]

    prioritized["priority_scores"] = scores.values
    return prioritized

def prioritize_pumps(data: pd.DataFrame) -> None:
    prioritized = _compute_priority_scores(data)
    for status in LABEL_MAP.keys():
        sorted_df = prioritized[prioritized["status_group"] == status].sort_values("priority_scores", ascending=False)
        safe_status = status.replace(" ", "_")
        sorted_df.to_csv(OUTPUT_DIR / f"priority_{safe_status}.csv")
        # TODO: fix scale for presentation?
        plot_priority_scatter(sorted_df, status)



if __name__ == "__main__":
    train_df, train_labels_df, test_df = read_data()
    test_ids = test_df["id"]

    y = train_labels_df["status_group"].map(LABEL_MAP)

    train_data, val_data, y_train, y_val = train_test_split(train_df, y, test_size=0.2, random_state=SEED, stratify=y) 

    X_train, encoders = transform_data(train_data, fit_encoders=True)
    X_val, _ = transform_data(val_data, fit_encoders=False, encoders=encoders)
    X_test, _ = transform_data(test_df, fit_encoders=False, encoders=encoders)

    rfc = train_rfc(X_train, y_train)
    evaluate_rfc(rfc, X_val, y_val)

    predicted_labels = pd.Series(rfc.predict(X_test)).map(LABEL_NAMES)

    train_df["status_group"] = train_labels_df["status_group"].values
    test_df["status_group"] = predicted_labels.values
    data = pd.concat([train_df, test_df]) 

    prioritize_pumps(data)
