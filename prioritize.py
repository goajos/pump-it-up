from sklearn.model_selection import train_test_split
from scipy.spatial import cKDTree
from read_data import read_data
from transform_data import transform_data
from model import train_rfc, evaluate_rfc
from utils import SEED, LABEL_MAP, LABEL_NAMES, OUTPUT_DIR, plot_priority_scatter

import pandas as pd
import numpy as np

NON_FUNCTIONAL_WEIGHTS = {"population": 0.5, "nearest_functional": 0.5}
NEEDS_REPAIR_WEIGHTS = {
    "population": 1 / 3,
    "nearest_functional": 1 / 3,
    "construction_year": 1 / 3,
}
WEIGHT_MAP = {
    "non functional": NON_FUNCTIONAL_WEIGHTS,
    "functional needs repair": NEEDS_REPAIR_WEIGHTS,
}


def _normalize(series: pd.Series) -> pd.Series:
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def _nearest_functional_distances(
    non_functional: pd.DataFrame, functional: pd.DataFrame
) -> pd.Series:
    functional_coords = functional.dropna(subset=["latitude", "longitude"])[
        ["latitude", "longitude"]
    ].values
    has_coords = (
        non_functional["latitude"].notna() & non_functional["longitude"].notna()
    )

    distances = pd.Series(np.nan, index=non_functional.index)

    tree = cKDTree(functional_coords)
    non_functional_coords = non_functional.loc[
        has_coords, ["latitude", "longitude"]
    ].values
    dists, _ = tree.query(non_functional_coords, k=1)
    distances.loc[has_coords] = dists
    # median distance for pumps with no coords
    distances.loc[~has_coords] = np.nanmedian(dists)

    return distances


def _compute_priority_scores(data: pd.DataFrame, status: str) -> pd.DataFrame:
    prioritized = data[data["status_group"] == status].copy()
    functional = data[data["status_group"] == "functional"].copy()

    prioritized["nearest_functional_dist"] = _nearest_functional_distances(
        prioritized, functional
    )

    year = prioritized["construction_year"].fillna(
        prioritized["construction_year"].mean()
    )
    population = prioritized["population"].fillna(prioritized["population"].mean())

    weights = WEIGHT_MAP[status]
    if status == "non functional":
        prioritized["priority_scores"] = weights["population"] * _normalize(
            population
        ) + weights["nearest_functional"] * _normalize(
            prioritized["nearest_functional_dist"]
        )
    elif status == "functional needs repair":
        prioritized["priority_scores"] = (
            weights["construction_year"] * (1 - _normalize(year))
            + weights["population"] * _normalize(population)
            + weights["nearest_functional"]
            * _normalize(prioritized["nearest_functional_dist"])
        )

    return prioritized


def prioritize_pumps(data: pd.DataFrame) -> None:
    for status in ["non functional", "functional needs repair"]:
        prioritized = _compute_priority_scores(data, status)
        sorted_df = prioritized.sort_values("priority_scores", ascending=False)
        safe_status = status.replace(" ", "_")
        sorted_df.to_csv(OUTPUT_DIR / f"priority_{safe_status}.csv")
        plot_priority_scatter(sorted_df, status)


if __name__ == "__main__":
    train_df, train_labels_df, test_df = read_data()
    test_ids = test_df["id"]

    y = train_labels_df["status_group"].map(LABEL_MAP)

    train_data, val_data, y_train, y_val = train_test_split(
        train_df, y, test_size=0.2, random_state=SEED, stratify=y
    )

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
