from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import matplotlib.pyplot as plt

LABEL_NAMES = {0: "functional", 1: "functional needs repair", 2: "non functional"}
LABEL_COLORS = {"functional": "green", "functional needs repair": "orange", "non functional": "red"}
LABEL_MAP = {"functional": 0, "functional needs repair": 1, "non functional": 2}
OUTPUT_DIR = Path("outputs")
SEED = 8080

def print_column_stats(df: pd.DataFrame, columns: list | None = None) -> None:
    """Print column statistics.

    For all columns it prints: dtype, amount of NA values, amount of missing values, missing percentage, amount of unique values.
    For numerical columns it also prints: min value, q1 value, median value, q3 value, max value, mean value.
    For categorical columns it also prints: mode value, mode frequency, mode percentage of NA values.
    For datetime columns it also prints: min value,  max value, mean value.
    """
    total_rows = len(df)

    if columns is None:
        columns = df.columns.tolist()

    for col in columns:
        rows = df[col]
        not_na = int(rows.notna().sum())
        missing = int(rows.isna().sum())
        missing_pct = (missing / total_rows) * 100
        unique = int(rows.nunique(dropna=True))

        print(f"\nColumn: {col}")
        print(f"Type: {rows.dtype}")
        print(f"Not-na: {not_na}\nMissing: {missing} ({missing_pct:.1f}%)\nUnique: {unique}")

        if rows.dtype in ["int64", "float64"]:
            print(f"Min: {rows.min():.2f}\nQ1: {rows.quantile(0.25):.2f}\nMedian: {rows.median():.2f}\nQ3: {rows.quantile(0.75):.2f}\nMax: {rows.max():.2f}\nMean: {rows.mean():.2f}")


        elif rows.dtype in ["str", "object"]:
            mode_val = rows.mode().iloc[0] if not rows.mode().empty else None
            mode_freq = int(rows.value_counts().iloc[0]) if not_na > 0 else 0
            mode_pct = (mode_freq / not_na) * 100 if not_na > 0 else 0
            print(f"Mode: '{mode_val}' ({mode_freq} times, {mode_pct:.1f}% of not-NA)")
            # # to print the column values to the console for some manual checks
            # vals = sorted(rows.dropna().unique().tolist())
            # print(f"Values: {vals}")

        elif rows.dtype in ["datetime64[us]"]:
            print(f"Min: {rows.min()}\nMax: {rows.max()}\nMean: {rows.mean()}")

def print_column_stats_many_unique(df: pd.DataFrame, threshold: int) -> None:
    """Wrapper function to only print column statistics that have more than threshold unique values."""
    columns = []
    for col in df.columns.tolist():
        if df[col].nunique(dropna=True) > threshold:
            columns.append(col)

    print_column_stats(df, columns)

def plot_coordinates(df: pd.DataFrame, title: str) -> None:
    fig, ax = plt.subplots()
    coords = df[["latitude", "longitude"]].dropna()
    ax.scatter(coords["longitude"], coords["latitude"])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    fig.savefig(OUTPUT_DIR / f"{title}_coords.png")
    plt.close(fig)

def plot_histogram(df: pd.DataFrame, column: str, title: str) -> None:
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=50)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{column} {title} histogram")
    fig.savefig(OUTPUT_DIR / f"{column}_{title}_hist.png")
    plt.close(fig)

# TODO: external library for speed up?
def group_fuzzy_matches(df: pd.DataFrame, column: str, threshold: float) -> pd.Series:
    """Group similar categorical values by fuzzily matching strings.

    This function creates representatives for near-duplicate entries in a categorical column.
    The more frequent value of a matched pair becomes the representative.
    """
    unique_vals = df[column].dropna().unique()

    representatives = {val: val for val in unique_vals}

    def _find(string: str):
        if representatives[string] != string:
            representatives[string] = _find(representatives[string])
        return representatives[string]

    def _union(string_x, string_y):
        representative_x, representative_y = _find(string_x), _find(string_y)
        if representative_x != representative_y:
            if (df[column] == representative_x).sum() >= (df[column] == representative_y).sum():
                representatives[representative_y] = representative_x
            else:
                representatives[representative_x] = representative_y

    for idx, val in enumerate(unique_vals):
        for other_val in unique_vals[idx+1:]:
            # ratio 1.0 is identical, 0.0 nothing in common
            if SequenceMatcher(None, val, other_val).ratio() >= threshold:
                _union(val, other_val)

    mapping = {val: _find(val) for val in unique_vals}
    # _print_fuzzy_map(mapping)

    return df[column].map(mapping)

def _print_fuzzy_map(mapping: dict) -> None:
    parents = {}
    for variant, parent in mapping.items():
        if parent not in parents:
            parents[parent] = []
        parents[parent].append(variant)
 
    print("\n")
    for parent, variants in parents.items():
        if len(variants) > 1:
            print(f"{parent}: {variants}")

# montage <imgage1>.png <image2>.png <image3>.png -tile 2x2 -geometry +0+0 stack.png
# display stack.png
def plot_status_map(df: pd.DataFrame, labels: pd.Series, status: str | None = None) -> None:
    fig, ax = plt.subplots()
    if status:
        mask = labels == status
        coords = df[mask][["longitude", "latitude"]].dropna()
        ax.scatter(coords["longitude"], coords["latitude"], c=LABEL_COLORS[status], label=status, alpha=.25, s=2)
    else:
        for status, color in LABEL_COLORS.items():
            mask = labels == status
            coords = df[mask][["longitude", "latitude"]].dropna()
            ax.scatter(coords["longitude"], coords["latitude"], c=color, label=status, alpha=.25, s=2)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    if status:
        ax.set_title(f"Water pump {status}")
        safe_status = status.replace(" ", "_")
        fig.savefig(OUTPUT_DIR / f"{safe_status}_map.png")
    else:
        ax.set_title("All water pump status")
        fig.savefig(OUTPUT_DIR / "all_status_map.png")
    plt.close(fig)

def plot_priority_scatter(df: pd.DataFrame, status: str) -> None:
    fig, ax = plt.subplots()

    population_min, population_max = df["population"].min(), df["population"].max()
    sizes = 5 + 200 * (df["population"] - population_min) / (population_max - population_min) if population_max > population_min else 20

    scatter = ax.scatter(df["longitude"], df["latitude"], c=df["priority_scores"], s=sizes, cmap="YlOrRd", edgecolors="none")
    plt.colorbar(scatter, ax=ax, label="Priority Score")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Priority map: {status}")
    safe_status = status.replace(" ", "_")
    fig.savefig(OUTPUT_DIR / f"priority_{safe_status}_map.png")
    plt.close(fig)
