from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def print_column_stats(df: pd.DataFrame, columns: list | None = None) -> None:
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
            # print(f"Mode: '{mode_val}' ({mode_freq} times, {mode_pct:.1f}% of non-null)")
            # # to print the column values to the console for some manual checks
            # vals = sorted(rows.dropna().unique().tolist())
            # print(f"Values: {vals}")

        elif rows.dtype in ["datetime64[us]"]:
            print(f"Min: {rows.min()}\nMax: {rows.max()}\nMean: {rows.mean()}")

def plot_coordinates(df: pd.DataFrame, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots()
    coords = df[["latitude", "longitude"]].dropna()
    ax.scatter(coords["longitude"], coords["latitude"])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(title)
    fig.savefig(output_path / f"{title}_coords.png")
    plt.close(fig)

def plot_histogram(df: pd.DataFrame, column: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=50)
    ax.set_xlabel(column)
    ax.set_ylabel("frequency")
    ax.set_title(f"{column} {title} histogram")
    fig.savefig(output_path / f"{column}_{title}_hist.png")
    plt.close(fig)
