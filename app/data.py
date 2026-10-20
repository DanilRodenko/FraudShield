from pathlib import Path

import pandas as pd

from app.features import FEATURE_COLS

DATA = Path("data/raw")
VAL_START = "2020-04-21"  # last ~2 months of train become validation


def load_data() -> pd.DataFrame:
    """Reads train + test CSVs into one frame with a `split` column."""
    train = pd.read_csv(DATA / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(DATA / "fraudTest.csv", index_col=0)
    train["split"] = "train"
    test["split"] = "test"
    # ignore_index=True: avoid duplicate index labels (feature functions use sort_index)
    return pd.concat([train, test], ignore_index=True)


def split_xy(df: pd.DataFrame):
    """Time-based split: train (fit) / val (compare models) / test (final, once)."""
    ts = pd.to_datetime(df["trans_date_trans_time"])
    from_train_file = df["split"] == "train"

    masks = {
        "train": from_train_file & (ts < VAL_START),
        "val": from_train_file & (ts >= VAL_START),
        "test": df["split"] == "test",
    }
    # {"train": (X, y), "val": (X, y), "test": (X, y)}
    return {name: (df.loc[m, FEATURE_COLS], df.loc[m, "is_fraud"]) for name, m in masks.items()}