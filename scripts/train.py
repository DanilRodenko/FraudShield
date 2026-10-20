from pathlib import Path

import pandas as pd

from app.features import FEATURE_COLS, build_features

DATA = Path("data/raw")


def load_data() -> pd.DataFrame:
    """Reads train + test CSVs into one frame with a `split` column."""
    train = pd.read_csv(DATA / "fraudTrain.csv", index_col=0)
    test = pd.read_csv(DATA / "fraudTest.csv", index_col=0)

    train['split'] = "train"
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    return df


def main():
    df = load_data()
    df = build_features(df)  # history features see train past when computing test
    X_train, y_train = df.loc[df["split"] == "train", FEATURE_COLS], df.loc[df['split'] == 'train', 'is_fraud']
    X_test, y_test = df.loc[df["split"] == "test", FEATURE_COLS], df.loc[df['split'] == "test", "is_fraud"]


    print(f"train: {X_train.shape}, fraud rate {y_train.mean():.2%}")
    print(f"test:  {X_test.shape}, fraud rate {y_test.mean():.2%}")


if __name__ == "__main__":
    main()