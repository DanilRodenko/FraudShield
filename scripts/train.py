from app.data import load_data, split_xy
from app.features import build_features


def main():
    df = build_features(load_data())
    splits = split_xy(df)
    for name, (X, y) in splits.items():
        print(f"{name:5}: {X.shape}, fraud rate {y.mean():.2%}")


if __name__ == "__main__":
    main()