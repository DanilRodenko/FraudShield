import numpy as np
import pandas as pd

# Night hours with elevated fraud rate (see notebooks/01_eda.ipynb, Q5)
NIGHT_HOURS = {22, 23, 0, 1, 2, 3}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `hour` (0–23) and `is_night` (1 if hour in NIGHT_HOURS, else 0)."""
    df = df.copy()
    # Parse here so the function works on raw CSV strings too
    ts = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = ts.dt.hour
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)
    return df


def add_online_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `is_online`: 1 if category ends with '_net', else 0."""
    df = df.copy()
    # "_net" categories are online, card-not-present purchases
    df["is_online"] = df["category"].str.endswith("_net").astype(int)
    return df


def add_age(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `age`: cardholder age in full years AT THE TIME of the transaction."""
    df = df.copy()
    ts = pd.to_datetime(df["trans_date_trans_time"])
    dob = pd.to_datetime(df["dob"])

    years = ts.dt.year - dob.dt.year
    # Subtract one year if the birthday hasn't happened yet that year.
    # (days / 365.25 is off by one around birthdays — caught by tests)
    before_birthday = (ts.dt.month < dob.dt.month) | (
        (ts.dt.month == dob.dt.month) & (ts.dt.day < dob.dt.day)
    )
    df["age"] = years - before_birthday.astype(int)
    return df


def add_distance_km(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `distance_km` between cardholder home (lat, long)
    and merchant (merch_lat, merch_long) — haversine formula."""
    df = df.copy()
    lat1 = np.radians(df["lat"])
    lon1 = np.radians(df["long"])
    lat2 = np.radians(df["merch_lat"])
    lon2 = np.radians(df["merch_long"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2

    # 6371 = Earth's mean radius in km
    df["distance_km"] = 2 * 6371 * np.arcsin(np.sqrt(a))
    return df


def add_amt_vs_card_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `card_mean_amt` (mean of this card's PREVIOUS amounts) and
    `amt_ratio` = amt / card_mean_amt. First transaction of a card -> NaN."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    # "Previous" only makes sense once rows are in time order within each card
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    # shift(1) excludes the current transaction -> no leakage from the present/future
    df["card_mean_amt"] = df.groupby("cc_num")["amt"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df["amt_ratio"] = df["amt"] / df["card_mean_amt"]

    df = df.sort_index()  # restore original row order for the caller
    return df


def add_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `txn_count_1h` and `txn_count_24h`: number of this card's
    PREVIOUS transactions in the last 1h / 24h (current one excluded)."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df = df.sort_values(["cc_num", "trans_date_trans_time"])

    for window, col in [("1h", "txn_count_1h"), ("24h", "txn_count_24h")]:
        # Time-based window per card: counts rows in (t - window, t], current included
        counts = (
            df.groupby("cc_num")
            .rolling(window, on="trans_date_trans_time")["amt"]
            .count()
        )
        # Result comes back in the same (card, time) order as df -> align by position.
        # Subtract 1 to exclude the current transaction itself.
        df[col] = counts.to_numpy().astype(int) - 1

    df = df.sort_index()  # restore original row order
    return df