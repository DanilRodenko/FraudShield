import pandas as pd
import pytest

from app.features import add_age, add_distance_km, add_online_flag, add_time_features


# --- distance: known answer on real cities ----------------------------------

def test_distance_dublin_limerick():
    df = pd.DataFrame({"lat": [53.35], "long": [-6.26],
                       "merch_lat": [52.66], "merch_long": [-8.63]})
    result = add_distance_km(df)
    # approx: floats are never exactly equal
    assert result["distance_km"].iloc[0] == pytest.approx(175, abs=5)


def test_distance_same_point_is_zero():
    df = pd.DataFrame({"lat": [52.66], "long": [-8.63],
                       "merch_lat": [52.66], "merch_long": [-8.63]})
    assert add_distance_km(df)["distance_km"].iloc[0] == pytest.approx(0)


# --- time: boundary hours ---------------------------------------------------

@pytest.mark.parametrize("time, expected_night", [
    ("2020-01-01 21:59:59", 0),   # still day
    ("2020-01-01 22:00:00", 1),   # night starts
    ("2020-01-01 03:59:59", 1),   # last night minute
    ("2020-01-01 04:00:00", 0),   # night ends
    ("2020-01-01 12:00:00", 0),   # midday
])
def test_is_night(time, expected_night):
    df = pd.DataFrame({"trans_date_trans_time": [time]})
    assert add_time_features(df)["is_night"].iloc[0] == expected_night


# --- online flag ------------------------------------------------------------

@pytest.mark.parametrize("category, expected", [
    ("shopping_net", 1),
    ("misc_net", 1),
    ("grocery_pos", 0),
    ("net_something", 0),   # "net" at the start, not the end -> not online
])
def test_is_online(category, expected):
    df = pd.DataFrame({"category": [category]})
    assert add_online_flag(df)["is_online"].iloc[0] == expected


# --- age: birthday edge cases -----------------------------------------------

@pytest.mark.parametrize("dob, trans_time, expected_age", [
    ("2000-01-01", "2020-06-15 10:00:00", 20),   # regular case
    ("2000-03-01", "2001-03-01 10:00:00", 1),    # exactly on the birthday
    ("2000-03-01", "2001-02-28 10:00:00", 0),    # one day before the birthday
])
def test_age(dob, trans_time, expected_age):
    df = pd.DataFrame({"dob": [dob], "trans_date_trans_time": [trans_time]})
    assert add_age(df)["age"].iloc[0] == expected_age


# --- functions must not modify their input (df.copy() check) ----------------

def test_does_not_modify_input():
    df = pd.DataFrame({"category": ["shopping_net"]})
    add_online_flag(df)
    assert "is_online" not in df.columns   # column must not appear in the original df