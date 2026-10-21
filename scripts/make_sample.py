from pathlib import Path

import pandas as pd

from app.data import load_data
from app.features import FEATURE_COLS, build_features

OUT = Path("data/sample/transactions.csv")
N_LEGIT, N_FRAUD = 2000, 300

# Only what the app needs. No names, address, dob, job: PII never leaves the laptop.
KEEP = ["trans_num", "trans_date_trans_time", "cc_num", "merchant", "category",
        "amt", "city", "state", "is_fraud", *FEATURE_COLS]


def main():
    df = build_features(load_data())  # full history -> correct card features
    test = df[df["split"] == "test"]

    # Demo sample: fraud is oversampled so the queue has something to show.
    # The UI must say so: real fraud rate is 0.39%, not ~13%.
    legit = test[test["is_fraud"] == 0].sample(N_LEGIT, random_state=42)
    fraud = test[test["is_fraud"] == 1].sample(N_FRAUD, random_state=42)
    sample = pd.concat([legit, fraud])[KEEP].sort_values("trans_date_trans_time")

    # Sparkov prefixes EVERY merchant with "fraud_" — a generator artefact, not a signal
    sample["merchant"] = sample["merchant"].str.removeprefix("fraud_")
    # Mask the card number: keep last 4 digits only
    sample["cc_num"] = "**** " + sample["cc_num"].astype(str).str[-4:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT, index=False)
    print(f"saved {len(sample)} rows ({sample['is_fraud'].mean():.1%} fraud) to {OUT}")


if __name__ == "__main__":
    main()