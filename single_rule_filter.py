# single_rule_filter.py

from __future__ import annotations

import pandas as pd

INPUT_FILE = "output_signals.csv"
OUTPUT_FILE = "output_signals_single_rule.csv"

KEEP_PREFIXES = [
    "matched_tiebreak_up_uptrend_mom_up",
]

def main():
    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("No data.")
        return

    reasons = df["reason"].astype(str)

    keep_mask = reasons.apply(lambda x: any(x.startswith(p) for p in KEEP_PREFIXES))

    df["signal"] = df["signal"].where(keep_mask, 0)
    df["reason"] = df["reason"].where(keep_mask, "single_rule_filtered_out")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved: {OUTPUT_FILE}")
    print("\nSignal counts:")
    print(df["signal"].value_counts(dropna=False).to_string())
    print("\nTop reasons:")
    print(df["reason"].astype(str).value_counts().head(20).to_string())

if __name__ == "__main__":
    main()