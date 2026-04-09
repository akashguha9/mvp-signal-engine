# core_signal_filter.py

from __future__ import annotations

import pandas as pd

from config import OUTPUT_SIGNALS

CORE_LONG_REASONS = {
    "news_led",
    "matched_tiebreak_up_uptrend_mom_up",
    "void_inferred_up_uptrend_low_vol",
}

CORE_SHORT_REASONS = {
    "market_led",
}

def base_reason(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).split("|")[0].split("_high_vol")[0].split("_low_vol")[0]

def main():
    df = pd.read_csv(OUTPUT_SIGNALS)

    if df.empty:
        print("No data.")
        return

    new_signal = []
    new_reason = []

    for _, row in df.iterrows():
        r = str(row.get("reason", ""))
        b = base_reason(r)

        if b in CORE_LONG_REASONS:
            new_signal.append(1)
            new_reason.append(f"core_{b}")
        elif b in CORE_SHORT_REASONS:
            new_signal.append(-1)
            new_reason.append(f"core_{b}")
        else:
            new_signal.append(0)
            new_reason.append("core_filtered_out")

    df["signal"] = new_signal
    df["reason"] = new_reason

    df.to_csv("output_signals_core.csv", index=False)
    print("Saved: output_signals_core.csv")
    print(df["signal"].value_counts(dropna=False).to_string())

if __name__ == "__main__":
    main()