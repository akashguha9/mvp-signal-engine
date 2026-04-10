import pandas as pd


def make_signal(series: pd.Series, threshold: float) -> pd.Series:
    signal = pd.Series(0, index=series.index)
    signal[series > threshold] = 1
    signal[series < -threshold] = -1
    return signal


def score(df: pd.DataFrame, signal_col: str) -> dict:
    active = df[df[signal_col] != 0].copy()

    if len(active) == 0:
        return {
            "active_signals": 0,
            "long_signals": 0,
            "short_signals": 0,
            "acc_1d": None,
            "acc_5d": None,
        }

    active["hit_1d"] = (
        ((active[signal_col] == 1) & (active["return_1d"] > 0)) |
        ((active[signal_col] == -1) & (active["return_1d"] < 0))
    ).astype(int)

    active["hit_5d"] = (
        ((active[signal_col] == 1) & (active["return_5d"] > 0)) |
        ((active[signal_col] == -1) & (active["return_5d"] < 0))
    ).astype(int)

    return {
        "active_signals": int(len(active)),
        "long_signals": int((active[signal_col] == 1).sum()),
        "short_signals": int((active[signal_col] == -1).sum()),
        "acc_1d": active["hit_1d"].mean(),
        "acc_5d": active["hit_5d"].mean(),
    }


def main() -> None:
    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Date", "symbol"]).reset_index(drop=True)

    split_date = df["Date"].quantile(0.7)

    train = df[df["Date"] <= split_date].copy()
    test = df[df["Date"] > split_date].copy()

    train["price_only_signal"] = make_signal(train["price_score"], 0.25)
    train["overlay_signal"] = make_signal(train["overlay_score"], 0.25)

    test["price_only_signal"] = make_signal(test["price_score"], 0.25)
    test["overlay_signal"] = make_signal(test["overlay_score"], 0.25)

    summary = pd.DataFrame([
        {"dataset": "train", "rule": "price_only", **score(train, "price_only_signal")},
        {"dataset": "train", "rule": "overlay", **score(train, "overlay_signal")},
        {"dataset": "test", "rule": "price_only", **score(test, "price_only_signal")},
        {"dataset": "test", "rule": "overlay", **score(test, "overlay_signal")},
    ])

    summary.to_csv("data/processed/walkforward_overlay_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()