import pandas as pd


def score(active: pd.DataFrame, signal_col: str) -> dict:
    active = active[active[signal_col] != 0].copy()

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
        "active_signals": len(active),
        "long_signals": int((active[signal_col] == 1).sum()),
        "short_signals": int((active[signal_col] == -1).sum()),
        "acc_1d": active["hit_1d"].mean(),
        "acc_5d": active["hit_5d"].mean(),
    }


def make_signal(df: pd.DataFrame, score_col: str, threshold: float = 0.25) -> pd.Series:
    signal = pd.Series(0, index=df.index)
    signal[df[score_col] > threshold] = 1
    signal[df[score_col] < -threshold] = -1
    return signal


def main() -> None:
    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["price_only_signal"] = make_signal(df, "price_score", threshold=0.25)
    df["overlay_signal"] = make_signal(df, "overlay_score", threshold=0.25)
    df["belief_only_signal"] = make_signal(df, "belief_score", threshold=0.05)

    summary = pd.DataFrame([
        {"rule": "price_only", **score(df, "price_only_signal")},
        {"rule": "overlay", **score(df, "overlay_signal")},
        {"rule": "belief_only", **score(df, "belief_only_signal")},
    ])

    summary.to_csv("data/processed/overlay_ablation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()