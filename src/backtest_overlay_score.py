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


def main() -> None:
    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    summary = pd.DataFrame([
        {"rule": "overlay_signal", "rows_total": len(df), **score(df, "overlay_signal")}
    ])

    summary.to_csv("data/processed/overlay_backtest_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()