import pandas as pd
from utils import save_csv, print_stage


def zscore_by_symbol(df: pd.DataFrame, col: str, out_col: str, window: int = 20) -> pd.DataFrame:
    df[out_col] = df.groupby("symbol")[col].transform(
        lambda s: (
            (s - s.rolling(window, min_periods=5).mean()) /
            s.rolling(window, min_periods=5).std().replace(0, pd.NA)
        )
    )
    return df


def main() -> None:
    print_stage("Building belief overlay score")

    df = pd.read_csv("data/processed/final_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Core price factor z-scores
    df = zscore_by_symbol(df, "momentum_20d", "z_momentum_20d", window=20)
    df = zscore_by_symbol(df, "trend_50_200", "z_trend_50_200", window=20)
    df = zscore_by_symbol(df, "drawdown", "z_drawdown", window=20)

    # Belief z-score already exists, but keep numeric safety
    df["b_zscore_3"] = pd.to_numeric(df["b_zscore_3"], errors="coerce")
    df["belief_mean"] = pd.to_numeric(df["belief_mean"], errors="coerce")

    # Base price score
    df["price_score"] = (
        0.50 * df["z_momentum_20d"].fillna(0) +
        0.30 * df["z_trend_50_200"].fillna(0) -
        0.20 * df["z_drawdown"].fillna(0)
    )

    # Belief score
    df["belief_score"] = (
        0.70 * df["belief_mean"].fillna(0) +
        0.30 * df["b_zscore_3"].fillna(0)
    )

    # Overlay score: price engine first, belief as narrative modifier
    df["overlay_score"] = (
        0.80 * df["price_score"].fillna(0) +
        0.20 * df["belief_score"].fillna(0)
    )

    # Directional overlay signal
    df["overlay_signal"] = 0
    df.loc[df["overlay_score"] > 0.25, "overlay_signal"] = 1
    df.loc[df["overlay_score"] < -0.25, "overlay_signal"] = -1

    save_csv(df, "data/processed/final_dataset_with_overlay.csv")

    print(f"Rows: {len(df)}")
    print("Columns added:")
    print([
        "z_momentum_20d",
        "z_trend_50_200",
        "z_drawdown",
        "price_score",
        "belief_score",
        "overlay_score",
        "overlay_signal",
    ])


if __name__ == "__main__":
    main()