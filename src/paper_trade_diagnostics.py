import pandas as pd
from utils import save_csv, print_stage


def main() -> None:
    print_stage("Building paper trade diagnostics")

    df = pd.read_csv("data/processed/paper_trade_overlay_log.csv")
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")

    # Side label
    df["side"] = df["direction"].map({1: "long", -1: "short"})

    # Symbol-level summary
    by_symbol = (
        df.groupby("symbol", dropna=False)
        .agg(
            n_trades=("symbol", "size"),
            mean_ret_1d=("ret_1d_realized", "mean"),
            mean_ret_5d=("ret_5d_realized", "mean"),
            hit_1d=("hit_1d_realized", "mean"),
            hit_5d=("hit_5d_realized", "mean"),
        )
        .reset_index()
        .sort_values("mean_ret_5d", ascending=False)
    )

    # Side-level summary
    by_side = (
        df.groupby("side", dropna=False)
        .agg(
            n_trades=("side", "size"),
            mean_ret_1d=("ret_1d_realized", "mean"),
            mean_ret_5d=("ret_5d_realized", "mean"),
            hit_1d=("hit_1d_realized", "mean"),
            hit_5d=("hit_5d_realized", "mean"),
        )
        .reset_index()
        .sort_values("mean_ret_5d", ascending=False)
    )

    # Strength buckets from overlay score magnitude
    df["score_abs"] = df["overlay_score"].abs()
    df["score_bucket"] = pd.cut(
        df["score_abs"],
        bins=[0, 0.25, 0.50, 0.75, 10],
        labels=["0-0.25", "0.25-0.50", "0.50-0.75", "0.75+"],
        include_lowest=True,
    )

    by_bucket = (
        df.groupby("score_bucket", dropna=False)
        .agg(
            n_trades=("score_bucket", "size"),
            mean_ret_1d=("ret_1d_realized", "mean"),
            mean_ret_5d=("ret_5d_realized", "mean"),
            hit_1d=("hit_1d_realized", "mean"),
            hit_5d=("hit_5d_realized", "mean"),
        )
        .reset_index()
    )

    save_csv(by_symbol, "data/processed/paper_trade_by_symbol.csv")
    save_csv(by_side, "data/processed/paper_trade_by_side.csv")
    save_csv(by_bucket, "data/processed/paper_trade_by_bucket.csv")

    print("\nBY SYMBOL")
    print(by_symbol.to_string(index=False))

    print("\nBY SIDE")
    print(by_side.to_string(index=False))

    print("\nBY SCORE BUCKET")
    print(by_bucket.to_string(index=False))


if __name__ == "__main__":
    main()