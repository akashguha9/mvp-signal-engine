import pandas as pd
from utils import save_csv, print_stage


def main() -> None:
    print_stage("Building gated belief overlay score")

    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df["belief_present"] = df["belief_mean"].notna().astype(int)

    # Gated overlay:
    # if belief exists -> 80/20 blend
    # if belief missing -> pure price score
    df["overlay_score_gated"] = df["price_score"]

    mask = df["belief_present"] == 1
    df.loc[mask, "overlay_score_gated"] = (
        0.80 * df.loc[mask, "price_score"].fillna(0) +
        0.20 * df.loc[mask, "belief_score"].fillna(0)
    )

    df["overlay_signal_gated"] = 0
    df.loc[df["overlay_score_gated"] > 0.25, "overlay_signal_gated"] = 1
    df.loc[df["overlay_score_gated"] < -0.25, "overlay_signal_gated"] = -1

    save_csv(df, "data/processed/final_dataset_with_overlay_gated.csv")

    print(f"Rows: {len(df)}")
    print("Columns added:")
    print(["belief_present", "overlay_score_gated", "overlay_signal_gated"])


if __name__ == "__main__":
    main()