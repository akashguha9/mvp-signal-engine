import pandas as pd
from utils import save_csv, print_stage


def main():
    print_stage("Dynamic belief weighting")

    df = pd.read_csv("data/processed/final_dataset_with_overlay.csv")

    # strength = absolute belief signal
    df["belief_strength"] = df["belief_score"].abs()

    # normalize (0 → 1)
    max_strength = df["belief_strength"].max()
    if max_strength == 0:
        df["w"] = 0
    else:
        df["w"] = df["belief_strength"] / max_strength

    # cap weight (avoid overtrust)
    df["w"] = df["w"].clip(0, 0.3)

    df["overlay_dynamic"] = (
        (1 - df["w"]) * df["price_score"].fillna(0) +
        df["w"] * df["belief_score"].fillna(0)
    )

    df["signal_dynamic"] = 0
    df.loc[df["overlay_dynamic"] > 0.25, "signal_dynamic"] = 1
    df.loc[df["overlay_dynamic"] < -0.25, "signal_dynamic"] = -1

    save_csv(df, "data/processed/final_dataset_dynamic.csv")

    print("Added:", ["belief_strength", "w", "overlay_dynamic", "signal_dynamic"])


if __name__ == "__main__":
    main()