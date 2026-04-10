import pandas as pd


def score(df, col):
    active = df[df[col] != 0].copy()

    if len(active) == 0:
        return {"acc_1d": None, "acc_5d": None}

    active["hit_5d"] = (
        ((active[col] == 1) & (active["return_5d"] > 0)) |
        ((active[col] == -1) & (active["return_5d"] < 0))
    ).astype(int)

    return {"acc_5d": active["hit_5d"].mean()}


def main():
    df = pd.read_csv("data/processed/final_dataset_dynamic.csv")

    print("Dynamic overlay:")
    print(score(df, "signal_dynamic"))


if __name__ == "__main__":
    main()