# lead_lag.py

import pandas as pd

from config import OUTPUT_EVENT_PANEL


def load_panel(path=OUTPUT_EVENT_PANEL):
    df = pd.read_csv(path)

    if "platform" in df.columns:
        df["platform"] = df["platform"].astype(str).str.strip()

    if "timestamp_utc" in df.columns:
        df["timestamp_utc"] = pd.to_datetime(
            df["timestamp_utc"].astype(str).str.strip().replace({"nan": None, "NaT": None}),
            utc=True,
            errors="coerce",
            format="mixed"
        )

    df = df[df["timestamp_utc"].notna()].copy()
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def first_news_timestamp(df):
    news_df = df[df["platform"] == "market_news"].copy()
    if news_df.empty:
        return None
    return news_df["timestamp_utc"].min()


def first_polymarket_event_timestamp(df):
    pm_df = df[df["platform"] == "polymarket_market"].copy()
    if pm_df.empty:
        return None
    return pm_df["timestamp_utc"].min()


def compute_event_vs_news_timing(df):
    t_news = first_news_timestamp(df)
    t_pm = first_polymarket_event_timestamp(df)

    result = {
        "t_news": t_news,
        "t_polymarket_event": t_pm,
        "lead_lag_minutes": None,
        "leader": None
    }

    if t_news is None and t_pm is None:
        result["leader"] = "no_signal"
        return result
    if t_news is not None and t_pm is None:
        result["leader"] = "news_only"
        return result
    if t_news is None and t_pm is not None:
        result["leader"] = "polymarket_event_only"
        return result

    delta_minutes = (t_pm - t_news).total_seconds() / 60.0
    result["lead_lag_minutes"] = delta_minutes

    if delta_minutes > 0:
        result["leader"] = "news_led"
    elif delta_minutes < 0:
        result["leader"] = "polymarket_event_led"
    else:
        result["leader"] = "simultaneous"

    return result


if __name__ == "__main__":
    df = load_panel()
    print("Platform counts:")
    print(df["platform"].value_counts(dropna=False).to_string())

    print("\nFirst news timestamp:", first_news_timestamp(df))
    print("First polymarket timestamp:", first_polymarket_event_timestamp(df))

    result = compute_event_vs_news_timing(df)

    print("\nEvent vs News Timing Result:")
    print(result)