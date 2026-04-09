# event_matcher.py

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OUTPUT_EVENT_PANEL,
    OUTPUT_NEWS_PANEL,
    OUTPUT_EVENT_MATCHES,
)


STOPWORDS = {
    "the", "a", "an", "of", "to", "for", "in", "on", "at", "by", "with", "from",
    "and", "or", "is", "are", "be", "will", "would", "could", "should", "has",
    "have", "had", "after", "before", "into", "over", "under", "about", "new",
    "latest", "report", "reports", "says", "say", "amid", "as", "that", "this",
}


def normalize_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).lower().strip()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(x: object) -> list[str]:
    s = normalize_text(x)
    if not s:
        return []
    return [tok for tok in s.split() if tok not in STOPWORDS and len(tok) > 1]


def jaccard_score(a: object, b: object) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def sequence_score(a: object, b: object) -> float:
    na = normalize_text(a)
    nb = normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def date_proximity_score(event_time: pd.Timestamp, news_time: pd.Timestamp) -> float:
    event_time = pd.to_datetime(event_time, utc=True, errors="coerce")
    news_time = pd.to_datetime(news_time, utc=True, errors="coerce")

    if pd.isna(event_time) or pd.isna(news_time):
        return 0.0

    delta_minutes = abs((news_time - event_time).total_seconds()) / 60.0

    if delta_minutes <= 60:
        return 1.0
    if delta_minutes <= 6 * 60:
        return 0.85
    if delta_minutes <= 24 * 60:
        return 0.60
    if delta_minutes <= 3 * 24 * 60:
        return 0.35
    if delta_minutes <= 7 * 24 * 60:
        return 0.15
    return 0.0


def weighted_match_score(
    event_title: object,
    news_title: object,
    event_time: pd.Timestamp,
    news_time: pd.Timestamp,
) -> float:
    jac = jaccard_score(event_title, news_title)
    seq = sequence_score(event_title, news_title)
    prox = date_proximity_score(event_time, news_time)
    score = 0.45 * jac + 0.35 * seq + 0.20 * prox
    return float(round(score, 6))


def _safe_read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_event_panel() -> pd.DataFrame:
    df = _safe_read_csv(OUTPUT_EVENT_PANEL)
    if df.empty:
        return df

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce", format="mixed")

    # Normalize to matcher schema
    df["polymarket_time"] = df["timestamp_utc"]
    df["polymarket_title"] = df["market_title"]
    df["leader"] = df.get("source_provider", np.nan)
    df["symbol"] = df.get("market_ref", np.nan)
    df["country"] = np.nan
    df["bucket"] = np.nan
    df["source"] = df.get("platform", "event_panel")

    return df


def load_news_panel() -> pd.DataFrame:
    df = _safe_read_csv(OUTPUT_NEWS_PANEL)
    if df.empty:
        return df

    df = df.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce", format="mixed")

    # Normalize to matcher schema
    df["news_time"] = df["timestamp_utc"]
    df["news_title"] = df["market_title"]

    return df


def best_news_match_for_event(
    event_title: object,
    event_time: pd.Timestamp,
    news_df: pd.DataFrame,
    seed_label: object = None,
    min_match_score: float = 0.15,
) -> dict | None:
    """
    Strict window:
    - same seed_label if available
    - news within [event_time - 7 days, event_time + 2 days]
    - no fallback to full dataset
    - reject weak matches
    """
    event_time = pd.to_datetime(event_time, utc=True, errors="coerce")
    if pd.isna(event_time) or news_df.empty:
        return None

    candidates = news_df.copy()

    if seed_label is not None and "seed_label" in candidates.columns:
        candidates = candidates[candidates["seed_label"] == seed_label]

    candidates = candidates[
        (candidates["news_time"].notna()) &
        (candidates["news_time"] >= event_time - pd.Timedelta(days=7)) &
        (candidates["news_time"] <= event_time + pd.Timedelta(days=2))
    ].copy()

    if candidates.empty:
        return None

    candidates["match_score"] = candidates.apply(
        lambda r: weighted_match_score(
            event_title,
            r.get("news_title"),
            event_time,
            r.get("news_time"),
        ),
        axis=1,
    )

    candidates = candidates.sort_values(
        ["match_score", "news_time"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if candidates.empty:
        return None

    best = candidates.iloc[0].to_dict()

    if float(best.get("match_score", 0.0)) < min_match_score:
        return None

    return best


def build_event_matches(
    events_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lag_cap_minutes: float = 10080.0,
) -> pd.DataFrame:
    """
    lead_lag_minutes = news_time - polymarket_time
    positive => news after market event
    negative => news before market event
    """
    if events_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for _, ev in events_df.iterrows():
        event_title = ev.get("polymarket_title")
        event_time = pd.to_datetime(ev.get("polymarket_time"), utc=True, errors="coerce")
        seed_label = ev.get("seed_label")

        best = best_news_match_for_event(
            event_title=event_title,
            event_time=event_time,
            news_df=news_df,
            seed_label=seed_label,
            min_match_score=0.15,
        )

        match_score = np.nan
        news_title = np.nan
        news_time = pd.NaT
        lead_lag_minutes = np.nan

        if best is not None:
            news_title = best.get("news_title")
            news_time = pd.to_datetime(best.get("news_time"), utc=True, errors="coerce")
            match_score = best.get("match_score", np.nan)

            if pd.notna(event_time) and pd.notna(news_time):
                lead_lag_minutes = (news_time - event_time).total_seconds() / 60.0
                if abs(lead_lag_minutes) > lag_cap_minutes:
                    lead_lag_minutes = np.nan

        rows.append(
            {
                "seed_label": seed_label,
                "symbol": ev.get("symbol"),
                "country": ev.get("country"),
                "bucket": ev.get("bucket"),
                "leader": ev.get("leader"),
                "source": ev.get("source"),
                "polymarket_title": event_title,
                "news_title": news_title,
                "polymarket_time": event_time,
                "news_time": news_time,
                "event_time": news_time if pd.notna(news_time) else event_time,
                "match_score": match_score,
                "lead_lag_minutes": lead_lag_minutes,
            }
        )

    out = pd.DataFrame(rows)

    if not out.empty:
        out = out.sort_values(["seed_label", "event_time", "symbol"], na_position="last").reset_index(drop=True)

    return out


def main() -> None:
    events_df = load_event_panel()
    news_df = load_news_panel()

    if events_df.empty:
        print("No event panel rows found.")
        pd.DataFrame().to_csv(OUTPUT_EVENT_MATCHES, index=False)
        print(f"Saved empty file: {OUTPUT_EVENT_MATCHES}")
        return

    if news_df.empty:
        print("No news panel rows found.")
        out = events_df.copy()
        out["news_title"] = np.nan
        out["news_time"] = pd.NaT
        out["event_time"] = out["polymarket_time"]
        out["match_score"] = np.nan
        out["lead_lag_minutes"] = np.nan
        out.to_csv(OUTPUT_EVENT_MATCHES, index=False)
        print(f"Saved fallback file: {OUTPUT_EVENT_MATCHES}")
        return

    out = build_event_matches(events_df, news_df, lag_cap_minutes=10080.0)
    out.to_csv(OUTPUT_EVENT_MATCHES, index=False)

    print("Event matching complete.")
    print(f"Rows: {len(out)}")

    if "match_score" in out.columns:
        print("\nMatch score summary:")
        print(out["match_score"].describe().to_string())

    if "lead_lag_minutes" in out.columns:
        print("\nLead/lag summary:")
        print(out["lead_lag_minutes"].describe().to_string())

    print(f"\nSaved: {OUTPUT_EVENT_MATCHES}")


if __name__ == "__main__":
    main()