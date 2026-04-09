# event_matcher.py

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    OUTPUT_EVENT_MATCHES,
    OUTPUT_KALSHI_EVENTS,
    OUTPUT_POLYMARKET_EVENTS,
    OUTPUT_NEWS_DATASET,
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


def date_proximity_score(pm_time: pd.Timestamp, news_time: pd.Timestamp) -> float:
    pm_time = pd.to_datetime(pm_time, utc=True, errors="coerce")
    news_time = pd.to_datetime(news_time, utc=True, errors="coerce")

    if pd.isna(pm_time) or pd.isna(news_time):
        return 0.0

    delta_minutes = abs((news_time - pm_time).total_seconds()) / 60.0

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
    pm_title: object,
    news_title: object,
    pm_time: pd.Timestamp,
    news_time: pd.Timestamp,
) -> float:
    jac = jaccard_score(pm_title, news_title)
    seq = sequence_score(pm_title, news_title)
    prox = date_proximity_score(pm_time, news_time)
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


def load_event_sources() -> pd.DataFrame:
    pm = _safe_read_csv(OUTPUT_POLYMARKET_EVENTS)
    ka = _safe_read_csv(OUTPUT_KALSHI_EVENTS)

    frames: list[pd.DataFrame] = []

    if not pm.empty:
        pm = pm.copy()
        if "title" in pm.columns and "polymarket_title" not in pm.columns:
            pm["polymarket_title"] = pm["title"]
        if "timestamp" in pm.columns and "polymarket_time" not in pm.columns:
            pm["polymarket_time"] = pm["timestamp"]
        if "source" not in pm.columns:
            pm["source"] = "polymarket"
        frames.append(pm)

    if not ka.empty:
        ka = ka.copy()
        if "title" in ka.columns and "polymarket_title" not in ka.columns:
            ka["polymarket_title"] = ka["title"]
        if "timestamp" in ka.columns and "polymarket_time" not in ka.columns:
            ka["polymarket_time"] = ka["timestamp"]
        if "source" not in ka.columns:
            ka["source"] = "kalshi"
        frames.append(ka)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True, sort=False)

    if "seed_label" not in df.columns:
        df["seed_label"] = np.nan
    if "polymarket_title" not in df.columns:
        df["polymarket_title"] = np.nan
    if "polymarket_time" not in df.columns:
        df["polymarket_time"] = np.nan
    if "leader" not in df.columns:
        df["leader"] = np.nan

    df["polymarket_time"] = pd.to_datetime(df["polymarket_time"], utc=True, errors="coerce")
    return df


def load_news() -> pd.DataFrame:
    news = _safe_read_csv(OUTPUT_NEWS_DATASET)
    if news.empty:
        return news

    news = news.copy()

    if "title" in news.columns and "news_title" not in news.columns:
        news["news_title"] = news["title"]

    if "timestamp_utc" in news.columns and "news_time" not in news.columns:
        news["news_time"] = news["timestamp_utc"]
    elif "published_at" in news.columns and "news_time" not in news.columns:
        news["news_time"] = news["published_at"]
    elif "timestamp" in news.columns and "news_time" not in news.columns:
        news["news_time"] = news["timestamp"]

    if "news_title" not in news.columns:
        news["news_title"] = np.nan
    if "news_time" not in news.columns:
        news["news_time"] = np.nan

    news["news_time"] = pd.to_datetime(news["news_time"], utc=True, errors="coerce")
    return news


def best_news_match_for_event(
    pm_title: object,
    pm_time: pd.Timestamp,
    news_df: pd.DataFrame,
    min_match_score: float = 0.15,
) -> dict | None:
    """
    Strict match window:
    - news must be within [pm_time - 7 days, pm_time + 2 days]
    - no fallback to full dataset
    - reject weak matches
    """
    pm_time = pd.to_datetime(pm_time, utc=True, errors="coerce")
    if pd.isna(pm_time) or news_df.empty:
        return None

    candidates = news_df[
        (news_df["news_time"].notna()) &
        (news_df["news_time"] >= pm_time - pd.Timedelta(days=7)) &
        (news_df["news_time"] <= pm_time + pd.Timedelta(days=2))
    ].copy()

    if candidates.empty:
        return None

    candidates["match_score"] = candidates.apply(
        lambda r: weighted_match_score(pm_title, r.get("news_title"), pm_time, r.get("news_time")),
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
    lead_lag_minutes convention:
      news_time - polymarket_time
      positive => news after event
      negative => news before event
    """
    if events_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for _, ev in events_df.iterrows():
        pm_title = ev.get("polymarket_title")
        pm_time = pd.to_datetime(ev.get("polymarket_time"), utc=True, errors="coerce")

        best = best_news_match_for_event(pm_title, pm_time, news_df)

        match_score = np.nan
        news_title = np.nan
        news_time = pd.NaT
        lead_lag_minutes = np.nan

        if best is not None:
            news_title = best.get("news_title")
            news_time = pd.to_datetime(best.get("news_time"), utc=True, errors="coerce")
            match_score = best.get("match_score", np.nan)

            if pd.isna(pm_time) or pd.isna(news_time):
                lead_lag_minutes = np.nan
            else:
                lead_lag_minutes = (news_time - pm_time).total_seconds() / 60.0

                if abs(lead_lag_minutes) > lag_cap_minutes:
                    lead_lag_minutes = np.nan

        rows.append(
            {
                "seed_label": ev.get("seed_label"),
                "symbol": ev.get("symbol"),
                "country": ev.get("country"),
                "bucket": ev.get("bucket"),
                "leader": ev.get("leader"),
                "source": ev.get("source"),
                "polymarket_title": pm_title,
                "news_title": news_title,
                "polymarket_time": pm_time,
                "news_time": news_time,
                "event_time": news_time if pd.notna(news_time) else pm_time,
                "match_score": match_score,
                "lead_lag_minutes": lead_lag_minutes,
            }
        )

    out = pd.DataFrame(rows)

    sort_cols = [c for c in ["seed_label", "event_time", "symbol"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def main() -> None:
    events_df = load_event_sources()
    news_df = load_news()

    if events_df.empty:
        print("No event source rows found.")
        pd.DataFrame().to_csv(OUTPUT_EVENT_MATCHES, index=False)
        print(f"Saved empty file: {OUTPUT_EVENT_MATCHES}")
        return

    if news_df.empty:
        print("No news rows found.")
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