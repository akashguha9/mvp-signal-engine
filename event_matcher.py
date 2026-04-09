# event_matcher.py

import math
import pandas as pd
from difflib import SequenceMatcher

from config import OUTPUT_EVENT_MATCHES, OUTPUT_EVENT_PANEL


STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "by", "and",
    "or", "is", "are", "be", "will", "with", "from", "as", "that", "this",
    "it", "its", "their", "his", "her", "than", "into", "over", "under",
    "up", "down", "more", "less", "before", "after", "between", "about"
}


def norm_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def text_similarity(a, b):
    return SequenceMatcher(None, norm_text(a), norm_text(b)).ratio()


def tokenize(text):
    text = norm_text(text)
    tokens = []
    for word in text.replace("?", " ").replace(",", " ").replace(".", " ").replace("-", " ").split():
        word = word.strip()
        if len(word) > 2 and word not in STOPWORDS:
            tokens.append(word)
    return set(tokens)


def keyword_overlap(a, b):
    ta = tokenize(a)
    tb = tokenize(b)

    if not ta or not tb:
        return 0.0

    inter = len(ta & tb)
    union = len(ta | tb)

    if union == 0:
        return 0.0

    return inter / union


def date_proximity_score(t1, t2, half_life_days=30):
    """
    Exponential decay by time distance.
    1.0 when same timestamp, falls as gap widens.
    """
    if pd.isna(t1) or pd.isna(t2):
        return 0.0

    delta_days = abs((t1 - t2).total_seconds()) / 86400.0
    return math.exp(-delta_days / half_life_days)


def load_event_panel(path=OUTPUT_EVENT_PANEL):
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


def split_panel(df):
    pm = df[df["platform"] == "polymarket_market"].copy()
    news = df[df["platform"] == "market_news"].copy()
    return pm, news


def weighted_match_score(pm_title, news_title, pm_time, news_time):
    sim = text_similarity(pm_title, news_title)
    overlap = keyword_overlap(pm_title, news_title)
    prox = date_proximity_score(pm_time, news_time)

    score = 0.5 * sim + 0.3 * overlap + 0.2 * prox

    return {
        "text_similarity": round(sim, 4),
        "keyword_overlap": round(overlap, 4),
        "date_proximity": round(prox, 4),
        "weighted_score": round(score, 4),
    }


def match_polymarket_to_news(pm_df, news_df, min_score=0.15):
    rows = []

    for _, pm_row in pm_df.iterrows():
        pm_title = pm_row.get("headline") or pm_row.get("market_title") or ""
        pm_time = pm_row.get("timestamp_utc")

        best = None
        best_score = -1.0

        for _, news_row in news_df.iterrows():
            news_title = news_row.get("headline") or ""
            news_time = news_row.get("timestamp_utc")

            scores = weighted_match_score(pm_title, news_title, pm_time, news_time)

            if scores["weighted_score"] > best_score:
                best_score = scores["weighted_score"]
                best = {
                    "news_title": news_title,
                    "news_time": news_time,
                    **scores
                }

        if best is not None and best["weighted_score"] >= min_score:
            delta_minutes = (pm_time - best["news_time"]).total_seconds() / 60.0

            if delta_minutes > 0:
                leader = "news_led"
            elif delta_minutes < 0:
                leader = "polymarket_led"
            else:
                leader = "simultaneous"

            rows.append({
                "seed_label": pm_row.get("seed_label"),
                "polymarket_ref": pm_row.get("market_ref"),
                "polymarket_title": pm_title,
                "polymarket_time": pm_time,
                "news_title": best["news_title"],
                "news_time": best["news_time"],
                "text_similarity": best["text_similarity"],
                "keyword_overlap": best["keyword_overlap"],
                "date_proximity": best["date_proximity"],
                "match_score": best["weighted_score"],
                "lead_lag_minutes": delta_minutes,
                "leader": leader,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        ["match_score", "seed_label", "polymarket_time"],
        ascending=[False, True, True]
    ).reset_index(drop=True)


if __name__ == "__main__":
    df = load_event_panel()
    pm_df, news_df = split_panel(df)

    print(f"Polymarket rows: {len(pm_df)}")
    print(f"News rows: {len(news_df)}")

    matched_df = match_polymarket_to_news(pm_df, news_df, min_score=0.15)

    if matched_df.empty:
        print("\nNo usable event matches found.")
    else:
        print("\nMatched event rows:")
        print(matched_df.head(20).to_string(index=False))
        matched_df.to_csv(OUTPUT_EVENT_MATCHES, index=False)
        print(f"\nSaved: {OUTPUT_EVENT_MATCHES}")