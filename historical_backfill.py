# historical_backfill.py

import os
import time
import random
import pandas as pd

from gdelt_news import fetch_gdelt_articles
from seed_pairs import SEED_MARKETS


HISTORICAL_ROOT = os.path.join("historical", "news", "gdelt")


def month_ranges(start_date, end_date):
    start_ts = pd.to_datetime(start_date, utc=True)
    end_ts = pd.to_datetime(end_date, utc=True)

    current = start_ts.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    while current <= end_ts:
        next_month = current + pd.offsets.MonthBegin(1)
        month_end = min(next_month - pd.Timedelta(seconds=1), end_ts)
        label = current.strftime("%Y-%m")
        yield current, month_end, label
        current = next_month


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fetch_with_retries(query, start_date, end_date, theme_bucket, max_records=150, max_attempts=5):
    """
    Retry GDELT with backoff when 429 happens.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fetch_gdelt_articles(
                query=query,
                start_date=start_date,
                end_date=end_date,
                max_records=max_records,
                theme_bucket=theme_bucket,
            )
        except Exception as e:
            msg = str(e)

            if "429" in msg:
                sleep_secs = min(60, (2 ** attempt) + random.uniform(0.5, 2.5))
                print(f"429 hit. Sleeping {sleep_secs:.1f}s before retry {attempt}/{max_attempts}...")
                time.sleep(sleep_secs)
                continue

            raise

    raise RuntimeError(f"GDELT failed after {max_attempts} attempts.")


def backfill_seed(seed, start_date, end_date, max_records=150):
    theme_bucket = seed.get("theme_bucket", "unknown")
    query = seed.get("news_symbol_or_query") or seed.get("label")

    seed_dir = os.path.join(HISTORICAL_ROOT, seed["label"])
    ensure_dir(seed_dir)

    for month_start, month_end, label in month_ranges(start_date, end_date):
        out_path = os.path.join(seed_dir, f"{label}.csv")

        if os.path.exists(out_path):
            print(f"Skipping existing: {out_path}")
            continue

        print(f"Backfilling {seed['label']} | {label}")

        try:
            df = fetch_with_retries(
                query=query,
                start_date=month_start.isoformat(),
                end_date=month_end.isoformat(),
                theme_bucket=theme_bucket,
                max_records=max_records,
                max_attempts=5,
            )
        except Exception as e:
            print(f"Failed {seed['label']} {label}: {e}")
            continue

        df.to_csv(out_path, index=False)
        print(f"Saved {out_path} | rows={len(df)}")

        # polite spacing between successful requests
        time.sleep(random.uniform(2.0, 5.0))


if __name__ == "__main__":
    target_labels = {"macro_spy"}
    end_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for seed in SEED_MARKETS:
        if seed["label"] not in target_labels:
            continue

        start_date = seed.get("historical_start", "2020-01-01")

        backfill_seed(
            seed=seed,
            start_date=start_date,
            end_date=end_date,
            max_records=150,
        )

    print("\nHistorical backfill complete.")