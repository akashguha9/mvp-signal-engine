"""
milk_test_polymarket_history.py
================================
STANDALONE script. No imports from your project.
No writes to your project files.
No changes to src/ or data/.

Purpose: test ONE hypothesis in isolation before touching fetch_polymarket.py.

HYPOTHESIS
----------
Polymarket GAMMA /prices-history endpoint returns daily price data
when called with condition_id (not clob token_id).

Endpoint under test:
  GET https://gamma-api.polymarket.com/prices-history
  ?market={condition_id}
  &startTs={unix_30_days_ago}
  &endTs={unix_now}
  &fidelity=1440

condition_id is read from your existing data/processed/polymarket_markets.csv.
The script picks the first valid row automatically.
"""

from __future__ import annotations
import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 15


# -------------------------
# Helpers
# -------------------------

def get_condition_id_from_csv(csv_path: str):
    import csv
    p = Path(csv_path)

    if not p.exists():
        print(f"[FAIL] CSV not found: {p.resolve()}")
        return None, None

    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        print(f"[INFO] CSV headers: {headers}")

        cid_col = next((h for h in headers if h.lower() in ("conditionid", "condition_id", "id")), None)
        q_col = next((h for h in headers if h.lower() in ("question", "title", "slug")), None)

        if not cid_col:
            print("[FAIL] No condition_id column found.")
            return None, None

        for row in reader:
            cid = row.get(cid_col, "").strip()
            q = row.get(q_col, "") if q_col else ""
            if cid and len(cid) > 10:
                return cid, q

    print("[FAIL] No valid condition_id found in CSV.")
    return None, None


def test_gamma_history(condition_id: str, question: str = "", days_back: int = 30):
    now = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())

    url = f"{GAMMA_BASE}/prices-history"
    params = {
        "market": condition_id,
        "startTs": start_ts,
        "endTs": now,
        "fidelity": 1440,
    }

    print("\n========== TESTING GAMMA ==========")
    print(f"condition_id: {condition_id}")
    print(f"question    : {question}")
    print(f"url         : {url}")
    print(f"params      : {params}")
    print("==================================\n")

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
    except Exception as e:
        print(f"[FAIL] Network error: {e}")
        return False

    print(f"[INFO] Status: {resp.status_code}")
    print(f"[INFO] Body (first 300 chars):\n{resp.text[:300]}\n")

    if resp.status_code != 200:
        print("[FAIL] Non-200 response")
        return False

    try:
        data = resp.json()
    except Exception as e:
        print(f"[FAIL] JSON parse error: {e}")
        return False

    history = data if isinstance(data, list) else data.get("history", [])

    if not history:
        print("[FAIL] No history returned (empty array)")
        return False

    print(f"[PASS] Got {len(history)} rows")
    print(f"First row: {history[0]}")
    print(f"Last row : {history[-1]}")

    print("\n===== MILK TEST: PASS =====")
    return True


def test_clob_fallback(token_id: str):
    now = int(datetime.now(timezone.utc).timestamp())
    start_ts = now - 30 * 86400

    url = "https://clob.polymarket.com/prices-history"
    params = {
        "market": token_id,
        "startTs": start_ts,
        "endTs": now,
        "fidelity": 1440,
    }

    print("\n========== TESTING CLOB ==========")
    print(f"token_id: {token_id}")
    print(f"url     : {url}")
    print("=================================\n")

    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
    except Exception as e:
        print(f"[FAIL] Network error: {e}")
        return False

    print(f"[INFO] Status: {resp.status_code}")
    print(f"[INFO] Body (first 300 chars):\n{resp.text[:300]}\n")

    if resp.status_code != 200:
        print("[FAIL] CLOB failed")
        return False

    data = resp.json()
    history = data if isinstance(data, list) else data.get("history", [])

    if not history:
        print("[FAIL] No history in CLOB")
        return False

    print(f"[PASS] CLOB got {len(history)} rows")
    return True


# -------------------------
# MAIN
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="data/processed/polymarket_markets.csv")
    parser.add_argument("--condition-id", default=None)
    parser.add_argument("--clob-fallback", action="store_true")

    args = parser.parse_args()

    if args.condition_id:
        cid = args.condition_id
        question = ""
    else:
        cid, question = get_condition_id_from_csv(args.csv_path)
        if not cid:
            print("[ABORT] No condition_id found.")
            sys.exit(1)

    success = test_gamma_history(cid, question)

    if not success and args.clob_fallback:
        print("\nTrying fallback...")
        test_clob_fallback(cid)


if __name__ == "__main__":
    main()