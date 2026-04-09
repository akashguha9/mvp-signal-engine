# signal_diff_audit.py

from __future__ import annotations

import sys
import pandas as pd


def vc(series: pd.Series) -> pd.Series:
    return series.astype(str).value_counts(dropna=False)


def main():
    if len(sys.argv) < 3:
        print("Usage: python signal_diff_audit.py old.csv new.csv")
        return

    old_file = sys.argv[1]
    new_file = sys.argv[2]

    old = pd.read_csv(old_file)
    new = pd.read_csv(new_file)

    print("=== SIGNAL DIFF AUDIT ===")
    print(f"old_file: {old_file}")
    print(f"new_file: {new_file}")
    print(f"old rows: {len(old)}")
    print(f"new rows: {len(new)}")

    print("\n=== SIGNAL COUNTS ===")
    print("OLD:")
    if "signal" in old.columns:
        print(old["signal"].value_counts(dropna=False).to_string())
    else:
        print("signal column missing")

    print("\nNEW:")
    if "signal" in new.columns:
        print(new["signal"].value_counts(dropna=False).to_string())
    else:
        print("signal column missing")

    print("\n=== TOP REASONS DELTA ===")
    if "reason" in old.columns and "reason" in new.columns:
        old_r = vc(old["reason"]).rename("old_count")
        new_r = vc(new["reason"]).rename("new_count")
        reason_cmp = pd.concat([old_r, new_r], axis=1).fillna(0)
        reason_cmp["old_count"] = reason_cmp["old_count"].astype(int)
        reason_cmp["new_count"] = reason_cmp["new_count"].astype(int)
        reason_cmp["delta"] = reason_cmp["new_count"] - reason_cmp["old_count"]

        print("\nMost increased:")
        print(reason_cmp.sort_values("delta", ascending=False).head(30).to_string())

        print("\nMost decreased:")
        print(reason_cmp.sort_values("delta", ascending=True).head(30).to_string())
    else:
        print("reason column missing in one of the files")

    if "regime" in old.columns and "regime" in new.columns:
        print("\n=== REGIME COUNTS DELTA ===")
        old_g = vc(old["regime"]).rename("old_count")
        new_g = vc(new["regime"]).rename("new_count")
        regime_cmp = pd.concat([old_g, new_g], axis=1).fillna(0)
        regime_cmp["old_count"] = regime_cmp["old_count"].astype(int)
        regime_cmp["new_count"] = regime_cmp["new_count"].astype(int)
        regime_cmp["delta"] = regime_cmp["new_count"] - regime_cmp["old_count"]
        print(regime_cmp.to_string())

    key_cols = ["event_time", "seed_label", "symbol"]
    if all(c in old.columns for c in key_cols) and all(c in new.columns for c in key_cols):
        old_key = old[key_cols + [c for c in ["reason", "signal", "regime"] if c in old.columns]].copy()
        new_key = new[key_cols + [c for c in ["reason", "signal", "regime"] if c in new.columns]].copy()

        merged = old_key.merge(
            new_key,
            on=key_cols,
            how="outer",
            suffixes=("_old", "_new"),
            indicator=True
        )

        compare_cols = []
        for c in ["signal", "reason", "regime"]:
            if f"{c}_old" in merged.columns and f"{c}_new" in merged.columns:
                compare_cols.append(c)

        changed_mask = merged["_merge"] != "both"

        if "signal" in compare_cols:
            changed_mask |= merged["signal_old"] != merged["signal_new"]
        if "reason" in compare_cols:
            changed_mask |= merged["reason_old"].astype(str) != merged["reason_new"].astype(str)
        if "regime" in compare_cols:
            changed_mask |= merged["regime_old"].astype(str) != merged["regime_new"].astype(str)

        changed = merged[changed_mask].copy()

        print("\n=== ROW-LEVEL CHANGES ===")
        print(f"changed rows: {len(changed)}")

        if not changed.empty:
            show_cols = key_cols + [c for c in changed.columns if c not in key_cols and c != "_merge"] + ["_merge"]
            print(changed[show_cols].head(50).to_string(index=False))
    else:
        print("\n=== ROW-LEVEL CHANGES ===")
        print("Cannot compare row-level changes because one of these key columns is missing:")
        print(", ".join(key_cols))

    for label, df in [("OLD", old), ("NEW", new)]:
        if "signal" in df.columns:
            directional = df[df["signal"].isin([1, -1])].copy()
            print(f"\n=== {label} DIRECTIONAL SUMMARY ===")
            print(f"directional rows: {len(directional)}")

            for metric in ["signal_success_1d", "signal_success_3d", "signal_success_5d", "signal_success_20d"]:
                if metric in directional.columns:
                    val = pd.to_numeric(directional[metric], errors="coerce").dropna()
                    print(f"{metric}: {float(val.mean()) if len(val) else 'nan'}")


if __name__ == "__main__":
    main()