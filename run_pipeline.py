# run_pipeline.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPELINE_STEPS = [
    "matcher.py",
    "lead_lag.py",
    "event_matcher.py",
    "build_seed_index_panel.py",
    "index_mapper.py",
    "build_timeseries_dataset.py",
    "signal_engine.py",
    "backtest_signals.py",
    "archive_run.py",
]


def run_step(script_name: str) -> None:
    script_path = Path(script_name)

    if not script_path.exists():
        raise FileNotFoundError(f"Missing pipeline step: {script_name}")

    print("\n" + "=" * 80)
    print(f"RUNNING: {script_name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_name],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name} (exit code {result.returncode})")

    print(f"COMPLETED: {script_name}")


def main() -> None:
    print("Starting full pipeline...")

    for step in PIPELINE_STEPS:
        run_step(step)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()