# run_pipeline.py

import subprocess
import sys

MANDATORY_FILES = [
    "matcher.py",
    "lead_lag.py",
    "event_matcher.py",
    "build_dataset.py",
    "archive_run.py",
]

OPTIONAL_FILES = [
    "yahoo_enrich.py",
    "index_patterns.py",
    "index_mapper.py",
    "build_timeseries_dataset.py",
    "signal_engine.py",
]


def run_file(filename, required=True):
    print("\n" + "=" * 80)
    print(f"RUNNING: {filename}")
    print("=" * 80)

    result = subprocess.run([sys.executable, filename], capture_output=False, text=True)

    if result.returncode != 0:
        if required:
            raise SystemExit(f"Pipeline stopped. Failed at: {filename}")
        print(f"Optional step failed, continuing: {filename}")


if __name__ == "__main__":
    # Current event/news pipeline
    for file in MANDATORY_FILES[:3]:
        run_file(file, required=True)

    # Optional enrichments and downstream layers
    for file in OPTIONAL_FILES:
        run_file(file, required=False)

    # Summary dataset + archive
    for file in MANDATORY_FILES[3:]:
        run_file(file, required=True)

    print("\nPipeline completed successfully.")