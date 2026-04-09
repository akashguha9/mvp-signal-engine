# archive_run.py

import os
import shutil
from datetime import datetime, timezone

from config import (
    OUTPUT_SEED_SUMMARY,
    OUTPUT_PRICE_PANEL,
    OUTPUT_NEWS_PANEL,
    OUTPUT_EVENT_PANEL,
    OUTPUT_EVENT_MATCHES,
    OUTPUT_YAHOO_ENRICH,
    OUTPUT_MODEL_DATASET,
)


ARCHIVE_ROOT = "archive"


FILES_TO_ARCHIVE = [
    OUTPUT_SEED_SUMMARY,
    OUTPUT_PRICE_PANEL,
    OUTPUT_NEWS_PANEL,
    OUTPUT_EVENT_PANEL,
    OUTPUT_EVENT_MATCHES,
    OUTPUT_YAHOO_ENRICH,
    OUTPUT_MODEL_DATASET,
]


def make_archive_dir():
    now = datetime.now(timezone.utc)
    folder_name = now.strftime("%Y-%m-%d_%H-%M-%S_UTC")
    archive_dir = os.path.join(ARCHIVE_ROOT, folder_name)
    os.makedirs(archive_dir, exist_ok=True)
    return archive_dir, now


def archive_files(archive_dir):
    copied = []
    missing = []

    for file_name in FILES_TO_ARCHIVE:
        if os.path.exists(file_name):
            dst = os.path.join(archive_dir, os.path.basename(file_name))
            shutil.copy2(file_name, dst)
            copied.append(file_name)
        else:
            missing.append(file_name)

    return copied, missing


def write_manifest(archive_dir, run_time, copied, missing):
    manifest_path = os.path.join(archive_dir, "manifest.txt")

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(f"archive_created_utc: {run_time.isoformat()}\n")
        f.write("\nCOPIED FILES\n")
        f.write("=" * 40 + "\n")
        for item in copied:
            f.write(f"{item}\n")

        f.write("\nMISSING FILES\n")
        f.write("=" * 40 + "\n")
        for item in missing:
            f.write(f"{item}\n")

    return manifest_path


if __name__ == "__main__":
    archive_dir, run_time = make_archive_dir()
    copied, missing = archive_files(archive_dir)
    manifest_path = write_manifest(archive_dir, run_time, copied, missing)

    print(f"Archive folder: {archive_dir}")
    print("\nCopied files:")
    for item in copied:
        print(f"- {item}")

    if missing:
        print("\nMissing files:")
        for item in missing:
            print(f"- {item}")

    print(f"\nManifest written: {manifest_path}")