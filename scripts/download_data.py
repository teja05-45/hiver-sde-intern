#!/usr/bin/env python3
"""
Download the "Customer Support on Twitter" dataset from Kaggle.

Usage:
    python scripts/download_data.py

Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables (see
.env.example) and the `kaggle` CLI package. If you don't have Kaggle API
credentials, download manually instead:

  1. Go to https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter
  2. Download and extract the archive
  3. Place twcs.csv at data/raw/twcs.csv

This script is a thin, documented wrapper -- it does not hide what it's
doing or silently retry forever. If the Kaggle CLI isn't installed or
credentials are missing, it prints the manual fallback instructions above
and exits non-zero, rather than failing confusingly deep in a stack trace.
"""
from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
DATASET_SLUG = "thoughtvector/customer-support-on-twitter"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "twcs.csv"
    if target.exists():
        print(f"{target} already exists. Delete it first if you want to re-download.")
        return

    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print(
            "KAGGLE_USERNAME / KAGGLE_KEY not set.\n\n"
            "Manual fallback:\n"
            f"  1. Download from https://www.kaggle.com/datasets/{DATASET_SLUG}\n"
            "  2. Extract the archive\n"
            f"  3. Place twcs.csv at {target}\n"
        )
        sys.exit(1)

    try:
        subprocess.run(["kaggle", "--version"], check=True, capture_output=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("The `kaggle` CLI is not installed. Run `pip install kaggle` first, or use the "
              "manual fallback described above.")
        sys.exit(1)

    print(f"Downloading {DATASET_SLUG} via Kaggle CLI ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET_SLUG, "-p", str(RAW_DIR)], check=True)

    zip_path = RAW_DIR / "customer-support-on-twitter.zip"
    if zip_path.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    if not target.exists():
        print(f"Download/extract completed but {target} was not found. Check {RAW_DIR} contents "
              "and rename/move the CSV manually if needed.")
        sys.exit(1)

    print(f"Done: {target}")


if __name__ == "__main__":
    main()
