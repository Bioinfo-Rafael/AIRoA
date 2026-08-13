#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from airoa.data.libero_plus import download_metadata, inspect_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_metadata.json"))
    args = parser.parse_args()
    snapshot = download_metadata(args.cache_dir)
    report = inspect_metadata(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
