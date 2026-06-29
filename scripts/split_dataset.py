"""scripts/split_dataset.py

Split train.csv into N batch files of exactly 10 rows each.
The SalePrice (target) column is dropped — batches simulate raw incoming data.

Usage
-----
    python scripts/split_dataset.py \
        --input  ~/Downloads/house-prices/train.csv \
        --out    data/raw_data \
        --num-files 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from scripts.features import FEATURE_COLUMNS

ROWS_PER_FILE = 10


def split(input_csv: str, out_dir: str, num_files: int) -> None:
    src = Path(input_csv)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)

    # Keep only the 15 feature columns (drop SalePrice and anything else)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    df = df[FEATURE_COLUMNS].reset_index(drop=True)

    total_rows_needed = num_files * ROWS_PER_FILE
    if len(df) < total_rows_needed:
        raise ValueError(
            f"Not enough rows: need {total_rows_needed}, got {len(df)}"
        )

    for i in range(num_files):
        batch = df.iloc[i * ROWS_PER_FILE : (i + 1) * ROWS_PER_FILE]
        assert len(batch) == ROWS_PER_FILE, f"Batch {i} has {len(batch)} rows"
        out_path = out / f"batch_{i:04d}.csv"
        batch.to_csv(out_path, index=False)

    print(f"✅ Created {num_files} files of {ROWS_PER_FILE} rows in '{out}'")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split train.csv into batch files.")
    parser.add_argument("--input",     required=True, help="Path to train.csv")
    parser.add_argument("--out",       required=True, help="Output directory (data/raw_data)")
    parser.add_argument("--num-files", type=int, default=30, help="Number of batch files (default 30)")
    args = parser.parse_args(argv)

    split(args.input, args.out, args.num_files)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
