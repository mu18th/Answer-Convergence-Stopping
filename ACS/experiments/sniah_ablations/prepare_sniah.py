"""Prepare and validate the 250-row S-NIAH ablation build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    # Keep this experiment runnable from its own directory while reusing the
    # canonical root benchmark builder instead of carrying another drifting copy.
    sys.path.append(str(SOURCE_ROOT))

from benchmark_data.sniah_data import sniah
from common import load_config

EXPECTED_ROWS = 250
EXPECTED_BUCKETS = {8192, 16384, 32768, 65536, 131072}


def validate_rows(rows: list[dict]) -> list[dict]:
    """Validate the complete 250-row ablation cohort without applying a split."""
    ids = [str(row["id"]) for row in rows]
    buckets = {int(row["bucket"]) for row in rows}
    counts = {bucket: sum(int(row["bucket"]) == bucket for row in rows)
              for bucket in sorted(buckets)}
    if len(rows) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} unique rows, found {len(rows)} rows and "
            f"{len(set(ids))} unique IDs"
        )
    if buckets != EXPECTED_BUCKETS or set(counts.values()) != {50}:
        raise RuntimeError(f"Expected 50 rows in each S-NIAH bucket, found {counts}")
    print(
        f"Validated S-NIAH: all {len(rows)} rows, bucket counts={counts}",
        flush=True,
    )
    return rows


def validate(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return validate_rows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_sniah_qwen14b_base.yaml")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    cfg = load_config(str(config_path))
    data_path = root / "data" / "sniah.jsonl"
    if not args.validate_only:
        sniah(cfg)
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    validate(data_path)


if __name__ == "__main__":
    main()
