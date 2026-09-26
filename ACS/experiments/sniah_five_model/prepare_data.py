"""Prepare the pinned S-NIAH replication dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from benchmark_data import sniah
from common import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", choices=["sniah"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    try:
        import datasets  # noqa: F401
    except ImportError as error:
        raise SystemExit(
            "Install the release requirements before data preparation"
        ) from error
    sniah(load_config(args.config))


if __name__ == "__main__":
    main()
