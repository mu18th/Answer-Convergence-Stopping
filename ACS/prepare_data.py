"""CLI dispatcher for benchmark-specific data builders.

python prepare_data.py longbench_v2 --config CONFIG
python prepare_data.py sniah --config CONFIG
"""

from __future__ import annotations

import argparse

from benchmark_data import longbench_v2, sniah
from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bench",
        choices=[
            "longbench_v2",
            "sniah",
        ],
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit(
            f"'{args.bench}' needs the `datasets` package, which is not installed.\n"
            "  pip install -r requirements.txt"
        )
    {
        "longbench_v2": longbench_v2,
        "sniah": sniah,
    }[args.bench](config)


if __name__ == "__main__":
    main()
