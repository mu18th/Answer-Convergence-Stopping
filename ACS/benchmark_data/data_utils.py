from __future__ import annotations

from pathlib import Path


def balanced_quota(sizes, n):
    quota = {key: 0 for key in sizes}
    remaining = n
    while remaining > 0 and any(quota[key] < sizes[key] for key in sizes):
        for key in sorted(sizes):
            if remaining == 0:
                break
            if quota[key] < sizes[key]:
                quota[key] += 1
                remaining -= 1
    return quota


def output_path(cfg, name):
    path = Path(cfg["paths"]["data_dir"]) / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
