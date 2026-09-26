"""Evidence-relative RULER analysis using the first literal gold-answer mention.

This is explicitly a lower-bound proxy for complete supporting evidence.  It
never treats the question text as evidence and excludes yes/no-style answers,
whose literal occurrences are not meaningful evidence locations.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import read_jsonl
from analyze_ruler_hotpotqa import (
    official_score, stop_gate, stop_verbal, stop_windowed, validate,
)


def answers(value) -> list[str]:
    raw = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in raw
            if len(str(item).strip()) >= 3 and str(item).strip().casefold()
            not in {"yes", "no", "true", "false"}]


def first_proxy_chunk(data_row: dict, chunk_chars: int) -> int | None:
    context = str(data_row["context"])
    folded = context.casefold()
    positions = [folded.find(answer.casefold()) for answer in answers(data_row["answer"])]
    positions = [position for position in positions if position >= 0]
    return min(positions) // chunk_chars + 1 if positions else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--theta", type=float, default=0.995)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--verbal-threshold", type=float, default=99.5)
    parser.add_argument("--expected-located", type=int, default=455)
    args = parser.parse_args()
    data = [row for path in args.data for row in read_jsonl(path)]
    trajectories = validate(
        [row for path in args.traj for row in read_jsonl(path)], data
    )
    data_by_id = {row["id"]: row for row in data}
    located = []
    for tr in trajectories:
        proxy = first_proxy_chunk(data_by_id[tr["id"]], int(tr["chunk_chars"]))
        if proxy is not None:
            located.append((tr, proxy))
    if args.expected_located and len(located) != args.expected_located:
        raise RuntimeError(
            f"Expected {args.expected_located} located proxies, found {len(located)}; "
            "do not report a silently changed evidence cohort"
        )

    def fixed25(tr):
        return max(1, math.ceil(0.25 * len(tr["steps"])))

    policies = [
        ("fixed at 25%", fixed25),
        ("verbalized gate @99.5",
         lambda tr: stop_verbal(tr, args.verbal_threshold)),
        ("END gate", stop_gate),
        ("ACS, fixed",
         lambda tr: stop_windowed(tr, args.theta, args.eps, args.window)),
        ("full reading", lambda tr: len(tr["steps"])),
    ]
    rows = []
    for name, stopper in policies:
        pre = over = accuracy = late_mass = 0.0
        for tr, evidence in located:
            stop = stopper(tr)
            pre += stop < evidence
            if stop >= evidence:
                over += stop - evidence
                late_mass += 1
            accuracy += official_score(tr["steps"][stop - 1]["draft"], tr["answer"])
        n = len(located)
        rows.append({"policy": name, "n": n, "pre_proxy": pre / n,
                     "over_read": over / late_mass if late_mass else None,
                     "accuracy": accuracy / n})

    # Exact expectation under a uniform random stop for every trajectory.
    pre = over = accuracy = late_mass = 0.0
    for tr, evidence in located:
        total = len(tr["steps"])
        for stop in range(1, total + 1):
            weight = 1.0 / total
            pre += weight * (stop < evidence)
            if stop >= evidence:
                over += weight * (stop - evidence)
                late_mass += weight
            accuracy += weight * official_score(
                tr["steps"][stop - 1]["draft"], tr["answer"]
            )
    rows.insert(1, {"policy": "random stop (exact expectation)",
                    "n": len(located), "pre_proxy": pre / len(located),
                    "over_read": over / late_mass if late_mass else None,
                    "accuracy": accuracy / len(located)})

    payload = {
        "definition": "first case-insensitive literal gold-answer occurrence in context; answers shorter than 3 characters and yes/no answers excluded",
        "interpretation": "lower-bound proxy for complete supporting evidence",
        "all_trajectories": len(trajectories), "located": len(located),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
