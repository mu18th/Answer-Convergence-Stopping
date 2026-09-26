"""Theta sweep for complete RULER-HotpotQA trajectories."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import read_jsonl
from analyze_ruler_hotpotqa import evaluate, stop_windowed, validate

DEFAULT_THETAS = [0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--theta", type=float, nargs="+", default=DEFAULT_THETAS)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args()
    trajectories = validate(
        [row for path in args.traj for row in read_jsonl(path)],
        [row for path in args.data for row in read_jsonl(path)],
    )
    full_stops = {tr["id"]: len(tr["steps"]) for tr in trajectories}
    full, _ = evaluate("full", trajectories, full_stops, "none")
    rows = []
    for theta in args.theta:
        stops = {tr["id"]: stop_windowed(tr, theta, args.eps, args.window)
                 for tr in trajectories}
        result, _ = evaluate(f"theta={theta:g}", trajectories, stops, "measured")
        rows.append({
            "theta": theta, "accuracy": result["accuracy"],
            "mean_tokens": result["mean_tokens"],
            "saving": 1.0 - result["mean_tokens"] / full["mean_tokens"],
            "early_stop_rate": result["early_stop_rate"],
            "harms_vs_full": result["harms_vs_full"],
            "gains_vs_full": result["gains_vs_full"],
        })
    rows.append({
        "theta": "full", "accuracy": full["accuracy"],
        "mean_tokens": full["mean_tokens"], "saving": 0.0,
        "early_stop_rate": 0.0, "harms_vs_full": 0, "gains_vs_full": 0,
    })
    best = min(rows[:-1], key=lambda row: (-row["accuracy"], row["mean_tokens"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({
        "n": len(trajectories), "epsilon": args.eps, "window": args.window,
        "best_theta_same_questions_upper_bound": best["theta"], "rows": rows,
    }, indent=2), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
