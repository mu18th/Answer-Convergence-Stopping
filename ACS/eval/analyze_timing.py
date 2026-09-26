"""Estimate end-to-end policy time from each trajectory's mean time per chunk.

The reported value is a full-dataset total, not a per-sample latency.  Because
complete trajectories record optional signals together, this is explicitly an
estimate: sample time is scaled by the policy's read fraction.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import analyze as az
from common import dedupe_prefer_success, load_config, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verbalized-at", type=float, default=99.5)
    parser.add_argument("--best-theta", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fixed = dict(cfg["online_stop"]); fixed["rule"] = "windowed"
    trajectories = [row for row in dedupe_prefer_success(read_jsonl(args.traj))
                    if row.get("steps") and not row.get("failed")]
    if not trajectories or any(float(row.get("wall_s", 0)) <= 0 for row in trajectories):
        raise RuntimeError("Complete trajectories with positive wall_s are required")

    policies = [
        ("full reading", lambda tr: len(tr["steps"])),
        (f"verbalized gate @{args.verbalized_at:g}",
         lambda tr: az.stop_verbalized(tr, args.verbalized_at)),
        ("END gate", az.stop_verbalized_gate),
        ("ACS, fixed", lambda tr: az.stop_online(tr, cfg=fixed)),
    ]
    if args.best_theta is not None:
        policies.append((f"ACS, best theta={args.best_theta:g}",
                         lambda tr: az.stop_measured(
                             tr, args.best_theta, float(fixed["eps"]),
                             w=int(fixed["window"]))))
    full_seconds = sum(float(tr["wall_s"]) for tr in trajectories)
    rows = []
    for name, stopper in policies:
        seconds = sum(float(tr["wall_s"]) * stopper(tr) / len(tr["steps"])
                      for tr in trajectories)
        rows.append({"policy": name, "n": len(trajectories),
                     "estimated_total_seconds": seconds,
                     "estimated_total_hours": seconds / 3600,
                     "time_saving": 1.0 - seconds / full_seconds})
    # Exact expected read fraction for a uniform discrete random stop.
    random_seconds = sum(float(tr["wall_s"]) *
                         ((len(tr["steps"]) + 1) / 2) / len(tr["steps"])
                         for tr in trajectories)
    rows.insert(1, {"policy": "random stop (exact expectation)",
                    "n": len(trajectories),
                    "estimated_total_seconds": random_seconds,
                    "estimated_total_hours": random_seconds / 3600,
                    "time_saving": 1.0 - random_seconds / full_seconds})
    payload = {
        "estimator": "sum_i wall_i * stop_chunk_i / full_chunks_i",
        "caveat": "estimated from average time per chunk; not directly timed policy execution",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(f"{row['policy']:<36} {row['estimated_total_hours']:9.2f} h "
              f"saving={row['time_saving']:.1%}")


if __name__ == "__main__":
    main()
