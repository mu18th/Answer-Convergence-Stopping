"""Reproduce the reported theta sweeps from complete trajectories."""
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

DEFAULT_THETAS = [0.80, 0.90, 0.92, 0.95, 0.98, 0.99, 0.995]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--theta", type=float, nargs="+", default=DEFAULT_THETAS)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-official", action="store_true")
    parser.add_argument("--judge-cache", default=None)
    args = parser.parse_args()
    if any(not 0 <= theta <= 1 for theta in args.theta):
        raise SystemExit("theta values must lie in [0,1]")
    cfg = load_config(args.config)
    az._enable_judge(args, cfg)
    trajectories = [row for row in dedupe_prefer_success(read_jsonl(args.traj))
                    if row.get("steps") and not row.get("failed")]
    if not trajectories:
        raise RuntimeError("No successful complete trajectories")
    az._check_build(trajectories, "threshold-sweep trajectories")
    if any(row.get("online_stop_config") for row in trajectories):
        raise RuntimeError("Threshold sweeps require complete, non-truncated trajectories")

    full_scores, full_tokens = [], []
    for tr in trajectories:
        stop = len(tr["steps"])
        answer, _ = az.step_answer(tr, stop)
        full_scores.append(az.score_answer(tr, answer))
        full_tokens.append(az.policy_tokens(tr, stop, uses=()))
    full_mean_tokens = sum(full_tokens) / len(full_tokens)

    rows = []
    for theta in args.theta:
        scores, tokens, premature = [], [], []
        evidence_n = 0
        for tr in trajectories:
            stop = az.stop_measured(
                tr, theta=theta, eps=args.eps, w=args.window
            )
            answer, _ = az.step_answer(tr, stop)
            scores.append(az.score_answer(tr, answer))
            tokens.append(az.policy_tokens(tr, stop, uses=("probe",)))
            evidence = az._first_evidence_chunk(tr)
            if evidence is not None:
                evidence_n += 1
                premature.append(stop < evidence)
        mean_tokens = sum(tokens) / len(tokens)
        rows.append({
            "theta": theta, "epsilon": args.eps, "window": args.window,
            "n": len(trajectories), "accuracy": sum(scores) / len(scores),
            "mean_tokens": mean_tokens,
            "saving": 1.0 - mean_tokens / full_mean_tokens,
            "evidence_n": evidence_n,
            "premature": (sum(premature) / len(premature)
                          if premature else None),
        })
    rows.append({
        "theta": "full", "epsilon": args.eps, "window": args.window,
        "n": len(trajectories), "accuracy": sum(full_scores) / len(full_scores),
        "mean_tokens": full_mean_tokens, "saving": 0.0,
        "evidence_n": sum(az._first_evidence_chunk(tr) is not None
                          for tr in trajectories),
        "premature": 0.0,
    })
    numeric = rows[:-1]
    best = min(numeric, key=lambda row: (-row["accuracy"], row["mean_tokens"]))
    payload = {
        "trajectory": str(args.traj), "configuration": str(args.config),
        "selection_note": "best theta is exploratory: highest accuracy on these same questions; ties use lower cost",
        "best_theta": best["theta"], "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with args.out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print("theta accuracy mean_tokens saving premature")
    for row in rows:
        premature = "--" if row["premature"] is None else f"{row['premature']:.1%}"
        print(f"{str(row['theta']):>5} {row['accuracy']:.3f} "
              f"{row['mean_tokens']:,.0f} {row['saving']:.1%} {premature}")
    if az.JUDGE is not None:
        az.JUDGE.flush()


if __name__ == "__main__":
    main()
