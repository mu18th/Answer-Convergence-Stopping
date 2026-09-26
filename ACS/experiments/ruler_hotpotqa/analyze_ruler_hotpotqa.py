"""Offline policy replay for complete RULER-HotpotQA trajectories.

No parameter is tuned here.  The caller supplies the frozen policy values, and
the script evaluates all prepared rows overall and by official context bucket.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.append(str(EVAL_ROOT))

import analyze as shared_analysis
from common import dedupe_prefer_success, f1_tokens, is_non_answer, read_jsonl


def official_score(prediction: str, gold: object) -> int:
    """NVIDIA/RULER string_match_part: any lowercased gold substring."""
    answers = gold if isinstance(gold, list) else [gold]
    pred = str(prediction).lower()
    return int(any(str(answer).lower() in pred for answer in answers))


def adjacent_divergences(steps: list[dict], lo: int, hi: int) -> list[float]:
    return [
        1.0 - f1_tokens(steps[i - 1]["draft_norm"], steps[i]["draft_norm"])
        for i in range(lo + 1, hi)
    ]


def stop_windowed(
    tr: dict, theta: float, eps: float, window: int
) -> int:
    steps = tr["steps"]
    for t in range(window, len(steps) + 1):
        recent = steps[t - window:t]
        if any(
            is_non_answer(step.get("draft", ""))
            or float(step.get("draft_conf", 0.0)) < theta
            for step in recent
        ):
            continue
        values = adjacent_divergences(steps, t - window, t)
        instability = statistics.mean(values)
        if instability <= eps:
            return t
    return len(steps)


def stop_verbal(tr: dict, threshold: float) -> int:
    for t, step in enumerate(tr["steps"], 1):
        value = step.get("verbalized")
        if isinstance(value, (int, float)) and value >= threshold:
            return t
    return len(tr["steps"])


def stop_gate(tr: dict) -> int:
    for t, step in enumerate(tr["steps"], 1):
        if step.get("verbalized_stop", False):
            return t
    return len(tr["steps"])


def logical_cost(tr: dict, t: int, signal: str) -> float:
    uses = {
        "none": (),
        "measured": ("probe",),
        "verb": ("verb",),
        "gate": ("gate",),
    }
    if signal not in uses:
        raise ValueError(signal)
    return float(shared_analysis.policy_tokens(tr, t, uses=uses[signal]))


def percentile_median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def evaluate(name: str, trajectories: list[dict], stops: dict[str, int], signal: str) -> tuple[dict, list[dict]]:
    sample_rows = []
    for tr in trajectories:
        t = stops[tr["id"]]
        full_t = len(tr["steps"])
        prediction = tr["steps"][t - 1]["draft"]
        full_prediction = tr["steps"][-1]["draft"]
        correct = official_score(prediction, tr["answer"])
        full_correct = official_score(full_prediction, tr["answer"])
        tokens = logical_cost(tr, t, signal)
        full_tokens = logical_cost(tr, full_t, "none")
        sample_rows.append({
            "policy": name, "id": tr["id"], "source_id": tr.get("source_id"),
            "context_length_tokens": tr["context_length_tokens"],
            "gold": json.dumps(tr["answer"], ensure_ascii=False),
            "prediction": prediction, "full_prediction": full_prediction,
            "correct": correct, "full_correct": full_correct,
            "gain_vs_full": int(correct and not full_correct),
            "harm_vs_full": int(full_correct and not correct),
            "same_correct": int(correct and full_correct),
            "same_wrong": int(not correct and not full_correct),
            "stop_chunk": t, "total_chunks": full_t,
            "read_fraction": t / full_t, "stopped_early": int(t < full_t),
            "tokens": tokens,
            "full_tokens": full_tokens,
            "token_saving_fraction": 1.0 - tokens / full_tokens if full_tokens else 0.0,
        })
    n = len(sample_rows)
    summary = {
        "policy": name, "n": n,
        "correct": sum(row["correct"] for row in sample_rows),
        "accuracy": sum(row["correct"] for row in sample_rows) / n,
        "early_stop_count": sum(row["stopped_early"] for row in sample_rows),
        "early_stop_rate": sum(row["stopped_early"] for row in sample_rows) / n,
        "gains_vs_full": sum(row["gain_vs_full"] for row in sample_rows),
        "harms_vs_full": sum(row["harm_vs_full"] for row in sample_rows),
        "same_correct": sum(row["same_correct"] for row in sample_rows),
        "same_wrong": sum(row["same_wrong"] for row in sample_rows),
        "mean_tokens": statistics.mean(row["tokens"] for row in sample_rows),
        "median_tokens": percentile_median([row["tokens"] for row in sample_rows]),
        "mean_stop_chunk": statistics.mean(row["stop_chunk"] for row in sample_rows),
        "median_stop_chunk": percentile_median([row["stop_chunk"] for row in sample_rows]),
        "mean_read_fraction": statistics.mean(row["read_fraction"] for row in sample_rows),
        "mean_token_saving_fraction_vs_full": statistics.mean(
            row["token_saving_fraction"] for row in sample_rows
        ),
    }
    return summary, sample_rows


def evaluate_random(
    trajectories: list[dict], draws: int = 200, seed: int = 0
) -> tuple[dict, list[dict]]:
    """Average uniform random stopping over the paper's fixed 200 draws."""
    rng = random.Random(seed)
    summaries, samples_by_id = [], {tr["id"]: [] for tr in trajectories}
    for _ in range(draws):
        stops = {
            tr["id"]: rng.randint(1, len(tr["steps"])) for tr in trajectories
        }
        summary, samples = evaluate("random stop", trajectories, stops, "none")
        summaries.append(summary)
        for row in samples:
            samples_by_id[row["id"]].append(row)

    summary = {"policy": "random stop", "n": len(trajectories)}
    for key, value in summaries[0].items():
        if key not in {"policy", "n"} and isinstance(value, (int, float)):
            summary[key] = statistics.mean(row[key] for row in summaries)

    sample_rows = []
    for tr in trajectories:
        rows = samples_by_id[tr["id"]]
        row = dict(rows[0])
        row["policy"] = "random stop"
        row["prediction"] = "<mean over 200 random draws>"
        for key, value in rows[0].items():
            if isinstance(value, (int, float)):
                row[key] = statistics.mean(item[key] for item in rows)
        sample_rows.append(row)
    return summary, sample_rows


def validate(trajectories: list[dict], data: list[dict] | None) -> list[dict]:
    trajectories = dedupe_prefer_success(trajectories)
    failed = [tr["id"] for tr in trajectories if tr.get("failed")]
    if failed:
        raise RuntimeError(f"Analysis input still has {len(failed)} failed IDs; first={failed[:5]}")
    if data is not None:
        data_by_id = {row["id"]: row for row in data}
        if len(data_by_id) != len(data):
            raise RuntimeError("Prepared data contain duplicate sample IDs")
        expected = set(data_by_id)
        found = {tr["id"] for tr in trajectories}
        if expected != found:
            raise RuntimeError(
                f"Trajectory/data mismatch: missing={sorted(expected-found)[:5]}, "
                f"extra={sorted(found-expected)[:5]}"
            )
        mismatched_builds = [
            tr["id"] for tr in trajectories
            if tr.get("build_id") != data_by_id[tr["id"]].get("build_id")
        ]
        if mismatched_builds:
            raise RuntimeError(
                f"Trajectory/data build mismatch; first={mismatched_builds[:5]}"
            )
    for tr in trajectories:
        if tr.get("benchmark") != "ruler_hotpotqa":
            raise RuntimeError(f"{tr.get('id')}: wrong benchmark")
        if tr.get("recording_mode") != "full_trajectory":
            raise RuntimeError(f"{tr['id']}: not a full-trajectory recording")
        if tr.get("chunks_read") != tr.get("n_chunks") or len(tr.get("steps", [])) != tr.get("n_chunks"):
            raise RuntimeError(f"{tr['id']}: truncated trajectory")
        for key in ("answer", "context_length_tokens"):
            if key not in tr:
                raise RuntimeError(f"{tr['id']}: missing {key}")
    if not trajectories:
        raise RuntimeError("No valid trajectories")
    builds = {tr.get("build_id") for tr in trajectories}
    models = {tr.get("model") for tr in trajectories}
    if None in builds:
        raise RuntimeError(f"Analysis found a missing build_id: {builds}")
    if None in models or len(models) != 1:
        raise RuntimeError(f"Analysis requires exactly one model, found {models}")
    return sorted(trajectories, key=lambda tr: (tr["context_length_tokens"], tr["id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--theta", type=float, required=True)
    parser.add_argument("--eps", type=float, required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--verbal-threshold", type=float, default=99.5)
    args = parser.parse_args()
    if not (0 <= args.theta <= 1 and 0 < args.eps <= 1 and args.window >= 2):
        raise SystemExit("Require theta in [0,1], eps in (0,1], and window >=2")
    trajectories = validate(
        [row for path in args.traj for row in read_jsonl(path)],
        [row for path in args.data for row in read_jsonl(path)],
    )
    # Invalid comparator outputs conservatively mean CONTINUE; retain the
    # baseline whenever the comparator was recorded on every step.
    has_verb = all(
        "verbalized" in step for tr in trajectories for step in tr["steps"]
    )
    has_gate = all(
        "verbalized_stop" in step for tr in trajectories for step in tr["steps"]
    )

    policy_specs = [
        ("full reading", {tr["id"]: len(tr["steps"]) for tr in trajectories}, "none"),
        ("random stop", None, "random-200"),
    ]
    if has_verb:
        policy_specs.append((f"verbalized gate @{args.verbal_threshold:g}", {
            tr["id"]: stop_verbal(tr, args.verbal_threshold) for tr in trajectories
        }, "verb"))
    if has_gate:
        policy_specs.append(("END gate", {
            tr["id"]: stop_gate(tr) for tr in trajectories
        }, "gate"))
    policy_specs.append(("ACS, fixed", {
        tr["id"]: stop_windowed(
            tr, args.theta, args.eps, args.window
        ) for tr in trajectories
    }, "measured"))

    all_summaries, all_samples = [], []
    groups = [("overall", trajectories)] + [
        (f"{length // 1024}K", [
            tr for tr in trajectories if tr["context_length_tokens"] == length
        ])
        for length in sorted({tr["context_length_tokens"] for tr in trajectories})
    ]
    for group_name, group in groups:
        ids = {tr["id"] for tr in group}
        for policy, stops, signal in policy_specs:
            if signal == "random-200":
                summary, samples = evaluate_random(group)
            else:
                summary, samples = evaluate(
                    policy, group,
                    {sample_id: t for sample_id, t in stops.items() if sample_id in ids},
                    signal,
                )
            summary["group"] = group_name
            all_summaries.append(summary)
            # Per-sample rows are written once. Context-bucket membership is
            # already present in context_length_tokens; duplicating them under
            # both "overall" and a bucket would inflate downstream counts.
            if group_name == "overall":
                for sample in samples:
                    sample["group"] = group_name
                    all_samples.append(sample)

    output = {
        "benchmark": "NVIDIA/RULER qa_2 (HotpotQA)",
        "official_scoring": "string_match_part",
        "trajectory_files": [str(path.resolve()) for path in args.traj],
        "data_files": [str(path.resolve()) for path in args.data],
        "n": len(trajectories),
        "build_ids": sorted({tr["build_id"] for tr in trajectories}),
        "model": trajectories[0]["model"],
        "base_url": trajectories[0].get("base_url"),
        "frozen_parameters": {
            "theta": args.theta, "eps": args.eps, "window": args.window,
            "verbal_threshold": args.verbal_threshold,
            "divergence_stat": "mean",
        },
        "summaries": all_summaries,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.out_dir / "policy_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summaries[0]))
        writer.writeheader(); writer.writerows(all_summaries)
    with (args.out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_samples[0]))
        writer.writeheader(); writer.writerows(all_samples)
    print("group policy n correct accuracy early harms gains mean_tokens savings")
    for row in all_summaries:
        print(
            f"{row['group']:>7} {row['policy']:<28} {row['n']:>3} "
            f"{row['correct']:>5.1f} {row['accuracy']:.3f} "
            f"{row['early_stop_rate']:.1%} {row['harms_vs_full']:>3} "
            f"{row['gains_vs_full']:>3} {row['mean_tokens']:,.0f} "
            f"{row['mean_token_saving_fraction_vs_full']:.1%}"
        )
    print(f"Saved analysis to {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
