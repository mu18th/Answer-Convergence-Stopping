"""Create a Table-3-style report from the five matched full-trajectory arms."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path


THETA = 0.995
WINDOW_EPS = 0.05
WINDOW = 3
VERBAL_THRESHOLD = 99.5
EXPECTED_ROWS = 80
EXPECTED_IDS_SHA256 = (
    "66fd3251bdca1d9299ebd8ce9d43e76f1f93a844b7dc300e342270802164be03"
)

ARM_FILES = {
    "base": "base_24k_notes6k.jsonl",
    "chunk12": "chunk12k_notes6k.jsonl",
    "chunk48": "chunk48k_notes6k.jsonl",
    "notes3": "chunk24k_notes3k.jsonl",
    "notes12": "chunk24k_notes12k.jsonl",
}


def read_success(path: Path) -> dict[str, dict]:
    by: dict[str, dict] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row.get("id"))
            if key not in by or (by[key].get("failed") and not row.get("failed")):
                by[key] = row
    return {key: row for key, row in by.items() if not row.get("failed") and row.get("steps")}


def answer_at(tr: dict, t: int) -> str:
    posterior = tr["steps"][t - 1]["posterior"]
    return max(posterior, key=posterior.get)


def full_stop(tr: dict) -> int:
    return len(tr["steps"])


def confidence_only(tr: dict) -> int:
    for index, step in enumerate(tr["steps"], 1):
        if max(step["posterior"].values()) >= THETA:
            return index
    return len(tr["steps"])


def asked_numeric(tr: dict) -> int:
    for index, step in enumerate(tr["steps"], 1):
        value = step.get("verbalized")
        if isinstance(value, (int, float)) and value >= VERBAL_THRESHOLD:
            return index
    return len(tr["steps"])


def asked_gate(tr: dict) -> int:
    for index, step in enumerate(tr["steps"], 1):
        if step.get("verbalized_stop", False):
            return index
    return len(tr["steps"])


def replay_runtime(tr: dict, runtime, rule: str) -> int:
    steps = []
    stop_config = {
        "rule": rule,
        "theta": THETA,
        "eps": WINDOW_EPS,
        "window": WINDOW,
        "min_chunks": 2,
    }
    for source in tr["steps"]:
        step = copy.deepcopy(source)
        for key in ("divergence", "should_stop"):
            step.pop(key, None)
        step["confidence"] = max(step["posterior"].values())
        steps.append(step)
        stop, _reason, _evidence = runtime.online_stop_decision(
            steps, "mcq", stop_config
        )
        if stop:
            return len(steps)
    return len(steps)


def policy_tokens(tr: dict, t: int, uses: tuple[str, ...]) -> int:
    step = tr["steps"][t - 1]
    probe = step.get("cum_probe_tokens", 0) or 0
    verb = step.get("cum_verb_tokens", 0) or 0
    gate = step.get("cum_gate_tokens", 0) or 0
    total_recorded = step["cum_tokens"]
    fold = total_recorded - probe - verb - gate
    previous = tr["steps"][t - 2] if t >= 2 else None
    one_probe = probe - ((previous.get("cum_probe_tokens", 0) or 0) if previous else 0)
    total = fold + (probe if "probe" in uses else one_probe)
    if "verb" in uses:
        total += verb
    if "gate" in uses:
        total += gate
    return int(total)


def evaluate(rows: list[dict], stopper, uses: tuple[str, ...]) -> dict:
    records = []
    for tr in rows:
        t = stopper(tr)
        full_t = len(tr["steps"])
        prediction = answer_at(tr, t)
        full_prediction = answer_at(tr, full_t)
        correct = prediction == tr["answer"]
        full_correct = full_prediction == tr["answer"]
        records.append({
            "id": tr["id"],
            "t": t,
            "full_t": full_t,
            "correct": correct,
            "full_correct": full_correct,
            "premature_harm": t < full_t and full_correct and not correct,
            "tokens": policy_tokens(tr, t, uses),
            "read_fraction": t / full_t,
        })
    n = len(records)
    return {
        "accuracy": sum(r["correct"] for r in records) / n,
        "correct": sum(r["correct"] for r in records),
        "premature": sum(r["premature_harm"] for r in records) / n,
        "premature_count": sum(r["premature_harm"] for r in records),
        "tokens": statistics.mean(r["tokens"] for r in records),
        "median_tokens": statistics.median(r["tokens"] for r in records),
        "mean_stop_chunk": statistics.mean(r["t"] for r in records),
        "mean_read_fraction": statistics.mean(r["read_fraction"] for r in records),
        "early_count": sum(r["t"] < r["full_t"] for r in records),
        "records": records,
    }


def notes_diagnostics(rows: list[dict]) -> dict:
    steps = [step for row in rows for step in row["steps"]]
    raw = [int(step.get("raw_notes_chars", len(step.get("notes", "")))) for step in steps]
    kept = [int(step.get("notes_chars", len(step.get("notes", "")))) for step in steps]
    overflow = [step for step in steps if step.get("notes_overflow")]
    return {
        "raw_notes_max": max(raw),
        "raw_notes_p95": statistics.quantiles(raw, n=100, method="inclusive")[94],
        "raw_notes_p99": statistics.quantiles(raw, n=100, method="inclusive")[98],
        "kept_notes_max": max(kept),
        "overflow_steps": len(overflow),
        "overflow_rate": len(overflow) / len(steps),
    }


def probe_overhead(rows: list[dict], runtime) -> dict:
    records = []
    for tr in rows:
        t = replay_runtime(tr, runtime, "windowed")
        step = tr["steps"][t - 1]
        probe = int(step.get("cum_probe_tokens", 0) or 0)
        total = policy_tokens(tr, t, ("probe",))
        records.append({"id": tr["id"], "probe_tokens": probe,
                        "policy_tokens": total, "probe_fraction": probe / total})
    probe_total = sum(r["probe_tokens"] for r in records)
    policy_total = sum(r["policy_tokens"] for r in records)
    return {
        "arm": "24K chunks / 6K notes (default)",
        "samples": len(records),
        "total_probe_tokens": probe_total,
        "mean_probe_tokens_per_sample": statistics.mean(r["probe_tokens"] for r in records),
        "mean_probe_tokens_per_executed_chunk": probe_total / sum(
            replay_runtime(tr, runtime, "windowed") for tr in rows),
        "aggregate_probe_fraction": probe_total / policy_total,
        "mean_sample_probe_fraction": statistics.mean(r["probe_fraction"] for r in records),
        "probe_latency_available": False,
        "latency_note": "Trajectories store combined per-step latency, not probe-only latency.",
    }


def latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Axis & Configuration & Acc & Premature & Tokens \\",
        r"\midrule",
    ]
    previous_axis = None
    for row in rows:
        if previous_axis is not None and row["axis"] != previous_axis:
            lines.append(r"\midrule")
        axis = row["axis"] if row["axis"] != previous_axis else ""
        lines.append(
            f"{axis} & {row['configuration']} & {row['accuracy']:.3f} & "
            f"{100 * row['premature']:.1f}\\% & {row['tokens']:,.0f} \\\\"
        )
        previous_axis = row["axis"]
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=here.parent.parent)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "data" / "longbench_v2_matched_80.jsonl",
    )
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--out-dir", type=Path, default=here / "analysis")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    import run_fold as runtime

    ids = [str(json.loads(line)["id"]) for line in args.data.open(encoding="utf-8") if line.strip()]
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected the frozen {EXPECTED_ROWS}-ID manifest, "
            f"found {len(ids)}/{len(set(ids))}"
        )
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if digest != EXPECTED_IDS_SHA256:
        raise RuntimeError("Input IDs/order differ from the frozen 80-ID cohort")
    arms = {name: read_success(args.results_dir / filename) for name, filename in ARM_FILES.items()}
    missing = {name: sorted(set(ids) - set(rows)) for name, rows in arms.items()}
    if any(missing.values()):
        raise RuntimeError(
            "Arms are incomplete for the frozen 80-ID cohort: "
            f"{ {k: len(v) for k, v in missing.items()} }"
        )
    # Restrict every arm to the frozen, ordered evaluation cohort.
    ordered = {name: [rows[sample_id] for sample_id in ids] for name, rows in arms.items()}

    base = ordered["base"]
    specifications = [
        ("Stopping signal", "asked, END/CONTINUE gate", base, asked_gate, ("gate",)),
        ("Stopping signal", "verbalized P(True) >= 99.5", base, asked_numeric, ("verb",)),
        ("Stopping signal", "measured, confidence only", base, confidence_only, ("probe",)),
        ("Stopping signal", "measured, confidence + W3", base, lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        # This single-model stability axis repeats the corresponding measured
        # rows by construction.
        ("Stability test (Kimi)", "confidence test only", base, confidence_only, ("probe",)),
        ("Stability test (Kimi)", "both tests (default)", base, lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Chunk size L", "12K", ordered["chunk12"], lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Chunk size L", "24K (default)", base, lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Chunk size L", "48K", ordered["chunk48"], lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Notes cap B", "3K", ordered["notes3"], lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Notes cap B", "6K (default)", base, lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
        ("Notes cap B", "12K", ordered["notes12"], lambda tr: replay_runtime(tr, runtime, "windowed"), ("probe",)),
    ]

    table = []
    sample_rows = []
    for axis, configuration, trajectories, stopper, uses in specifications:
        result = evaluate(trajectories, stopper, uses)
        row = {"axis": axis, "configuration": configuration,
               **{key: value for key, value in result.items() if key != "records"}}
        table.append(row)
        for record in result["records"]:
            sample_rows.append({"axis": axis, "configuration": configuration, **record})

    full = evaluate(base, full_stop, ())
    diagnostics = {name: notes_diagnostics(rows) for name, rows in ordered.items()}
    default_probe_overhead = probe_overhead(base, runtime)
    summary = {
        "model": "moonshotai/kimi-k2.5",
        "n": len(ids),
        "fixed_parameters": {"theta": THETA, "window_eps": WINDOW_EPS,
                             "window": WINDOW,
                             "verbal_threshold": VERBAL_THRESHOLD},
        "premature_definition": "stopped early and wrong when full-context prediction is correct",
        "full_reference": {key: value for key, value in full.items() if key != "records"},
        "notes_diagnostics": diagnostics,
        "default_24k6k_probe_overhead": default_probe_overhead,
        "table": table,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "table.tex").write_text(latex(table), encoding="utf-8")
    with (args.out_dir / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        columns = ["axis", "configuration", "accuracy", "correct", "premature",
                   "premature_count", "tokens", "median_tokens", "mean_stop_chunk",
                   "mean_read_fraction", "early_count"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row[key] for key in columns} for row in table)
    with (args.out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0]))
        writer.writeheader()
        writer.writerows(sample_rows)

    print(f"Full reference: {full['correct']}/{len(ids)} = {full['accuracy']:.3f}")
    print("24K/6K probe overhead: "
          f"{default_probe_overhead['mean_probe_tokens_per_sample']:,.0f} tokens/sample, "
          f"{100 * default_probe_overhead['aggregate_probe_fraction']:.2f}% of policy tokens")
    for row in table:
        print(f"{row['axis']:16} | {row['configuration']:34} | "
              f"acc={row['accuracy']:.3f} premature={100*row['premature']:.1f}% "
              f"tokens={row['tokens']:,.0f}")
    print(f"Saved analysis: {args.out_dir}")


if __name__ == "__main__":
    main()
