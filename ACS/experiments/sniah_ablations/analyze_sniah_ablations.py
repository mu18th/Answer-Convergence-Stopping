"""Produce Qwen3-14B S-NIAH ablation rows over all matched 250 questions."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.append(str(EVAL_ROOT))

import analyze as core
from common import dedupe_prefer_success, read_jsonl
from prepare_sniah import EXPECTED_ROWS
from run_sniah_ablations import ARMS, EXPECTED_MODEL

THETA = 0.995
EPSILON = 0.05
WINDOW = 3
RUNTIME_STOP = {
    "rule": "windowed",
    "theta": THETA,
    "eps": EPSILON,
    "window": WINDOW,
    "min_chunks": 2,
}


def load_arm(path: Path) -> list[dict]:
    records = [row for row in dedupe_prefer_success(read_jsonl(path))
               if not row.get("failed") and row.get("steps")]
    if len(records) != EXPECTED_ROWS:
        raise RuntimeError(f"{path.name}: expected {EXPECTED_ROWS} successes, found {len(records)}")
    wrong = {str(row.get("model")) for row in records
             if str(row.get("model")) != EXPECTED_MODEL}
    if wrong:
        raise RuntimeError(f"{path.name}: unexpected model(s) {sorted(wrong)}")
    return records


def probe_tokens_at(record: dict, stop: int) -> int:
    return int(record["steps"][stop - 1].get("cum_probe_tokens", 0) or 0)


def summarize(name: str, records: list[dict]) -> dict:
    stops = [core.stop_online(record, cfg=RUNTIME_STOP) for record in records]
    scores, tokens, probes, fractions = [], [], [], []
    early = 0
    oracle_n = 0
    for record, stop in zip(records, stops):
        answer, _confidence = core.step_answer(record, stop)
        scores.append(core.score_answer(record, answer))
        total = core.policy_tokens(record, stop, uses=("probe",))
        probe = probe_tokens_at(record, stop)
        tokens.append(total)
        probes.append(probe)
        fractions.append(stop / len(record["steps"]))
        evidence = core._first_evidence_chunk(record)
        if evidence is not None:
            oracle_n += 1
            early += int(stop < evidence)
    return {
        "arm": name,
        "samples": len(records),
        "correct": int(round(sum(scores))),
        "accuracy": sum(scores) / len(scores),
        "premature_count": early,
        "oracle_labeled_samples": oracle_n,
        "premature_rate": early / oracle_n if oracle_n else None,
        "mean_tokens": sum(tokens) / len(tokens),
        "mean_stop_chunk": sum(stops) / len(stops),
        "mean_read_fraction": sum(fractions) / len(fractions),
        # Aggregate token share is the scientifically relevant cost fraction.
        "probe_token_share": sum(probes) / sum(tokens),
        "mean_probe_tokens": sum(probes) / len(probes),
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    core.STABILITY_W = WINDOW
    records_by_arm = {}
    for arm, (_config, output_name) in ARMS.items():
        path = root / "results" / output_name
        if not path.exists():
            raise FileNotFoundError(f"Missing arm output: {path}")
        records_by_arm[arm] = load_arm(path)

    reference_ids = {row["id"] for row in records_by_arm["base24_notes6"]}
    for arm, records in records_by_arm.items():
        if {row["id"] for row in records} != reference_ids:
            raise RuntimeError(f"{arm}: IDs do not match the shared 250 rows")

    summaries = {arm: summarize(arm, records) for arm, records in records_by_arm.items()}
    table_rows = [
        {"axis": "chunk_size", "configuration": "12K", **summaries["chunk12"]},
        {"axis": "chunk_size", "configuration": "24K", **summaries["base24_notes6"]},
        {"axis": "chunk_size", "configuration": "48K", **summaries["chunk48"]},
        {"axis": "notes_cap", "configuration": "3K", **summaries["notes3"]},
        {"axis": "notes_cap", "configuration": "6K", **summaries["base24_notes6"]},
        {"axis": "notes_cap", "configuration": "12K", **summaries["notes12"]},
    ]

    results = root / "results"
    csv_path = results / "sniah_qwen14b_three_ablations.csv"
    json_path = results / "sniah_qwen14b_three_ablations.json"
    text_path = results / "sniah_qwen14b_three_ablations.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)
    payload = {
        "model": EXPECTED_MODEL,
        "benchmark": "RULER niah_single_1 (S-NIAH)",
        "total_trajectories_per_arm": EXPECTED_ROWS,
        "reported_samples": EXPECTED_ROWS,
        "theta": THETA,
        "epsilon": EPSILON,
        "window": WINDOW,
        "rows": table_rows,
        "probe_share_baseline": summaries["base24_notes6"]["probe_token_share"],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "Qwen3-14B S-NIAH ablations | all n=250 | theta=.995 eps=.05 W=3",
        "",
        f"{'Axis':12s} {'Config':>7s} {'Acc':>8s} {'Premature':>11s} "
        f"{'Tokens':>12s} {'Probe share':>12s}",
    ]
    for row in table_rows:
        premature = (f"{row['premature_rate']:.1%}"
                      if row["premature_rate"] is not None else "n/a")
        lines.append(
            f"{row['axis']:12s} {row['configuration']:>7s} "
            f"{row['accuracy']:8.3f} {premature:>11s} "
            f"{row['mean_tokens']:12,.0f} {row['probe_token_share']:12.1%}"
        )
    lines.extend([
        "",
        "Probe share for the ACS 24K/6K reference arm: "
        f"{summaries['base24_notes6']['probe_token_share']:.1%}",
    ])
    text = "\n".join(lines) + "\n"
    text_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {csv_path}\nSaved: {json_path}\nSaved: {text_path}")


if __name__ == "__main__":
    main()
