"""Generate the five-model S-NIAH comparison table and machine-readable summary."""
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

import analyze as az
from common import dedupe_prefer_success, read_jsonl
from run_five_models import MODELS, ROOT

LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "gemma3_12b": "Gemma-3-12B",
    "gemma3_27b": "Gemma-3-27B",
}
THETA, EPS, WINDOW = 0.995, 0.05, 3
VERBALIZED_AT = 99.5
RUNTIME_STOP = {
    "rule": "windowed",
    "theta": THETA,
    "eps": EPS,
    "window": WINDOW,
    "min_chunks": 2,
}


def policy_row(trajectories: list[dict], name: str, stopper, uses=("probe",)) -> dict:
    premature = correct = full_correct = stopped = 0
    stop_chunks, full_chunks, tokens = [], [], []
    for tr in trajectories:
        evidence = az._first_evidence_chunk(tr)
        if evidence is None:
            raise RuntimeError(f"{tr['id']} lacks evidence_char_starts")
        t = int(stopper(tr))
        full_t = len(tr["steps"])
        prediction, _ = az.step_answer(tr, t)
        full_prediction, _ = az.step_answer(tr, full_t)
        correct += int(bool(az.score_answer(tr, prediction)))
        full_correct += int(bool(az.score_answer(tr, full_prediction)))
        premature += int(t < evidence)
        stopped += int(t < full_t)
        stop_chunks.append(t)
        full_chunks.append(full_t)
        tokens.append(az.policy_tokens(tr, t, uses=uses))
    n = len(trajectories)
    return {
        "policy": name,
        "n": n,
        "stops_before_evidence": premature,
        "stops_before_evidence_rate": premature / n,
        "early_stop_count": stopped,
        "accuracy": correct / n,
        "correct": correct,
        "full_accuracy": full_correct / n,
        "mean_stop_chunk": sum(stop_chunks) / n,
        "mean_full_chunks": sum(full_chunks) / n,
        "mean_tokens": sum(tokens) / n,
    }


def main() -> None:
    results = ROOT / "results"
    comparison = results / "five_model_comparison"
    comparison.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for key, model in MODELS.items():
        path = results / f"sniah_{key}_n250_full.jsonl"
        trajectories = [
            row for row in dedupe_prefer_success(read_jsonl(path))
            if row.get("steps") and not row.get("failed")
        ]
        if len(trajectories) != 250:
            raise RuntimeError(
                f"{key}: expected 250 completed trajectories, found {len(trajectories)}"
            )
        if {str(row.get("model")) for row in trajectories} != {model}:
            raise RuntimeError(f"{key}: model identity mismatch")
        if not all(
            all(field in step for field in ("verbalized", "draft_conf", "draft_norm"))
            and step.get("verbalized_valid") is True
            for row in trajectories for step in row["steps"]
        ):
            raise RuntimeError(f"{key}: required verbalized/measured signals are missing")

        verbalized = policy_row(
            trajectories,
            "verbalized",
            lambda tr: az.stop_verbalized(tr, VERBALIZED_AT),
            uses=("verb",),
        )
        measured = policy_row(
            trajectories,
            "measured",
            lambda tr: az.stop_online(tr, cfg=RUNTIME_STOP),
            uses=("probe",),
        )
        for row in (verbalized, measured):
            row.update({
                "model_key": key,
                "model_label": LABELS[key],
                "model_id": model,
                "theta": THETA,
                "eps": EPS,
                "window": WINDOW,
                "verbalized_at": VERBALIZED_AT,
            })
            all_rows.append(row)

    fields = list(all_rows[0])
    with (comparison / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    (comparison / "summary.json").write_text(
        json.dumps({
            "configuration": {
                "theta": THETA,
                "eps": EPS,
                "window": WINDOW,
                "questions_per_model": 250,
            },
            "rows": all_rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Saved table and JSON under {comparison}")


if __name__ == "__main__":
    main()
