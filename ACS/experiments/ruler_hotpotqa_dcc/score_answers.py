"""Score normalized method outputs against the paired prepared examples.

HotpotQA uses normalized exact match and token F1. RULER-HotpotQA uses the
official qa_2 string_match_part criterion. Results need an id and prediction;
gold answers are joined from the prepared input file, not from model prompts.
"""
from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    result = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{path}:{number}: invalid JSON") from error
    return result


def normalize_hotpot(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def hotpot_scores(prediction: str, gold: str) -> tuple[float, float]:
    pred = normalize_hotpot(prediction).split()
    ref = normalize_hotpot(gold).split()
    em = float(pred == ref)
    if not pred or not ref:
        return em, float(pred == ref)
    overlap = sum((Counter(pred) & Counter(ref)).values())
    if overlap == 0:
        return em, 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return em, 2 * precision * recall / (precision + recall)


def prediction_from(row: dict) -> str:
    if "prediction" in row:
        return str(row["prediction"] or "")
    steps = row.get("steps")
    if isinstance(steps, list) and steps:
        return str(steps[-1].get("draft", "") or "")
    raise RuntimeError(f"{row.get('id')}: no prediction or trajectory steps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared ACS-schema JSONL used by all methods")
    parser.add_argument("--predictions", type=Path, required=True,
                        help="JSONL with id and prediction, or ACS trajectories")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit("Score output already exists; choose a new path or pass --overwrite")

    data_rows = read_jsonl(args.data)
    pred_rows = read_jsonl(args.predictions)
    gold_by_id = {str(row["id"]): row for row in data_rows}
    if len(gold_by_id) != len(data_rows):
        raise SystemExit("Prepared data contains duplicate IDs")
    pred_by_id: dict[str, dict] = {}
    for row in pred_rows:
        if row.get("status", "ok") != "ok" or row.get("failed"):
            continue
        row_id = str(row.get("id", ""))
        if not row_id or row_id in pred_by_id:
            raise SystemExit(f"Missing/duplicate successful prediction ID: {row_id!r}")
        pred_by_id[row_id] = row
    unknown = set(pred_by_id) - set(gold_by_id)
    if unknown:
        raise SystemExit(f"Predictions contain IDs outside paired data: {sorted(unknown)[:5]}")

    scored = []
    for row_id, result in pred_by_id.items():
        if result.get("status", "ok") != "ok" or result.get("failed"):
            continue
        source = gold_by_id[row_id]
        prediction = prediction_from(result)
        benchmark = source.get("benchmark")
        gold = source.get("answer", source.get("answers"))
        if benchmark == "ruler_hotpotqa":
            references = gold if isinstance(gold, list) else [gold]
            correct = int(any(str(ref).lower() in prediction.lower() for ref in references))
            record = {"id": row_id, "correct": correct}
        elif benchmark == "hotpotqa":
            if isinstance(gold, list):
                pairs = [hotpot_scores(prediction, str(ref)) for ref in gold]
                em = max(pair[0] for pair in pairs)
                f1 = max(pair[1] for pair in pairs)
            else:
                em, f1 = hotpot_scores(prediction, str(gold))
            record = {"id": row_id, "exact_match": em, "f1": f1}
        else:
            raise SystemExit(f"Unsupported benchmark in paired data: {benchmark!r}")

        record.update({
            "prediction": prediction,
            "benchmark": benchmark,
            "split": source.get("source_split", source.get("split")),
            "context_length_tokens": source.get("context_length_tokens"),
            "prompt_tokens": result.get("prompt_tokens"),
            "completion_tokens": result.get("completion_tokens"),
            "total_tokens": result.get("total_tokens") or (
                result.get("steps", [{}])[-1].get("cum_tokens")
                if result.get("steps") else None
            ),
            "latency_seconds": result.get("latency_seconds", result.get("wall_s")),
        })
        scored.append(record)

    if not scored:
        raise SystemExit("No successful paired predictions to score")
    benchmark = scored[0]["benchmark"]
    if any(row["benchmark"] != benchmark for row in scored):
        raise SystemExit("Score one benchmark per invocation")
    non_fitting_ids = {
        str(row.get("id")) for row in pred_rows if row.get("status") == "non_fitting"
    }
    failed_ids = {
        str(row.get("id")) for row in pred_rows
        if row.get("status") == "error" or row.get("failed")
    }
    summary = {
        "benchmark": benchmark,
        "n_data": len(data_rows),
        "n_scored": len(scored),
        "coverage": len(scored) / len(data_rows) if data_rows else 0.0,
        "n_non_fitting": len(non_fitting_ids),
        "n_failed": len(failed_ids),
    }
    if benchmark == "ruler_hotpotqa":
        summary["accuracy"] = sum(row["correct"] for row in scored) / len(scored)
    else:
        summary["exact_match"] = sum(row["exact_match"] for row in scored) / len(scored)
        summary["f1"] = sum(row["f1"] for row in scored) / len(scored)
    for metric in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"):
        vals = [row[metric] for row in scored if isinstance(row.get(metric), (int, float))]
        if vals:
            summary[f"mean_{metric}"] = sum(vals) / len(vals)

    if benchmark == "ruler_hotpotqa":
        data_by_length: dict[int, list[dict]] = {}
        for row in data_rows:
            data_by_length.setdefault(int(row["context_length_tokens"]), []).append(row)
        scored_by_id = {row["id"]: row for row in scored}
        predictions_by_id = {str(row.get("id", "")): row for row in pred_rows}
        by_length = {}
        for length, bucket in sorted(data_by_length.items()):
            bucket_scored = [scored_by_id[row["id"]] for row in bucket if row["id"] in scored_by_id]
            bucket_prediction_ids = {str(row["id"]) for row in bucket}
            bucket_non_fitting = sum(
                predictions_by_id.get(row_id, {}).get("status") == "non_fitting"
                for row_id in bucket_prediction_ids
            )
            bucket_failed = sum(
                predictions_by_id.get(row_id, {}).get("status") == "error"
                or bool(predictions_by_id.get(row_id, {}).get("failed"))
                for row_id in bucket_prediction_ids
            )
            record = {
                "n_data": len(bucket),
                "n_scored": len(bucket_scored),
                "coverage": len(bucket_scored) / len(bucket) if bucket else 0.0,
                "n_non_fitting": int(bucket_non_fitting),
                "n_failed": int(bucket_failed),
                "accuracy": (
                    sum(row["correct"] for row in bucket_scored) / len(bucket_scored)
                    if bucket_scored else None
                ),
            }
            for metric in ("prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds"):
                values = [row[metric] for row in bucket_scored
                          if isinstance(row.get(metric), (int, float))]
                if values:
                    record[f"mean_{metric}"] = sum(values) / len(values)
            by_length[str(length)] = record
        summary["by_context_length_tokens"] = by_length

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary, "examples": scored}
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
