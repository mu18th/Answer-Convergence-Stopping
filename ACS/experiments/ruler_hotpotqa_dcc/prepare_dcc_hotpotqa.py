"""Prepare RULER-HotpotQA examples in the official DCC repository's schema.

For HotpotQA, ``gold_information`` is the smallest contiguous context span from
the first to the last annotated supporting sentence. DCC's published code uses
the end of this span as its sufficiency-label boundary. This is a conservative
multi-hop evidence proxy, not a claim that intervening text is itself evidence.
Answers are retained only as evaluation labels and are never put in the prompt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def support_span(context: str, sentences: object, row_id: str) -> str:
    if not isinstance(sentences, list) or not sentences:
        raise ValueError(f"Missing RULER training supporting facts for {row_id}")
    spans = []
    for sentence in sentences:
        value = str(sentence).strip()
        start = context.find(value) if value else -1
        if start < 0:
            raise ValueError(
                f"RULER context for {row_id} does not contain supporting sentence: {value[:100]!r}"
            )
        spans.append((start, start + len(value)))
    start = min(left for left, _ in spans)
    end = max(right for _, right in spans)
    result = context[start:end]
    if not result:
        raise ValueError(f"Empty supporting-fact span for {row_id}")
    return result


def convert_acs_jsonl(
    rows: list[dict], use_answer_proxy: bool, use_supporting_facts: bool
) -> list[dict]:
    converted = []
    seen: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        row_id = str(row.get("id", "")).strip()
        question = str(row.get("question", "")).strip()
        context = row.get("context")
        if not row_id or row_id in seen or not question or not isinstance(context, str):
            raise ValueError(f"Malformed or duplicate prepared row {row_number}: {row_id!r}")
        seen.add(row_id)
        answer = row.get("answer", row.get("answers", ""))
        answers = answer if isinstance(answer, list) else [answer]
        answers = [str(value) for value in answers if str(value)]
        if not answers:
            raise ValueError(f"Missing answer label for {row_id}")
        gold_information = ""
        gold_information_type = "unavailable"
        if use_supporting_facts:
            if row.get("benchmark") != "ruler_hotpotqa" or row.get("source_split") != "train":
                raise ValueError(
                    f"Supporting-fact supervision is restricted to RULER train rows: {row_id}"
                )
            gold_information = support_span(
                context, row.get("supporting_fact_sentences"), row_id
            )
            gold_information_type = "supporting_fact_span"
        elif use_answer_proxy:
            occurrences = [(context.lower().find(value.lower()), value) for value in answers]
            occurrences = [(position, value) for position, value in occurrences if position >= 0]
            if occurrences:
                position, value = min(occurrences, key=lambda item: item[0])
                gold_information = context[position:position + len(value)]
                gold_information_type = "answer_mention_proxy"
        converted.append({
            "id": row_id,
            "input": question,
            "context": context,
            "answers": answers,
            "gold_information": gold_information,
            "gold_information_type": gold_information_type,
            "task": "multihop_qa",
            "benchmark": row.get("benchmark"),
            "ruler_task": row.get("ruler_task"),
            "source_id": row.get("source_id"),
            "source_hotpotqa_id": row.get("source_hotpotqa_id"),
            "source_split": row.get("source_split"),
            "context_length_tokens": row.get("context_length_tokens"),
            "question_index": row.get("question_index"),
            "build_id": row.get("build_id"),
        })
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Prepared RULER-HotpotQA train/dev ACS-schema JSONL")
    parser.add_argument("--format", choices=("acs_jsonl",), required=True,
                        help="Only RULER-HotpotQA cohorts are accepted in this experiment")
    parser.add_argument("--answer-position-proxy", action="store_true",
                        help="For RULER dev evaluation, retain the first literal answer mention as metadata only")
    parser.add_argument("--supporting-facts", action="store_true",
                        help="Use exact RULER-HotpotQA train supporting sentences as DCC sufficiency labels")
    parser.add_argument("--out", type=Path, required=True,
                        help="DCC task JSON file (normally named multihop_qa.json)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {args.out}; choose a new path or pass --overwrite")

    if args.answer_position_proxy and args.supporting_facts:
        raise SystemExit("Choose either --answer-position-proxy or --supporting-facts, not both")
    if not (args.answer_position_proxy or args.supporting_facts):
        raise SystemExit("Specify --supporting-facts for RULER train or --answer-position-proxy for RULER dev")
    converted = convert_acs_jsonl(
        read_jsonl(args.source), args.answer_position_proxy, args.supporting_facts
    )

    if not converted:
        raise SystemExit("No rows were converted")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(converted, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.out)
    print(f"Saved DCC task data: {args.out} ({len(converted)} rows)")
    if args.answer_position_proxy:
        print("WARNING: gold_information is only a first-answer-mention proxy; do not interpret it as complete evidence timing.")
    if args.supporting_facts:
        print("Training labels use exact supporting-sentence spans from the separate RULER train cohort.")


if __name__ == "__main__":
    main()
