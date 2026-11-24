"""Validate a paired RULER-HotpotQA cohort at five context lengths."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_LENGTHS = (8192, 16384, 32768, 65536, 131072)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--per-length", type=int, default=100)
    parser.add_argument("--cohort", choices=("train", "eval"), default="eval")
    args = parser.parse_args()

    lengths = tuple(sorted(args.lengths))
    if len(set(lengths)) != len(lengths) or not lengths or args.per_length < 1:
        raise SystemExit("Lengths must be unique and --per-length must be positive")

    rows: list[dict] = []
    with args.data.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise SystemExit(f"{args.data}:{line_number}: invalid JSON") from error

    expected_total = len(lengths) * args.per_length
    if len(rows) != expected_total:
        raise SystemExit(f"Expected {expected_total} trajectories, found {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise SystemExit("Cohort IDs must be non-empty and unique across lengths")

    by_length: dict[int, list[dict]] = defaultdict(list)
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        length = row.get("context_length_tokens")
        if length not in lengths:
            raise SystemExit(f"Unexpected context length {length!r} in {row.get('id')}")
        if (row.get("benchmark") != "ruler_hotpotqa"
                or row.get("ruler_task") != "qa_2"
                or row.get("source_split") != ("train" if args.cohort == "train" else "dev")
                or not isinstance(row.get("question"), str)
                or not row["question"].strip()
                or not isinstance(row.get("context"), str)
                or not row["context"].strip()
                or not isinstance(row.get("answer"), list)
                or not row["answer"]):
            raise SystemExit(f"Malformed RULER qa_2 row: {row.get('id')!r}")
        source_id = str(row.get("source_id", ""))
        source_hotpotqa_id = str(row.get("source_hotpotqa_id", ""))
        if not source_id:
            raise SystemExit(f"Missing source_id in {row.get('id')!r}")
        if (not str(row.get("id", "")).startswith(f"ruler_hotpotqa_{args.cohort}_")
                or not source_hotpotqa_id
                or source_id != f"hotpotqa_{args.cohort}_{source_hotpotqa_id}"):
            raise SystemExit(f"Cohort namespace mismatch in {row.get('id')!r}")
        if args.cohort == "train":
            support = row.get("supporting_fact_sentences")
            if (not isinstance(support, list) or not support
                    or any(not isinstance(sentence, str) or not sentence
                           or sentence not in row["context"] for sentence in support)):
                raise SystemExit(
                    f"Training row {row.get('id')!r} lacks fully located supporting-fact sentences"
                )
        by_length[int(length)].append(row)
        by_source[source_id].append(row)

    for length in lengths:
        bucket = by_length[length]
        if len(bucket) != args.per_length:
            raise SystemExit(
                f"Length {length}: expected {args.per_length}, found {len(bucket)}"
            )
        source_ids = {str(row["source_id"]) for row in bucket}
        if len(source_ids) != args.per_length:
            raise SystemExit(f"Length {length}: duplicate source question IDs")

    if len(by_source) != args.per_length:
        raise SystemExit(
            f"Expected {args.per_length} unique questions repeated across lengths; "
            f"found {len(by_source)}"
        )
    for source_id, repeated_rows in by_source.items():
        if len(repeated_rows) != len(lengths):
            raise SystemExit(
                f"{source_id}: expected {len(lengths)} length variants, "
                f"found {len(repeated_rows)}"
            )
        if {int(row["context_length_tokens"]) for row in repeated_rows} != set(lengths):
            raise SystemExit(f"{source_id}: missing or duplicate context-length variant")
        questions = {str(row["question"]) for row in repeated_rows}
        answers = {json.dumps(row["answer"], sort_keys=True) for row in repeated_rows}
        if len(questions) != 1 or len(answers) != 1:
            raise SystemExit(f"{source_id}: question/gold answer changed across lengths")

    build_ids = {str(row.get("build_id", "")) for row in rows}
    if len(build_ids) != 1 or not next(iter(build_ids)):
        raise SystemExit("All rows must carry the same non-empty build_id")

    counts = Counter(int(row["context_length_tokens"]) for row in rows)
    print(
        f"PASS: {len(rows)} RULER-HotpotQA {args.cohort} trajectories; "
        f"{len(by_source)} questions x {len(lengths)} lengths; "
        f"build_id={next(iter(build_ids))}"
    )
    print("Per-length rows: " + ", ".join(f"{length}={counts[length]}" for length in lengths))


if __name__ == "__main__":
    main()
