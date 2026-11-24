"""Build paired, source-disjoint RULER-HotpotQA train/evaluation cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The released RULER-HotpotQA helpers, used unchanged.
RULER_HELPER_DIR = HERE.parent / "ruler_hotpotqa"
sys.path.insert(0, str(RULER_HELPER_DIR))
from prepare_ruler_hotpotqa import (  # noqa: E402
    read_jsonl,
    source_file,
    split_official_prompt,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_sentences(source_row: dict) -> list[str]:
    docs = {
        str(title): [str(sentence) for sentence in sentences]
        for title, sentences in source_row.get("context", [])
    }
    facts = source_row.get("supporting_facts", [])
    result = []
    for fact in facts:
        if not isinstance(fact, list) or len(fact) != 2:
            raise ValueError(f"Malformed supporting fact in {source_row.get('_id')}")
        title, sentence_index = str(fact[0]), fact[1]
        if title not in docs or not isinstance(sentence_index, int):
            raise ValueError(f"Invalid supporting fact in {source_row.get('_id')}: {fact}")
        if sentence_index < 0 or sentence_index >= len(docs[title]):
            raise ValueError(f"Supporting-fact index out of range: {fact}")
        sentence = docs[title][sentence_index]
        if sentence and sentence not in result:
            result.append(sentence)
    if not result:
        raise ValueError(f"No supporting facts for {source_row.get('_id')}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ruler-root", type=Path, required=True,
                        help="Root containing <length>/data/qa_2/validation.jsonl")
    parser.add_argument("--source-json", type=Path, required=True,
                        help="The exact HotpotQA JSON array used by this RULER generation")
    parser.add_argument("--cohort", choices=("train", "eval"), required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--samples-per-length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    lengths = sorted(set(args.lengths))
    if len(lengths) != len(args.lengths) or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain unique positive integers")
    if not 1 <= args.samples_per_length <= 500:
        raise SystemExit("--samples-per-length must be between 1 and 500")
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    if not args.overwrite and (args.out.exists() or manifest_path.exists()):
        raise SystemExit("Output exists; choose a new path or pass --overwrite")

    source_path = args.source_json.resolve()
    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list):
        raise SystemExit("--source-json must be a HotpotQA JSON array")
    source_ids = [str(row.get("_id", "")).strip() for row in source_rows]
    if not source_rows or any(not value for value in source_ids):
        raise SystemExit("Source rows need non-empty HotpotQA _id values")
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("Source HotpotQA IDs are not unique")

    raw_paths = {
        length: source_file(args.ruler_root.resolve(), length)
        for length in lengths
    }
    raw_rows = {length: read_jsonl(path) for length, path in raw_paths.items()}
    indexed: dict[int, dict[int, dict]] = {}
    for length, rows in raw_rows.items():
        by_index = {int(row["index"]): row for row in rows}
        if len(by_index) != len(rows):
            raise SystemExit(f"Duplicate RULER indices in {raw_paths[length]}")
        indexed[length] = by_index

    common = set.intersection(*(set(rows) for rows in indexed.values()))
    if len(common) < args.samples_per_length:
        raise SystemExit(
            f"Only {len(common)} question indices are shared across lengths; "
            f"need {args.samples_per_length}"
        )
    ranked = sorted(
        common,
        key=lambda index: hashlib.sha256(
            f"{args.seed}:{index}".encode("utf-8")
        ).hexdigest(),
    )
    chosen = ranked[:args.samples_per_length]
    if max(chosen) >= len(source_rows):
        raise SystemExit(
            "RULER indices exceed the supplied source JSON; use the exact source "
            "array used to generate these files"
        )

    raw_hashes = {str(length): sha256(path) for length, path in raw_paths.items()}
    source_hash = sha256(source_path)
    build_material = {
        "benchmark": "NVIDIA/RULER qa_2 (HotpotQA)",
        "cohort": args.cohort,
        "source_split": "HotpotQA train" if args.cohort == "train" else "HotpotQA dev",
        "source_sha256": source_hash,
        "lengths": lengths,
        "samples_per_length": args.samples_per_length,
        "selection_seed": args.seed,
        "question_indices": chosen,
        "ruler_source_sha256": raw_hashes,
    }
    build_id = hashlib.sha256(
        json.dumps(build_material, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]

    rows_out = []
    identities: dict[int, tuple[str, tuple[str, ...], str]] = {}
    for length in lengths:
        for index in chosen:
            raw = indexed[length][index]
            context, question = split_official_prompt(raw, raw_paths[length])
            source = source_rows[index]
            answers = [str(value).strip() for value in raw.get("outputs", [])]
            expected_answers = [str(source.get("answer", "")).strip()]
            if not answers or answers != expected_answers:
                raise SystemExit(
                    f"RULER/source gold-answer mismatch at {args.cohort} index {index}: "
                    f"{answers!r} != {expected_answers!r}"
                )
            if question != str(source.get("question", "")).strip():
                raise SystemExit(
                    f"RULER/source question mismatch at {args.cohort} index {index}"
                )
            hotpot_id = str(source["_id"])
            supporting = support_sentences(source)
            missing_support = [sentence for sentence in supporting if sentence not in context]
            if missing_support:
                raise SystemExit(
                    f"RULER context at {args.cohort} index {index}, length {length} "
                    f"omits {len(missing_support)} annotated supporting sentence(s); "
                    "refusing to create an incomplete evidence-labelled cohort"
                )
            signature = (question, tuple(answers), hotpot_id)
            if index in identities and identities[index] != signature:
                raise SystemExit(f"Question/gold changes across context lengths: {index}")
            identities[index] = signature
            rows_out.append({
                "id": f"ruler_hotpotqa_{args.cohort}_{length}_{index:04d}",
                "source_id": f"hotpotqa_{args.cohort}_{hotpot_id}",
                "source_hotpotqa_id": hotpot_id,
                "source_split": "train" if args.cohort == "train" else "dev",
                "benchmark": "ruler_hotpotqa",
                "ruler_task": "qa_2",
                "context_length_tokens": length,
                "official_length": int(raw.get("length", length)),
                "question_index": index,
                "question": question,
                "answer": answers,
                "answers": answers,
                "supporting_fact_sentences": supporting,
                "answer_type": "ruler_string_match_part",
                "task_shape": "open",
                "context": context,
                "n_chars": len(context),
                "answer_prefix": str(raw.get("answer_prefix", " Answer:")),
                "build_id": build_id,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    manifest = {
        **build_material,
        "build_id": build_id,
        "rows": len(rows_out),
        "unique_questions": len(identities),
        "selection_before_model_outputs": True,
        "source_json": str(source_path),
        "raw_files": {str(k): str(v.resolve()) for k, v in raw_paths.items()},
        "output": str(args.out.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Saved {len(rows_out)} {args.cohort} RULER trajectories "
        f"({args.samples_per_length} questions x {len(lengths)} lengths): {args.out}"
    )
    print(f"Build ID: {build_id}")


if __name__ == "__main__":
    main()
