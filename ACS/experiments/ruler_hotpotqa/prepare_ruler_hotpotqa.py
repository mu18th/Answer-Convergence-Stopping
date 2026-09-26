"""Convert official NVIDIA RULER qa_2 files into ACS open-QA rows.

The same HotpotQA question indices are selected at every requested context
length.  No model outputs are inspected during selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


QA_HEADER = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\nThe following are given documents.\n\n"
)
QA_MARKER = (
    "\n\nAnswer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\nQuestion: "
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_file(root: Path, length: int) -> Path:
    candidates = [
        root / str(length) / "data" / "qa_2" / "validation.jsonl",
        root / "qa_2" / f"{length}.jsonl",
        root / str(length) / "qa_2" / "validation.jsonl",
    ]
    found = [path for path in candidates if path.exists()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one RULER qa_2 file for {length}; checked: "
            + ", ".join(str(path) for path in candidates)
        )
    return found[0]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def split_official_prompt(row: dict, source: Path) -> tuple[str, str]:
    text = str(row.get("input", ""))
    answer_prefix = str(row.get("answer_prefix", ""))
    # Some mirrors retain the answer prefix inside input; official current files
    # split it into a separate field. Support both without changing task content.
    if answer_prefix and text.endswith(answer_prefix):
        text = text[: -len(answer_prefix)]
    if not text.startswith(QA_HEADER):
        raise RuntimeError(
            f"{source}: row {row.get('index')} does not use NVIDIA RULER's "
            "official base qa template. Regenerate with model_template_type=base."
        )
    marker = text.rfind(QA_MARKER)
    if marker < len(QA_HEADER):
        raise RuntimeError(f"{source}: row {row.get('index')} has no QA marker")
    context = text[len(QA_HEADER):marker]
    question = text[marker + len(QA_MARKER):].strip()
    if not context.strip() or not question:
        raise RuntimeError(f"{source}: row {row.get('index')} parsed empty content")
    return context, question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-root", type=Path, required=True,
                        help="root containing <length>/data/qa_2/validation.jsonl")
    parser.add_argument("--lengths", type=int, nargs="+", required=True,
                        help="token buckets, e.g. 8192 16384 32768 65536 131072")
    parser.add_argument("--samples-per-length", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection-offset", type=int, default=0,
        help="take the next ranked block; 50 creates a disjoint second 5x50 cohort",
    )
    parser.add_argument("--out", type=Path,
                        default=Path("data/ruler_hotpotqa.jsonl"))
    args = parser.parse_args()
    lengths = sorted(set(args.lengths))
    if len(lengths) != len(args.lengths) or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain unique positive integers")
    if not 1 <= args.samples_per_length <= 500:
        raise SystemExit("--samples-per-length must be between 1 and 500")
    if args.selection_offset < 0:
        raise SystemExit("--selection-offset must be nonnegative")

    sources = {length: source_file(args.ruler_root.resolve(), length) for length in lengths}
    raw = {length: read_jsonl(path) for length, path in sources.items()}
    indexed: dict[int, dict[int, dict]] = {}
    for length, rows in raw.items():
        by_index = {int(row["index"]): row for row in rows}
        if len(by_index) != len(rows):
            raise RuntimeError(f"{sources[length]} contains duplicate indices")
        indexed[length] = by_index

    common = set.intersection(*(set(rows) for rows in indexed.values()))
    selection_end = args.selection_offset + args.samples_per_length
    if len(common) < selection_end:
        raise RuntimeError(
            f"Only {len(common)} question indices are shared by every length; "
            f"requested ranked slice [{args.selection_offset}:{selection_end}]"
        )
    # Seeded hash ordering is deterministic and independent of all model outputs.
    ranked = sorted(
        common,
        key=lambda index: hashlib.sha256(f"{args.seed}:{index}".encode()).hexdigest(),
    )
    chosen = ranked[args.selection_offset:selection_end]

    source_hashes = {str(length): sha256(path) for length, path in sources.items()}
    build_material = {
        "benchmark": "NVIDIA/RULER qa_2 (HotpotQA)",
        "lengths": lengths,
        "samples_per_length": args.samples_per_length,
        "seed": args.seed,
        "indices": chosen,
        "source_sha256": source_hashes,
    }
    # Preserve the original first-cohort build ID byte-for-byte. Only additional
    # cohorts add the offset field to their provenance material.
    if args.selection_offset:
        build_material["selection_offset"] = args.selection_offset
    build_id = hashlib.sha256(
        json.dumps(build_material, sort_keys=True).encode()
    ).hexdigest()[:20]

    converted = []
    identity: dict[int, tuple[str, tuple[str, ...]]] = {}
    for length in lengths:
        for index in chosen:
            source_row = indexed[length][index]
            context, question = split_official_prompt(source_row, sources[length])
            outputs = [str(value).strip() for value in source_row.get("outputs", [])]
            if not outputs or any(not value for value in outputs):
                raise RuntimeError(f"{sources[length]} index {index}: empty gold output")
            signature = (question, tuple(outputs))
            if index in identity and identity[index] != signature:
                raise RuntimeError(
                    f"Question index {index} changes question/gold across lengths"
                )
            identity[index] = signature
            converted.append({
                "id": f"ruler_hotpotqa_{length}_{index:04d}",
                "source_id": f"hotpotqa_{index:04d}",
                "benchmark": "ruler_hotpotqa",
                "ruler_task": "qa_2",
                "context_length_tokens": length,
                "official_length": int(source_row.get("length", length)),
                "question_index": index,
                "question": question,
                "answer": outputs,
                "answers": outputs,
                "answer_type": "ruler_string_match_part",
                "task_shape": "open",
                "context": context,
                "n_chars": len(context),
                "answer_prefix": str(source_row.get("answer_prefix", " Answer:")),
                "build_id": build_id,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    manifest = {
        **build_material,
        "build_id": build_id,
        "rows": len(converted),
        "selection_before_model_outputs": True,
        "source_files": {str(k): str(v.resolve()) for k, v in sources.items()},
        "output": str(args.out.resolve()),
    }
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(converted)} rows ({args.samples_per_length} x {len(lengths)}): {args.out}")
    print(f"Build ID: {build_id}")


if __name__ == "__main__":
    main()
