"""Select a reproducible HotpotQA-train question cohort for RULER DCC training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Official HotpotQA training JSON array")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=120,
                        help="120 questions x five RULER lengths = 600 DCC candidates")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    if not args.overwrite and (args.out.exists() or manifest_path.exists()):
        raise SystemExit("Output exists; choose a new path or pass --overwrite")
    rows = json.loads(args.source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Expected official HotpotQA train JSON array")
    ids = [str(row.get("_id", "")).strip() for row in rows]
    if not rows or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise SystemExit("Training source must have unique, non-empty HotpotQA _id values")
    if args.questions < 1 or len(rows) < args.questions:
        raise SystemExit(f"Need at least {args.questions} HotpotQA training questions")

    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{args.seed}:{row['_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected = ranked[:args.questions]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(selected, ensure_ascii=False)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(args.out)
    manifest = {
        "benchmark_source": "official HotpotQA train",
        "source_file": str(args.source.resolve()),
        "source_sha256": sha256(args.source),
        "selected_questions": len(selected),
        "seed": args.seed,
        "selection": "lowest SHA-256 ranks of seed:HotpotQA _id",
        "selected_hotpot_ids": [str(row["_id"]) for row in selected],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(row["_id"]) for row in selected)).encode("utf-8")
        ).hexdigest(),
        "output": str(args.out.resolve()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(selected)} disjoint HotpotQA train questions: {args.out}")
    print(f"Train-source manifest: {manifest_path}")


if __name__ == "__main__":
    main()
