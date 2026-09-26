"""Build all 830 BrowseComp-Plus questions in one deterministic pass.

Every query is retained. Each context contains at least ten documents and all
mandatory evidence/gold documents; a query with more than ten mandatory
documents therefore receives more than ten documents rather than being split,
discarded, or stripped of evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from benchmark_data.browsecomp_plus_data import (
    CHARS_PER_TOKEN,
    CORPUS_DATASET,
    CORPUS_REVISION,
    QUERY_DATASET,
    QUERY_REVISION,
    _bcp_bucket,
    _bcp_decrypt,
    _build_context,
    _distractor_pool,
    _mandatory_documents,
    _oracle_fields,
    _sample_documents,
)
from common import load_config

EXPECTED_ROWS = 830
MINIMUM_DOCUMENTS = 10
CONSTRUCTION = "ACS_min10doc_all830_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def publish(source: Path, destination: Path) -> None:
    last_error = None
    for attempt in range(10):
        try:
            source.replace(destination)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(min(0.25 * 2**attempt, 5.0))
    raise last_error


def validate(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    ids = [str(row.get("id", "")) for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_ROWS} unique rows, found {len(rows)} rows and "
            f"{len(set(ids))} unique IDs"
        )
    if {row.get("construction") for row in rows} != {CONSTRUCTION}:
        raise RuntimeError("Unexpected BrowseComp construction tag")
    if any(int(row.get("n_documents", 0)) < MINIMUM_DOCUMENTS for row in rows):
        raise RuntimeError("A BrowseComp row contains fewer than ten documents")
    if len({str(row.get("build_id", "")) for row in rows}) != 1:
        raise RuntimeError("The 830 rows do not share one build ID")
    print(f"Validated one BrowseComp build: {EXPECTED_ROWS}/830 unique rows", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_full.yaml")
    parser.add_argument("--out", default="data/browsecomp_plus_all830.jsonl")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = Path(args.config)
    output_path = Path(args.out)
    if not config_path.is_absolute():
        config_path = root / config_path
    if not output_path.is_absolute():
        output_path = root / output_path
    if args.validate_only:
        validate(output_path)
        return
    if output_path.exists():
        try:
            validate(output_path)
        except Exception as error:
            print(f"Existing build is invalid; rebuilding: {error}", flush=True)
        else:
            print(f"Reusing {output_path}", flush=True)
            return

    cfg = load_config(str(config_path))
    bench = cfg["benchmarks"]["browsecomp_plus"]
    if int(bench["documents_per_query"]) != MINIMUM_DOCUMENTS:
        raise RuntimeError("The all-830 protocol is locked to a ten-document minimum")
    seed = int(bench["seed"])
    max_distractor = int(bench["max_distractor_chars"])
    os.environ.setdefault("HF_HOME", cfg["server"]["cache_dir"])

    from datasets import load_dataset

    queries = load_dataset(
        QUERY_DATASET,
        revision=QUERY_REVISION,
        split="test",
        cache_dir=cfg["server"]["cache_dir"],
    )
    corpus = load_dataset(
        CORPUS_DATASET,
        revision=CORPUS_REVISION,
        split="train",
        cache_dir=cfg["server"]["cache_dir"],
    )
    if len(queries) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Pinned protocol expects 830 official questions; dataset exposes {len(queries)}"
        )

    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    pool = _distractor_pool(corpus, max_distractor)
    rng = random.Random(seed)
    payload = {
        "construction": CONSTRUCTION,
        "query_dataset": QUERY_DATASET,
        "query_revision": QUERY_REVISION,
        "corpus_dataset": CORPUS_DATASET,
        "corpus_revision": CORPUS_REVISION,
        "seed": seed,
        "minimum_documents_per_query": MINIMUM_DOCUMENTS,
        "mandatory_documents_are_never_dropped": True,
        "max_distractor_chars": max_distractor,
        "rows": EXPECTED_ROWS,
    }
    build_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]

    rows = []
    for query_index in order:
        query = queries[query_index]
        mandatory, evidence_ids = _mandatory_documents(query)
        if not evidence_ids:
            raise RuntimeError(f"Official query {query['query_id']} has no evidence documents")
        target = max(MINIMUM_DOCUMENTS, len(mandatory))
        documents = _sample_documents(corpus, mandatory, target, rng, pool)
        context, evidence_starts, evidence_ends = _build_context(documents, evidence_ids)
        answer = _bcp_decrypt(query["answer"])
        oracle, _ = _oracle_fields(context, answer, evidence_starts, evidence_ends)
        rows.append({
            "id": f"bcp_{query['query_id']}",
            "bucket": _bcp_bucket(len(context) / CHARS_PER_TOKEN),
            "question": _bcp_decrypt(query["query"]),
            "answer": answer,
            "answer_type": "string",
            "construction": CONSTRUCTION,
            "build_id": build_id,
            "n_documents": len(documents),
            "n_evidence": len(evidence_ids),
            "n_negatives": len(documents) - len(evidence_ids),
            "document_ids": [document["docid"] for document in documents],
            **oracle,
            "context": context,
            "n_chars": len(context),
            "task_shape": "open",
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    publish(temporary, output_path)
    validate(output_path)
    chunks = sum((int(row["n_chars"]) + 23999) // 24000 for row in rows)
    print(f"Saved {output_path} | rows=830 | chunks@24K={chunks:,}", flush=True)


if __name__ == "__main__":
    main()
