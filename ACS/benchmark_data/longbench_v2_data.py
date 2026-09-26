from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from .data_utils import output_path

LONG_BENCH_DATASET = "THUDM/LongBench-v2"
LONG_BENCH_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"


def longbench_v2_rows(cfg, domain=None, *, strict=False):
    os.environ.setdefault("HF_HOME", cfg["server"]["cache_dir"])
    from datasets import load_dataset

    dataset = load_dataset(
        LONG_BENCH_DATASET,
        revision=LONG_BENCH_REVISION,
        split="train",
        cache_dir=cfg["server"]["cache_dir"],
    )
    for row in dataset:
        if domain and str(row.get("domain")) != domain:
            continue
        choices = {
            letter: str(row.get(f"choice_{letter}"))
            for letter in "ABCD"
            if row.get(f"choice_{letter}") is not None
        }
        answer = str(row.get("answer", "")).strip().upper()[:1]
        if len(choices) != 4 or answer not in "ABCD":
            if strict:
                raise ValueError(
                    f"LongBench-v2 row {row.get('_id')!r} is not a valid "
                    "four-choice MCQ; refusing to call a partial export the "
                    "whole benchmark"
                )
            continue
        yield {
            "id": str(row["_id"]),
            "domain": str(row.get("domain")),
            "difficulty": str(row.get("difficulty")),
            "length_bin": str(row.get("length")),
            "question": str(row["question"]),
            "choices": choices,
            "answer": answer,
            "context": row["context"] or "",
            "n_chars": len(row["context"] or ""),
            "task_shape": "mcq",
        }


def longbench_v2(cfg):
    config = cfg["benchmarks"]["longbench_v2"]
    if config.get("all_samples", False):
        # Whole-benchmark mode deliberately performs no character-length filter,
        # shuffling, domain balancing, or n-sample truncation.  Source order is
        # retained, making the exported file auditable against the HF dataset.
        # Stream to an atomic temporary file: the 503 contexts occupy roughly
        # 450 MB, so materializing a second Python list is unnecessary and made
        # Windows preparation look hung for several minutes.
        expected_n = config.get("expected_n")
        output = output_path(
            cfg, str(config.get("output_name", "longbench_v2_all"))
        )
        temporary = output.with_suffix(output.suffix + ".tmp")
        ids, count = set(), 0
        with temporary.open("w", encoding="utf-8") as file:
            for row in longbench_v2_rows(cfg, strict=True):
                if row["id"] in ids:
                    raise RuntimeError(
                        f"Duplicate LongBench-v2 sample ID: {row['id']}"
                    )
                ids.add(row["id"])
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                if count % 25 == 0:
                    print(f"longbench_v2 preparation: {count} rows written", flush=True)
        if expected_n is not None and count != int(expected_n):
            raise RuntimeError(
                f"Expected {expected_n} LongBench-v2 rows, found {count}. "
                "Refusing to replace the data file with an incomplete export."
            )
        temporary.replace(output)
        print(
            f"longbench_v2: WHOLE BENCHMARK, {count} rows, "
            f"no length filter -> {output}"
        )
        return

    rng = random.Random(config["seed"])
    by_domain = defaultdict(list)
    for row in longbench_v2_rows(cfg):
        if config["min_chars"] <= row["n_chars"] <= config["max_chars"]:
            by_domain[row["domain"]].append(row)
    for domain_rows in by_domain.values():
        rng.shuffle(domain_rows)
    picked, index, domains = [], 0, sorted(by_domain)
    while len(picked) < config["n"] and any(by_domain.values()):
        domain = domains[index % len(domains)]
        if by_domain[domain]:
            picked.append(by_domain[domain].pop())
        index += 1
    output = output_path(cfg, "longbench_v2")
    with output.open("w", encoding="utf-8") as file:
        for row in picked:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"longbench_v2: {len(picked)} -> {output}")
