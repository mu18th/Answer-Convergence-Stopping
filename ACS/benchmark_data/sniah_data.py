"""Load RULER ``niah_single_1`` rows and adapt them to the local runner schema.

The pinned Hugging Face artifact was generated with NVIDIA/RULER's official
pipeline and the Qwen3-4B-Instruct-2507 tokenizer. This module changes only the
container shape: RULER's prompt is split into the context and question fields
consumed by this repository's runners.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Final, TypedDict

from .data_utils import balanced_quota, output_path

DATASET_ID: Final = "VenusChenyy/RULER_50"
DATASET_REVISION: Final = "8ade4ed3c880f380dc1507046775f9910b5f52a8"
RULER_REVISION: Final = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
TASK_NAME: Final = "niah_single_1"


class RulerRow(TypedDict):
    task: str
    index: int
    input: str
    outputs: list[str]
    length: int
    length_w_model_temp: int
    answer_prefix: str
    token_position_answer: int


class SniahRow(TypedDict):
    id: str
    bucket: int
    question: str
    answer: str
    answer_type: str
    task: str
    context: str
    n_chars: int
    n_tokens_budget: int
    source_position: int
    answer_char_index: int
    answer_token_position: int
    answer_prefix: str
    source_dataset: str
    source_revision: str
    ruler_revision: str
    task_shape: str


@dataclass(frozen=True, slots=True)
class InvalidRulerRowError(Exception):
    bucket: int
    source_position: int
    reason: str

    def __str__(self) -> str:
        return (
            f"invalid RULER row for bucket {self.bucket} at source position "
            f"{self.source_position}: {self.reason}"
        )


@dataclass(frozen=True, slots=True)
class InsufficientRulerRowsError(Exception):
    requested: int
    available: int

    def __str__(self) -> str:
        return (
            f"requested {self.requested} S-NIAH rows, but the configured RULER "
            f"files contain only {self.available}"
        )


def _parse_source_row(raw) -> RulerRow:
    return {
        "task": str(raw["task"]),
        "index": int(raw["index"]),
        "input": str(raw["input"]),
        "outputs": [str(output) for output in raw["outputs"]],
        "length": int(raw["length"]),
        "length_w_model_temp": int(raw["length_w_model_temp"]),
        "answer_prefix": str(raw["answer_prefix"]),
        "token_position_answer": int(raw["token_position_answer"]),
    }


def adapt_ruler_row(
    source: RulerRow, bucket: int, source_position: int
) -> SniahRow:
    outputs = source["outputs"]
    if len(outputs) != 1:
        raise InvalidRulerRowError(
            bucket=bucket,
            source_position=source_position,
            reason=f"expected one answer, found {len(outputs)}",
        )
    context, separator, question = source["input"].rpartition("\n")
    if not separator:
        raise InvalidRulerRowError(
            bucket=bucket,
            source_position=source_position,
            reason="input does not end with a separate question line",
        )
    return {
        "id": f"ruler_niah_{bucket}_{source_position}",
        "bucket": bucket,
        "question": question + source["answer_prefix"],
        "answer": outputs[0],
        "answer_type": "substring",
        "task": TASK_NAME,
        "context": context,
        "n_chars": len(context),
        "n_tokens_budget": source["length_w_model_temp"],
        "source_position": source_position,
        "answer_char_index": source["index"],
        "answer_token_position": source["token_position_answer"],
        "answer_prefix": source["answer_prefix"],
        "source_dataset": DATASET_ID,
        "source_revision": DATASET_REVISION,
        "ruler_revision": RULER_REVISION,
        "task_shape": "open",
    }


def sniah(cfg) -> None:
    os.environ.setdefault("HF_HOME", cfg["server"]["cache_dir"])
    from datasets import load_dataset

    config = cfg["benchmarks"]["sniah"]
    buckets = cfg["benchmarks"]["buckets_tokens"]
    source = load_dataset(
        DATASET_ID,
        data_files={
            f"bucket_{bucket}": f"{TASK_NAME}/{bucket}.jsonl"
            for bucket in buckets
        },
        revision=DATASET_REVISION,
        cache_dir=cfg["server"]["cache_dir"],
    )
    splits = {bucket: source[f"bucket_{bucket}"] for bucket in buckets}
    quota = balanced_quota(
        {bucket: len(dataset) for bucket, dataset in splits.items()}, config["n"]
    )
    available = sum(quota.values())
    if available != config["n"]:
        raise InsufficientRulerRowsError(
            requested=config["n"], available=available
        )

    rng = random.Random(config["seed"])
    output = output_path(cfg, "sniah")
    counts: dict[int, int] = {}
    with output.open("w", encoding="utf-8") as file:
        for bucket in buckets:
            dataset = splits[bucket]
            positions = list(range(len(dataset)))
            rng.shuffle(positions)
            for source_position in positions[: quota[bucket]]:
                source_row = _parse_source_row(dataset[source_position])
                row = adapt_ruler_row(source_row, bucket, source_position)
                file.write(json.dumps(row) + "\n")
            counts[bucket] = quota[bucket]
    print(
        f"sniah[{DATASET_ID}@{DATASET_REVISION[:8]}]: "
        f"{sum(counts.values())} across buckets {counts} -> {output}"
    )
