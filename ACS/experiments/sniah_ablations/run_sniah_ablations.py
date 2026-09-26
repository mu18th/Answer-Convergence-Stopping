"""Run the five full-trajectory S-NIAH ablation arms."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import dedupe_prefer_success, load_config, make_client, read_jsonl
from run_fold import (
    execution_config,
    execution_fingerprint,
    validate_execution_resume,
)
from prepare_sniah import EXPECTED_ROWS, validate

EXPECTED_MODEL = "qwen/qwen3-14b"
ARMS = {
    # Shared control: used once in both the chunk and notes-cap comparisons.
    "base24_notes6": (
        "config_sniah_qwen14b_base.yaml",
        "sniah_qwen14b_chunk24k_notes6k_full.jsonl",
    ),
    "chunk12": (
        "config_sniah_qwen14b_chunk12k.yaml",
        "sniah_qwen14b_chunk12k_notes6k_full.jsonl",
    ),
    "chunk48": (
        "config_sniah_qwen14b_chunk48k.yaml",
        "sniah_qwen14b_chunk48k_notes6k_full.jsonl",
    ),
    "notes3": (
        "config_sniah_qwen14b_notes3k.yaml",
        "sniah_qwen14b_chunk24k_notes3k_full.jsonl",
    ),
    "notes12": (
        "config_sniah_qwen14b_notes12k.yaml",
        "sniah_qwen14b_chunk24k_notes12k_full.jsonl",
    ),
}


def successes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [row for row in dedupe_prefer_success(read_jsonl(path))
            if not row.get("failed") and row.get("steps")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
        help="Run selected arms; default runs all five unique configurations.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit(
            "$env:OPENROUTER_API_KEY is missing. Set it before running; "
            "never place the key in YAML or source code."
        )
    data_path = root / "data" / "sniah.jsonl"
    base_config = root / "config_sniah_qwen14b_base.yaml"
    if args.prepare:
        subprocess.run(
            [sys.executable, "-u", str(root / "prepare_sniah.py"),
             "--config", str(base_config)],
            cwd=root, check=True,
        )
    if not data_path.exists():
        raise SystemExit(f"Missing {data_path}; rerun with --prepare")
    inputs = validate(data_path)
    input_ids = {str(row["id"]) for row in inputs}

    # Fail before launching 250 jobs if the configured endpoint lacks the model.
    base_cfg = load_config(str(base_config))
    models = make_client(base_cfg).models.list()
    available = {str(model.id) for model in models.data}
    if EXPECTED_MODEL not in available:
        raise SystemExit(
            f"configured endpoint does not expose {EXPECTED_MODEL!r}; "
            f"models={sorted(available)}"
        )

    expected = args.limit if args.limit else EXPECTED_ROWS
    for arm in args.arms:
        config_name, output_name = ARMS[arm]
        config_path = root / config_name
        output_path = root / "results" / output_name
        cfg = load_config(str(config_path))
        if cfg["server"]["model"] != EXPECTED_MODEL:
            raise SystemExit(f"{arm}: wrong model {cfg['server']['model']!r}")
        if cfg["online_stop"]["rule"] != "none":
            raise SystemExit(f"{arm}: expected full-trajectory rule:none")
        stop_cfg = dict(cfg["online_stop"])
        stop_cfg.update({"rule": "none", "record_verbalized": True, "record_gate": True})
        expected_spec = execution_config(cfg, stop_cfg)
        expected_fingerprint = execution_fingerprint(expected_spec)

        completed = successes(output_path)
        for row in completed:
            validate_execution_resume(row, expected_spec, expected_fingerprint)
        completed_ids = {str(row["id"]) for row in completed}
        if not completed_ids.issubset(input_ids):
            raise SystemExit(f"{arm}: output contains rows outside this S-NIAH build")
        if len(completed_ids) < expected:
            command = [
                sys.executable, "-u", str(PROJECT_ROOT / "run_fold.py"),
                "--data", str(data_path),
                "--out", str(output_path),
                "--config", str(config_path),
                "--workers", str(args.workers),
                "--rule", "none",
                "--record-verbalized",
                "--record-gate",
            ]
            if args.limit:
                command.extend(["--limit", str(args.limit)])
            print(f"\n===== {arm} =====", flush=True)
            print("Running:", " ".join(command), flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        completed = successes(output_path)
        for row in completed:
            validate_execution_resume(row, expected_spec, expected_fingerprint)
        if len({str(row["id"]) for row in completed}) < expected:
            raise SystemExit(
                f"{arm}: incomplete ({len(completed)}/{expected}). "
                "Rerun the same command; successful samples are checkpointed."
            )
        wrong_models = {str(row.get("model")) for row in completed
                        if str(row.get("model")) != EXPECTED_MODEL}
        if wrong_models:
            raise SystemExit(f"{arm}: unexpected model(s) {sorted(wrong_models)}")

        # Oracle positions are deterministic labels from the prepared needle text.
        if not args.limit and any(not row.get("evidence_char_starts") for row in completed):
            subprocess.run(
                [sys.executable, "-u", str(PROJECT_ROOT / "eval" / "add_oracle_labels.py"),
                 str(output_path), str(data_path)],
                cwd=PROJECT_ROOT, check=True,
            )
        print(f"{arm}: COMPLETE -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
