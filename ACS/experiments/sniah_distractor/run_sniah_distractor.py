"""Run the full-trajectory S-NIAH distractor arm (paper Table 3)."""

from __future__ import annotations

import argparse
import hashlib
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
from prepare_sniah_distractor import (
    DATA_PATH, EXPECTED_ROWS, SNIAH_ROOT, SOURCE_PATH, expected_build_id, validate,
)

EXPECTED_MODEL = "qwen/qwen3-14b"
# The shared Qwen3-14B S-NIAH reference configuration (24K chunks, 6K notes).
CONFIG_PATH = SNIAH_ROOT / "config_sniah_qwen14b_base.yaml"
OUTPUT_NAME = "sniah_distractor_qwen14b_chunk24k_notes6k_full.jsonl"


def successes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [row for row in dedupe_prefer_success(read_jsonl(path))
            if not row.get("failed") and row.get("steps")]


def main() -> None:
    parser = argparse.ArgumentParser()
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
    if args.prepare:
        subprocess.run(
            [sys.executable, "-u", str(root / "prepare_sniah_distractor.py")],
            cwd=root, check=True,
        )
    if not DATA_PATH.exists():
        raise SystemExit(f"Missing {DATA_PATH}; rerun with --prepare")
    inputs = validate(DATA_PATH)
    input_ids = {str(row["id"]) for row in inputs}
    # Table 3 is the 3-distractor, seed-0 build of the shared S-NIAH file; refuse any other.
    if not SOURCE_PATH.exists():
        raise SystemExit(f"Missing {SOURCE_PATH}; rerun with --prepare")
    paper_build = expected_build_id(hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest())
    if {str(row["build_id"]) for row in inputs} != {paper_build}:
        raise SystemExit(
            f"{DATA_PATH.name} is not the Table 3 build (3 distractors, seed 0, from "
            f"{SOURCE_PATH}); rerun with --prepare"
        )

    # Fail before launching 250 jobs if the configured endpoint lacks the model.
    cfg = load_config(str(CONFIG_PATH))
    models = make_client(cfg).models.list()
    available = {str(model.id) for model in models.data}
    if EXPECTED_MODEL not in available:
        raise SystemExit(
            f"configured endpoint does not expose {EXPECTED_MODEL!r}; "
            f"models={sorted(available)}"
        )
    if cfg["server"]["model"] != EXPECTED_MODEL:
        raise SystemExit(f"wrong model {cfg['server']['model']!r}")
    if cfg["online_stop"]["rule"] != "none":
        raise SystemExit("expected full-trajectory rule:none")
    stop_cfg = dict(cfg["online_stop"])
    stop_cfg.update({"rule": "none", "record_verbalized": True, "record_gate": True})
    expected_spec = execution_config(cfg, stop_cfg)
    expected_fingerprint = execution_fingerprint(expected_spec)

    expected = args.limit if args.limit else EXPECTED_ROWS
    output_path = root / "results" / OUTPUT_NAME
    completed = successes(output_path)
    for row in completed:
        validate_execution_resume(row, expected_spec, expected_fingerprint)
    completed_ids = {str(row["id"]) for row in completed}
    if not completed_ids.issubset(input_ids):
        raise SystemExit("output contains rows outside this distractor build")
    if len(completed_ids) < expected:
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "run_fold.py"),
            "--data", str(DATA_PATH),
            "--out", str(output_path),
            "--config", str(CONFIG_PATH),
            "--workers", str(args.workers),
            "--rule", "none",
            "--record-verbalized",
            "--record-gate",
        ]
        if args.limit:
            command.extend(["--limit", str(args.limit)])
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    completed = successes(output_path)
    for row in completed:
        validate_execution_resume(row, expected_spec, expected_fingerprint)
    if len({str(row["id"]) for row in completed}) < expected:
        raise SystemExit(
            f"incomplete ({len(completed)}/{expected}). "
            "Rerun the same command; successful samples are checkpointed."
        )
    wrong_models = {str(row.get("model")) for row in completed
                    if str(row.get("model")) != EXPECTED_MODEL}
    if wrong_models:
        raise SystemExit(f"unexpected model(s) {sorted(wrong_models)}")
    # The build stores the target needle's position; run_fold passes it through.
    if any(not row.get("evidence_char_starts") for row in completed):
        raise SystemExit("trajectories lack evidence_char_starts; rerun on the prepared build")
    print(f"COMPLETE -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
