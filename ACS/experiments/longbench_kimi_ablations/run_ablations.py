"""Run matched Kimi LongBench-v2 folding ablations with resumable JSONL output."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import load_config
from run_fold import (
    execution_config,
    execution_fingerprint,
    validate_execution_resume,
)

ARMS = {
    "base_24k_notes6k": "config_base_24k_notes6k.yaml",
    "chunk12k_notes6k": "config_chunk12k_notes6k.yaml",
    "chunk48k_notes6k": "config_chunk48k_notes6k.yaml",
    "chunk24k_notes3k": "config_chunk24k_notes3k.yaml",
    "chunk24k_notes12k": "config_chunk24k_notes12k.yaml",
}
EXPECTED_ROWS = 80
EXPECTED_IDS_SHA256 = (
    "66fd3251bdca1d9299ebd8ce9d43e76f1f93a844b7dc300e342270802164be03"
)


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def successful_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("id"))
            if sample_id not in rows or (rows[sample_id].get("failed") and not row.get("failed")):
                rows[sample_id] = row
    return {key: row for key, row in rows.items() if not row.get("failed") and row.get("steps")}


def ids_in_order(path: Path) -> list[str]:
    ids = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.append(str(json.loads(line)["id"]))
    if len(ids) != EXPECTED_ROWS or len(set(ids)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Expected the frozen {EXPECTED_ROWS} unique IDs, "
            f"found {len(ids)}/{len(set(ids))}"
        )
    return ids


def verify_frozen_manifest(here: Path, data_path: Path, wanted: list[str]) -> None:
    canonical_path = here / "matched80_ids.json"
    materialized_path = data_path.with_name(data_path.stem + "_manifest.json")
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    materialized = json.loads(materialized_path.read_text(encoding="utf-8"))
    frozen = [str(value) for value in canonical["ids"]]
    materialized_ids = [str(value) for value in materialized["ids"]]
    if (
        ids_sha256(frozen) != EXPECTED_IDS_SHA256
        or wanted != frozen
        or materialized_ids != frozen
    ):
        raise RuntimeError("Input IDs/order differ from the frozen 80-ID manifest")


def validate_execution_rows(
    rows: dict[str, dict], expected_spec: dict, expected_fingerprint: str
) -> None:
    for row in rows.values():
        validate_execution_resume(row, expected_spec, expected_fingerprint)


def extract_existing_base(
    source: Path, output: Path, wanted: list[str],
    expected_spec: dict, expected_fingerprint: str,
) -> None:
    rows = successful_by_id(source)
    missing = set(wanted) - set(rows)
    if missing:
        raise RuntimeError(f"Existing 503-run is missing {len(missing)} selected IDs")
    validate_execution_rows(rows, expected_spec, expected_fingerprint)
    for sample_id in wanted:
        row = rows[sample_id]
        if row.get("model") != "moonshotai/kimi-k2.5":
            raise RuntimeError(f"{sample_id}: unexpected model {row.get('model')!r}")
        if int(row.get("chunk_chars", 0)) != 24_000:
            raise RuntimeError(f"{sample_id}: existing baseline is not 24K chunks")
        for step in row["steps"]:
            required = {
                "posterior", "verbalized", "verbalized_valid",
                "verbalized_stop", "verbalized_gate_valid",
                "cum_probe_tokens", "cum_verb_tokens", "cum_gate_tokens",
            }
            absent = required - set(step)
            if absent:
                raise RuntimeError(f"{sample_id}: baseline step lacks {sorted(absent)}")
            if not step["verbalized_valid"] or not step["verbalized_gate_valid"]:
                raise RuntimeError(
                    f"{sample_id}: baseline contains an invalid optional signal"
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in wanted:
            handle.write(json.dumps(rows[sample_id], ensure_ascii=False) + "\n")
    temporary.replace(output)
    print(
        f"Reused {len(wanted)} matched baseline trajectories -> {output}",
        flush=True,
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    default_source_root = here.parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=default_source_root)
    parser.add_argument(
        "--data",
        type=Path,
        default=here / "data" / "longbench_v2_matched_80.jsonl",
    )
    parser.add_argument("--results-dir", type=Path, default=here / "results")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-arm-passes", type=int, default=25,
                        help="maximum automatic passes used to retry only failed/missing IDs")
    parser.add_argument("--retry-wait-seconds", type=int, default=60,
                        help="pause between incomplete passes (useful after provider 429s)")
    parser.add_argument("--arm", choices=["all", *ARMS], default="all")
    parser.add_argument(
        "--reuse-existing-base",
        type=Path,
        default=None,
        help="optional compatible 503-run JSONL; default reruns the baseline",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Run make_matched_80.py first: {args.data}")
    runner = args.source_root / "run_fold.py"
    if not runner.exists():
        raise FileNotFoundError(f"Missing unchanged source runner: {runner}")
    wanted = ids_in_order(args.data)
    verify_frozen_manifest(here, args.data, wanted)
    target_n = len(wanted)
    selected_arms = list(ARMS) if args.arm == "all" else [args.arm]
    args.results_dir.mkdir(parents=True, exist_ok=True)

    for arm in selected_arms:
        output = args.results_dir / f"{arm}.jsonl"
        config = here / "configs" / ARMS[arm]
        cfg = load_config(str(config))
        stop_cfg = dict(cfg["online_stop"])
        stop_cfg["rule"] = "none"
        if arm == "base_24k_notes6k":
            stop_cfg.update({"record_verbalized": True, "record_gate": True})
        expected_spec = execution_config(cfg, stop_cfg)
        expected_fingerprint = execution_fingerprint(expected_spec)
        if arm == "base_24k_notes6k" and args.reuse_existing_base is not None:
            existing = args.reuse_existing_base.resolve()
            if args.dry_run:
                print(f"WOULD EXTRACT BASE: {existing} -> {output}")
            else:
                extract_existing_base(
                    existing, output, wanted,
                    expected_spec, expected_fingerprint,
                )
            continue

        if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY", "").strip():
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        command = [
            sys.executable, "-u", str(runner),
            "--data", str(args.data),
            "--out", str(output),
            "--config", str(config),
            "--workers", str(args.workers),
            "--rule", "none",
        ]
        # Only the reused/default arm is needed for the asked-vs-measured
        # stopping-signal rows. Chunk-size and notes-cap arms replay only the
        # measured posterior, so verbal-confidence and END/CONTINUE calls would
        # double API requests without contributing to those ablations.
        if arm == "base_24k_notes6k":
            command.extend(["--record-verbalized", "--record-gate"])
        print(f"\nARM {arm}\n{' '.join(command)}", flush=True)
        if args.dry_run:
            continue

        existing_rows = successful_by_id(output)
        validate_execution_rows(
            existing_rows, expected_spec, expected_fingerprint
        )
        if not set(existing_rows).issubset(wanted):
            raise RuntimeError(f"{arm}: output contains IDs outside the frozen cohort")
        previous_done = len(set(wanted) & set(existing_rows))
        for pass_index in range(1, args.max_arm_passes + 1):
            remaining_before = target_n - previous_done
            if remaining_before == 0:
                print(f"{arm}: {target_n}/{target_n} already successful", flush=True)
                break
            print(
                f"{arm}: pass {pass_index}/{args.max_arm_passes}; "
                f"successful={previous_done}/{target_n}; remaining={remaining_before}",
                flush=True,
            )
            # run_fold records ordinary per-sample failures durably and returns
            # success; ProbeInvalid or another process-level failure can return
            # nonzero. Either way, inspect durable successes and retry only what
            # remains. run_fold.resume_ids skips every successful ID.
            completed = subprocess.run(command, cwd=args.source_root, check=False)
            existing_rows = successful_by_id(output)
            validate_execution_rows(
                existing_rows, expected_spec, expected_fingerprint
            )
            if not set(existing_rows).issubset(wanted):
                raise RuntimeError(
                    f"{arm}: output contains IDs outside the frozen cohort"
                )
            done_now = len(set(wanted) & set(existing_rows))
            remaining_now = target_n - done_now
            print(
                f"{arm}: pass {pass_index} returncode={completed.returncode}; "
                f"successful={done_now}/{target_n}; remaining={remaining_now}",
                flush=True,
            )
            if remaining_now == 0:
                print(f"{arm}: COMPLETE {target_n}/{target_n}", flush=True)
                break
            if pass_index == args.max_arm_passes:
                raise RuntimeError(
                    f"{arm}: {remaining_now} samples remain after "
                    f"{args.max_arm_passes} automatic passes"
                )
            if done_now == previous_done:
                print(f"{arm}: no progress in this pass", flush=True)
            previous_done = done_now
            print(
                f"{arm}: waiting {args.retry_wait_seconds}s before retrying "
                "failed/missing samples only",
                flush=True,
            )
            time.sleep(max(0, args.retry_wait_seconds))


if __name__ == "__main__":
    main()

