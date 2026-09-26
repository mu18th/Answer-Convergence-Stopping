"""Plan, run, and resume the single 830-row BrowseComp-Plus benchmark."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import (
    OPENROUTER_429_ATTEMPTS,
    OPENROUTER_429_MAX_DELAY_S,
    dedupe_prefer_success,
    load_config,
    read_jsonl,
)
from run_fold import (
    execution_config,
    execution_fingerprint,
    validate_execution_resume,
)
from prepare_browsecomp_all830 import validate

ARMS = {
    "qwen14": {
        "model": "qwen/qwen3-14b",
        "config": "config_full.yaml",
        "output": "traj_bcp_full_14b.jsonl",
        "workers": 8,
    },
    "kimi": {
        "model": "moonshotai/kimi-k2.5",
        "config": "config_full.yaml",
        "output": "browsecomp_plus_kimi_k25_all830_full.jsonl",
        "workers": 8,
    },
    "qwen35": {
        "model": "qwen/qwen3.5-397b-a17b",
        "config": "config_full.yaml",
        "output": "browsecomp_plus_qwen35_397b_all830_full.jsonl",
        "workers": 8,
    },
}


def successful_ids(
    path: Path, expected_spec: dict | None = None,
    expected_fingerprint: str | None = None,
) -> set[str]:
    if not path.exists():
        return set()
    rows = [
        row for row in dedupe_prefer_success(read_jsonl(path))
        if not row.get("failed") and row.get("steps")
    ]
    if expected_spec is not None and expected_fingerprint is not None:
        for row in rows:
            validate_execution_resume(
                row, expected_spec, expected_fingerprint
            )
    return {str(row["id"]) for row in rows}


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--data", type=Path, default=root / "data" / "browsecomp_plus_all830.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-passes", type=int, default=15)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--confirm-paid-run", action="store_true")
    args = parser.parse_args()

    arm = ARMS[args.arm]
    data = args.data if args.data.is_absolute() else root / args.data
    output = args.output or root / "results" / arm["output"]
    if not output.is_absolute():
        output = root / output
    config = root / arm["config"]
    rows = validate(data)
    selected = rows[:args.limit] if args.limit else rows
    wanted = {str(row["id"]) for row in selected}
    workers = args.workers or int(arm["workers"])
    if workers < 1 or args.limit < 0 or args.max_passes < 1:
        raise SystemExit("workers/max-passes must be positive; limit must be nonnegative")
    chunks = sum(math.ceil(len(str(row["context"])) / 24000) for row in selected)
    print(json.dumps({
        "arm": args.arm,
        "model": arm["model"],
        "samples": len(selected),
        "cohort": "single_deterministic_all830",
        "chunks": chunks,
        "nominal_physical_calls_before_retries": chunks * 4,
        "workers": workers,
        "full_context_trajectories": True,
        "output": str(output.resolve()),
    }, indent=2), flush=True)
    if args.plan_only:
        return
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("Set OPENROUTER_API_KEY in the environment; never store it in files")

    cfg = load_config(str(config))
    cfg["server"]["model"] = arm["model"]
    if cfg["online_stop"]["rule"] != "none":
        raise RuntimeError("BrowseComp collection must record complete trajectories")
    stop_cfg = dict(cfg["online_stop"])
    stop_cfg.update({"rule": "none", "record_verbalized": True, "record_gate": True})
    expected_spec = execution_config(cfg, stop_cfg)
    # The child process explicitly removes any provider pin.
    expected_spec["openrouter_provider"] = ""
    expected_fingerprint = execution_fingerprint(expected_spec)

    child_env = os.environ.copy()
    child_env["MODEL"] = arm["model"]
    child_env.pop("OPENROUTER_PROVIDER", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(PROJECT_ROOT / "run_fold.py"),
        "--data", str(data.resolve()), "--out", str(output.resolve()),
        "--config", str(config), "--workers", str(workers),
        "--rule", "none", "--record-verbalized", "--record-gate",
    ]
    if args.limit:
        command += ["--limit", str(args.limit)]
    print(
        f"429 handling: {OPENROUTER_429_ATTEMPTS} attempts/call; exponential "
        f"backoff capped at {OPENROUTER_429_MAX_DELAY_S}s",
        flush=True,
    )
    for pass_no in range(1, args.max_passes + 1):
        completed_ids = successful_ids(
            output, expected_spec, expected_fingerprint
        )
        if not completed_ids.issubset(wanted):
            raise RuntimeError("Output contains IDs outside this all-830 build")
        done = completed_ids & wanted
        print(f"PASS {pass_no}/{args.max_passes}: {len(done)}/{len(wanted)}", flush=True)
        if done == wanted:
            break
        subprocess.run(command, cwd=PROJECT_ROOT, env=child_env, check=True)
    completed_ids = successful_ids(output, expected_spec, expected_fingerprint)
    if not completed_ids.issubset(wanted):
        raise RuntimeError("Output contains IDs outside this all-830 build")
    done = completed_ids & wanted
    if done != wanted:
        raise SystemExit(f"Incomplete: {len(done)}/{len(wanted)}; rerun the same command")
    print(f"COMPLETE: {len(done)}/{len(wanted)} -> {output}", flush=True)


if __name__ == "__main__":
    main()

