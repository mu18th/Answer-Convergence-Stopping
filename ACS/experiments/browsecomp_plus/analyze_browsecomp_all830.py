"""Analyze one deterministic complete 830-row BrowseComp build."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import dedupe_prefer_success, read_jsonl


ROOT = Path(__file__).resolve().parent
ARMS = {
    "qwen14": {
        "model": "qwen/qwen3-14b",
        "traj": "traj_bcp_full_14b.jsonl",
        "config": "config_full.yaml",
        "stem": "browsecomp_plus_qwen3_14b_all830",
    },
    "kimi": {
        "model": "moonshotai/kimi-k2.5",
        "traj": "browsecomp_plus_kimi_k25_all830_full.jsonl",
        "config": "config_full.yaml",
        "stem": "browsecomp_plus_kimi_k25_all830",
    },
    "qwen35": {
        "model": "qwen/qwen3.5-397b-a17b",
        "traj": "browsecomp_plus_qwen35_397b_all830_full.jsonl",
        "config": "config_full.yaml",
        "stem": "browsecomp_plus_qwen35_397b_all830",
    },
}


def validate(path: Path, expected_model: str) -> None:
    rows = [
        row for row in dedupe_prefer_success(read_jsonl(path))
        if not row.get("failed") and row.get("steps")
    ]
    ids = {str(row["id"]) for row in rows}
    if len(rows) != 830 or len(ids) != 830:
        raise RuntimeError(f"Invalid all-830 file: rows={len(rows)}, ids={len(ids)}")
    if {str(row.get("model")) for row in rows} != {expected_model}:
        raise RuntimeError("All-830 file contains an unexpected model")
    if {str(row.get("construction")) for row in rows} != {
        "ACS_min10doc_all830_v1"
    }:
        raise RuntimeError("All-830 trajectories were not produced from the frozen build")
    if len({str(row.get("build_id")) for row in rows}) != 1:
        raise RuntimeError("All-830 file lacks one audited build ID")


def tee(command: list[str], destination: Path) -> None:
    print("Running:", " ".join(command), flush=True)
    with destination.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            sink.write(line)
            sink.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument(
        "--judge-model", required=True,
        help="OpenRouter/local model ID for the deterministic semantic judge; "
             "the model must pass a one-item judge preflight on the selected endpoint",
    )
    parser.add_argument("--skip-evidence", action="store_true")
    args = parser.parse_args()
    arm = ARMS[args.arm]
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("Set OPENROUTER_API_KEY for the resumable semantic judge")
    trajectory = ROOT / "results" / str(arm["traj"])
    config = ROOT / str(arm["config"])
    validate(trajectory, str(arm["model"]))
    # Keep the judge explicit: endpoint compatibility changes and silently falling
    # back to the runner would change the evaluation protocol.
    common = [
        "--traj", str(trajectory), "--config", str(config),
        "--judge", "--judge-model", args.judge_model,
    ]
    stem = str(arm["stem"])
    tee(
        [sys.executable, "-u", str(PROJECT_ROOT / "eval" / "analyze.py"), "policies", *common],
        ROOT / "results" / f"{stem}_policy_analysis.txt",
    )
    if not args.skip_evidence:
        tee(
            [sys.executable, "-u", str(PROJECT_ROOT / "eval" / "analyze.py"), "evidence", *common],
            ROOT / "results" / f"{stem}_evidence_analysis.txt",
        )


if __name__ == "__main__":
    main()
