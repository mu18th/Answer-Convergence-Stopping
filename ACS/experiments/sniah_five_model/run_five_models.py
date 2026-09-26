"""Run the same 250 S-NIAH questions to full execution on five OpenRouter models."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import dedupe_prefer_success, read_jsonl

ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "config_sniah_five_model.yaml"
MODELS = {
    "qwen25_7b": "qwen/qwen-2.5-7b-instruct",
    "qwen3_14b": "qwen/qwen3-14b",
    "qwen3_32b": "qwen/qwen3-32b",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
}


def completed(path: Path, wanted: set[str]) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["id"])
        for row in dedupe_prefer_success(read_jsonl(path))
        if str(row.get("id")) in wanted and row.get("steps") and not row.get("failed")
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all 250")
    parser.add_argument("--max-passes", type=int, default=15)
    parser.add_argument("--retry-delay", type=float, default=30.0)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("Set OPENROUTER_API_KEY first; the key is never stored in files.")

    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    data = ROOT / "data" / "sniah.jsonl"
    if args.prepare or not data.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "prepare_data.py"), "sniah",
             "--config", str(BASE_CONFIG)], cwd=ROOT, check=True,
        )
    rows = read_jsonl(data)
    if len(rows) != 250:
        raise SystemExit(f"Expected the frozen 250-row S-NIAH set, found {len(rows)}")
    selected = rows[:args.limit] if args.limit else rows
    wanted = {str(row["id"]) for row in selected}

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        model = MODELS[name]
        cfg = {**base, "server": {**base["server"], "model": model}}
        results = ROOT / "results"
        results.mkdir(parents=True, exist_ok=True)
        cfg_path = results / f"config_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        out = results / f"sniah_{name}_n250_full.jsonl"

        env = os.environ.copy()
        env["MODEL"] = model
        env.pop("OPENROUTER_PROVIDER", None)
        # Refuse to spend on a bulk run until this exact model passes every API
        # call shape used by S-NIAH, including generated-token logprobs.
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "preflight_models.py"),
             "--model", name], cwd=ROOT, env=env, check=True,
        )
        command = [
            sys.executable, "-u", str(PROJECT_ROOT / "run_fold.py"),
            "--data", str(data), "--out", str(out), "--config", str(cfg_path),
            "--workers", str(args.workers), "--rule", "none",
            "--record-verbalized", "--record-gate",
        ]
        if args.limit:
            command += ["--limit", str(args.limit)]

        print(f"\n{'=' * 78}\n{name}: {model}\n{'=' * 78}", flush=True)
        for pass_no in range(1, args.max_passes + 1):
            before = completed(out, wanted)
            print(f"pass {pass_no}: {len(before)}/{len(wanted)} completed", flush=True)
            subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=True)
            after = completed(out, wanted)
            if after == wanted:
                break
            if pass_no < args.max_passes:
                time.sleep(args.retry_delay)
        else:
            raise SystemExit(f"{name}: unresolved samples remain; rerun to resume")

        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "eval" / "add_oracle_labels.py"), str(out), str(data)],
            cwd=PROJECT_ROOT, check=True,
        )
        print(f"{name}: {len(wanted)}/{len(wanted)} complete -> {out}", flush=True)


if __name__ == "__main__":
    main()
