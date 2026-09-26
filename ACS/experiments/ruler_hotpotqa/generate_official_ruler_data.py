"""Fetch pinned NVIDIA RULER source and generate qa_2 with a chosen tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


RULER_COMMIT = "38da79d79519ef87aa46ae804f838e1eab7f86d7"
HOTPOT_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
    "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_dev_distractor_v1.json"
)
HOTPOT_SHA256 = "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8"
QA_HEADER = (
    "Answer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\nThe following are given documents.\n\n"
)
QA_MARKER = (
    "\n\nAnswer the question based on the given documents. Only give me the answer "
    "and do not output any other words.\n\nQuestion: "
)
OFFICIAL_QA_TEMPLATE = QA_HEADER + "{context}" + QA_MARKER + "{query} Answer:"
OFFICIAL_RETRY_BLOCK = """            except:
                if used_docs > incremental:
                    used_docs -= incremental
"""
GUARDED_RETRY_BLOCK = """            except Exception as error:
                if used_docs > incremental:
                    used_docs -= incremental
                else:
                    raise RuntimeError(
                        f\"qa_2 sample {index} cannot fit the minimum \"
                        f\"{incremental} documents in {length} tokens\"
                    ) from error
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def atomic_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def install_nontermination_guard(repo: Path) -> Path:
    """Guard only RULER's impossible minimum-size retry; preserve valid output.

    Official qa.py correctly fills short HotpotQA rows with distractors.  Its
    retry loop, however, never terminates if even the minimum document count
    cannot fit.  This exact-source patch changes only that terminal failure:
    every successful generation path and every sampled distractor is untouched.
    """
    qa_file = repo / "scripts" / "data" / "synthetic" / "qa.py"
    source = qa_file.read_text(encoding="utf-8")
    if GUARDED_RETRY_BLOCK in source:
        return qa_file
    occurrences = source.count(OFFICIAL_RETRY_BLOCK)
    if occurrences != 1:
        raise RuntimeError(
            f"Expected exactly one pinned RULER retry block in {qa_file}; "
            f"found {occurrences}. Refusing an unverified source modification."
        )
    qa_file.write_text(
        source.replace(OFFICIAL_RETRY_BLOCK, GUARDED_RETRY_BLOCK),
        encoding="utf-8",
    )
    return qa_file


def validate_official_output(path: Path, expected_rows: int) -> None:
    """Reject truncated templates before they can enter an experiment."""
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(f"{path}:{line_number}: invalid JSON") from error
    if len(rows) != expected_rows:
        raise RuntimeError(f"{path}: expected {expected_rows} rows, found {len(rows)}")
    for position, row in enumerate(rows):
        text = str(row.get("input", ""))
        if not text.startswith(QA_HEADER) or QA_MARKER not in text:
            raise RuntimeError(
                f"{path}: row {position} has a truncated/non-base qa template"
            )
        if str(row.get("answer_prefix", "")) != " Answer:":
            raise RuntimeError(
                f"{path}: row {position} has unexpected answer_prefix "
                f"{row.get('answer_prefix')!r}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--tokenizer-type", choices=["hf", "nemo", "openai"], required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=500)
    args = parser.parse_args()

    if any(length <= 0 for length in args.lengths):
        raise SystemExit("Every requested context length must be positive")

    repo = args.repo_dir.resolve()
    if not repo.exists():
        run(["git", "clone", "https://github.com/NVIDIA/RULER.git", str(repo)])
    has_pinned_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{RULER_COMMIT}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    if not has_pinned_commit:
        run(["git", "fetch", "origin", RULER_COMMIT], cwd=repo)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if actual != RULER_COMMIT:
        run(["git", "checkout", "--detach", RULER_COMMIT], cwd=repo)
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    if actual != RULER_COMMIT:
        raise RuntimeError(f"Expected RULER {RULER_COMMIT}, got {actual}")

    hotpot = repo / "scripts" / "data" / "synthetic" / "json" / "hotpotqa.json"
    hotpot.parent.mkdir(parents=True, exist_ok=True)
    if not hotpot.exists() or sha256(hotpot) != HOTPOT_SHA256:
        print(f"Downloading pinned HotpotQA source to {hotpot}", flush=True)
        urllib.request.urlretrieve(HOTPOT_URL, hotpot)
    if sha256(hotpot) != HOTPOT_SHA256:
        raise RuntimeError("HotpotQA SHA-256 mismatch; refusing to generate data")

    guarded_source = install_nontermination_guard(repo)
    print(
        "Installed a failure-only nontermination guard in "
        f"{guarded_source}; valid official generation is unchanged.",
        flush=True,
    )

    generation_manifest = {
        "ruler_commit": RULER_COMMIT,
        "hotpot_sha256": HOTPOT_SHA256,
        "tokenizer_type": args.tokenizer_type,
        "tokenizer_path": args.tokenizer_path,
        "model_template_type": "base",
        "task": "qa_2",
        "lengths": sorted(set(args.lengths)),
        "num_samples_per_length": args.num_samples,
        "random_seed": 42,
    }
    manifest_path = args.output_root.resolve() / "generation_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != generation_manifest:
            raise RuntimeError(
                f"{manifest_path} describes different generation settings. "
                "Use a new --output-root; never mix tokenizer/build variants."
            )
    else:
        atomic_manifest(manifest_path, generation_manifest)

    qa_script = repo / "scripts" / "data" / "synthetic" / "qa.py"
    for length in sorted(set(args.lengths)):
        save_dir = args.output_root.resolve() / str(length) / "data"
        run([
            # NVIDIA prepare.py constructs this same invocation as a multiline
            # shell string.  On Windows that truncates --template at its first
            # newline.  Passing an argv list preserves the official template
            # byte-for-byte without changing qa.py or its generation logic.
            sys.executable, str(qa_script),
            "--save_dir", str(save_dir),
            "--save_name", "qa_2",
            "--subset", "validation",
            "--tokenizer_path", args.tokenizer_path,
            "--tokenizer_type", args.tokenizer_type,
            "--max_seq_length", str(length),
            "--tokens_to_generate", "32",
            "--num_samples", str(args.num_samples),
            "--random_seed", "42",
            "--dataset", "hotpotqa",
            "--pre_samples", "0",
            "--template", OFFICIAL_QA_TEMPLATE,
        ], cwd=repo / "scripts" / "data" / "synthetic")
        output = save_dir / "qa_2" / "validation.jsonl"
        validate_official_output(output, args.num_samples)
    print("Official RULER qa_2 generation complete.")
    print(f"Generation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
