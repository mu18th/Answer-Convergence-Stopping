"""Generate RULER qa_2 training contexts from a disjoint HotpotQA train subset.

This runs the pinned NVIDIA RULER qa.py without modifying its data-generation
logic. Use a dedicated RULER checkout because qa.py reads hotpotqa.json from
inside that checkout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent
# The released RULER-HotpotQA helpers, used unchanged.
RULER_HELPER_DIR = HERE.parent / "ruler_hotpotqa"
sys.path.insert(0, str(RULER_HELPER_DIR))
from generate_official_ruler_data import (  # noqa: E402
    OFFICIAL_QA_TEMPLATE,
    RULER_COMMIT,
    install_nontermination_guard,
    validate_official_output,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path | None = None) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_pinned_repo(repo: Path) -> None:
    if not repo.exists():
        run(["git", "clone", "https://github.com/NVIDIA/RULER.git", str(repo)])
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{RULER_COMMIT}^{{commit}}"], cwd=repo,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    ).returncode == 0
    if not present:
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
        raise RuntimeError(f"Expected pinned RULER {RULER_COMMIT}; found {actual}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, required=True,
                        help="Dedicated pinned RULER checkout for this train source")
    parser.add_argument("--hotpotqa-source", type=Path, required=True,
                        help="Selected official HotpotQA train JSON array (120 questions)")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    lengths = sorted(set(args.lengths))
    if len(lengths) != len(args.lengths) or any(length <= 0 for length in lengths):
        raise SystemExit("--lengths must contain unique positive integers")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be positive")

    source = args.hotpotqa_source.resolve()
    if not source.is_file():
        raise SystemExit(f"Missing HotpotQA train source: {source}")
    source_rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(source_rows, list) or len(source_rows) != args.num_samples:
        raise SystemExit(
            f"Expected exactly {args.num_samples} selected train questions in {source}"
        )
    repo = args.repo_dir.resolve()
    ensure_pinned_repo(repo)

    dataset_path = repo / "scripts" / "data" / "synthetic" / "json" / "hotpotqa.json"
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    if dataset_path.exists() and sha256(dataset_path) != source_hash:
        raise SystemExit(
            f"Refusing to replace existing RULER input {dataset_path}; "
            "use a separate --repo-dir for the train source"
        )
    if not dataset_path.exists():
        temporary = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
        temporary.write_bytes(source.read_bytes())
        temporary.replace(dataset_path)

    install_nontermination_guard(repo)
    output_root = args.output_root.resolve()
    manifest = {
        "ruler_commit": RULER_COMMIT,
        "hotpotqa_source_split": "official_train_selected_subset",
        "hotpotqa_source_path": str(source),
        "hotpotqa_source_sha256": source_hash,
        "tokenizer_type": "hf",
        "tokenizer_path": str(Path(args.tokenizer_path).resolve()),
        "model_template_type": "base",
        "task": "qa_2",
        "lengths": lengths,
        "num_samples_per_length": args.num_samples,
        "tokens_to_generate": 32,
        "random_seed": args.seed,
    }
    manifest_path = output_root / "generation_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise SystemExit(
                f"{manifest_path} describes a different build; use a new output root"
            )
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qa_script = repo / "scripts" / "data" / "synthetic" / "qa.py"
    for length in lengths:
        save_dir = output_root / str(length) / "data"
        output = save_dir / "qa_2" / "validation.jsonl"
        if output.exists():
            raise SystemExit(f"Refusing to overwrite existing generated file: {output}")
        run([
            sys.executable, str(qa_script),
            "--save_dir", str(save_dir),
            "--save_name", "qa_2",
            "--subset", "validation",
            "--tokenizer_path", str(Path(args.tokenizer_path).resolve()),
            "--tokenizer_type", "hf",
            "--max_seq_length", str(length),
            "--tokens_to_generate", "32",
            "--num_samples", str(args.num_samples),
            "--random_seed", str(args.seed),
            "--dataset", "hotpotqa",
            "--pre_samples", "0",
            "--template", OFFICIAL_QA_TEMPLATE,
        ], cwd=repo / "scripts" / "data" / "synthetic")
        validate_official_output(output, args.num_samples)
    print(
        f"Saved {args.num_samples} RULER-HotpotQA train rows at each of "
        f"{len(lengths)} lengths ({args.num_samples * len(lengths)} total trajectories)."
    )
    print(f"Generation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
