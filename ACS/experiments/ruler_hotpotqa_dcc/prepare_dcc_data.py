"""Prepare everything the Table 4 DCC comparison reads.

1. The official DCC code at the pinned commit (external/dcc_official), used
   read-only; every adaptation is in dcc_overlay.py and run_dcc.py.
2. The paper's 500 RULER-HotpotQA evaluation documents, rebuilt with the
   released helpers in ../ruler_hotpotqa (official RULER qa_2 pool of 500
   questions per length, Qwen3-14B tokenizer, seed 42, cohorts at offsets 0
   and 50), then converted for DCC without changing any question, context or
   answer. The two cohort build ids printed here must equal the build_id
   values of the paper's ACS trajectories.
3. The separate DCC training cohort: 120 official HotpotQA train questions x
   5 RULER qa_2 lengths, source-disjoint from the evaluation questions and
   labelled with the exact annotated supporting-fact spans.

Every step is skipped when its output already exists, so rerunning resumes.
Needs network access once and Qwen/Qwen3-14B's tokenizer in the local
Hugging Face cache.

Usage (from this folder)
    python prepare_dcc_data.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RULER_HELPERS = ROOT.parent / "ruler_hotpotqa"
if str(RULER_HELPERS) not in sys.path:
    sys.path.append(str(RULER_HELPERS))

# The pinned HotpotQA dev file RULER's qa_2 is generated from.
from generate_official_ruler_data import HOTPOT_SHA256, HOTPOT_URL  # noqa: E402

MODEL_ID = "Qwen/Qwen3-14B"
LENGTHS = ("8192", "16384", "32768", "65536", "131072")
SEED = "42"

DCC_REPO_URL = "https://github.com/ruoyuxie/when-to-stop"
DCC_COMMIT = "337c20b6561e13fc1af34d706158cd74783765da"
DCC_DIR = ROOT / "external" / "dcc_official"

# Official HotpotQA train v1.1, accepted only with the official SHA-256. The
# original host is often unreachable; this mirror serves the same bytes.
HOTPOT_TRAIN_URL = (
    "https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/"
    "7e54db4656209750ff487f6fdf8e39a66dba136b/hotpot_train_v1.json"
)
HOTPOT_TRAIN_SHA256 = "26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316"

DATA = ROOT / "data"
PAPER_EVAL = DATA / "paper_eval"
EVAL_ACS_JSONL = PAPER_EVAL / "ruler_hotpotqa_paper500.jsonl"
EVAL_DCC_JSON = DATA / "dcc" / "paper_dev_500" / "multihop_qa.json"
EVAL_SOURCE = DATA / "source" / "hotpot_dev_distractor_v1.json"
TRAIN_SOURCE = DATA / "source" / "hotpot_train_v1.1.json"
TRAIN_QUESTIONS = DATA / "source" / "ruler_hotpotqa_train_120.json"
TRAIN_ACS_JSONL = DATA / "ruler_hotpotqa" / "ruler_hotpotqa_train_120x5.jsonl"
TRAIN_DCC_JSON = DATA / "dcc" / "ruler_train_120x5" / "multihop_qa.json"


def run(*command: object) -> None:
    print("Running:", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_checked(url: str, expected_sha256: str, target: Path, label: str) -> None:
    """Download target unless it already holds the official bytes; accept only those bytes."""
    if not target.is_file() or sha256(target) != expected_sha256:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".part")
        print(f"Downloading {label} -> {target}", flush=True)
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)
    if sha256(target) != expected_sha256:
        raise SystemExit(f"{target} is not the official {label} file")


def tokenizer_path() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, local_files_only=True)


def fetch_dcc() -> None:
    if not (DCC_DIR / ".git").is_dir():
        DCC_DIR.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--quiet", DCC_REPO_URL, DCC_DIR)
    run("git", "-C", DCC_DIR, "checkout", "--quiet", "--detach", DCC_COMMIT)
    status = subprocess.run(["git", "-C", str(DCC_DIR), "status", "--porcelain"],
                            check=True, capture_output=True, text=True).stdout
    if status.strip():
        raise SystemExit(f"{DCC_DIR} has local changes; the DCC code must stay unmodified")
    print(f"DCC official code at {DCC_COMMIT}: {DCC_DIR}", flush=True)


def build_paper_eval(tokenizer: str) -> None:
    raw = DATA / "ruler_raw_paper"
    if not (raw / LENGTHS[-1] / "data" / "qa_2" / "validation.jsonl").is_file():
        run(sys.executable, "-u", RULER_HELPERS / "generate_official_ruler_data.py",
            "--repo-dir", ROOT / "external" / "RULER_paper", "--output-root", raw,
            "--tokenizer-path", tokenizer, "--tokenizer-type", "hf",
            "--lengths", *LENGTHS, "--num-samples", "500")
    cohorts = []
    for name, offset in (("c1", "0"), ("c2", "50")):
        cohort = PAPER_EVAL / f"ruler_hotpotqa_{name}.jsonl"
        if not cohort.is_file():
            run(sys.executable, "-u", RULER_HELPERS / "prepare_ruler_hotpotqa.py",
                "--ruler-root", raw, "--lengths", *LENGTHS, "--samples-per-length", "50",
                "--seed", SEED, "--selection-offset", offset, "--out", cohort)
        cohorts.append(cohort)
    if not (EVAL_ACS_JSONL.is_file() and EVAL_DCC_JSON.is_file()):
        # The HotpotQA file the RULER generator downloaded; when the raw RULER data
        # came from elsewhere, the same file (same SHA-256) is fetched here instead.
        hotpot_source = (ROOT / "external" / "RULER_paper" / "scripts" / "data" / "synthetic"
                         / "json" / "hotpotqa.json")
        if not hotpot_source.is_file() or sha256(hotpot_source) != HOTPOT_SHA256:
            hotpot_source = EVAL_SOURCE
            download_checked(HOTPOT_URL, HOTPOT_SHA256, hotpot_source,
                             "HotpotQA dev distractor v1")
        run(sys.executable, "-u", ROOT / "prepare_paper_eval_for_dcc.py",
            "--paper-eval", *cohorts,
            "--hotpot-source", hotpot_source,
            "--out-acs", EVAL_ACS_JSONL, "--out-dcc", EVAL_DCC_JSON)


def build_training_cohort(tokenizer: str) -> None:
    if not TRAIN_ACS_JSONL.is_file():
        download_checked(HOTPOT_TRAIN_URL, HOTPOT_TRAIN_SHA256, TRAIN_SOURCE, "HotpotQA train v1.1")
        if not TRAIN_QUESTIONS.is_file():
            run(sys.executable, "-u", ROOT / "select_ruler_train_questions.py",
                "--source", TRAIN_SOURCE, "--questions", "120", "--seed", SEED,
                "--out", TRAIN_QUESTIONS)
        raw = DATA / "ruler_raw_train"
        if not (raw / LENGTHS[-1] / "data" / "qa_2" / "validation.jsonl").is_file():
            run(sys.executable, "-u", ROOT / "generate_ruler_hotpotqa_train.py",
                "--repo-dir", ROOT / "external" / "RULER_train",
                "--hotpotqa-source", TRAIN_QUESTIONS, "--output-root", raw,
                "--tokenizer-path", tokenizer, "--lengths", *LENGTHS,
                "--num-samples", "120", "--seed", SEED)
        run(sys.executable, "-u", ROOT / "prepare_ruler_cohort.py",
            "--ruler-root", raw, "--source-json", TRAIN_QUESTIONS, "--cohort", "train",
            "--lengths", *LENGTHS, "--samples-per-length", "120", "--seed", SEED,
            "--out", TRAIN_ACS_JSONL)
    run(sys.executable, "-u", ROOT / "validate_ruler_cohort.py",
        "--data", TRAIN_ACS_JSONL, "--cohort", "train", "--per-length", "120")
    if not TRAIN_DCC_JSON.is_file():
        run(sys.executable, "-u", ROOT / "prepare_dcc_hotpotqa.py",
            "--source", TRAIN_ACS_JSONL, "--format", "acs_jsonl", "--supporting-facts",
            "--out", TRAIN_DCC_JSON)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    fetch_dcc()
    tokenizer = tokenizer_path()
    build_paper_eval(tokenizer)
    build_training_cohort(tokenizer)
    print("\nDCC inputs ready:")
    for path in (DCC_DIR, EVAL_ACS_JSONL, EVAL_DCC_JSON, TRAIN_DCC_JSON):
        print(f"  {path}")
    builds = sorted({json.loads(line)["build_id"]
                     for name in ("c1", "c2")
                     for line in (PAPER_EVAL / f"ruler_hotpotqa_{name}.jsonl").open(encoding="utf-8")
                     if line.strip()})
    print(f"Paper cohort build ids: {builds}; they must equal the build_id values "
          "of the paper's ACS trajectories.")


if __name__ == "__main__":
    main()
