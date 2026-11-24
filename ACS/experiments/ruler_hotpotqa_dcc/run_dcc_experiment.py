"""Run the two DCC arms of paper Table 4 on the paper's 500 RULER-HotpotQA documents.

train_8k_32k        probe trained on the 8K-32K training documents
                    (120 questions x 3 lengths = 360); one run_dcc.py process
                    trains, evaluates full-context and chunked DCC, then both
                    are scored.
train_all_lengths   probe trained on all five lengths (600 documents). The
                    64K/128K documents need the low-memory feature path:
                    extract features, train once from the cache, evaluate,
                    merge, score.

Every stage resumes from its outputs. run_dcc.py picks the device itself
(CUDA, MPS or CPU, with device_map="auto"), so nothing here assumes a device
type or count. Stages may be split across processes with --shard-index and
--shard-count (extract and evaluate), then finished with --stage score.

Usage (from this folder, after prepare_dcc_data.py)
    python run_dcc_experiment.py                       # both arms, every stage
    python run_dcc_experiment.py --arms train_8k_32k   # the main Table 4 row
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "dcc"

from prepare_dcc_data import (  # noqa: E402
    DCC_DIR, EVAL_ACS_JSONL, EVAL_DCC_JSON, MODEL_ID, TRAIN_DCC_JSON,
)

# The Table 4 DCC settings: DCC's defaults (10 percentage chunks, 5 probe heads,
# threshold 0.95, greedy 32-token answers, seed 42) plus the Qwen3 adaptations
# documented in dcc_overlay.py and run_dcc.py.
DCC_ARGS = (
    "--chunk-count", "10",
    "--threshold", "0.95",
    "--num-heads", "5",
    "--max-new-tokens", "32",
    "--context-limit-tokens", "131072",
    "--seed", "42",
    "--attn-implementation", "sdpa",
    # Qwen3 answers DCC's raw completion prompt by reasoning aloud and runs out
    # of its 32-token budget; the chat template with thinking disabled is the
    # setting ACS uses. DCC's prompt text is unchanged.
    "--chat-template",
    # Qwen's documented YaRN setting for 131,072 tokens, applied in memory.
    "--yarn-128k",
)


@dataclass(frozen=True)
class Arm:
    name: str
    train_lengths: str
    low_memory: bool

    @property
    def train_dir(self) -> Path:
        return RESULTS / self.name

    @property
    def evaluation_dir(self) -> Path:
        # The 8K-32K arm trains and evaluates in one process and folder.
        return self.train_dir if not self.low_memory else RESULTS / f"{self.name}_eval"

    @property
    def feature_cache(self) -> Path:
        return ROOT / "cache" / f"dcc_{self.name}"


ARMS = {
    "train_8k_32k": Arm("train_8k_32k", "8192,16384,32768", low_memory=False),
    "train_all_lengths": Arm("train_all_lengths", "8192,16384,32768,65536,131072",
                             low_memory=True),
}
STAGES = ("all", "extract", "train", "evaluate", "score")


def run_dcc(*arguments: object, model_path: str) -> None:
    command = [sys.executable, "-B", "-u", str(ROOT / "run_dcc.py"),
               "--dcc-repo", str(DCC_DIR), "--model-path", model_path,
               "--eval-dcc-json", str(EVAL_DCC_JSON),
               "--eval-acs-jsonl", str(EVAL_ACS_JSONL),
               *DCC_ARGS, *(str(part) for part in arguments)]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def score(run_dir: Path) -> None:
    for mode in ("full_context", "chunked"):
        predictions = run_dir / f"dcc_{mode}.jsonl"
        if not predictions.is_file():
            raise SystemExit(f"Missing {predictions}; run the evaluation stage before --stage score")
        if '"status": "ok"' not in predictions.read_text(encoding="utf-8"):
            print(f"No document fit the context window for DCC {mode}; scoring skipped.")
            continue
        subprocess.run([sys.executable, "-u", str(ROOT / "score_answers.py"),
                        "--data", str(EVAL_ACS_JSONL), "--predictions", str(predictions),
                        "--out", str(run_dir / f"dcc_{mode}_scores.json"), "--overwrite"],
                       cwd=ROOT, check=True)


def shard_dir(arm: Arm, kind: str, index: int, count: int) -> Path:
    return RESULTS / f"{arm.name}_{kind}_{index}_of_{count}"


def run_main_arm(arm: Arm, stage: str, model_path: str) -> None:
    if stage in ("all", "train", "evaluate") and \
            not (arm.train_dir / "dcc_chunked.jsonl").is_file():
        # Training and evaluation share one process (as in DCC), so an unfinished
        # run cannot resume mid-way: it restarts in the same folder.
        if arm.train_dir.is_dir() and any(arm.train_dir.iterdir()):
            print(f"{arm.train_dir} holds an unfinished run; restarting it from the beginning.",
                  flush=True)
        run_dcc("--train-dcc-json", TRAIN_DCC_JSON, "--train-lengths", arm.train_lengths,
                "--out-dir", arm.train_dir, "--overwrite", model_path=model_path)
    if stage in ("all", "score"):
        score(arm.train_dir)


def run_low_memory_arm(arm: Arm, stage: str, index: int, count: int, model_path: str) -> None:
    train = ("--train-dcc-json", TRAIN_DCC_JSON, "--train-lengths", arm.train_lengths,
             "--low-memory-training", "--feature-cache", arm.feature_cache)
    probe = arm.train_dir / "probe" / "probe_state.pt"
    if stage in ("all", "extract") and not probe.is_file():
        marker = arm.feature_cache / f"extract_shard_{index}_of_{count}.json"
        if not marker.is_file():
            run_dcc(*train, "--extract-only", "--shard-index", index, "--shard-count", count,
                    "--out-dir", shard_dir(arm, "extract", index, count), "--overwrite",
                    model_path=model_path)
    if stage in ("all", "train") and not probe.is_file():
        run_dcc(*train, "--train-only", "--out-dir", arm.train_dir, "--overwrite",
                model_path=model_path)
    if stage in ("all", "evaluate"):
        out = shard_dir(arm, "eval", index, count)
        if not (out / "dcc_chunked.jsonl").is_file():
            run_dcc("--probe-dir", arm.train_dir / "probe", "--shard-index", index,
                    "--shard-count", count, "--out-dir", out, "--overwrite",
                    model_path=model_path)
    if stage in ("all", "score"):
        shards = [shard_dir(arm, "eval", i, count) for i in range(count)]
        missing = [str(path) for path in shards if not (path / "dcc_chunked.jsonl").is_file()]
        if missing:
            raise SystemExit(f"{arm.name}: evaluation shards missing: {missing}")
        subprocess.run([sys.executable, "-u", str(ROOT / "merge_dcc_shards.py"),
                        "--eval-acs-jsonl", str(EVAL_ACS_JSONL),
                        "--out-dir", str(arm.evaluation_dir), *map(str, shards)],
                       cwd=ROOT, check=True)
        score(arm.evaluation_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="split extract/evaluate of the all-lengths arm across processes")
    parser.add_argument("--model-path", default=MODEL_ID,
                        help="local checkpoint folder or a hub id in the local cache")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1 and args.stage == "all":
        raise SystemExit("With --shard-count > 1 run --stage extract, train, evaluate "
                         "and score separately")
    for path in (DCC_DIR / "src" / "run.py", EVAL_ACS_JSONL, EVAL_DCC_JSON, TRAIN_DCC_JSON):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; run prepare_dcc_data.py first")

    for name in args.arms:
        arm = ARMS[name]
        print(f"\n===== {name} ({args.stage}) =====", flush=True)
        if arm.low_memory:
            run_low_memory_arm(arm, args.stage, args.shard_index, args.shard_count,
                               args.model_path)
        else:
            if args.shard_count > 1:
                raise SystemExit(f"{name} trains and evaluates in one process; drop --shard-count")
            run_main_arm(arm, args.stage, args.model_path)
        print(f"{name}: stage {args.stage} complete", flush=True)


if __name__ == "__main__":
    main()
