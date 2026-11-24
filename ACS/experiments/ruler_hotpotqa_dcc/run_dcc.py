"""Train official DCC on RULER-HotpotQA train rows and evaluate on dev rows.

The DCC repository is imported unmodified. This experiment driver supplies its
configuration in memory, then uses the official probe and early-stopping
classes. Output cost is counted from actual model forward input IDs, including
chunk/probe and generation passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

RULER_LENGTHS = (8192, 16384, 32768, 65536, 131072)
RULER_EVAL_QUESTIONS = 100
RULER_TRAIN_QUESTIONS = 120

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"{path}:{number}: invalid JSON") from error
    ids = [str(row.get("id", "")) for row in rows]
    if not rows or any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path} must contain unique, non-empty IDs")
    return rows


def read_dcc_json(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Expected non-empty DCC JSON array: {path}")
    ids = [str(row.get("id", "")) for row in rows]
    if any(not row_id for row_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path} must contain unique, non-empty IDs")
    for row in rows:
        if (not isinstance(row.get("input"), str) or not row["input"]
                or not isinstance(row.get("context"), str) or not row["context"]
                or not isinstance(row.get("answers"), list)
                or not isinstance(row.get("gold_information", ""), str)
                or row.get("task") != "multihop_qa"):
            raise ValueError(f"Malformed DCC example: {row.get('id')}")
        gold_information = row.get("gold_information", "")
        if gold_information and gold_information not in row["context"]:
            raise ValueError(f"DCC gold span is not an exact context substring: {row.get('id')}")
    return rows


def resolve_model_dir(model_path_arg: str) -> Path:
    """A local checkpoint directory, or the cached snapshot folder of a hub id. Nothing is downloaded.

    Newer huggingface_hub versions make snapshot_download(local_files_only=True) fail when ANY file
    of the repository is absent from the cache, including README.md, LICENSE and .gitattributes,
    which a model server never fetches. The model itself needs only its config, weights and
    tokenizer, so if that call fails the snapshot folder is located through the cached config.json
    and checked for those files.
    """
    import os
    if Path(model_path_arg).is_dir():
        return Path(model_path_arg).resolve()
    first_error = None
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(model_path_arg, local_files_only=True)).resolve()
    except Exception as error:  # incomplete snapshot, or not cached at all
        first_error = error
    try:
        from huggingface_hub import try_to_load_from_cache
        cached_config = try_to_load_from_cache(model_path_arg, "config.json")
    except Exception as error:
        cached_config, first_error = None, error
    if not isinstance(cached_config, str) or not Path(cached_config).is_file():
        raise SystemExit(
            f"{model_path_arg!r} is neither a local directory nor a model in the HF cache "
            f"(HF_HOME): {first_error}"
        )
    # Not .resolve(): cached files are links into blobs/, the snapshot folder is their parent.
    snapshot = Path(os.path.abspath(cached_config)).parent
    weights = sorted(snapshot.glob("*.safetensors")) or sorted(snapshot.glob("*.bin"))
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer_config.json")
                       if (snapshot / name).is_file()]
    if not weights or not tokenizer_files:
        raise SystemExit(
            f"The cached snapshot of {model_path_arg!r} at {snapshot} has no model weights or no "
            f"tokenizer files. Original error: {first_error}"
        )
    print(f"Using the cached snapshot without its README/LICENSE files: {snapshot} "
          f"({len(weights)} weight files)")
    return snapshot


def device_info(torch) -> dict:
    """The accelerator this process uses: type, count and names. CPU counts as one device."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        return {"type": "cuda", "count": count,
                "names": sorted({torch.cuda.get_device_name(i) for i in range(count)})}
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return {"type": "mps", "count": 1, "names": ["mps"]}
    return {"type": "cpu", "count": 1, "names": ["cpu"]}


def synchronize(torch) -> None:
    """Wait for queued kernels so latency covers the work, on any device."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.synchronize()


def empty_cache(torch) -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def split_ruler_question_groups(
    examples: list[dict], ratio: float, lengths: tuple[int, ...] = RULER_LENGTHS
) -> tuple[list[dict], list[dict]]:
    """Split complete source-question groups, never individual length variants."""
    groups: dict[str, list[dict]] = {}
    for example in examples:
        source_id = str(example.get("source_id", ""))
        if not source_id:
            raise ValueError("DCC training row lacks RULER source_id")
        groups.setdefault(source_id, []).append(example)
    for source_id, variants in groups.items():
        if ({int(row.get("context_length_tokens", -1)) for row in variants} != set(lengths)
                or len(variants) != len(lengths)):
            raise ValueError(f"Incomplete RULER length group before DCC split: {source_id}")
    group_ids = list(groups)
    random.shuffle(group_ids)
    split_point = int(len(group_ids) * ratio)
    left_ids, right_ids = set(group_ids[:split_point]), set(group_ids[split_point:])
    left = [row for row in examples if str(row["source_id"]) in left_ids]
    right = [row for row in examples if str(row["source_id"]) in right_ids]
    return left, right


def validate_ruler_rows(rows: list[dict], *, split: str, questions: int) -> None:
    expected_prefix = "eval" if split == "dev" else "train"
    by_length: dict[int, list[dict]] = {length: [] for length in RULER_LENGTHS}
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        hotpot_id = str(row.get("source_hotpotqa_id", ""))
        if (row.get("benchmark") != "ruler_hotpotqa"
                or row.get("ruler_task") != "qa_2"
                or row.get("source_split") != split
                or not source_id.startswith(f"hotpotqa_{expected_prefix}_")
                or not hotpot_id
                or source_id != f"hotpotqa_{expected_prefix}_{hotpot_id}"):
            raise SystemExit(
                f"Expected only RULER-HotpotQA {split} examples; invalid row {row.get('id')!r}"
            )
        length = row.get("context_length_tokens")
        if length not in by_length:
            raise SystemExit(f"Unexpected RULER context length {length!r}")
        by_length[length].append(row)
        by_source.setdefault(source_id, []).append(row)

    if len(rows) != questions * len(RULER_LENGTHS):
        raise SystemExit(
            f"RULER {split} cohort must contain {questions * len(RULER_LENGTHS)} rows; found {len(rows)}"
        )
    if len(by_source) != questions:
        raise SystemExit(f"RULER {split} cohort must contain {questions} unique questions")
    for length, bucket in by_length.items():
        if len(bucket) != questions or len({row["source_id"] for row in bucket}) != questions:
            raise SystemExit(f"RULER {split} length {length} must have {questions} unique questions")
    for source_id, group in by_source.items():
        if {row["context_length_tokens"] for row in group} != set(RULER_LENGTHS) or len(group) != len(RULER_LENGTHS):
            raise SystemExit(f"{source_id}: expected one row at each of the five RULER lengths")
        question_values = {
            str(row.get("input", row.get("question", ""))) for row in group
        }
        answer_values = {
            json.dumps(row.get("answers", row.get("answer", [])), sort_keys=True)
            for row in group
        }
        if len(question_values) != 1 or len(answer_values) != 1:
            raise SystemExit(f"{source_id}: RULER question/gold answer changed across lengths")
        if split == "train" and any(
            row.get("gold_information_type") != "supporting_fact_span"
            or not row.get("gold_information") for row in group
        ):
            raise SystemExit(f"{source_id}: DCC training needs RULER supporting-fact span labels")
    build_ids = {str(row.get("build_id", "")) for row in rows}
    if len(build_ids) != 1 or not next(iter(build_ids)):
        raise SystemExit(f"RULER {split} cohort must have one non-empty build_id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dcc-repo", type=Path, required=True,
                        help="Official DCC checkout at the pinned commit (prepare_dcc_data.py)")
    parser.add_argument("--model-path", required=True,
                        help="Local checkpoint directory, or a hub id already in the HF cache "
                             "(e.g. Qwen/Qwen3-14B); nothing is downloaded")
    parser.add_argument("--yarn-128k", action="store_true",
                        help="Apply Qwen's documented YaRN setting for 131072 tokens in memory "
                             "(rope_scaling yarn, factor 4, original 32768). No file is edited.")
    parser.add_argument("--eval-dcc-json", type=Path, required=True,
                        help="DCC-schema JSON array for evaluation")
    parser.add_argument("--eval-acs-jsonl", type=Path, required=True,
                        help="Paired prepared ACS data JSONL for scoring")
    parser.add_argument("--expected-eval-rows", type=int, default=500,
                        help="This experiment evaluates exactly the paired 100-question x five-length RULER cohort")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-dcc-json", type=Path,
                        help="DCC-schema training JSON array; omit only with --probe-dir")
    parser.add_argument("--probe-dir", type=Path,
                        help="Reuse a DCC-trained probe directory for transfer evaluation")
    parser.add_argument("--chunk-count", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--num-heads", type=int, default=5)
    parser.add_argument("--samples-per-task", type=int, default=None,
                        help="DCC training only: 120 disjoint RULER train questions x the number of "
                             "--train-lengths (360 for 8K-32K, 600 for all five). Default: derived.")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--context-limit-tokens", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="sdpa",
                        help="sdpa (default): DCC hooks the o_proj output, which does not need "
                             "materialised attention weights; eager at 64K+ tokens cannot fit")
    parser.add_argument("--train-lengths", default="8192,16384,32768",
                        help="RULER lengths used for probe training. Upstream DCC stores "
                             "every token's activations on the CPU before pooling, so "
                             "64K/128K training rows do not fit in host memory. DCC's own "
                             "experiments stop at ~40K tokens (paper Table 6). Evaluation "
                             "still covers all five lengths.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Inference only (--probe-dir): evaluate rows whose position in the "
                             "eval file satisfies index %% shard_count == shard_index. Run one "
                             "process per shard with a separate --out-dir, then merge_dcc_shards.py.")
    parser.add_argument("--chat-template", action="store_true",
                        help="Wrap DCC's prompt in the model's chat template with thinking "
                             "disabled (needed for Qwen3, which otherwise reasons aloud instead "
                             "of answering). Applied to probe training and inference alike.")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"),
                        default="auto",
                        help="auto: float16 (the upstream DCC default) on an accelerator, "
                             "float32 on CPU")
    parser.add_argument("--low-memory-training", action="store_true",
                        help="Probe training only. Each layer's activations are reduced inside the "
                             "hook to the prefix means at the chunk ends (the values upstream computes "
                             "afterwards from per-token activations kept in host RAM, about 107 GB for "
                             "one 128K document), and the feature forward skips the LM head and the KV "
                             "cache. Needed to train on the 64K and 128K rows. See dcc_overlay.py.")
    parser.add_argument("--feature-cache", type=Path,
                        help="With --low-memory-training: directory holding the reduced activations "
                             "of each training document. Fill it with --extract-only in several processes, "
                             "then train with the same directory.")
    parser.add_argument("--extract-only", action="store_true",
                        help="With --train-dcc-json and --feature-cache: run the feature forward for "
                             "this shard of the training documents (--shard-index/--shard-count), "
                             "fill the cache and exit. No split, no training, no evaluation.")
    parser.add_argument("--train-only", action="store_true",
                        help="Train and save the probe, write the manifests, skip the evaluation. "
                             "Evaluate afterwards with --probe-dir (shardable).")
    parser.add_argument("--drop-nonfitting-train-questions", action="store_true",
                        help="If a training document does not fit --context-limit-tokens, drop its "
                             "source question at every length (groups stay complete) instead of "
                             "exiting. The dropped questions are listed in the manifest.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train_lengths = tuple(sorted(int(x) for x in args.train_lengths.split(",")))
    if any(length not in RULER_LENGTHS for length in train_lengths):
        raise SystemExit(f"--train-lengths must be a subset of {RULER_LENGTHS}")
    if args.samples_per_task is None:
        args.samples_per_task = RULER_TRAIN_QUESTIONS * len(train_lengths)
    if args.feature_cache is not None and not args.low_memory_training:
        raise SystemExit("--feature-cache needs --low-memory-training")
    if args.extract_only and (args.train_dcc_json is None or args.feature_cache is None):
        raise SystemExit("--extract-only needs --train-dcc-json and --feature-cache")
    if (args.extract_only or args.train_only) and args.probe_dir is not None:
        raise SystemExit("--extract-only and --train-only are training modes; do not pass --probe-dir")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1 and args.probe_dir is None and not args.extract_only:
        raise SystemExit("Sharding is for --extract-only or for inference with --probe-dir")

    repo = args.dcc_repo.resolve()
    src = repo / "src"
    if not (src / "run.py").is_file() or not (src / "probe.py").is_file():
        raise SystemExit(f"Not an official DCC source checkout: {repo}")
    model_path = resolve_model_dir(args.model_path)
    print(f"Model files: {model_path}")
    eval_rows = read_dcc_json(args.eval_dcc_json)
    paired = read_jsonl(args.eval_acs_jsonl)
    if len(eval_rows) != args.expected_eval_rows:
        raise SystemExit(
            f"Expected exactly {args.expected_eval_rows} evaluation rows; found {len(eval_rows)}"
        )
    if [str(row["id"]) for row in eval_rows] != [str(row["id"]) for row in paired]:
        raise SystemExit("DCC evaluation rows and paired ACS JSONL must have identical IDs and order")
    if any(str(row.get("input", "")) != str(pair.get("question", ""))
           or str(row.get("context", "")) != str(pair.get("context", ""))
           or row.get("answers") != pair.get("answer", pair.get("answers"))
           for row, pair in zip(eval_rows, paired)):
        raise SystemExit("DCC/ACS question-context mismatch; refusing an unpaired comparison")
    if args.expected_eval_rows != 500:
        raise SystemExit("This experiment's common evaluation cohort is fixed at 500 RULER trajectories")
    validate_ruler_rows(eval_rows, split="dev", questions=RULER_EVAL_QUESTIONS)
    validate_ruler_rows(paired, split="dev", questions=RULER_EVAL_QUESTIONS)
    if any(row.get("source_id") != pair.get("source_id")
           or row.get("build_id") != pair.get("build_id")
           for row, pair in zip(eval_rows, paired)):
        raise SystemExit("DCC and ACS evaluation provenance mismatch")
    if (args.train_dcc_json is None) == (args.probe_dir is None):
        raise SystemExit("Supply exactly one of --train-dcc-json or --probe-dir")
    if (args.chunk_count < 2 or not 0 < args.threshold <= 1
            or args.max_new_tokens < 1 or args.samples_per_task < 1
            or args.context_limit_tokens < 1):
        raise SystemExit("Invalid chunk count, threshold, or generation length")
    expected_train_rows = RULER_TRAIN_QUESTIONS * len(train_lengths)
    if args.train_dcc_json is not None and args.samples_per_task != expected_train_rows:
        raise SystemExit(
            f"--samples-per-task must equal {expected_train_rows} "
            f"({RULER_TRAIN_QUESTIONS} questions x {len(train_lengths)} training lengths)"
        )
    args.out_dir = args.out_dir.resolve()
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output directory is not empty: {args.out_dir}; use a new path or --overwrite")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device_info(torch)
    if args.dtype == "auto":
        args.dtype = "float32" if device["type"] == "cpu" else "float16"
    print(f"Device: {device['type']} x{device['count']} {device['names']}; dtype {args.dtype}")

    sys.path.insert(0, str(src))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import config as dcc_config
    from dataset_manager import DatasetManager
    from early_stopping import EarlyOutputModel
    from probe import SufficiencyProbe
    from utils import prepare_sufficiency_input, get_suffix_prompt, setup_random_seeds
    import activation_manager as dcc_activation_manager
    import dcc_overlay
    # Qwen3 detection, no GQA head duplication, optional in-hook token pooling.
    # Upstream files are untouched; see dcc_overlay.py for each change.
    dcc_overlay.apply(dcc_activation_manager.ActivationManager)
    chat_wrapper = None
    if args.chat_template:
        import utils as dcc_utils
        import early_stopping as dcc_early_stopping
        chat_wrapper = dcc_overlay.apply_chat_template(
            AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False),
            {"utils": dcc_utils, "activation_manager": dcc_activation_manager,
             "early_stopping": dcc_early_stopping},
        )
        dcc_overlay.apply_full_context_decode_fix(EarlyOutputModel, dcc_early_stopping)
        prepare_sufficiency_input = dcc_utils.prepare_sufficiency_input
        get_suffix_prompt = dcc_utils.get_suffix_prompt
        print(f"Chat template wrapper on: prefix={chat_wrapper['prefix']!r} "
              f"postfix={chat_wrapper['postfix']!r}")

    class RulerGroupedSplitDatasetManager(DatasetManager):
        """Split by source question so the length variants cannot cross splits."""

        def __init__(self, *manager_args, **manager_kwargs):
            super().__init__(*manager_args, **manager_kwargs)
            self._paper_split_call = 0

        def _split_examples(self, examples, split_ratio):
            phase = self._paper_split_call % 2
            expected_upstream_ratio = (0.9, 0.8)[phase]
            group_ratio = (0.9, 8 / 9)[phase]
            if abs(split_ratio - expected_upstream_ratio) > 1e-12:
                raise RuntimeError(
                    "Pinned DCC split call order changed; refusing a non-paper 80/10/10 split"
                )
            self._paper_split_call += 1
            try:
                return split_ruler_question_groups(examples, group_ratio, train_lengths)
            except ValueError as error:
                raise RuntimeError(str(error)) from error

    # These settings are an explicit experiment overlay; the cloned upstream files stay unchanged.
    dcc_config.MODEL_CONFIG.update({
        "model_name": str(model_path),
        "trust_remote_code": False,
        "torch_dtype": args.dtype,
        "device_map": "auto",
        "attn_implementation": args.attn_implementation,
        "eval_model": str(model_path),
    })
    dcc_config.PATH_CONFIG["output_dir"] = str(args.out_dir)
    dcc_config.PATH_CONFIG["model_output_dir"] = str(args.out_dir)
    dcc_config.TRAINING_CONFIG.update({
        "samples_per_task": args.samples_per_task,
        "num_heads": args.num_heads,
        "batch_size": 1,
        "dev_ratio": 0.9,
        "seed": args.seed,
        "train_tasks": {"multihop_qa"},
        "eval_tasks": {"multihop_qa"},
    })
    dcc_config.DATA_CONFIG["chunking"] = {
        "strategy": "percentage",
        "percentage": {"percentage_num": args.chunk_count},
    }
    dcc_config.EARLY_OUTPUT_CONFIG.update({
        "min_sufficient_confidence": args.threshold * 100,
        "chunking_strategy": "percentage",
        "percentage_num": args.chunk_count,
        "number_of_evaluation_examples": len(eval_rows),
        "batch_size": 1,
        "eval_method": "match",
    })
    dcc_config.GENERATION_CONFIG.update({
        "temperature": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
    })
    setup_random_seeds(args.seed)

    def load_local_model():
        from transformers import AutoConfig
        model_config_obj = AutoConfig.from_pretrained(str(model_path), trust_remote_code=False)
        if args.yarn_128k:
            yarn = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768}
            # Newer transformers keep rope_theta INSIDE config.rope_parameters, and rope_scaling is an alias of
            # it, so assigning a fresh dict dropped the base frequency (YaRN init: None ** tensor). Merge into
            # the existing parameters and carry rope_theta over (from the config, else from config.json).
            theta = None
            if getattr(model_config_obj, "rope_parameters", None):
                theta = dict(model_config_obj.rope_parameters).get("rope_theta")
            theta = theta or getattr(model_config_obj, "rope_theta", None)
            if theta is None:
                theta = json.loads((Path(model_path) / "config.json").read_text())["rope_theta"]
            if hasattr(model_config_obj, "rope_parameters"):
                rp = dict(model_config_obj.rope_parameters or {})
                rp.update(yarn, rope_theta=theta)
                model_config_obj.rope_parameters = rp
            else:                                              # older transformers: rope_theta is its own field
                model_config_obj.rope_scaling = dict(yarn)
            model_config_obj.max_position_embeddings = 131072
            print(f"YaRN 128k: rope {getattr(model_config_obj, 'rope_parameters', None) or model_config_obj.rope_scaling}, "
                  f"rope_theta {theta}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), config=model_config_obj, trust_remote_code=False,
            torch_dtype=getattr(torch, args.dtype),
            device_map="auto", attn_implementation=args.attn_implementation,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return model, tokenizer

    training_wall_seconds = None
    training_device_count = None
    sampled_candidate_count = None
    internal_training_split = None
    if args.train_dcc_json is not None:
        train_candidates = read_dcc_json(args.train_dcc_json)
        validate_ruler_rows(
            train_candidates, split="train", questions=RULER_TRAIN_QUESTIONS
        )
        # Keep only the training lengths (default 8K-32K); the file still holds all five.
        train_candidates = [
            row for row in train_candidates
            if int(row.get("context_length_tokens", -1)) in train_lengths
        ]
        if {str(row["id"]) for row in train_candidates} & {str(row["id"]) for row in eval_rows}:
            raise SystemExit("Training/evaluation ID overlap detected")
        train_source_ids = {str(row["source_id"]) for row in train_candidates}
        train_hotpot_ids = {str(row.get("source_hotpotqa_id", "")) for row in train_candidates}
        eval_hotpot_ids = {str(row.get("source_hotpotqa_id", "")) for row in eval_rows}
        if not all(train_hotpot_ids) or not all(eval_hotpot_ids):
            raise SystemExit("RULER train/dev rows must preserve the original HotpotQA source IDs")
        if train_hotpot_ids & eval_hotpot_ids:
            raise SystemExit("RULER train/dev HotpotQA source-question overlap detected")
        if len(train_candidates) != args.samples_per_task:
            raise SystemExit(
                f"Expected exactly {args.samples_per_task} RULER training rows, found {len(train_candidates)}"
            )
        train_rows = train_candidates
        sampled_candidate_count = len(train_rows)
        if any(not row.get("gold_information") for row in train_rows):
            raise SystemExit("DCC training rows must have non-empty supporting-fact spans")
        print(
            f"DCC train cohort: {len(train_rows)} separate RULER-HotpotQA train trajectories "
            f"({RULER_TRAIN_QUESTIONS} questions x {len(train_lengths)} context lengths "
            f"{train_lengths}); split by question into 80/10/10 groups."
        )
        training_started = time.perf_counter()
        model, tokenizer = load_local_model()
        training_device_count = device["count"]
        # The feature forward generates nothing, so with --low-memory-training a training
        # document fits when its prompt fits the window. The older path keeps its 64-token margin.
        train_reserve = 0 if args.low_memory_training else 64
        too_long = [str(row["id"]) for row in train_rows
                    if len(tokenizer.encode(
                        prepare_sufficiency_input(row["input"], row["context"]),
                        add_special_tokens=True,
                    )) + train_reserve > args.context_limit_tokens]
        dropped_train_sources = []
        if too_long:
            if not args.drop_nonfitting_train_questions:
                raise SystemExit(
                    f"DCC training prompts exceed the configured context limit; first IDs: {too_long[:10]} "
                    f"({len(too_long)} documents). Pass --drop-nonfitting-train-questions to drop "
                    f"their source questions."
                )
            too_long_ids = set(too_long)
            dropped_train_sources = sorted({
                str(row["source_id"]) for row in train_rows if str(row["id"]) in too_long_ids
            })
            train_rows = [row for row in train_rows
                          if str(row["source_id"]) not in set(dropped_train_sources)]
            train_source_ids = {str(row["source_id"]) for row in train_rows}
            train_hotpot_ids = {str(row.get("source_hotpotqa_id", "")) for row in train_rows}
            sampled_candidate_count = len(train_rows)
            print(
                f"{len(too_long)} training documents do not fit {args.context_limit_tokens} tokens; "
                f"dropped their {len(dropped_train_sources)} source questions at every length. "
                f"Training uses {len(train_source_ids)} questions, {len(train_rows)} documents."
            )
        n_train_groups = len(train_source_ids)
        dev_groups = int(n_train_groups * 0.9)             # same expressions as the grouped split
        test_groups = n_train_groups - dev_groups
        fit_groups = int(dev_groups * (8 / 9))
        val_groups = dev_groups - fit_groups
        if min(fit_groups, val_groups, test_groups) < 1:
            raise SystemExit("Too few training questions left for an 80/10/10 split")

        feature_cache_meta = None
        if args.feature_cache is not None:
            args.feature_cache = args.feature_cache.resolve()
            args.feature_cache.mkdir(parents=True, exist_ok=True)
            # One cache holds one setting. A second process with another setting is refused.
            feature_cache_meta = {
                "model_path": str(model_path),
                "dtype": args.dtype,
                "chunk_count": args.chunk_count,
                "attn_implementation": args.attn_implementation,
                "yarn_128k": bool(args.yarn_128k),
                "chat_template_wrapper": chat_wrapper,
                "reduction": "prefix means at upstream percentage chunk ends, float64 sums, float32 stored",
            }
            meta_path = args.feature_cache / "cache_meta.json"
            if meta_path.is_file():
                if json.loads(meta_path.read_text(encoding="utf-8")) != feature_cache_meta:
                    raise SystemExit(f"{meta_path} was written with other settings; use a new --feature-cache")
            else:
                meta_path.write_text(json.dumps(feature_cache_meta, indent=2), encoding="utf-8")

        if args.extract_only:
            probe = SufficiencyProbe(dcc_config, model=model, tokenizer=tokenizer)
            manager_am = probe.activation_manager
            manager_am.reduce_tokens = False
            manager_am.prefix_chunks = args.chunk_count
            manager_am.feature_cache_dir = str(args.feature_cache)
            manager_am.model = dcc_overlay.FeatureForward(model)
            # Longest documents first, dealt round-robin, so every shard gets the same mix of lengths.
            ordered = sorted(train_rows, key=lambda row: (-int(row["context_length_tokens"]), str(row["id"])))
            shard_train_rows = [row for position, row in enumerate(ordered)
                                if position % args.shard_count == args.shard_index]
            print(f"Extracting features for shard {args.shard_index}/{args.shard_count}: "
                  f"{len(shard_train_rows)} of {len(train_rows)} training documents", flush=True)
            extract_started = time.perf_counter()
            for done, row in enumerate(shard_train_rows, 1):
                document_started = time.perf_counter()
                manager_am.compute_activations(
                    query=row["input"], context=row["context"],
                    gold_information=row.get("gold_information", []),
                )
                manager_am.attention_activations.clear()
                empty_cache(torch)
                print(f"[{done}/{len(shard_train_rows)}] {row['id']} "
                      f"({row['context_length_tokens']} tokens) "
                      f"{time.perf_counter() - document_started:.1f}s", flush=True)
            extract_seconds = time.perf_counter() - extract_started
            shard_manifest = {
                "shard_index": args.shard_index, "shard_count": args.shard_count,
                "documents": len(shard_train_rows),
                "document_ids_sha256": ids_sha256([str(row["id"]) for row in shard_train_rows]),
                "cache_misses_computed_here": getattr(manager_am, "feature_cache_misses", 0),
                "cache_hits_already_present": getattr(manager_am, "feature_cache_hits", 0),
                "extract_wall_seconds": extract_seconds,
                "model_load_and_check_seconds": extract_started - training_started,
                "device_type": device["type"],
                "visible_device_count": device["count"],
                "device_names": device["names"],
                "dropped_train_source_ids": dropped_train_sources,
            }
            (args.feature_cache / f"extract_shard_{args.shard_index}_of_{args.shard_count}.json").write_text(
                json.dumps(shard_manifest, indent=2), encoding="utf-8")
            (args.out_dir / "dcc_extract_manifest.json").write_text(
                json.dumps(shard_manifest, indent=2), encoding="utf-8")
            print(f"Shard {args.shard_index} done in {extract_seconds / 60:.1f} min. "
                  f"Cache: {args.feature_cache}", flush=True)
            return

        dcc_train_dir = args.out_dir / "dcc_training_input"
        dcc_train_dir.mkdir(parents=True, exist_ok=True)
        task_path = dcc_train_dir / "multihop_qa.json"
        task_path.write_text(json.dumps(train_rows, ensure_ascii=False), encoding="utf-8")
        dcc_config.PATH_CONFIG["data_dir"] = str(dcc_train_dir)
        dcc_config.CLASSIFIER_CONFIG["visualize"] = False

        probe = SufficiencyProbe(dcc_config, model=model, tokenizer=tokenizer)
        probe.activation_manager.reduce_tokens = False  # training needs prefix means
        if args.low_memory_training:
            # Same features, computed in the hook; see dcc_overlay.py items 2(c) and 3.
            probe.activation_manager.prefix_chunks = args.chunk_count
            probe.activation_manager.model = dcc_overlay.FeatureForward(model)
            if args.feature_cache is not None:
                probe.activation_manager.feature_cache_dir = str(args.feature_cache)
        manager = RulerGroupedSplitDatasetManager(str(dcc_train_dir), probe=probe)
        train_data, val_data, internal_test_data, statistics = manager.prepare_splits(
            str(args.out_dir),
            train_tasks={"multihop_qa"},
            eval_tasks={"multihop_qa"},
            development_ratio=0.9,
            samples_per_task=len(train_rows),
            batch_size=1,
        )
        feature_cache_use = None
        if args.low_memory_training:
            feature_cache_use = {
                "documents_loaded_from_cache": getattr(probe.activation_manager, "feature_cache_hits", 0),
                "documents_computed_in_this_process": getattr(probe.activation_manager, "feature_cache_misses", 0),
            }
            print(f"Training features: {feature_cache_use}", flush=True)
            # Back to upstream behaviour for everything after feature extraction.
            probe.activation_manager.prefix_chunks = 0
            probe.activation_manager.feature_cache_dir = None
            probe.activation_manager.model = model
        if manager._paper_split_call != 2:
            raise RuntimeError(
                f"Expected two DCC split calls for one task, observed {manager._paper_split_call}"
            )
        expected_source_group_counts = {"train": fit_groups, "val": val_groups, "test": test_groups}
        expected_split_counts = {
            name: groups * len(train_lengths) for name, groups in expected_source_group_counts.items()
        }
        internal_training_split = {
            "protocol": "80/10/10 RULER train-source-question-group split; no source question crosses splits",
            "dropped_nonfitting_source_ids": dropped_train_sources,
            "seed": args.seed,
            "sampled_candidate_count": len(train_rows),
            "train_build_ids": sorted({str(row["build_id"]) for row in train_rows}),
            "train_source_question_count": len(train_source_ids),
            "train_source_ids_sha256": ids_sha256(sorted(train_source_ids)),
            "train_hotpotqa_ids_sha256": ids_sha256(sorted(train_hotpot_ids)),
            "context_lengths_tokens": list(train_lengths),
            "expected_counts": expected_split_counts,
            "expected_source_question_counts": expected_source_group_counts,
            "splits": {},
        }
        all_split_ids = set()
        for split_name, expected_count in expected_split_counts.items():
            split_path = args.out_dir / "data" / f"{split_name}.json"
            if not split_path.is_file():
                raise RuntimeError(f"DCC split file was not saved: {split_path}")
            split_rows = json.loads(split_path.read_text(encoding="utf-8"))
            split_ids = sorted(str(row["id"]) for row in split_rows)
            if len(split_ids) != expected_count:
                raise RuntimeError(
                    f"DCC {split_name} split has {len(split_ids)} rows; expected {expected_count}"
                )
            if len(set(split_ids)) != len(split_ids) or all_split_ids.intersection(split_ids):
                raise RuntimeError("DCC paper split contains duplicate or overlapping example IDs")
            split_source_ids = sorted({str(row.get("source_id", "")) for row in split_rows})
            if len(split_source_ids) != expected_source_group_counts[split_name]:
                raise RuntimeError(
                    f"DCC {split_name} has {len(split_source_ids)} source questions; "
                    f"expected {expected_source_group_counts[split_name]}"
                )
            if any(not source_id for source_id in split_source_ids):
                raise RuntimeError(f"DCC {split_name} has a missing source_id")
            for source_id in split_source_ids:
                variants = [row for row in split_rows if str(row["source_id"]) == source_id]
                if len(variants) != len(train_lengths) or {
                    int(row["context_length_tokens"]) for row in variants
                } != set(train_lengths):
                    raise RuntimeError(f"DCC {split_name} split has an incomplete RULER group: {source_id}")
            if any(
                set(split_source_ids) & set(previous["source_ids"])
                for previous in internal_training_split["splits"].values()
            ):
                raise RuntimeError("RULER source question leaked across DCC train/validation/test")
            all_split_ids.update(split_ids)
            internal_training_split["splits"][split_name] = {
                "count": len(split_ids),
                "source_question_count": len(split_source_ids),
                "source_ids": split_source_ids,
                "ids": split_ids,
                "ids_sha256": ids_sha256(split_ids),
            }
        if len(all_split_ids) != len(train_rows):
            raise RuntimeError("DCC paper split does not cover every sampled training candidate")
        (args.out_dir / "dcc_split_manifest.json").write_text(
            json.dumps(internal_training_split, indent=2), encoding="utf-8"
        )
        probe.train_and_select_heads(train_data, val_data, num_heads=args.num_heads)
        probe.save(str(args.out_dir / "probe"))
        # Keep this internal classifier diagnostic separate from official-dev scores.
        internal_metrics, _ = probe.evaluator.evaluate_sufficiency_performance(probe, internal_test_data)
        (args.out_dir / "dcc_internal_sufficiency_diagnostic.json").write_text(
            json.dumps({"split_statistics": statistics, "heldout_sufficiency_metrics": internal_metrics}, indent=2),
            encoding="utf-8",
        )
        training_wall_seconds = time.perf_counter() - training_started
        model, tokenizer = probe.model, probe.tokenizer
        del train_data, val_data, internal_test_data, manager
        print(f"Saved DCC probe to {args.out_dir / 'probe'}")
    else:
        probe_dir = args.probe_dir.resolve()
        if not (probe_dir / "probe_state.pt").is_file():
            raise SystemExit(f"Missing trained DCC probe: {probe_dir / 'probe_state.pt'}")
        model, tokenizer = load_local_model()
        dcc_config.PATH_CONFIG["model_output_dir"] = str(probe_dir.parent)
        # The saved probe holds pickled scikit-learn classifiers. torch >= 2.6 loads
        # with weights_only=True by default and rejects them, so load this file,
        # which this driver wrote itself, with weights_only=False.
        _torch_load = torch.load
        torch.load = lambda *la, **kw: _torch_load(*la, **{**kw, "weights_only": False})
        try:
            wrapper = EarlyOutputModel(str(probe_dir), model=model, tokenizer=tokenizer)
        finally:
            torch.load = _torch_load
        probe = wrapper.probe

    if args.train_dcc_json is not None:
        wrapper = EarlyOutputModel(str(args.out_dir / "probe"), model=model, tokenizer=tokenizer, probe=probe)
    # Inference pools over tokens inside the hook (same values as upstream's
    # post-hoc mean, without holding every token's activations on the CPU).
    wrapper.activation_manager.reduce_tokens = True

    inference_costs = {}

    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    if args.shard_count > 1 and args.probe_dir is None:
        raise SystemExit("Sharding is for inference with --probe-dir; train the probe once first")
    shard_rows = [row for position, row in enumerate(eval_rows)
                  if position % args.shard_count == args.shard_index]
    print(f"Evaluating shard {args.shard_index}/{args.shard_count}: {len(shard_rows)} of {len(eval_rows)} rows")

    def run_condition(mode: str, chunk: bool, threshold: float | None) -> Path:
        output_path = args.out_dir / f"dcc_{mode}.jsonl"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists: {output_path}; choose a new output directory or --overwrite")
        condition_started = time.perf_counter()
        records = []
        for example in shard_rows:
            full_prompt = prepare_sufficiency_input(example["input"], example["context"]) + get_suffix_prompt()
            prompt_length = len(tokenizer.encode(full_prompt, add_special_tokens=True))
            if prompt_length + args.max_new_tokens + 64 > args.context_limit_tokens:
                records.append({
                    "id": str(example["id"]),
                    "status": "non_fitting",
                    "method": "DCC",
                    "mode": mode,
                    "required_context_tokens": prompt_length,
                    "context_limit_tokens": args.context_limit_tokens,
                    "reason": "full prompt plus answer reserve exceeds the configured context limit",
                })
                continue
            forward_tokens = 0

            def count_forward(_module, forward_args, forward_kwargs):
                nonlocal forward_tokens
                input_ids = forward_kwargs.get("input_ids")
                if input_ids is None and forward_args and torch.is_tensor(forward_args[0]):
                    input_ids = forward_args[0]
                if torch.is_tensor(input_ids):
                    forward_tokens += int(input_ids.numel())

            hook = model.register_forward_pre_hook(count_forward, with_kwargs=True)
            synchronize(torch)
            started = time.perf_counter()
            try:
                result = wrapper.generate(example=example, chunk=chunk, threshold=threshold)
            finally:
                synchronize(torch)
                latency = time.perf_counter() - started
                hook.remove()
                wrapper.activation_manager.attention_activations.clear()
            prediction = str(result.get("generated_text", "")).strip()
            if not prediction:
                raise RuntimeError(f"DCC returned an empty prediction for {example['id']}")
            records.append({
                "id": str(example["id"]),
                "prediction": prediction,
                "status": "ok",
                "method": "DCC",
                "mode": mode,
                "chunks_processed": result["chunks_processed"],
                "total_chunks": result["total_chunks"],
                "model_forward_tokens": forward_tokens,
                "total_tokens": forward_tokens,
                "latency_seconds": latency,
                "sufficiency_confidence": result["sufficiency_confidence"],
                "pred_percentage": result["pred_percentage"],
                "gold_position_metadata": result["actual_gold_percentage"],
                "gold_position_metadata_type": example.get("gold_information_type", "unavailable"),
            })
        if len(records) != len(shard_rows):
            raise RuntimeError(f"Expected {len(shard_rows)} outputs, got {len(records)}")
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output_path)
        successful = [record for record in records if record.get("status") == "ok"]
        latencies = [float(record["latency_seconds"]) for record in successful]
        inference_costs[mode] = {
            "n_examples": len(records),
            "n_successful": len(successful),
            "n_non_fitting": sum(record.get("status") == "non_fitting" for record in records),
            "sum_model_forward_tokens": sum(int(record.get("model_forward_tokens", 0)) for record in successful),
            "mean_model_forward_tokens_per_success": (
                sum(int(record.get("model_forward_tokens", 0)) for record in successful) / len(successful)
                if successful else None
            ),
            "sum_per_example_latency_seconds": sum(latencies),
            "mean_latency_seconds_per_success": sum(latencies) / len(latencies) if latencies else None,
            "condition_wall_seconds_including_bookkeeping_and_write": time.perf_counter() - condition_started,
            "output": str(output_path.resolve()),
        }
        print(f"Saved {len(records)} {mode} DCC outputs: {output_path}")
        return output_path

    if args.train_only:
        print("--train-only: probe saved, evaluation skipped. Evaluate with --probe-dir.", flush=True)
    else:
        run_condition("full_context", chunk=False, threshold=None)
        run_condition("chunked", chunk=True, threshold=args.threshold)
    # Feature extraction done beforehand in other processes (--extract-only) is training cost too.
    feature_extraction = None
    extraction_device_seconds = 0.0
    if args.train_dcc_json is not None and args.feature_cache is not None:
        shard_files = sorted(args.feature_cache.glob("extract_shard_*_of_*.json"))
        shard_records = [json.loads(path.read_text(encoding="utf-8")) for path in shard_files]
        extraction_device_seconds = sum(
            (record["extract_wall_seconds"] + record["model_load_and_check_seconds"])
            * max(1, int(record.get("visible_device_count", record.get("visible_gpu_count", 1))))
            for record in shard_records
        )
        feature_extraction = {
            "shards": shard_records,
            "device_seconds_including_model_load": extraction_device_seconds,
            "use_in_training_process": feature_cache_use,
            "cache_settings": feature_cache_meta,
        }
    training_device_hours = (
        (training_wall_seconds * training_device_count + extraction_device_seconds) / 3600
        if training_wall_seconds is not None and training_device_count is not None else None
    )
    cost_breakdown = {
        "method": "DCC (official trained activation-classifier implementation)",
        "training": {
            "trained_component": "DCC sufficiency probe/head selection; the shared Qwen backbone is frozen",
            "status": "measured_in_this_run" if training_wall_seconds is not None else "not_run_probe_reused",
            "wall_seconds": training_wall_seconds,
            "device_type": device["type"],
            "visible_device_count": training_device_count,
            "device_hours_estimate": training_device_hours,
            "device_names": device["names"],
            "foundation_model_pretraining_included": False,
            "scope": (
                "from local model load through DCC feature/split preparation, probe head training and selection, "
                "and internal held-out sufficiency diagnostic"
                if training_wall_seconds is not None else
                "training was not performed in this transfer/evaluation invocation; use the originating training run's cost record"
            ),
        },
        "inference": {
            "status": "measured",
            "conditions": inference_costs,
            "scope": "per-example model-forward tokens and synchronized DCC generation latency; condition wall time also includes input preparation and output writing",
        },
        "interpretation": "Device-hours are wall-clock seconds multiplied by the visible device count (GPU-hours on GPUs); no energy or monetary cost is estimated.",
    }
    (args.out_dir / "dcc_cost_breakdown.json").write_text(
        json.dumps(cost_breakdown, indent=2), encoding="utf-8"
    )
    model_config = model_path / "config.json"
    (args.out_dir / "dcc_run_manifest.json").write_text(json.dumps({
        "method": "DCC (official implementation)",
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_config) if model_config.is_file() else None,
        "dcc_repo": str(repo),
        "dcc_upstream_commit": "337c20b6561e13fc1af34d706158cd74783765da",
        "train_candidates": str(args.train_dcc_json.resolve()) if args.train_dcc_json else None,
        "train_candidates_sha256": sha256_file(args.train_dcc_json.resolve()) if args.train_dcc_json else None,
        "prepared_training_sample": str((args.out_dir / "dcc_training_input" / "multihop_qa.json").resolve()) if args.train_dcc_json else None,
        "prepared_training_sample_sha256": sha256_file(args.out_dir / "dcc_training_input" / "multihop_qa.json") if args.train_dcc_json else None,
        "sampled_candidate_count": sampled_candidate_count,
        "training_source_questions": len(train_source_ids) if args.train_dcc_json else None,
        "training_source_ids_sha256": ids_sha256(sorted(train_source_ids)) if args.train_dcc_json else None,
        "training_hotpotqa_ids_sha256": ids_sha256(sorted(train_hotpot_ids)) if args.train_dcc_json else None,
        "training_build_ids": sorted({str(row.get("build_id")) for row in train_rows}) if args.train_dcc_json else None,
        "eval_data": str(args.eval_dcc_json.resolve()),
        "eval_data_sha256": sha256_file(args.eval_dcc_json.resolve()),
        "evaluation_rows": len(eval_rows),
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "shard_rows": len(shard_rows),
        "expected_evaluation_rows": args.expected_eval_rows,
        "evaluation_source_questions": len({str(row["source_id"]) for row in eval_rows}),
        "evaluation_source_ids_sha256": ids_sha256(sorted({str(row["source_id"]) for row in eval_rows})),
        "evaluation_hotpotqa_ids_sha256": ids_sha256(sorted({str(row["source_hotpotqa_id"]) for row in eval_rows})),
        "evaluation_build_ids": sorted({str(row.get("build_id")) for row in eval_rows}),
        "paired_eval_data": str(args.eval_acs_jsonl.resolve()),
        "paired_eval_data_sha256": sha256_file(args.eval_acs_jsonl.resolve()),
        "eval_ids_sha256": ids_sha256([str(row["id"]) for row in eval_rows]),
        "threshold": args.threshold,
        "chunk_count": args.chunk_count,
        "samples_per_task_cap": args.samples_per_task if args.train_dcc_json else None,
        "max_new_tokens": args.max_new_tokens,
        "context_limit_tokens": args.context_limit_tokens,
        "seed": args.seed,
        "training_wall_seconds": training_wall_seconds,
        "trained_component": "DCC sufficiency probe/head selection on a frozen Qwen backbone",
        "foundation_model_pretraining_included_in_training_cost": False,
        "device": device,
        "training_visible_device_count": training_device_count,
        "training_device_hours_estimate": training_device_hours,
        "inference_costs": inference_costs,
        "cost_breakdown_file": str((args.out_dir / "dcc_cost_breakdown.json").resolve()),
        "total_run_wall_seconds": time.perf_counter() - run_started,
        "internal_training_split": internal_training_split,
        "training_lengths_tokens": list(train_lengths),
        "training_split_adapter": "Upstream DCC feature extraction; source-ID grouped ratios 0.9 then 8/9 produce 96/12/12 QA groups",
        "decoding": "greedy",
        "tokens": "sum of actual model forward input_ids positions, including generation forwards",
        "upstream_code_modified": False,
        "driver_overlay": (
            "dcc_overlay.py: Qwen3 detected as qwen2 layout; no GQA head duplication; in-hook token "
            "mean at inference; with low_memory_training, in-hook prefix means at the chunk ends and "
            "a feature forward without LM head and KV cache during probe training"
        ),
        "low_memory_training": bool(args.low_memory_training),
        "train_only": bool(args.train_only),
        "feature_extraction": feature_extraction,
        "dropped_nonfitting_train_source_ids": dropped_train_sources if args.train_dcc_json else None,
        "training_device_hours_note": "training process wall time x its devices, plus the --extract-only shards (wall time including model load x their devices)",
        "attn_implementation": args.attn_implementation,
        "yarn_128k_in_memory": bool(args.yarn_128k),
        "chat_template_wrapper": chat_wrapper,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
