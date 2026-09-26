"""Fail closed on common release, anonymity, and configuration mistakes."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "ACS"
EXPECTED_SOURCE_FILES = 62
EXPECTED_PATHS_SHA256 = (
    "425e6a0930e7c2451f23a4d6a2efdb2bd9dd08bac4b9858d76f2724a2b358437"
)
MAX_SOURCE_BYTES = 100_000
FROZEN_FILE_SHA256 = {
    "ACS/run_fold.py": "fab4e9517841f2d38682499ca392a6aad1d7017e121a4ac9a8b9304c251bcd3c",
    "ACS/draft_cleaner.py": "05c4f1d3ffdd79cece7330a8e9c6c0219dd3cb01cb6aa61c4714051f1aa9a62d",
    "test_release_semantics.py": "ad0f6899c042dcb608cf7bb7891b92588663cd2aa8f024aa067130bfd373c3ec",
}
MATCHED80_IDS_SHA256 = (
    "66fd3251bdca1d9299ebd8ce9d43e76f1f93a844b7dc300e342270802164be03"
)
TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".json", ".txt", ".sh", ".jinja"
}
TEXT_NAMES = {"LICENSE", ".gitignore"}
REQUIRED = {
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "REPRODUCTION_MANIFEST.json",
    "ACS/__init__.py",
    "ACS/common.py",
    "ACS/draft_cleaner.py",
    "ACS/run_fold.py",
    "ACS/eval/__init__.py",
    "ACS/eval/analyze.py",
    "ACS/eval/analyze_threshold_sweep.py",
    "ACS/eval/analyze_timing.py",
    "ACS/prepare_data.py",
    "test_release_semantics.py",
    "ACS/experiments/browsecomp_plus/prepare_browsecomp_all830.py",
    "ACS/experiments/browsecomp_plus/run_browsecomp_all830.py",
    "ACS/experiments/browsecomp_plus/analyze_browsecomp_all830.py",
    "ACS/experiments/ruler_hotpotqa/analyze_threshold_sweep.py",
    "ACS/experiments/ruler_hotpotqa/analyze_evidence_proxy.py",
    "ACS/experiments/sniah_five_model/analyze_parameter_selection.py",
    "ACS/experiments/sniah_five_model/analyze_five_model_table.py",
    "ACS/experiments/sniah_five_model/config_sniah_five_model.yaml",
    "ACS/benchmark_data/browsecomp_plus_data.py",
    "ACS/benchmark_data/browsecomp_plus_types.py",
    "ACS/configs/longbench_full.yaml",
    "ACS/configs/sniah_full.yaml",
    "ACS/experiments/longbench_kimi_ablations/make_matched_80.py",
    "ACS/experiments/longbench_kimi_ablations/matched80_ids.json",
}
REDUNDANT_RUNTIME_COPIES = {
    "ACS/experiments/browsecomp_plus/run_fold.py",
    "ACS/experiments/browsecomp_plus/analyze.py",
    "ACS/experiments/browsecomp_plus/common.py",
    "ACS/experiments/browsecomp_plus/draft_cleaner.py",
    "ACS/experiments/sniah_ablations/run_fold.py",
    "ACS/experiments/sniah_ablations/analyze.py",
    "ACS/experiments/sniah_ablations/common.py",
    "ACS/experiments/sniah_ablations/draft_cleaner.py",
    "ACS/experiments/sniah_five_model/run_fold.py",
    "ACS/experiments/sniah_five_model/analyze.py",
    "ACS/experiments/sniah_five_model/common.py",
    "ACS/experiments/sniah_five_model/draft_cleaner.py",
}
FORBIDDEN_DIRS = {
    "data", "results", "checkpoints", "hf_cache", "cache", "vendor", "figures",
    "official_ruler", "__pycache__", ".pytest_cache", ".venv",
}
FORBIDDEN_DIR_PREFIXES = ("pytest-cache-files-", "official_ruler_")
FORBIDDEN_FILES = {".env", "analyze_sniah_qwen3_14b_250.py"}
PATTERNS = {
    "absolute Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    "absolute Unix home path": re.compile(r"/home/[^/\s]+/", re.I),
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9_-]{8,}"),
    "generic API key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "Hugging Face key": re.compile(r"\bhf_[A-Za-z0-9]{16,}\b"),
}
SOURCE_PATTERNS = {
    "unresolved release marker": re.compile(r"\b(?:TO" r"DO|TB" r"D)\b"),
    "fabricated missing-label probability": re.compile(
        r"(?:get\([^\n]{0,80},\s*1e-9\)|[=:]\s*1e-9\s*(?:[,}\]]|$))",
        re.I,
    ),
    "structured output budget below 32 tokens": re.compile(
        r"max_tokens\s*=\s*(?:8|16)\b"
    ),
}


def main() -> None:
    errors: list[str] = []
    missing = sorted(name for name in REQUIRED if not (ROOT / name).is_file())
    if missing:
        errors.append(f"missing required files: {missing}")

    redundant = sorted(
        name for name in REDUNDANT_RUNTIME_COPIES if (ROOT / name).exists()
    )
    if redundant:
        errors.append(f"redundant canonical runtime copies included: {redundant}")

    source_files = [path for path in ROOT.rglob("*") if path.is_file()]
    relative_files = sorted(path.relative_to(ROOT).as_posix() for path in source_files)
    if len(source_files) != EXPECTED_SOURCE_FILES:
        errors.append(
            f"frozen artifact must contain exactly {EXPECTED_SOURCE_FILES} files; "
            f"found {len(source_files)}"
        )
    paths_digest = hashlib.sha256("\n".join(relative_files).encode()).hexdigest()
    if paths_digest != EXPECTED_PATHS_SHA256:
        errors.append(
            "artifact path set differs from the frozen submission scope: "
            f"{paths_digest}"
        )
    folded = [name.casefold() for name in relative_files]
    if len(folded) != len(set(folded)):
        errors.append("artifact contains case-insensitive path collisions")

    for name, expected_digest in FROZEN_FILE_SHA256.items():
        path = ROOT / name
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected_digest:
                errors.append(
                    f"{name}: tested frozen file changed ({digest})"
                )

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symbolic link included: {relative}")
            continue
        if path.is_dir() and (
            path.name in FORBIDDEN_DIRS
            or path.name.startswith(FORBIDDEN_DIR_PREFIXES)
        ):
            errors.append(f"generated/private directory included: {relative}")
            continue
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_FILES:
            errors.append(f"credential file included: {relative}")
            continue
        if path.stat().st_size > MAX_SOURCE_BYTES:
            errors.append(
                f"unexpectedly large source file ({path.stat().st_size} bytes): {relative}"
            )
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_NAMES:
            errors.append(f"unexpected non-source file included: {relative}")
            continue
        if path == Path(__file__).resolve():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as error:
            errors.append(f"non-UTF-8 source file: {relative}: {error}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{relative}: {label}")
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".json"}:
            for label, pattern in SOURCE_PATTERNS.items():
                if pattern.search(text):
                    errors.append(f"{relative}: {label}")

    for path in ROOT.rglob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        server = config.get("server") or {}
        if server.get("api_key") not in (None, ""):
            errors.append(f"{path.relative_to(ROOT)}: api_key must be empty")
        base_url = str(server.get("base_url") or "")
        if not re.fullmatch(r"https?://[^\s]+", base_url):
            errors.append(f"{path.relative_to(ROOT)}: invalid or missing base_url")
        if not str(server.get("model") or "").strip():
            errors.append(f"{path.relative_to(ROOT)}: missing model")
        for key in ("cache_dir", "scratch_dir"):
            value = server.get(key)
            if value and Path(str(value)).is_absolute():
                errors.append(f"{path.relative_to(ROOT)}: {key} must be relative")
        fold = config.get("fold") or {}
        if fold:
            for key in ("chunk_chars", "notes_chars", "notes_tokens", "workers"):
                if int(fold.get(key, 0)) <= 0:
                    errors.append(f"{path.relative_to(ROOT)}: {key} must be positive")
            if int(fold.get("draft_tokens", 0)) != 32:
                errors.append(f"{path.relative_to(ROOT)}: draft_tokens must equal 32")
        online = config.get("online_stop") or {}
        if online:
            required_online = {
                "rule", "theta", "eps", "window", "min_chunks",
                "record_verbalized", "record_gate",
            }
            absent = sorted(required_online - set(online))
            if absent:
                errors.append(
                    f"{path.relative_to(ROOT)}: missing online_stop keys {absent}"
                )
            rule = online.get("rule")
            if rule not in {"none", "windowed"}:
                errors.append(
                    f"{path.relative_to(ROOT)}: unsupported online rule {rule!r}"
                )
            window = int(online.get("window", 3))
            minimum = int(online.get("min_chunks", window))
            if minimum != 2:
                errors.append(
                    f"{path.relative_to(ROOT)}: min_chunks must equal 2"
                )
            if float(online.get("theta", 0.995)) != 0.995:
                errors.append(
                    f"{path.relative_to(ROOT)}: online theta must equal 0.995"
                )
            if float(online.get("eps", 0.05)) != 0.05:
                errors.append(
                    f"{path.relative_to(ROOT)}: online eps must equal 0.05"
                )
            if window != 3:
                errors.append(
                    f"{path.relative_to(ROOT)}: online window must equal 3"
                )
            # Auxiliary baseline recording is intentionally disabled in some
            # single-axis ablation configs.  Require an explicit Boolean so a
            # typo cannot silently change behavior, without forcing baselines
            # into runs that did not record them.
            for key in ("record_verbalized", "record_gate"):
                if not isinstance(online.get(key), bool):
                    errors.append(
                        f"{path.relative_to(ROOT)}: {key} must be Boolean"
                    )
        stopping = config.get("stopping") or {}
        if "theta_grid" in stopping:
            theta_grid = [float(value) for value in stopping["theta_grid"]]
            if theta_grid != [0.995]:
                errors.append(
                    f"{path.relative_to(ROOT)}: theta_grid must be [0.995]"
                )

    manifest = __import__("json").loads(
        (ROOT / "REPRODUCTION_MANIFEST.json").read_text(encoding="utf-8")
    )
    shared = manifest.get("shared_default") or {}
    expected = {
        "chunk_chars": 24000,
        "notes_chars": 6000,
        "theta": 0.995,
        "epsilon": 0.05,
        "window_states": 3,
    }
    for key, value in expected.items():
        if shared.get(key) != value:
            errors.append(
                f"REPRODUCTION_MANIFEST.json: {key} must equal {value!r}"
            )
    if shared.get("verbalized_threshold_0_100") != 99.5:
        errors.append(
            "REPRODUCTION_MANIFEST.json: verbalized threshold must equal 99.5"
        )

    expected_models = [
        "qwen/qwen3.5-397b-a17b",
        "moonshotai/kimi-k2.5",
        "qwen/qwen3-14b",
        "qwen/qwen-2.5-7b-instruct",
        "qwen/qwen3-32b",
        "google/gemma-3-12b-it",
        "google/gemma-3-27b-it",
    ]
    if manifest.get("models") != expected_models:
        errors.append("REPRODUCTION_MANIFEST.json: runtime model IDs are incorrect")

    judge = manifest.get("browsecomp_judge") or {}
    if judge != {
        "model": "qwen/qwen3-32b", "temperature": 0.0, "max_tokens": 512
    }:
        errors.append("REPRODUCTION_MANIFEST.json: BrowseComp judge is not frozen")

    ruler = (manifest.get("benchmarks") or {}).get("ruler_hotpotqa") or {}
    if ruler.get("tokenizer_type") != "hf" or \
            ruler.get("tokenizer_path") != "Qwen/Qwen3-14B":
        errors.append("REPRODUCTION_MANIFEST.json: RULER tokenizer is not frozen")

    expected_baselines = [
        "full_reading",
        "random_stop_uniform_200_draws",
        "verbalized_gate_at_99.5",
        "end_continue_gate",
    ]
    if manifest.get("headline_baselines") != expected_baselines:
        errors.append(
            "REPRODUCTION_MANIFEST.json: headline baseline set/order is incorrect"
        )

    analyzer_text = (PACKAGE_ROOT / "eval" / "analyze.py").read_text(encoding="utf-8")
    if 'sub.add_parser("policies")' not in analyzer_text or \
            'sub.add_parser("evidence")' not in analyzer_text:
        errors.append("analyze.py: required paper analysis commands are missing")
    for obsolete in ('sub.add_parser("paired")', 'sub.add_parser("grid")'):
        if obsolete in analyzer_text:
            errors.append(f"analyze.py: obsolete command remains: {obsolete}")
    for obsolete in ("def tune_measured(", "if True:"):
        if obsolete in analyzer_text:
            errors.append(f"analyze.py: dead analysis construct remains: {obsolete}")
    if "saved += nn - s" in analyzer_text:
        errors.append(
            "analyze.py: capture credits premature stops; safety adjustment missing"
        )
    runtime_text = (PACKAGE_ROOT / "run_fold.py").read_text(encoding="utf-8")
    required_runtime_fragments = [
        'STOP = {"rule": "windowed", "theta": 0.995, "eps": 0.05, "window": 3',
        'max_tokens=32, logprobs=True',
        'not is_non_answer(s.get("draft", ""))',
        'step["verbalized"] = numeric',
        'step["verbalized_stop"] = gate_value',
        '"execution_fingerprint": execution_id',
        'validate_execution_resume(',
    ]
    for fragment in required_runtime_fragments:
        if fragment not in runtime_text:
            errors.append(f"run_fold.py: required frozen behavior missing: {fragment}")
    common_text = (PACKAGE_ROOT / "common.py").read_text(encoding="utf-8")
    if "posterior_from_logprobs(measured)" not in common_text:
        errors.append("common.py: complete A-D logprob preflight is missing")
    readme_text = (ROOT / "README.md").read_text(encoding="utf-8")
    for placeholder in ("results/FIRST.jsonl", "data/FIRST.jsonl", "JUDGE_MODEL_ID"):
        if placeholder in readme_text:
            errors.append(f"README.md: unresolved command placeholder: {placeholder}")
    required_readme_fragments = [
        "--tokenizer-type hf --tokenizer-path Qwen/Qwen3-14B",
        "--judge-model qwen/qwen3-32b",
        "analyze_five_model_table.py",
        "run_five_models.py --model all --workers 8 --prepare",
        "--arms base24_notes6 chunk12 chunk48 notes3 notes12",
        "--workers 8 --prepare",
        "The runner does **not** put `answer`, `answers`, `evidence_char_starts`",
        "record complete trajectories with --rule none",
    ]
    for fragment in required_readme_fragments:
        if fragment not in readme_text:
            errors.append(f"README.md: required reproduction detail missing: {fragment}")
    for config_path in (PACKAGE_ROOT / "experiments" / "sniah_ablations").glob("*.yaml"):
        config_text = config_path.read_text(encoding="utf-8")
        stale_heldout_label = "hash split yields " + "115 held-out rows"
        if stale_heldout_label in config_text:
            errors.append(
                f"{config_path.relative_to(ROOT)}: all-250 ablation is mislabeled as held-out"
            )
    ruler_analysis = (
        PACKAGE_ROOT / "experiments" / "ruler_hotpotqa" / "analyze_ruler_hotpotqa.py"
    ).read_text(encoding="utf-8")
    if 'isinstance(value, (int, float)) and value >= threshold' not in ruler_analysis:
        errors.append(
            "RULER verbalized baseline does not conservatively continue on invalid values"
        )
    fingerprinted_launchers = [
        PACKAGE_ROOT / "experiments" / "browsecomp_plus" / "run_browsecomp_all830.py",
        PACKAGE_ROOT / "experiments" / "sniah_ablations" / "run_sniah_ablations.py",
        PACKAGE_ROOT / "experiments" / "longbench_kimi_ablations" / "run_ablations.py",
    ]
    for path in fingerprinted_launchers:
        source = path.read_text(encoding="utf-8")
        if "validate_execution_resume" not in source:
            errors.append(
                f"{path.relative_to(ROOT)}: launcher bypasses execution fingerprints"
            )
    sniah_ablation_prepare = (
        PACKAGE_ROOT / "experiments" / "sniah_ablations" / "prepare_sniah.py"
    ).read_text(encoding="utf-8")
    for obsolete in ("EXPECTED_HELD_OUT", "from common import is_dev"):
        if obsolete in sniah_ablation_prepare:
            errors.append(
                f"S-NIAH all-250 ablation retains split-only dependency: {obsolete}"
            )
    for path in ROOT.rglob("*.py"):
        if path == Path(__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        if "matplotlib" in source or "savefig(" in source:
            errors.append(f"{path.relative_to(ROOT)}: figure-generation code remains")
    ablation = manifest.get("longbench_kimi_ablation") or {}
    if ablation.get("expected_rows") != 80:
        errors.append(
            "REPRODUCTION_MANIFEST.json: LongBench Kimi ablation must use 80 rows"
        )
    if ablation.get("theta") != 0.995:
        errors.append(
            "REPRODUCTION_MANIFEST.json: LongBench Kimi theta must equal 0.995"
        )

    benchmarks = manifest.get("benchmarks") or {}
    pinned_sources = {
        ("longbench_v2", "revision"):
            "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        ("browsecomp_plus", "query_revision"):
            "144cff8e35b5eaef7e526346aa60774a9deb941f",
        ("browsecomp_plus", "corpus_revision"):
            "b27b02bc3e45511b8b82a13e6f90ce761df726f6",
    }
    for (benchmark, key), revision in pinned_sources.items():
        if (benchmarks.get(benchmark) or {}).get(key) != revision:
            errors.append(
                f"REPRODUCTION_MANIFEST.json: {benchmark}.{key} is not pinned"
            )

    longbench_loader = (
        PACKAGE_ROOT / "benchmark_data" / "longbench_v2_data.py"
    ).read_text(encoding="utf-8")
    browsecomp_loader = (
        PACKAGE_ROOT / "benchmark_data" / "browsecomp_plus_data.py"
    ).read_text(encoding="utf-8")
    browsecomp_preparer = (
        PACKAGE_ROOT / "experiments" / "browsecomp_plus" /
        "prepare_browsecomp_all830.py"
    ).read_text(encoding="utf-8")
    if "revision=LONG_BENCH_REVISION" not in longbench_loader:
        errors.append("LongBench-v2 loader does not use its pinned revision")
    for obsolete in ("def browsecomp_plus(", "def construction(", "def _build_id("):
        if obsolete in browsecomp_loader:
            errors.append(
                f"BrowseComp helper module retains superseded builder: {obsolete}"
            )
    for path_name, source in (
        ("BrowseComp all-830 preparer", browsecomp_preparer),
    ):
        if "revision=QUERY_REVISION" not in source:
            errors.append(f"{path_name} does not use the pinned query revision")
        if "revision=CORPUS_REVISION" not in source:
            errors.append(f"{path_name} does not use the pinned corpus revision")

    matched_path = (
        PACKAGE_ROOT / "experiments" / "longbench_kimi_ablations" /
        "matched80_ids.json"
    )
    if matched_path.is_file():
        matched = __import__("json").loads(matched_path.read_text(encoding="utf-8"))
        ids = [str(value) for value in matched.get("ids", [])]
        if matched.get("n") != 80 or len(ids) != 80 or len(set(ids)) != 80:
            errors.append("matched80_ids.json: expected exactly 80 unique frozen IDs")
        digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
        if digest != MATCHED80_IDS_SHA256:
            errors.append("matched80_ids.json: IDs/order differ from frozen cohort")
        if matched.get("ordered_ids_sha256") != MATCHED80_IDS_SHA256:
            errors.append("matched80_ids.json: ordered_ids_sha256 is incorrect")

    for path in ROOT.rglob("*.py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")

    if errors:
        raise SystemExit("RELEASE VALIDATION FAILED:\n- " + "\n- ".join(errors))
    print("PASS: release structure, anonymity patterns, configs, and Python syntax")


if __name__ == "__main__":
    main()
