"""Reproduce the all-250 S-NIAH development/test operating-point analysis.

The five model trajectories share question IDs.  MD5 parity therefore creates
one common 135-question development set and one common 115-question held-out
test set.  Per-model tuning and the fixed controller are always evaluated on
the same held-out IDs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(EVAL_ROOT) not in sys.path:
    sys.path.append(str(EVAL_ROOT))

import analyze as az
from common import dedupe_prefer_success, is_dev, read_jsonl
from run_five_models import MODELS, ROOT

THETAS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.92, 0.94, 0.95,
          0.96, 0.97, 0.98, 0.99, 0.995]
EPSILONS = [0.005, 0.01, 0.02, 0.05]
WINDOW = 3
TOLERANCE = 0.02
FIXED = (0.995, 0.05)

LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "qwen3_14b": "Qwen3-14B",
    "qwen3_32b": "Qwen3-32B",
    "gemma3_12b": "Gemma-3-12B",
    "gemma3_27b": "Gemma-3-27B",
}
MODEL_ALIASES = {
    "qwen25_7b": {"qwen/qwen-2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"},
    "qwen3_14b": {"qwen/qwen3-14b", "Qwen/Qwen3-14B"},
    "qwen3_32b": {"qwen/qwen3-32b", "Qwen/Qwen3-32B", "Qwen/Qwen3-32B-FP8"},
    "gemma3_12b": {"google/gemma-3-12b-it"},
    "gemma3_27b": {"google/gemma-3-27b-it"},
}


def load_model(key: str, path: Path) -> list[dict]:
    rows = [row for row in dedupe_prefer_success(read_jsonl(path))
            if row.get("steps") and not row.get("failed")]
    if len(rows) != 250 or len({row["id"] for row in rows}) != 250:
        raise RuntimeError(f"{key}: expected 250 unique successful trajectories")
    recorded_models = {str(row["model"]) for row in rows if row.get("model")}
    if recorded_models and not recorded_models.issubset(MODEL_ALIASES[key]):
        raise RuntimeError(f"{key}: model identity mismatch: {recorded_models}")
    return rows


def metrics(rows: list[dict], theta: float, eps: float) -> dict:
    scores, tokens, premature = [], [], []
    for tr in rows:
        stop = az.stop_measured(tr, theta=theta, eps=eps, w=WINDOW)
        answer, _ = az.step_answer(tr, stop)
        scores.append(az.score_answer(tr, answer))
        tokens.append(az.policy_tokens(tr, stop, uses=("probe",)))
        evidence = az._first_evidence_chunk(tr)
        if evidence is None:
            raise RuntimeError(f"{tr['id']}: missing evidence location")
        premature.append(stop < evidence)
    return {
        "accuracy": sum(scores) / len(scores),
        "mean_tokens": sum(tokens) / len(tokens),
        "premature": sum(premature) / len(premature),
    }


def select(parts: list[list[dict]]) -> tuple[float, float, dict]:
    candidates = []
    for theta in THETAS:
        for eps in EPSILONS:
            per_model = [metrics(rows, theta, eps) for rows in parts]
            summary = {
                field: sum(row[field] for row in per_model) / len(per_model)
                for field in ("accuracy", "mean_tokens", "premature")
            }
            candidates.append((theta, eps, summary))
    best_accuracy = max(row[2]["accuracy"] for row in candidates)
    eligible = [row for row in candidates
                if row[2]["accuracy"] >= best_accuracy - TOLERANCE - 1e-12]
    theta, eps, summary = min(
        eligible,
        key=lambda row: (row[2]["mean_tokens"], -row[2]["accuracy"],
                         -row[0], -row[1]),
    )
    return theta, eps, {**summary, "best_dev_accuracy": best_accuracy}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--out-dir", type=Path, default=None)
    for key in MODELS:
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path, default=None)
    args = parser.parse_args()
    paths = {
        key: (getattr(args, key) or args.results_dir / f"sniah_{key}_n250_full.jsonl")
        for key in MODELS
    }
    data = {key: load_model(key, paths[key]) for key in MODELS}
    dev = {key: [row for row in rows if is_dev(row["id"])]
           for key, rows in data.items()}
    test = {key: [row for row in rows if not is_dev(row["id"])]
            for key, rows in data.items()}
    if any(len(dev[key]) != 135 or len(test[key]) != 115 for key in data):
        raise RuntimeError("Expected the shared deterministic 135/115 split")

    output_rows = []
    for key in MODELS:
        theta, eps, development = select([dev[key]])
        tuned = metrics(test[key], theta, eps)
        fixed = metrics(test[key], *FIXED)
        output_rows.append({
            "model": LABELS[key], "model_id": MODELS[key],
            "dev_n": len(dev[key]), "test_n": len(test[key]),
            "selected_theta": theta, "selected_epsilon": eps,
            "tuned_test_accuracy": tuned["accuracy"],
            "tuned_test_premature": tuned["premature"],
            "tuned_test_mean_tokens": tuned["mean_tokens"],
            "fixed_test_accuracy": fixed["accuracy"],
            "fixed_test_premature": fixed["premature"],
            "fixed_test_mean_tokens": fixed["mean_tokens"],
            "dev_selected_accuracy": development["accuracy"],
            "dev_best_accuracy": development["best_dev_accuracy"],
        })

    pooled_theta, pooled_eps, pooled = select(list(dev.values()))
    payload = {
        "split": "integer MD5(question_id) parity; even=development",
        "dev_n_per_model": 135, "test_n_per_model": 115,
        "theta_grid": THETAS, "epsilon_grid": EPSILONS,
        "window": WINDOW, "accuracy_tolerance": TOLERANCE,
        "selection_objective": "minimum mean tokens within 0.02 of best dev accuracy",
        "fixed_operating_point": {"theta": FIXED[0], "epsilon": FIXED[1]},
        "pooled_dev_selection": {"theta": pooled_theta, "epsilon": pooled_eps,
                                 **pooled},
        "models": output_rows,
    }
    results = args.out_dir or args.results_dir / "parameter_selection"
    results.mkdir(parents=True, exist_ok=True)
    (results / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (results / "table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader(); writer.writerows(output_rows)
    print("model selected tuned_acc fixed_acc fixed_premature")
    for row in output_rows:
        print(f"{row['model']:<14} ({row['selected_theta']:g},{row['selected_epsilon']:g}) "
              f"{row['tuned_test_accuracy']:.3f} {row['fixed_test_accuracy']:.3f} "
              f"{row['fixed_test_premature']:.1%}")
    print(f"pooled development selection: theta={pooled_theta:g}, epsilon={pooled_eps:g}")


if __name__ == "__main__":
    main()
