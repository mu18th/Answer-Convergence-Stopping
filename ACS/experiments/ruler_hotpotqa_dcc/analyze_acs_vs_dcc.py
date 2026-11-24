"""Paper Table 4: ACS against trained DCC on the paper's 500 RULER-HotpotQA documents.

Reads result files only; no model is called.

ACS     the paper's complete Qwen3-14B RULER-HotpotQA trajectories (both 250-row
        cohorts), replayed with the released runtime rule (theta=0.995,
        eps=0.05, w=3) through ACS/eval/analyze.py:stop_online and scored with
        the official RULER qa_2 criterion (any gold string is a case-insensitive
        substring of the answer).
DCC     one or two run folders written by run_dcc.py (or merge_dcc_shards.py)
        and scored by score_answers.py: dcc_chunked.jsonl,
        dcc_chunked_scores.json, dcc_full_context.jsonl,
        dcc_full_context_scores.json. The main row is the probe trained on 8K
        to 32K; --dcc-run-all-lengths adds the probe trained on all five lengths.

DCC cannot read documents whose prompt exceeds the context window. Its
accuracy is therefore reported on the documents it read, and also with the
unread documents counted as wrong over all 500. ACS is reported on all 500
and on the documents DCC read; the paired tests use only the latter.

Before any number is computed, two checks refuse a comparison that is not
Table 4's:
- same documents: the trajectories must be exactly the 500 documents in
  data/paper_eval/ruler_hotpotqa_paper500.jsonl (the file DCC was run on;
  same ids, questions, answers and paper build ids);
- same DCC settings: every dcc_run_manifest.json of a run must record the
  pinned DCC commit, the settings run_dcc_experiment.py passes (chat template,
  YaRN 128K, 10 chunks, threshold 0.95, 32 new tokens, seed 42, sdpa), that
  same evaluation file, and the probe's training lengths.

Usage (from this folder, after run_dcc_experiment.py)
  python analyze_acs_vs_dcc.py \
      --acs-traj <paper Qwen3-14B RULER-HotpotQA trajectories, both cohorts>
DCC run folders default to the two arms of run_dcc_experiment.py; the
all-lengths arm is skipped when its folder does not exist. Writes
results/acs_vs_dcc_table4.{csv,json,txt}.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import random
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "eval", HERE.parent / "ruler_hotpotqa"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import analyze as az  # noqa: E402  released analyzer
from analyze_ruler_hotpotqa import official_score, validate  # noqa: E402
from common import read_jsonl  # noqa: E402
from prepare_dcc_data import DCC_COMMIT, EVAL_ACS_JSONL, sha256 as file_sha256  # noqa: E402
from run_dcc_experiment import ARMS, DCC_ARGS  # noqa: E402

THETA = 0.995
EPSILON = 0.05
WINDOW = 3
LENGTHS = (8192, 16384, 32768, 65536, 131072)
GROUPS = (
    ("32K and above", lambda length: length >= 32768),
    ("8K and 16K", lambda length: length <= 16384),
    ("64K and 128K", lambda length: length >= 65536),
    ("all", lambda length: True),
)


class Tee(io.TextIOBase):
    """Write printed output to the console and to a buffer for the .txt report."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def check_same_documents(trajectories: list[dict], eval_path: Path) -> None:
    """The ACS trajectories must be the very documents DCC was evaluated on."""
    if not eval_path.is_file():
        raise SystemExit(f"Missing {eval_path}, the documents DCC was run on; "
                         "prepare_dcc_data.py writes it")
    rows = read_jsonl(str(eval_path))
    by_id = {str(row["id"]): row for row in rows}
    trajectory_ids = {str(tr["id"]) for tr in trajectories}
    if len(by_id) != len(rows) or set(by_id) != trajectory_ids:
        raise SystemExit(
            f"Trajectory ids differ from {eval_path.name}: "
            f"missing={sorted(set(by_id) - trajectory_ids)[:5]}, "
            f"extra={sorted(trajectory_ids - set(by_id))[:5]}")
    different = [tr["id"] for tr in trajectories
                 if str(tr.get("build_id")) != str(by_id[str(tr["id"])].get("paper_build_id"))
                 or tr.get("question") != by_id[str(tr["id"])].get("question")
                 or tr.get("answer") != by_id[str(tr["id"])].get("answer")]
    if different:
        raise SystemExit(
            f"{len(different)} trajectories are not the documents of {eval_path.name} "
            f"(build id, question or answer differ), first {different[:5]}. Rebuild the "
            "evaluation documents with prepare_dcc_data.py from the same RULER data.")


def expected_dcc_settings() -> dict:
    """The settings run_dcc_experiment.py passes, under their dcc_run_manifest.json names."""
    values, i = {}, 0
    while i < len(DCC_ARGS):
        if i + 1 < len(DCC_ARGS) and not DCC_ARGS[i + 1].startswith("--"):
            values[DCC_ARGS[i]] = DCC_ARGS[i + 1]
            i += 2
        else:
            values[DCC_ARGS[i]] = True
            i += 1
    return {
        "chunk_count": int(values["--chunk-count"]),
        "threshold": float(values["--threshold"]),
        "max_new_tokens": int(values["--max-new-tokens"]),
        "context_limit_tokens": int(values["--context-limit-tokens"]),
        "seed": int(values["--seed"]),
        "attn_implementation": values["--attn-implementation"],
        "yarn_128k_in_memory": bool(values.get("--yarn-128k")),
        "chat_template": bool(values.get("--chat-template")),
    }


def run_manifests(folder: Path) -> list[tuple[str, dict]]:
    """dcc_run_manifest.json of a run folder, or the shard manifests of a merged folder."""
    found = []
    single = folder / "dcc_run_manifest.json"
    if single.is_file():
        found.append((str(single), json.loads(single.read_text(encoding="utf-8"))))
    merged = folder / "merged_shard_manifests.json"
    if merged.is_file():
        found += [(f"{merged} [shard {i}]", manifest) for i, manifest in
                  enumerate(json.loads(merged.read_text(encoding="utf-8")))]
    return found


def check_dcc_run(name: str, run: Path, train_folder: Path, train_lengths: str,
                  eval_sha256: str) -> None:
    """Refuse a DCC run whose manifests do not record the Table 4 settings."""
    expected = expected_dcc_settings()
    evaluation = run_manifests(run)
    training = [(label, m) for label, m in run_manifests(train_folder)
                if m.get("train_candidates") is not None]
    problems = []
    if not evaluation:
        problems.append(f"no dcc_run_manifest.json in {run}")
    if not training:
        problems.append(f"no training manifest in {train_folder} (needed for the probe's "
                        "training lengths and cost)")
    lengths = [int(x) for x in train_lengths.split(",")]
    for label, manifest in evaluation + training:
        if manifest.get("dcc_upstream_commit") != DCC_COMMIT:
            problems.append(f"{label}: DCC commit {manifest.get('dcc_upstream_commit')!r}")
        if manifest.get("upstream_code_modified") is not False:
            problems.append(f"{label}: DCC code not recorded as unmodified")
        if manifest.get("paired_eval_data_sha256") != eval_sha256:
            problems.append(f"{label}: evaluated on another file than --eval-acs-jsonl "
                            "(SHA-256 differs)")
        for key, value in expected.items():
            recorded = (bool(manifest.get("chat_template_wrapper")) if key == "chat_template"
                        else manifest.get(key))
            if recorded != value:
                problems.append(f"{label}: {key}={recorded!r}, Table 4 uses {value!r}")
    for label, manifest in training:
        if manifest.get("training_lengths_tokens") != lengths:
            problems.append(f"{label}: probe trained on {manifest.get('training_lengths_tokens')}, "
                            f"this arm is {lengths}")
    if problems:
        raise SystemExit(f"DCC run '{name}' is not the Table 4 setting:\n  " + "\n  ".join(problems))


def acs_records(trajectories: list[dict], theta: float, eps: float, window: int) -> dict:
    cfg = {"rule": "windowed", "theta": theta, "eps": eps,
           "window": window, "min_chunks": 2}
    records = {}
    for tr in trajectories:
        t = int(az.stop_online(tr, cfg=cfg))
        answer, _ = az.step_answer(tr, t)
        full_answer, _ = az.step_answer(tr, len(tr["steps"]))
        records[str(tr["id"])] = {
            "correct": official_score(answer, tr["answer"]),
            "full_correct": official_score(full_answer, tr["answer"]),
            "early": t < len(tr["steps"]),
            "length": int(tr["context_length_tokens"]),
            "tokens": float(az.policy_tokens(tr, t, ("probe",))),
        }
    return records


def load_dcc(run: Path, cost_dir: Path | None = None) -> dict:
    cost_file = (cost_dir or run) / "dcc_cost_breakdown.json"
    def jsonl(name):
        return {str(row["id"]): row for row in read_jsonl(str(run / name))}

    def scores(name):
        payload = json.loads((run / name).read_text(encoding="utf-8"))
        return {str(e["id"]): e for e in payload["examples"]}

    rows = jsonl("dcc_chunked.jsonl")
    return {
        "rows": rows,
        "scores": scores("dcc_chunked_scores.json"),
        "full_scores": scores("dcc_full_context_scores.json"),
        "status": {s: sum(r.get("status") == s for r in rows.values())
                   for s in sorted({str(r.get("status")) for r in rows.values()})},
        "cost": (json.loads(cost_file.read_text(encoding="utf-8"))
                 if cost_file.is_file() else None),
    }


def paired(first: list[int], second: list[int]) -> dict:
    """Paired difference, 95% bootstrap interval (seed 0, 10,000 resamples), exact McNemar p."""
    n = len(first)
    diff = (sum(first) - sum(second)) / n
    only_first = sum(a == 1 and b == 0 for a, b in zip(first, second))
    only_second = sum(a == 0 and b == 1 for a, b in zip(first, second))
    rng = random.Random(0)
    boots = sorted(
        sum(first[j] - second[j] for j in (rng.randrange(n) for _ in range(n))) / n
        for _ in range(10000)
    )
    m, k = only_first + only_second, min(only_first, only_second)
    p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2 ** m) if m else 1.0
    return {"n": n, "first": sum(first) / n, "second": sum(second) / n, "diff": diff,
            "ci95": [boots[249], boots[9749]], "only_first": only_first,
            "only_second": only_second, "mcnemar_p": p}


def mean(values) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize_dcc(name: str, dcc: dict, acs: dict) -> dict:
    scored = dcc["scores"]
    unknown = set(scored) - set(acs)
    if unknown:
        raise SystemExit(f"{name}: DCC ids outside the ACS trajectories: {sorted(unknown)[:5]}")
    out = {"status_counts": dcc["status"], "by_length": {}, "paired_acs_minus_dcc": {}}
    print(f"\n##### DCC, probe {name}: status {dcc['status']}, scored {len(scored)} of {len(acs)}")
    print(f"{'length':>7} | {'read':>4} | {'DCC acc':>7} | {'DCC /all':>8} | {'early':>5} | "
          f"{'DCC tok':>8} | {'ACS acc':>7} | {'ACS same docs':>13} | {'ACS early':>9} | {'ACS tok':>8}")
    for length in LENGTHS + ("all",):
        every = [i for i in acs if length == "all" or acs[i]["length"] == length]
        read = [i for i in every if i in scored]
        if not read:
            continue
        row = {
            "documents": len(every), "dcc_read": len(read),
            "dcc_accuracy_on_read": mean(int(scored[i]["correct"]) for i in read),
            "dcc_accuracy_unread_as_wrong": sum(int(scored[i]["correct"]) for i in read) / len(every),
            "dcc_early_stops": sum(int(dcc["rows"][i]["chunks_processed"])
                                   < int(dcc["rows"][i]["total_chunks"]) for i in read),
            "dcc_mean_tokens": mean(float(dcc["rows"][i]["total_tokens"]) for i in read),
            "dcc_no_stop_accuracy": mean(int(dcc["full_scores"][i]["correct"])
                                         for i in read if i in dcc["full_scores"]),
            "acs_accuracy_all": mean(acs[i]["correct"] for i in every),
            "acs_accuracy_on_dcc_read": mean(acs[i]["correct"] for i in read),
            "acs_early_stops": sum(acs[i]["early"] for i in every),
            "acs_mean_tokens": mean(acs[i]["tokens"] for i in every),
        }
        out["by_length"][str(length)] = row
        print(f"{str(length):>7} | {len(read):4d} | {row['dcc_accuracy_on_read']:7.3f} | "
              f"{row['dcc_accuracy_unread_as_wrong']:8.3f} | {row['dcc_early_stops']:5d} | "
              f"{row['dcc_mean_tokens']:8.0f} | {row['acs_accuracy_all']:7.3f} | "
              f"{row['acs_accuracy_on_dcc_read']:13.3f} | {row['acs_early_stops']:9d} | "
              f"{row['acs_mean_tokens']:8.0f}")
    print(f"\n--- ACS minus DCC ({name}), paired on the documents DCC read")
    for label, keep in GROUPS:
        ids = [i for i in scored if keep(acs[i]["length"])]
        if not ids:
            continue
        result = paired([acs[i]["correct"] for i in ids], [int(scored[i]["correct"]) for i in ids])
        out["paired_acs_minus_dcc"][label] = result
        print(f"  {label:14s} n={result['n']:3d}  ACS {result['first']:.3f} vs DCC {result['second']:.3f}  "
              f"diff {100 * result['diff']:+5.1f} [{100 * result['ci95'][0]:+5.1f}, "
              f"{100 * result['ci95'][1]:+5.1f}]  only ACS {result['only_first']:3d}  "
              f"only DCC {result['only_second']:3d}  p={result['mcnemar_p']:.2g}")
    cost = (dcc["cost"] or {}).get("training") or {}
    out["training_cost"] = cost
    if cost:
        print(f"  DCC training: status={cost.get('status')} "
              f"device_hours={cost.get('device_hours_estimate', cost.get('gpu_hours_estimate'))} "
              f"device={cost.get('device_type', 'cuda')} "
              f"x{cost.get('visible_device_count', cost.get('visible_gpu_count'))}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acs-traj", type=Path, nargs="+", required=True,
                    help="complete Qwen3-14B RULER-HotpotQA trajectories (both cohorts)")
    ap.add_argument("--acs-data", type=Path, nargs="*", default=None,
                    help="optional prepared cohort files, to check ids and build ids")
    ap.add_argument("--dcc-run", type=Path, default=ARMS["train_8k_32k"].evaluation_dir,
                    help="DCC run folder, probe trained on 8K to 32K")
    ap.add_argument("--dcc-run-all-lengths", type=Path,
                    default=ARMS["train_all_lengths"].evaluation_dir,
                    help="DCC run folder, probe trained on all five lengths (skipped if absent)")
    ap.add_argument("--dcc-train-all-lengths", type=Path,
                    default=ARMS["train_all_lengths"].train_dir,
                    help="training folder of the all-lengths probe (its manifest and cost)")
    ap.add_argument("--eval-acs-jsonl", type=Path, default=EVAL_ACS_JSONL,
                    help="the 500 documents DCC was evaluated on (written by prepare_dcc_data.py)")
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--eps", type=float, default=EPSILON)
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    a = ap.parse_args()
    if not (a.dcc_run / "dcc_chunked_scores.json").is_file():
        raise SystemExit(f"Missing scored DCC run: {a.dcc_run}; run run_dcc_experiment.py")
    if a.dcc_run_all_lengths is not None and \
            not (a.dcc_run_all_lengths / "dcc_chunked_scores.json").is_file():
        a.dcc_run_all_lengths = None

    trajectories = validate(
        [row for path in a.acs_traj for row in read_jsonl(str(path))],
        [row for path in a.acs_data for row in read_jsonl(str(path))] if a.acs_data else None,
    )
    if len(trajectories) != 500:
        raise SystemExit(f"Expected the paper's 500 trajectories; found {len(trajectories)}")
    check_same_documents(trajectories, a.eval_acs_jsonl)
    eval_sha256 = file_sha256(a.eval_acs_jsonl)
    check_dcc_run("trained on 8K-32K", a.dcc_run, a.dcc_run,
                  ARMS["train_8k_32k"].train_lengths, eval_sha256)
    if a.dcc_run_all_lengths is not None:
        check_dcc_run("trained on all five lengths", a.dcc_run_all_lengths,
                      a.dcc_train_all_lengths, ARMS["train_all_lengths"].train_lengths,
                      eval_sha256)

    log = io.StringIO()
    with contextlib.redirect_stdout(Tee(sys.stdout, log)):
        summary = report(a, trajectories)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = a.out_dir / "acs_vs_dcc_table4.json"
    csv_path = a.out_dir / "acs_vs_dcc_table4.csv"
    text_path = a.out_dir / "acs_vs_dcc_table4.txt"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = [{"dcc_probe": name, "length": length, **row}
             for name, result in summary["dcc"].items()
             for length, row in result["by_length"].items()]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    text_path.write_text(log.getvalue(), encoding="utf-8")
    print(f"\nSaved: {csv_path}\nSaved: {json_path}\nSaved: {text_path}")


def report(a, trajectories: list[dict]) -> dict:
    """Print the Table 4 comparison and return it as a summary dictionary."""
    acs = acs_records(trajectories, a.theta, a.eps, a.window)
    summary = {
        "acs": {"n": len(acs), "theta": a.theta, "eps": a.eps, "window": a.window,
                "accuracy": mean(r["correct"] for r in acs.values()),
                "full_reading_accuracy": mean(r["full_correct"] for r in acs.values()),
                "mean_tokens": mean(r["tokens"] for r in acs.values()),
                "early_stops": sum(r["early"] for r in acs.values())},
        "dcc": {},
    }
    print(f"ACS on all {len(acs)}: accuracy {summary['acs']['accuracy']:.3f}, "
          f"full reading {summary['acs']['full_reading_accuracy']:.3f}, "
          f"tokens/doc {summary['acs']['mean_tokens']:,.0f}, early stops {summary['acs']['early_stops']}")

    runs = {"trained on 8K-32K": load_dcc(a.dcc_run)}
    if a.dcc_run_all_lengths is not None:
        # Its training cost is recorded in the training folder, not the merged evaluation.
        runs["trained on all five lengths"] = load_dcc(
            a.dcc_run_all_lengths, cost_dir=a.dcc_train_all_lengths)
    for name, dcc in runs.items():
        summary["dcc"][name] = summarize_dcc(name, dcc, acs)

    if len(runs) == 2:
        first, second = (runs[name]["scores"] for name in runs)
        print("\n--- DCC all five lengths minus DCC 8K-32K (documents both read)")
        summary["dcc_all_minus_dcc_8k32k"] = {}
        for label, keep in GROUPS:
            ids = [i for i in second if i in first and keep(acs[i]["length"])]
            if not ids:
                continue
            result = paired([int(second[i]["correct"]) for i in ids],
                            [int(first[i]["correct"]) for i in ids])
            summary["dcc_all_minus_dcc_8k32k"][label] = result
            print(f"  {label:14s} n={result['n']:3d}  {result['first']:.3f} vs {result['second']:.3f}  "
                  f"diff {100 * result['diff']:+5.1f}  p={result['mcnemar_p']:.2g}")
    return summary


if __name__ == "__main__":
    main()
