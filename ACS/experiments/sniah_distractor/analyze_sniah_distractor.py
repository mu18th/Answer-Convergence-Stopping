"""Paper Table 3: stopping under misleading needles placed before the evidence.

Replays every stopping policy offline on the complete trajectories recorded
(rule none) on the file written by prepare_sniah_distractor.py, with the
released implementations in ACS/eval/analyze.py. Scoring and token pricing
are the released ones: each policy pays the fold, the probe at its stop, and
only the signals it consumes.

Policies (the Table 3 rows)
  ACS                          released runtime rule: theta, eps=0.05, w=3
  Confidence alone             first step from chunk 1 whose answer has
  (first confident answer)     confidence >= theta (no window, no stability)
  END gate                     first asked END
  END gate, k ENDs in a row    first step ending k consecutive asked ENDs (k=4, 5)
  Verbalized gate              first numeric self-report >= 99.5
  Full reading                 every chunk
A diagnostic row keeps the window and turns only the stability test off.

Per policy: tokens (thousands, mean per question), stops before the evidence
(the chunk holding the target needle), accuracy, and how many stops return a
distractor's number. A second block keeps only the questions where a
distractor lies in an earlier chunk than the target. With --orig, every
question is paired with its run on the same document without distractors
(those trajectories need evidence labels from ACS/eval/add_oracle_labels.py).

Usage (from this folder, after run_sniah_distractor.py)
  python analyze_sniah_distractor.py
Inputs default to results/ here and, for the pairing, to the reference arm of
../sniah_ablations (labelled by run_sniah_ablations.py). Writes
results/sniah_distractor_qwen14b_table3.{csv,json,txt}.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "eval"):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import analyze as az  # noqa: E402  released analyzer
from common import dedupe_prefer_success, read_jsonl  # noqa: E402
from prepare_sniah_distractor import EXPECTED_ROWS, SNIAH_ROOT  # noqa: E402
from run_sniah_distractor import EXPECTED_MODEL, OUTPUT_NAME  # noqa: E402

THETA = 0.995
EPSILON = 0.05
WINDOW = 3
MIN_CHUNKS = 2
VERBALIZED_THRESHOLD = 99.5
REFERENCE_ARM = SNIAH_ROOT / "results" / "sniah_qwen14b_chunk24k_notes6k_full.jsonl"


def acs_cfg(theta: float, eps: float = EPSILON) -> dict:
    return {"rule": "windowed", "theta": theta, "eps": eps,
            "window": WINDOW, "min_chunks": MIN_CHUNKS}


def stop_end_k(tr: dict, k: int) -> int:
    """First step that completes k consecutive asked END decisions."""
    run = 0
    for step in tr["steps"]:
        run = run + 1 if step.get("verbalized_stop") is True else 0
        if run >= k:
            return int(step["t"])
    return len(tr["steps"])


def stop_conf_first(tr: dict, theta: float, min_chunks: int = 1) -> int:
    """First answering step with confidence >= theta; no window, no stability."""
    for step in tr["steps"]:
        t = int(step["t"])
        if t < min_chunks:
            continue
        if tr["task_shape"] == "mcq":
            confidence = max(step["posterior"].values())
        else:
            if az.is_non_answer(step.get("draft", "")):
                continue
            confidence = float(step.get("draft_conf", 0.0))
        if confidence >= theta:
            return t
    return len(tr["steps"])


def load(path: Path, expected_rows: int | None = EXPECTED_ROWS) -> list[dict]:
    rows = [t for t in dedupe_prefer_success(read_jsonl(str(path)))
            if t.get("steps") and not t.get("failed")]
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(f"{path.name}: expected {expected_rows} successes, found {len(rows)}")
    wrong = {str(row.get("model")) for row in rows if str(row.get("model")) != EXPECTED_MODEL}
    if wrong:
        raise RuntimeError(f"{path.name}: unexpected model(s) {sorted(wrong)}")
    az._check_build(rows, "fold trajectories")
    return rows


def evidence_chunk(tr: dict) -> int:
    chunk = az._first_evidence_chunk(tr)
    if chunk is None:
        raise SystemExit(f"{tr['id']}: no evidence_char_starts; label the trajectories first")
    return chunk


def policies(trajs: list[dict], theta: float) -> list[tuple]:
    rows = [
        ("ACS", lambda tr: az.stop_online(tr, cfg=acs_cfg(theta)), ("probe",)),
        ("Confidence alone (first confident answer)",
         lambda tr: stop_conf_first(tr, theta, 1), ("probe",)),
    ]
    if all("verbalized_stop" in s for t in trajs for s in t["steps"]):
        rows += [
            ("END gate", lambda tr: stop_end_k(tr, 1), ("gate",)),
            ("END gate, 4 ENDs in a row", lambda tr: stop_end_k(tr, 4), ("gate",)),
            ("END gate, 5 ENDs in a row", lambda tr: stop_end_k(tr, 5), ("gate",)),
        ]
    if all("verbalized" in s for t in trajs for s in t["steps"]):
        rows.append((f"Verbalized gate, {VERBALIZED_THRESHOLD:g}",
                     lambda tr: az.stop_verbalized(tr, VERBALIZED_THRESHOLD), ("verb",)))
    rows += [
        ("Full reading", az.stop_full, ()),
        ("diagnostic: window kept, stability off",
         lambda tr: az.stop_online(tr, cfg=acs_cfg(theta, 99.0)), ("probe",)),
    ]
    return rows


def evaluate(name: str, trajs: list[dict], stopper, uses, distractors: bool) -> dict:
    n = len(trajs)
    correct = tokens = premature = distractor_answers = premature_distractor = 0
    for tr in trajs:
        t = int(stopper(tr))
        answer, _ = az.step_answer(tr, t)
        correct += az.is_correct(tr, az.score_answer(tr, answer))
        tokens += az.policy_tokens(tr, t, uses)
        early = t < evidence_chunk(tr)
        premature += early
        if distractors:
            hit = any(value in str(answer) for value in tr["distractor_values"])
            distractor_answers += hit
            premature_distractor += early and hit
    row = {"policy": name, "n": n, "accuracy": correct / n,
           "mean_tokens_k": tokens / n / 1000, "stops_before_evidence": premature,
           "premature_rate": premature / n}
    if distractors:
        row.update({"distractor_answers": distractor_answers,
                    "premature_distractor_answers": premature_distractor})
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent
    ap.add_argument("--traj", type=Path, default=root / "results" / OUTPUT_NAME,
                    help="complete trajectories on the distractor build")
    ap.add_argument("--orig", type=Path, default=None,
                    help="trajectories of the same questions without distractors; default: "
                         "the ../sniah_ablations reference arm, skipped when it is absent "
                         "or incomplete")
    ap.add_argument("--theta", type=float, default=THETA)
    ap.add_argument("--allow-partial", action="store_true",
                    help="accept fewer than 250 trajectories (smoke tests only)")
    a = ap.parse_args()
    expected_rows = None if a.allow_partial else EXPECTED_ROWS
    az.STABILITY_W = WINDOW

    trajs = load(a.traj, expected_rows)
    missing = [t["id"] for t in trajs if not t.get("distractor_values")]
    if missing:
        raise SystemExit(f"{len(missing)} trajectories carry no distractor fields; they "
                         "were not run on the file written by prepare_sniah_distractor.py")
    if any(t.get("recording_mode") != "full_trajectory" for t in trajs):
        raise SystemExit("Replay needs complete trajectories recorded with --rule none")

    def chunk_of(tr, position):
        return position // int(tr.get("chunk_chars") or 24000) + 1

    subset = [t for t in trajs
              if min(chunk_of(t, p) for p in t["distractor_char_starts"]) < evidence_chunk(t)]
    out = {"model": EXPECTED_MODEL,
           "benchmark": "S-NIAH with 3 misleading needles before the target",
           "traj": a.traj.name, "n": len(trajs), "theta": a.theta, "epsilon": EPSILON,
           "window": WINDOW, "n_distractor_in_earlier_chunk": len(subset)}
    lines = [f"Qwen3-14B S-NIAH with misleading needles | n={len(trajs)} | "
             f"theta={a.theta:g} eps={EPSILON:g} W={WINDOW}"]
    for key, label, group in (("all", "all questions", trajs),
                              ("subset", "a distractor in an earlier chunk than the target",
                               subset)):
        if not group:
            continue
        lines += ["", f"=== {label} (n={len(group)}) ===",
                  f"{'stopping rule':44s} {'tokens (K)':>10s} {'stops before evidence':>22s} "
                  f"{'accuracy':>8s} {'answer = distractor':>20s}"]
        out[key] = []
        for name, stopper, uses in policies(group, a.theta):
            row = evaluate(name, group, stopper, uses, distractors=True)
            out[key].append(row)
            lines.append(f"{name:44s} {row['mean_tokens_k']:10.1f} "
                         f"{row['stops_before_evidence']:5d} ({row['premature_rate']:6.1%})      "
                         f"{row['accuracy']:8.3f} {row['distractor_answers']:9d} "
                         f"({row['premature_distractor_answers']} premature)")

    original = None
    if a.orig is not None:                    # given explicitly: it must be valid
        if not a.orig.exists():
            raise SystemExit(f"--orig {a.orig} does not exist")
        original = {t["id"]: t for t in load(a.orig, expected_rows)}
    elif REFERENCE_ARM.exists():              # the default: pair only with a complete run
        try:
            original = {t["id"]: t for t in load(REFERENCE_ARM, expected_rows)}
            if any(not t.get("evidence_char_starts") for t in original.values()):
                raise RuntimeError("no evidence labels (run_sniah_ablations.py adds them)")
        except (RuntimeError, SystemExit) as error:
            original = None
            print(f"Pairing skipped, the reference arm is not usable: {error}", flush=True)
    if original is not None:
        both = [t for t in trajs if t["id"] in original]
        paired = [original[t["id"]] for t in both]
        lines += ["", f"=== paired with the run without distractors (n={len(both)}) ===",
                  f"{'stopping rule':44s} {'without: acc / before evidence':>32s} "
                  f"{'with: acc / before evidence':>30s}"]
        out["paired"] = []
        # Policies available in both runs (a gate needs its signal on every step of both).
        for name, stopper, uses in policies(both + paired, a.theta):
            with_d = evaluate(name, both, stopper, uses, distractors=True)
            without = evaluate(name, paired, stopper, uses, distractors=False)
            out["paired"].append({"policy": name, "n": len(both),
                                  "without": without, "with": with_d})
            lines.append(f"{name:44s} {without['accuracy']:14.3f} / {without['premature_rate']:6.1%}"
                         f" {with_d['accuracy']:18.3f} / {with_d['premature_rate']:6.1%}")

    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    csv_path = results / "sniah_distractor_qwen14b_table3.csv"
    json_path = results / "sniah_distractor_qwen14b_table3.json"
    text_path = results / "sniah_distractor_qwen14b_table3.txt"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["subset", *out["all"][0]])
        writer.writeheader()
        for key in ("all", "subset"):
            for row in out.get(key, []):
                writer.writerow({"subset": key, **row})
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    text_path.write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved: {csv_path}\nSaved: {json_path}\nSaved: {text_path}")


if __name__ == "__main__":
    main()
