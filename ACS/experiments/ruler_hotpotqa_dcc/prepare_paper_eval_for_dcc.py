"""Put the PAPER's RULER-HotpotQA evaluation set into the form the DCC driver needs.

The paper's 500 evaluation trajectories (Tables 1 and 4) are two disjoint cohorts of
50 questions x 5 lengths, selected by the submitted prepare_ruler_hotpotqa.py from
the official RULER qa_2 pool. This script changes no question, context or answer.
It only adds the bookkeeping fields the DCC driver validates (HotpotQA id, split
tag, one combined build id) and writes the same rows in DCC's input schema, so DCC
is evaluated on exactly the documents the paper's ACS numbers come from. Row ids
are kept, so DCC outputs join one to one with the paper's ACS trajectories.

Usage:
  python prepare_paper_eval_for_dcc.py \
      --paper-eval data/paper_eval/ruler_hotpotqa_c1.jsonl data/paper_eval/ruler_hotpotqa_c2.jsonl \
      --hotpot-source external/RULER_paper/scripts/data/synthetic/json/hotpotqa.json \
      --out-acs data/paper_eval/ruler_hotpotqa_paper500.jsonl \
      --out-dcc data/dcc/paper_dev_500/multihop_qa.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

LENGTHS = (8192, 16384, 32768, 65536, 131072)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-eval", type=Path, nargs="+", required=True)
    ap.add_argument("--hotpot-source", type=Path, required=True,
                    help="the HotpotQA dev JSON array RULER generated from (pinned SHA)")
    ap.add_argument("--out-acs", type=Path, required=True)
    ap.add_argument("--out-dcc", type=Path, required=True)
    a = ap.parse_args()

    source = json.loads(a.hotpot_source.read_text(encoding="utf-8"))
    rows = [row for path in a.paper_eval for row in read_jsonl(path)]
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("duplicate row ids across the supplied paper cohorts")
    questions = sorted({int(r["question_index"]) for r in rows})
    if len(rows) != 500 or len(questions) != 100:
        raise SystemExit(f"expected 500 rows over 100 questions; got {len(rows)} rows, "
                         f"{len(questions)} questions")
    paper_builds = sorted({str(r["build_id"]) for r in rows})
    combined_build = "paper_" + hashlib.sha256("+".join(paper_builds).encode()).hexdigest()[:16]

    acs_rows, dcc_rows = [], []
    # Order: by length, then question index, so every length bucket is contiguous.
    for row in sorted(rows, key=lambda r: (int(r["context_length_tokens"]), int(r["question_index"]))):
        index = int(row["question_index"])
        if int(row["context_length_tokens"]) not in LENGTHS:
            raise SystemExit(f"unexpected length in {row['id']}")
        src = source[index]
        if str(src.get("question", "")).strip() != str(row["question"]).strip():
            raise SystemExit(f"{row['id']}: question does not match HotpotQA source index {index}; "
                             "wrong --hotpot-source file")
        answers = [str(v) for v in (row["answer"] if isinstance(row["answer"], list) else [row["answer"]])]
        if [str(src.get("answer", "")).strip()] != [v.strip() for v in answers]:
            raise SystemExit(f"{row['id']}: gold answer does not match the HotpotQA source")
        hotpot_id = str(src["_id"])
        extra = {
            "source_id": f"hotpotqa_eval_{hotpot_id}",
            "source_hotpotqa_id": hotpot_id,
            "source_split": "dev",
            "paper_source_id": row.get("source_id"),
            "paper_build_id": row.get("build_id"),
            "build_id": combined_build,
        }
        acs = {**row, **extra}
        acs_rows.append(acs)
        context = row["context"]
        # First literal answer mention: metadata only on dev, never used for training.
        hits = [(context.lower().find(v.lower()), v) for v in answers]
        hits = [(p, v) for p, v in hits if p >= 0]
        gold = ""
        if hits:
            p, v = min(hits)
            gold = context[p:p + len(v)]
        dcc_rows.append({
            "id": str(row["id"]), "input": row["question"], "context": context,
            "answers": answers, "gold_information": gold,
            "gold_information_type": "answer_mention_proxy" if gold else "unavailable",
            "task": "multihop_qa", "benchmark": "ruler_hotpotqa", "ruler_task": "qa_2",
            "context_length_tokens": int(row["context_length_tokens"]),
            "question_index": index, **extra,
        })

    a.out_acs.parent.mkdir(parents=True, exist_ok=True)
    a.out_dcc.parent.mkdir(parents=True, exist_ok=True)
    with a.out_acs.open("w", encoding="utf-8", newline="\n") as h:
        for r in acs_rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    a.out_dcc.write_text(json.dumps(dcc_rows, ensure_ascii=False), encoding="utf-8")
    print(f"paper cohort build ids: {paper_builds}")
    print(f"combined build id for the DCC driver: {combined_build}")
    print(f"wrote {len(acs_rows)} rows: {a.out_acs}")
    print(f"wrote {len(dcc_rows)} rows: {a.out_dcc}")


if __name__ == "__main__":
    main()
