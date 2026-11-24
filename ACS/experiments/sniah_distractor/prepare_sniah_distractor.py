"""Paper Table 3: S-NIAH with misleading needles placed before the decisive one.

S-NIAH has one needle, so nothing misleading precedes the evidence. This
script builds that case from the paper's own 250 S-NIAH documents.

For every question it inserts K needles BEFORE the target needle. Nothing else
changes: same documents, questions, ids and target needle. The inserted
needles are copied verbatim from K other questions of the same file, so they
are real RULER needles ("One of the special magic numbers for <key> is:
<value>.") with other keys and other values. This is RULER's multi-key setting
(niah_multikey_1: 4 needles, 1 queried) with the 3 other needles placed before
the queried one. Insertion follows RULER's generator (scripts/data/synthetic/
niah.py at 38da79d): on the noise haystack a needle is its own line, inserted
at a random line index; here the index is drawn among the lines before the
target. On an essay haystack the needle goes between sentences instead.

Each row gains evidence_char_starts (the target needle's value), the
distractor keys, values, positions and source ids, and a build_id derived
from the source file's SHA-256, K and the seed, so a rebuild from the same
S-NIAH file reproduces the same rows and build_id. answer_char_index and
n_tokens_budget are copied from the source row and describe the document
before insertion.

Usage (from this folder)
    python prepare_sniah_distractor.py
Reads ../sniah_ablations/data/sniah.jsonl, building it first with that
folder's prepare_sniah.py when it is missing, and writes
data/sniah_distractor.jsonl with a .manifest.json next to it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SNIAH_ROOT = ROOT.parent / "sniah_ablations"
if str(SNIAH_ROOT) not in sys.path:
    sys.path.append(str(SNIAH_ROOT))

from prepare_sniah import EXPECTED_ROWS, validate as validate_sniah  # noqa: E402

SOURCE_PATH = SNIAH_ROOT / "data" / "sniah.jsonl"
DATA_PATH = ROOT / "data" / "sniah_distractor.jsonl"
K_DISTRACTORS = 3
SEED = 0

# RULER needle (NVIDIA/RULER 38da79d, scripts/data/synthetic/niah.py):
#   needle = "One of the special magic {type_needle_v} for {key} is: {value}."
# A key is "<adjective>-<noun>" from wonderwords, and some entries hold a space
# ("ad hoc-picture"), so the key is matched up to " is: " on the same line.
NEEDLE_RE = re.compile(r"One of the special magic numbers for (?P<key>[^\n]+?) is: (?P<value>\d+)\.")
VARIANT = "distractors_before_target"


def expected_build_id(source_sha256: str, k: int = K_DISTRACTORS, seed: int = SEED) -> str:
    """Build id of the distractor file made from a source file with this SHA-256."""
    return hashlib.sha256(json.dumps(
        {"source_sha256": source_sha256, "variant": VARIANT, "k": k, "seed": seed},
        sort_keys=True).encode()).hexdigest()[:20]


def validate(path: Path, k: int = K_DISTRACTORS) -> list[dict]:
    """Validate a complete distractor build: 250 rows, one build, k needles before each target."""
    rows = read_rows(path)
    if len(rows) != EXPECTED_ROWS or len({str(r["id"]) for r in rows}) != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS} unique rows in {path.name}, found {len(rows)}")
    builds = {str(r.get("build_id", "")) for r in rows}
    if len(builds) != 1 or not next(iter(builds)):
        raise RuntimeError(f"{path.name} must carry one non-empty build_id; found {builds}")
    for row in rows:
        starts = row.get("distractor_char_starts") or []
        evidence = (row.get("evidence_char_starts") or [None])[0]
        if len(starts) != k:
            raise RuntimeError(f"{row.get('id')}: {len(starts)} distractors; "
                               f"this build should have {k}")
        if (row.get("variant") != VARIANT
                or len(starts) != len(row.get("distractor_values") or [])
                or not starts or evidence is None or max(starts) > evidence
                or len(NEEDLE_RE.findall(row["context"])) != len(starts) + 1):
            raise RuntimeError(f"{row.get('id')}: not a valid distractor row")
    print(f"Validated S-NIAH distractor build: {len(rows)} rows, build_id={next(iter(builds))}",
          flush=True)
    return rows


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


# ----------------------------------------------------------------------------- build
def find_target(row: dict) -> re.Match:
    """The single RULER needle of an S-NIAH document, checked against the row."""
    matches = list(NEEDLE_RE.finditer(row["context"]))
    if len(matches) != 1:
        pos = row["context"].find(str(row["answer"]))
        around = row["context"][max(0, pos - 120):pos + 40] if pos >= 0 else "<gold answer not in the document>"
        raise RuntimeError(f"{row['id']}: expected exactly one needle in the document, "
                           f"found {len(matches)}. Text around the gold answer: {around!r}")
    m = matches[0]
    if m["value"] != str(row["answer"]):
        raise RuntimeError(f"{row['id']}: needle value {m['value']} is not the gold "
                           f"answer {row['answer']}")
    if f" {m['key']} " not in f" {row['question']} ":
        raise RuntimeError(f"{row['id']}: needle key {m['key']!r} is not in the question")
    return m


def insertion_slots(context: str, start: int, end: int) -> tuple[str, list[int]]:
    """Places where a needle can go before the target needle (start..end).

    Returns (separator, character offsets). Noise haystack: the needle is a whole
    line, so the slots are the line starts up to the target line. Essay haystack:
    the needle sits between sentences, so the slots are the sentence starts up to
    the target sentence. The header line of the prompt is never split.
    """
    line_based = (start == 0 or context[start - 1] == "\n") and \
                 (end == len(context) or context[end] == "\n")
    first_newline = context.find("\n")
    header_end = first_newline + 1 if 0 <= first_newline < start and \
        "quiz you" in context[:first_newline] else 0
    if line_based:
        slots = [header_end] if header_end <= start else []
        slots += [m.end() for m in re.finditer(r"\n", context[:start]) if m.end() > header_end]
        return "\n", sorted(set(slots))
    slots = [header_end]
    slots += [header_end + m.end()
              for m in re.finditer(r"(?<=[.!?])\s", context[header_end:start])]
    slots.append(start)
    return " ", sorted(set(s for s in slots if s <= start))


def build(a) -> None:
    rows = read_rows(a.data)
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("duplicate ids in the input file")
    targets = {r["id"]: find_target(r) for r in rows}
    pool = [(r["id"], targets[r["id"]]["key"], targets[r["id"]]["value"],
             targets[r["id"]].group(0)) for r in rows]
    source_sha = hashlib.sha256(a.data.read_bytes()).hexdigest()
    build_id = expected_build_id(source_sha, a.k, a.seed)

    out_rows, stats = [], {"line_based": 0, "sentence_based": 0, "stacked": 0}
    for r in rows:
        rng = random.Random(f"{a.seed}:{r['id']}")
        m = targets[r["id"]]
        context, gold, key = r["context"], str(r["answer"]), m["key"]
        # K needles from K other questions: other key, other value, value not in this text.
        candidates = [p for p in pool if p[0] != r["id"] and p[1] != key and p[2] != gold
                      and p[2] not in context]
        rng.shuffle(candidates)
        chosen, seen_keys, seen_values = [], set(), set()
        for p in candidates:
            if p[1] in seen_keys or p[2] in seen_values:
                continue
            chosen.append(p)
            seen_keys.add(p[1])
            seen_values.add(p[2])
            if len(chosen) == a.k:
                break
        if len(chosen) != a.k:
            raise RuntimeError(f"{r['id']}: could not draw {a.k} distinct distractors")

        sep, slots = insertion_slots(context, m.start(), m.end())
        stats["line_based" if sep == "\n" else "sentence_based"] += 1
        if len(slots) >= a.k:
            picks = sorted(rng.sample(slots, a.k))
        else:                                   # target at the very start: stack before it
            picks = sorted(rng.choice(slots) for _ in range(a.k))
            stats["stacked"] += 1
        # Insert from the right so earlier offsets stay valid.
        for offset, p in sorted(zip(picks, chosen), key=lambda x: x[0], reverse=True):
            context = context[:offset] + p[3] + sep + context[offset:]

        evidence = context.find(gold)
        holders = [n for n in NEEDLE_RE.finditer(context) if n.start() <= evidence < n.end()]
        if evidence < 0 or len(holders) != 1 or holders[0]["value"] != gold \
                or holders[0]["key"] != key:
            raise RuntimeError(f"{r['id']}: gold answer is no longer located in the target needle")
        # Locate each distractor through its own needle (key and value), never by
        # a bare number search that could match inside another needle.
        by_key = {n["key"]: n for n in NEEDLE_RE.finditer(context)}
        d_starts = []
        for p in chosen:
            needle = by_key.get(p[1])
            if needle is None or needle["value"] != p[2]:
                raise RuntimeError(f"{r['id']}: distractor needle {p[1]!r} not found intact")
            d_starts.append(needle.start("value"))
        if any(s < 0 or s > evidence for s in d_starts):
            raise RuntimeError(f"{r['id']}: a distractor is not before the target")
        if len(NEEDLE_RE.findall(context)) != a.k + 1:
            raise RuntimeError(f"{r['id']}: expected {a.k + 1} needles after insertion")

        row = dict(r)
        row.update({
            "context": context, "n_chars": len(context),
            "evidence_char_starts": [evidence],
            "evidence_char_fraction": round(evidence / max(len(context), 1), 4),
            "distractor_keys": [p[1] for p in chosen],
            "distractor_values": [p[2] for p in chosen],
            "distractor_char_starts": d_starts,
            "distractor_source_ids": [p[0] for p in chosen],
            "target_key": key, "variant": VARIANT, "build_id": build_id,
        })
        out_rows.append(row)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", encoding="utf-8") as h:
        for row in out_rows:
            h.write(json.dumps(row) + "\n")
    chunk = a.chunk_chars
    earlier = sum(1 for row in out_rows
                  if min(row["distractor_char_starts"]) // chunk < row["evidence_char_starts"][0] // chunk)
    manifest = {
        "source": str(a.data), "source_sha256": source_sha, "variant": VARIANT,
        "k": a.k, "seed": a.seed, "build_id": build_id, "rows": len(out_rows),
        "output": str(a.out), "output_sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
        "haystack_structure": stats,
        f"rows_with_a_distractor_in_an_earlier_chunk_at_{chunk}_chars": earlier,
    }
    Path(str(a.out) + ".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    ex = out_rows[0]
    s = ex["distractor_char_starts"][0]
    print("\nexample", ex["id"], "| question:", ex["question"][:90])
    print("  first distractor :", ex["context"][max(0, s - 60):s + 12].replace("\n", " / "))
    e = ex["evidence_char_starts"][0]
    print("  target needle    :", ex["context"][max(0, e - 60):e + 12].replace("\n", " / "))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=SOURCE_PATH,
                    help="the paper's prepared 250-row S-NIAH file")
    ap.add_argument("--out", type=Path, default=DATA_PATH)
    ap.add_argument("--k", type=int, default=K_DISTRACTORS,
                    help="distractor needles per document (RULER niah_multikey_1 "
                         "has 4 needles: the target and 3 others)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chunk-chars", type=int, default=24000,
                    help="only for the count in the manifest")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args()
    if not a.validate_only:
        if a.data == SOURCE_PATH and not SOURCE_PATH.exists():
            # Reuse the canonical S-NIAH builder instead of a drifting copy.
            subprocess.run([sys.executable, "-u", str(SNIAH_ROOT / "prepare_sniah.py")],
                           cwd=SNIAH_ROOT, check=True)
        validate_sniah(a.data)
        build(a)
    validate(a.out, a.k)


if __name__ == "__main__":
    main()
