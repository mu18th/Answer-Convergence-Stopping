#!/usr/bin/env python3
"""Inject deterministic evidence offsets into S-NIAH trajectories offline.

RULER needle tasks have a known oracle: the needle contains the gold answer string,
so its character offset in the source context is the evidence position.
Trajectories drop the raw context, so this joins the prepared input by sample ID.

  python add_oracle_labels.py results/traj_sniah_14b.jsonl data/sniah.jsonl
"""
import json, os, shutil, sys

if len(sys.argv) != 3 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__.strip() + "\n\nPass the exact prepared data file used by the run.")

traj_path, data_path = sys.argv[1], sys.argv[2]
src = {}
for line in open(data_path, encoding="utf-8"):
    if line.strip():
        r = json.loads(line)
        src[r["id"]] = r

rows = [json.loads(l) for l in open(traj_path, encoding="utf-8") if l.strip()]
n_ok = n_miss = 0
for r in rows:
    s = src.get(r.get("id"))
    gold = str(r.get("answer", "") or "")
    if not s or not gold:
        n_miss += 1
        continue
    pos = s["context"].find(gold)
    if pos < 0:
        n_miss += 1
        continue
    r["evidence_char_starts"] = [pos]
    r["evidence_char_fraction"] = round(pos / max(len(s["context"]), 1), 4)
    n_ok += 1

if n_ok == 0:
    sys.exit("no needle positions recovered — check that data file matches this run")
shutil.copy(traj_path, traj_path + ".bak")
temporary = traj_path + ".tmp"
with open(temporary, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
os.replace(temporary, traj_path)
print(f"annotated {n_ok}/{len(rows)} trajectories with evidence_char_starts "
      f"({n_miss} unmatched; backup: {traj_path}.bak)")
