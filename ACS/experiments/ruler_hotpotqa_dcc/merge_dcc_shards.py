"""Merge per-process DCC inference shards back into one result folder.

Usage: python merge_dcc_shards.py --eval-acs-jsonl DEV.jsonl --out-dir MERGED SHARD_DIR [SHARD_DIR ...]
Rows are written in the order of the eval file; every eval id must appear exactly once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-acs-jsonl", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("shards", type=Path, nargs="+")
    a = ap.parse_args()
    order = [str(r["id"]) for r in read_jsonl(a.eval_acs_jsonl)]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("full_context", "chunked"):
        by_id: dict[str, dict] = {}
        for shard in a.shards:
            for row in read_jsonl(shard / f"dcc_{mode}.jsonl"):
                rid = str(row["id"])
                if rid in by_id:
                    raise SystemExit(f"{mode}: id {rid} appears in more than one shard")
                by_id[rid] = row
        missing = [rid for rid in order if rid not in by_id]
        if missing or len(by_id) != len(order):
            raise SystemExit(f"{mode}: {len(missing)} eval ids missing, first {missing[:3]}; "
                             f"{len(by_id)} rows for {len(order)} ids")
        with (a.out_dir / f"dcc_{mode}.jsonl").open("w", encoding="utf-8", newline="\n") as h:
            for rid in order:
                h.write(json.dumps(by_id[rid], ensure_ascii=False) + "\n")
        ok = [r for r in by_id.values() if r.get("status") == "ok"]
        print(f"{mode}: {len(by_id)} rows, {len(ok)} ok, "
              f"sum latency {sum(r['latency_seconds'] for r in ok) / 3600:.2f} hours")
    manifests = [json.loads((s / "dcc_run_manifest.json").read_text(encoding="utf-8")) for s in a.shards]
    (a.out_dir / "merged_shard_manifests.json").write_text(json.dumps(manifests, indent=2), encoding="utf-8")
    print(f"merged into {a.out_dir}")


if __name__ == "__main__":
    main()
