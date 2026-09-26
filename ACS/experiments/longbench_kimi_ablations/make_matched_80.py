"""Materialize the frozen 80-ID ablation cohort without consulting outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ROWS = 80
EXPECTED_IDS_SHA256 = (
    "66fd3251bdca1d9299ebd8ce9d43e76f1f93a844b7dc300e342270802164be03"
)


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=here.parent.parent / "data" / "longbench_v2_all.jsonl",
    )
    parser.add_argument(
        "--manifest", type=Path, default=here / "matched80_ids.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "data" / "longbench_v2_matched_80.jsonl",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    wanted = [str(value) for value in manifest["ids"]]
    if manifest.get("n") != EXPECTED_ROWS:
        raise RuntimeError(f"Frozen manifest n must equal {EXPECTED_ROWS}")
    if len(wanted) != EXPECTED_ROWS or len(set(wanted)) != EXPECTED_ROWS:
        raise RuntimeError(
            f"Frozen manifest must contain exactly {EXPECTED_ROWS} unique IDs"
        )
    if ids_sha256(wanted) != EXPECTED_IDS_SHA256:
        raise RuntimeError("Frozen manifest IDs/order do not match the release cohort")
    rows = [
        json.loads(line)
        for line in args.source.open(encoding="utf-8")
        if line.strip()
    ]
    by_id = {str(row["id"]): row for row in rows}
    if len(rows) != 503 or len(by_id) != 503:
        raise RuntimeError(
            f"Expected the official 503-row source; found {len(rows)} rows "
            f"and {len(by_id)} unique IDs"
        )
    missing = set(wanted) - set(by_id)
    if missing:
        raise RuntimeError(f"Frozen IDs missing from source: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in wanted:
            handle.write(json.dumps(by_id[sample_id], ensure_ascii=False) + "\n")
    temporary.replace(args.out)
    output_manifest = args.out.with_name(args.out.stem + "_manifest.json")
    output_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved frozen 80-ID cohort: {args.out}")


if __name__ == "__main__":
    main()
