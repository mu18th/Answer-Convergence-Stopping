# S-NIAH with misleading needles (paper Table 3)

Tests whether ACS stops on misleading information that appears before the
decisive evidence. Every one of the paper's 250 S-NIAH documents gets three
extra RULER needles with other keys and values, all placed **before** the
queried needle (RULER's `niah_multikey_1` layout: 4 needles, 1 queried).
Nothing else changes: same documents, questions, ids and target needle.

The reader, fold and probe use the shared Qwen3-14B S-NIAH configuration
(`../sniah_ablations/config_sniah_qwen14b_base.yaml`: 24K-character chunks,
6K-character notes; replay at theta=0.995, epsilon=0.05, w=3). Trajectories
are recorded in full (`--rule none`) and every stopping rule is replayed
offline.

## Run

From this folder, with `OPENROUTER_API_KEY` set:

```bash
python -u run_sniah_distractor.py --workers 8 --prepare
python -u analyze_sniah_distractor.py
```

- `prepare_sniah_distractor.py` (called by `--prepare`) reads
  `../sniah_ablations/data/sniah.jsonl`, building it with that folder's
  `prepare_sniah.py` if missing, and writes `data/sniah_distractor.jsonl`
  plus a manifest with the source SHA-256, build id and output SHA-256. The
  build is deterministic (seed 0): the same S-NIAH file gives the same rows
  and build id.
- `run_sniah_distractor.py` records the 250 trajectories through
  `../../run_fold.py` with the verbalized and END signals. It runs only on
  the Table 3 build (3 distractors, seed 0, from the shared S-NIAH file).
  Rerunning resumes; rows from any other setting are refused through the
  execution fingerprint.
- `analyze_sniah_distractor.py` writes
  `results/sniah_distractor_qwen14b_table3.{csv,json,txt}`. When the
  complete, labelled reference arm of `../sniah_ablations` exists (run with
  `run_sniah_ablations.py --arms base24_notes6`), every question is also
  paired with its run on the same document without distractors; an absent
  or incomplete reference arm only skips the pairing. A file given with
  `--orig` must be complete.

## Rows and definitions

| Table 3 row | Replayed rule |
|---|---|
| ACS | released runtime rule, theta=0.995, epsilon=0.05, w=3 |
| Confidence alone (first confident answer) | first step, from chunk 1, whose non-abstaining answer has confidence >= theta; no window, no stability test |
| END gate | first asked END |
| END gate, 4 / 5 ENDs in a row | first step that completes 4 / 5 consecutive asked ENDs |
| Verbalized gate | first numeric self-report >= 99.5 |
| Full reading | every chunk |

Tokens are the mean per question in thousands, priced as in the main
analysis: the fold, one probe at the stop, and only the signals a rule uses.
"Stops before the evidence" counts stops in a chunk before the one holding
the queried needle. The analysis also counts stops that return a
distractor's number, and repeats the table for the questions where a
distractor sits in an earlier chunk than the target. A diagnostic row keeps
the window and turns only the stability test off.

Reported values (Qwen3-14B, 250 questions):

| Stopping rule | Tokens (K) | Stops before the evidence | Accuracy |
|---|---:|---:|---:|
| ACS | 39.1 | 1 (0.4%) | 0.984 |
| Confidence alone (first confident answer) | 29.1 | 7 (2.8%) | 0.972 |
| END gate | 23.8 | 71 (28.4%) | 0.716 |
| END gate, 4 ENDs in a row | 38.0 | 21 (8.4%) | 0.904 |
| END gate, 5 ENDs in a row | 40.9 | 16 (6.4%) | 0.924 |
| Verbalized gate | 20.8 | 101 (40.4%) | 0.596 |
| Full reading | 54.1 | 0 | 0.988 |
