# ACS against trained DCC on RULER-HotpotQA (paper Table 4)

Compares ACS with the activation-based DCC method (Xie et al., 2025) using
DCC's official implementation, default settings and evaluation protocol, on
the same backbone (Qwen3-14B, thinking disabled) and the paper's own 500
RULER-HotpotQA documents (100 questions x 8K, 16K, 32K, 64K, 128K tokens).

- **ACS** needs no new run: Table 4 replays the paper's complete Qwen3-14B
  RULER-HotpotQA trajectories (both 250-row cohorts, see `../ruler_hotpotqa/`)
  with theta=0.995, epsilon=0.05, w=3.
- **DCC** trains its sufficiency probe on a separate, source-disjoint training
  cohort: 120 official HotpotQA *train* questions rendered by RULER `qa_2` at
  the same five lengths, labelled with the exact annotated supporting-fact
  spans. The 500 evaluation documents are never used for training,
  validation or threshold selection. Questions are split 80/10/10 by source
  question (96/12/12), so no question appears in two splits.

The official DCC code is fetched at commit `337c20b` and never edited. Every
adaptation is in `dcc_overlay.py` (applied in the driver process) and
`run_dcc.py`, each with its reason:

- Qwen3 is routed to DCC's existing Qwen2 attention hook (same `o_proj`
  layout), without upstream's duplication of grouped-query heads.
- At inference, tokens are pooled inside the hook (the same mean upstream
  computes afterwards), so 128K-token prefixes fit in host memory.
- `--chat-template` wraps DCC's unchanged prompt in Qwen3's chat template
  with thinking disabled, the setting ACS uses. Without it Qwen3 reasons
  aloud and exhausts the 32-token answer budget; that first run (6% accuracy
  for every arm) was discarded.
- `--yarn-128k` applies Qwen's documented YaRN setting for 131,072 tokens in
  memory. `sdpa` attention replaces `eager`, which cannot hold 64K+ tokens.
- `--low-memory-training` (all-lengths probe only) reduces activations to
  the prefix means DCC's trainer uses, inside the hook; tested equal to
  upstream's features up to float rounding.

DCC uses 10 percentage chunks, 5 probe heads, sufficiency threshold 0.95,
greedy 32-token answers and seed 42; `run_dcc_experiment.py` fixes these.
`run_dcc.py` uses whatever device PyTorch finds (CUDA, MPS or CPU, with
`device_map="auto"`; float16 on an accelerator, float32 on CPU). Costs are
actual model-forward tokens; training cost is wall time x visible devices
(GPU-hours on GPUs), reported separately from inference.

## Run

From this folder, in DCC's own environment (its repository's
`requirement.txt`, fetched into `external/dcc_official` by the first step),
with `Qwen/Qwen3-14B` in the local Hugging Face cache.

```bash
python -u prepare_dcc_data.py        # DCC code, the paper's 500 documents, the training cohort
python -u run_dcc_experiment.py      # both DCC arms, every stage
python -u analyze_acs_vs_dcc.py --acs-traj <paper Qwen3-14B RULER-HotpotQA trajectories>
```

- `prepare_dcc_data.py` fetches DCC at the pinned commit into
  `external/dcc_official`, rebuilds the paper's evaluation documents with the
  released `../ruler_hotpotqa` helpers and converts them for DCC
  (`prepare_paper_eval_for_dcc.py`), and builds the training cohort. It
  prints the two cohort build ids; `analyze_acs_vs_dcc.py` later refuses to
  compare unless they equal the `build_id` values of the ACS trajectories.
- `run_dcc_experiment.py` runs `train_8k_32k` (the main row: train, evaluate
  full-context and chunked DCC, score) and `train_all_lengths` (extract
  features, train from the cache, evaluate, merge, score). Finished stages
  are skipped on rerun. `train_8k_32k` trains and evaluates in one process,
  so an unfinished run of it restarts from the beginning. `--arms
  train_8k_32k` runs only the main row.
- `analyze_acs_vs_dcc.py` writes `results/acs_vs_dcc_table4.{csv,json,txt}`.

The all-lengths stages can be split across processes, for example two:

```bash
python -u run_dcc_experiment.py --arms train_all_lengths --stage extract --shard-index 0 --shard-count 2
python -u run_dcc_experiment.py --arms train_all_lengths --stage extract --shard-index 1 --shard-count 2
python -u run_dcc_experiment.py --arms train_all_lengths --stage train --shard-count 2
python -u run_dcc_experiment.py --arms train_all_lengths --stage evaluate --shard-index 0 --shard-count 2
python -u run_dcc_experiment.py --arms train_all_lengths --stage evaluate --shard-index 1 --shard-count 2
python -u run_dcc_experiment.py --arms train_all_lengths --stage score --shard-count 2
```

If a training document does not fit the 131,072-token window, `run_dcc.py`
stops; rerunning it with `--drop-nonfitting-train-questions` drops that
question at every length and lists it in the run manifest.

## Table 4

`analyze_acs_vs_dcc.py` prints accuracy by length for ACS and DCC, inference
tokens per document, early-stop counts, the paired ACS-minus-DCC difference
with a 95% bootstrap interval (10,000 resamples, seed 0) and an exact
McNemar p-value, and each probe's training cost.

Before computing anything it refuses a comparison that is not Table 4:

- The trajectories must be exactly the 500 documents DCC was evaluated on
  (`data/paper_eval/ruler_hotpotqa_paper500.jsonl`): same ids, questions,
  answers and paper build ids.
- Every `dcc_run_manifest.json` of a DCC run (for the all-lengths arm, the
  training folder and every evaluation shard) must record the pinned DCC
  commit, unmodified DCC files, that same evaluation file, the settings
  `run_dcc_experiment.py` passes (chat template, YaRN 128K, 10 chunks,
  threshold 0.95, 32 new tokens, seed 42, `sdpa`) and the arm's training
  lengths.

DCC cannot read documents whose prompt exceeds the context window (6 of the
500; DCC reads 494). DCC accuracy is reported on the documents it read and,
separately, with the unread ones counted as wrong; ACS is reported on all 500
and on the 494. The paired tests use only documents both methods read.

## Provenance

Each run folder holds `dcc_run_manifest.json`: data and training-set hashes,
the question-grouped split ids, the DCC commit, the device, whether the chat
template, YaRN and low-memory training were on,
and measured costs. `analyze_acs_vs_dcc.py` checks the settings above
against these fields.

Not part of Table 4, and therefore not included: the DCC self-prompting
reconstruction, the one-call direct-context arm, and GRU-Mem (no public code
or trained weights).
