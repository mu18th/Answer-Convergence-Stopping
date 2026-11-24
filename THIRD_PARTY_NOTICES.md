# Third-party notices

This package interoperates with the following projects and datasets. Their
licenses and terms continue to apply.

- NVIDIA/RULER, Apache-2.0: <https://github.com/NVIDIA/RULER>
- HotpotQA: <https://github.com/hotpotqa/hotpot>
- BrowseComp-Plus code and datasets, MIT:
  <https://github.com/texttron/BrowseComp-Plus> and
  `Tevatron/browsecomp-plus`
- LongBench-v2: `THUDM/LongBench-v2`
- Hugging Face Datasets: <https://github.com/huggingface/datasets>
- vLLM: <https://github.com/vllm-project/vllm>
- DCC, "Knowing When to Stop" (Xie et al., 2025), Apache-2.0:
  <https://github.com/ruoyuxie/when-to-stop>. Fetched at commit `337c20b`
  by `ACS/experiments/ruler_hotpotqa_dcc/prepare_dcc_data.py`; its files are
  not edited, and the Qwen3 adaptations are applied at runtime by
  `dcc_overlay.py`. It is not redistributed in this archive.
- HotpotQA train v1.1 (CC BY-SA 4.0), downloaded during data preparation and
  accepted only if its SHA-256 matches the official file.

BrowseComp-Plus queries and answers are decrypted only during local data
preparation. Decrypted benchmark text is deliberately excluded from this
archive.

