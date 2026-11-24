"""Driver-level overlay for the pinned official DCC code (external/dcc_official).

The upstream files stay byte-identical to github.com/ruoyuxie/when-to-stop
(commit 337c20b). This module patches two methods of ActivationManager at
import time in the driver process only. Every change is listed here with the
reason, so the comparison can be audited.

1. _detect_model_type
   Upstream keys on the class name containing "qwen2". Qwen3ForCausalLM
   therefore falls into the generic "llama" branch, which registers a hook on
   every module whose name contains "attn" (self_attn, q_proj, k_proj,
   v_proj, o_proj) and keeps whichever fires last. The patched detector
   returns "qwen2" for any Qwen class so the single o_proj hook upstream wrote
   for Qwen2 is used for Qwen3 as well (same attention layout: o_proj,
   num_key_value_heads).

2. _qwen2_process_activation
   (a) Upstream reshapes the o_proj output into num_heads slices and then
       repeat_interleave()s them by num_heads // kv_heads, which for a 40-head
       8-kv-head model stores 200 duplicated "heads" per layer (5x the memory
       and 5x the probe candidates, all duplicates). The o_proj output is not
       grouped by kv head, so the duplication carries no information. The
       patch keeps the num_heads x head_dim view and drops the duplication.
   (b) When manager.reduce_tokens is True (set by the driver for inference
       only), the hook returns the mean over the sequence dimension with
       keepdim, shape (batch, 1, heads, head_dim). Upstream's inference path
       (get_head_features_for_eval) takes exactly this mean afterwards, so the
       probe input is unchanged; what changes is that a 128K-token prefix no
       longer materialises seq x layers x hidden fp32 tensors on the CPU.
       Training keeps reduce_tokens False because get_chunk_features needs
       cumulative prefix means at every chunk boundary.
   (c) When manager.prefix_chunks = N (set by the driver for probe TRAINING
       with --low-memory-training), the hook returns the mean over the
       cumulative prefix at each of the N percentage chunk ends, shape
       (batch, N, heads, head_dim). These are the values upstream's
       get_chunk_features computes afterwards from the stored per-token
       activations (layer[:end].sum(0) / end). The chunk ends come from
       upstream's own ChunkingManager._create_percentage_chunks. Sums are
       accumulated in float64 and stored as float32, so the features equal
       upstream's up to float rounding. Without this, one 128K-token document
       needs about 107 GB of host RAM in training (tokens x layers x hidden,
       float32), which is why the first probe was trained on 8K to 32K only.

3. get_chunk_features, compute_activations (wrappers, upstream called inside)
   With prefix_chunks set, get_chunk_features reads the features straight
   from the reduced tensors after checking that the chunk ends it is asked
   for are the ones the hook reduced at. compute_activations records those
   chunk ends on the returned object and, when manager.feature_cache_dir is
   set, saves or loads the reduced tensors of a document (keyed by the SHA-256
   of its full model input), so several processes can extract features in parallel
   and the training process can reuse them. With prefix_chunks unset, both
   wrappers call upstream unchanged.

FeatureForward
   Stand-in for the model during training feature extraction only. Upstream
   calls model(**inputs) and throws the output away; the hooks on o_proj are
   all it needs. The stand-in runs the same decoder stack without the LM head
   and without a KV cache (at 128K tokens: about 40 GB of logits and 21 GB of
   cache that nothing reads). The hooked activations are unchanged.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def percentage_boundaries(total_tokens: int, num_chunks: int) -> list[int]:
    """Chunk ends for `total_tokens`, from upstream's own percentage chunker."""
    from chunking_manager import ChunkingManager
    stub = SimpleNamespace(model_token_ids=[range(int(total_tokens))])
    boundaries, _ = ChunkingManager._create_percentage_chunks(None, stub, int(num_chunks))
    return [int(b) for b in boundaries]


class FeatureForward:
    """The model's decoder stack without LM head and KV cache (see module docstring)."""

    def __init__(self, causal_lm):
        self._lm = causal_lm
        self._base = getattr(causal_lm, causal_lm.base_model_prefix)

    def __call__(self, **inputs):
        return self._base(**inputs, use_cache=False)

    def __getattr__(self, name):
        return getattr(self._lm, name)


def apply(activation_manager_cls) -> None:
    upstream_get_chunk_features = activation_manager_cls.get_chunk_features
    upstream_compute_activations = activation_manager_cls.compute_activations
    am_module = sys.modules[activation_manager_cls.__module__]
    def _detect_model_type(self) -> str:
        name = self.model.__class__.__name__.lower()
        if "phi" in name:
            return "phi"
        if "qwen" in name:
            return "qwen2"
        return "llama"

    def _qwen2_process_activation(self, activation):
        try:
            activation = self._validate_and_sanitize_activation(
                activation, "qwen2_process_activation")
            if activation is None or len(activation.shape) != 3:
                return None
            batch_size, seq_len, hidden_dim = activation.shape
            if hidden_dim != self.hidden_size:
                print(f"Warning: Unexpected hidden dimension. Expected {self.hidden_size}, got {hidden_dim}")
                return None
            processed = activation.view(batch_size, seq_len, self.num_heads, self.head_dim)
            n_prefix = int(getattr(self, "prefix_chunks", 0) or 0)
            if n_prefix:
                if batch_size != 1:
                    raise RuntimeError("prefix reduction expects one document per forward")
                boundaries = percentage_boundaries(seq_len, n_prefix)
                if boundaries[0] <= 0 or boundaries[-1] != seq_len:
                    raise RuntimeError(f"bad chunk ends {boundaries} for {seq_len} tokens")
                running = torch.zeros(self.num_heads, self.head_dim,
                                      dtype=torch.float64, device=processed.device)
                means, start = [], 0
                for end in boundaries:
                    running = running + processed[0, start:end].sum(dim=0, dtype=torch.float64)
                    means.append(running / end)          # mean over tokens [0, end)
                    start = end
                processed = torch.stack(means).to(torch.float32).unsqueeze(0)
                self._prefix_boundaries = boundaries
            elif getattr(self, "reduce_tokens", False):
                processed = processed.mean(dim=1, keepdim=True)
            return processed.detach().cpu()
        except Exception as error:  # mirror upstream behaviour: log, return None
            print(f"Error processing Qwen2 activation: {error}")
            return None

    def get_chunk_features(self, context_activations, chunk_boundaries):
        n_prefix = int(getattr(self, "prefix_chunks", 0) or 0)
        if not n_prefix:
            return upstream_get_chunk_features(self, context_activations, chunk_boundaries)
        reduced_at = getattr(context_activations, "prefix_boundaries", None)
        if reduced_at is None or [int(b) for b in chunk_boundaries] != list(reduced_at):
            raise RuntimeError(
                f"Chunk ends {list(chunk_boundaries)} are not the ones the activations were "
                f"reduced at ({reduced_at}); refusing to build features")
        layers = context_activations.attention_activations
        expected_layers = int(self.model.config.num_hidden_layers)
        if len(layers) != expected_layers:
            raise RuntimeError(f"Reduced activations for {len(layers)} layers, expected {expected_layers}")
        features = [{} for _ in chunk_boundaries]
        for layer_name, layer_tensor in layers.items():
            if tuple(layer_tensor.shape) != (len(chunk_boundaries), self.num_heads, self.head_dim):
                raise RuntimeError(f"{layer_name}: unexpected reduced shape {tuple(layer_tensor.shape)}")
            layer_num = layer_name.split('_')[1]            # same key naming as upstream
            values = layer_tensor.cpu().numpy()
            for chunk_idx in range(len(chunk_boundaries)):
                for head_idx in range(values.shape[1]):
                    features[chunk_idx][f"layer_{layer_num}_head_{head_idx}"] = \
                        values[chunk_idx, head_idx].copy()
        return features

    def compute_activations(self, query, context, gold_information):
        n_prefix = int(getattr(self, "prefix_chunks", 0) or 0)
        if not n_prefix:
            return upstream_compute_activations(self, query, context, gold_information)
        cache_dir = getattr(self, "feature_cache_dir", None)
        text = am_module.prepare_sufficiency_input(query, context)
        cache_path = None
        if cache_dir:
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            cache_path = Path(cache_dir) / f"{digest}.pt"
        if cache_path is not None and cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            token_ids = self.model_tokenizer(text, return_tensors="pt", padding=True).input_ids
            if int(payload["n_chunks"]) != n_prefix or int(payload["n_tokens"]) != int(token_ids.shape[1]):
                raise RuntimeError(f"Feature cache entry {cache_path.name} does not match this document")
            cached = am_module.ContextActivations(
                full_context=text,
                model_token_ids=token_ids.cpu(),
                attention_activations=dict(payload["activations"]),
                gold_location=self.find_gold_location(text, gold_information),
            )
            cached.prefix_boundaries = [int(b) for b in payload["boundaries"]]
            self.feature_cache_hits = getattr(self, "feature_cache_hits", 0) + 1
            return cached
        self._prefix_boundaries = None
        computed = upstream_compute_activations(self, query, context, gold_information)
        if self._prefix_boundaries is None:
            raise RuntimeError("The activation hook did not run in prefix mode")
        computed.prefix_boundaries = list(self._prefix_boundaries)
        # A hook that failed (for example out of GPU memory) logs and stores nothing, as upstream
        # does. Stop here, before a document with missing layers reaches the cache or the probe.
        expected_layers = int(self.model.config.num_hidden_layers)
        expected_shape = (n_prefix, self.num_heads, self.head_dim)
        bad = [name for name, tensor in computed.attention_activations.items()
               if tuple(tensor.shape) != expected_shape]
        if len(computed.attention_activations) != expected_layers or bad:
            raise RuntimeError(
                f"Feature extraction incomplete: {len(computed.attention_activations)} of "
                f"{expected_layers} layers, wrong shape in {bad[:3]}; see the hook error above")
        self.feature_cache_misses = getattr(self, "feature_cache_misses", 0) + 1
        if cache_path is not None:
            temporary = cache_path.with_suffix(".tmp")
            torch.save({
                "n_tokens": int(computed.model_token_ids.shape[1]),
                "n_chunks": n_prefix,
                "boundaries": list(computed.prefix_boundaries),
                "activations": dict(computed.attention_activations),
            }, temporary)
            temporary.replace(cache_path)
        return computed

    activation_manager_cls._detect_model_type = _detect_model_type
    activation_manager_cls._qwen2_process_activation = _qwen2_process_activation
    activation_manager_cls.get_chunk_features = get_chunk_features
    activation_manager_cls.compute_activations = compute_activations
    activation_manager_cls.overlay_applied = True


def apply_chat_template(tokenizer, modules) -> dict:
    """Wrap DCC's prompt in the model's own chat template with thinking disabled.

    Why: upstream DCC feeds a raw completion prompt. Qwen3 answers a raw prompt by
    reasoning aloud ("Okay, let's see. The user is asking ..."), which uses up the
    answer budget before any answer appears (first run: 6% accuracy for both
    full-context and chunked DCC, at every length). ACS runs the same model through
    the chat template with thinking disabled, so DCC gets the same setting.

    How: the template is rendered once around a sentinel and split into the text
    before and after the user content. prepare_sufficiency_input() gets the prefix,
    get_suffix_prompt() gets DCC's own suffix followed by the template's closing and
    assistant header. DCC's instruction, query and context text are unchanged, and
    the wrapper is applied to training and inference alike, so the probe is trained
    on the same prompts it is evaluated on.
    """
    sentinel = "␞DCC_USER_CONTENT␞"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": sentinel}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    if rendered.count(sentinel) != 1:
        raise RuntimeError("Chat template did not render the user content exactly once")
    prefix, postfix = rendered.split(sentinel)
    utils = modules["utils"]
    original_prepare, original_suffix = utils.prepare_sufficiency_input, utils.get_suffix_prompt

    def prepare_sufficiency_input(query: str, context: str) -> str:
        return prefix + original_prepare(query, context)

    def get_suffix_prompt() -> str:
        return original_suffix() + postfix

    for module in modules.values():
        if hasattr(module, "prepare_sufficiency_input"):
            module.prepare_sufficiency_input = prepare_sufficiency_input
        if hasattr(module, "get_suffix_prompt"):
            module.get_suffix_prompt = get_suffix_prompt
    return {"prefix": prefix, "postfix": postfix}


def apply_full_context_decode_fix(early_output_model_cls, early_stopping_module) -> None:
    """Decode only the newly generated tokens in full-context mode.

    Upstream decodes prompt + answer with skip_special_tokens=True and slices the
    string at len(prompt). Once the prompt contains chat-template special tokens,
    the decoded prompt is shorter than the prompt string and the slice cuts into
    the answer. Slicing by token count returns exactly the generated answer, which
    is what upstream intends and what its chunked path already does.
    """
    import torch

    def _full_context_generate(self, query, context, task_name, example, generation_kwargs):
        text = (early_stopping_module.prepare_sufficiency_input(query, context)
                + early_stopping_module.get_suffix_prompt())
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.size(1):], skip_special_tokens=True)
        del inputs, outputs
        early_stopping_module.clear_gpu_memory()
        return {
            "generated_text": generated_text, "sufficient_chunk_idx": 0, "chunks_processed": 1,
            "sufficiency_confidence": 0, "used_chunk": context, "early_stopped": False,
            "total_chunks": 1, "pred_percentage": 1, "actual_gold_percentage": 1,
            "percentage_diff": 1,
        }

    early_output_model_cls._full_context_generate = _full_context_generate
