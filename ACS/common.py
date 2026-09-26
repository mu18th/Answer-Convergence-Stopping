"""ACS shared client, strict probes, scoring, I/O, and statistics.

Candidate posteriors are accepted only when every option has a measured score;
incomplete instrumentation raises instead of inventing probabilities.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
import yaml
from openai import OpenAI
from draft_cleaner import is_non_answer as _is_non_answer

# Public re-export used by runners and analyzers.
is_non_answer = _is_non_answer

LETTERS = ["A", "B", "C", "D"]

# OpenRouter can temporarily exhaust the upstream provider's shared quota even
# at low local concurrency.  Fifteen total 429 attempts with exponential
# backoff wait for roughly 11 minutes before abandoning the call.  Other error
# error types use the configured ordinary retry budget.
OPENROUTER_429_ATTEMPTS = 15
OPENROUTER_429_MAX_DELAY_S = 60


def load_config(path):
    """Load a required config and resolve its local cache paths deterministically."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    root = Path(path).resolve().parent

    def _abs(v):
        return str((root / os.path.expandvars(os.path.expanduser(str(v)))).resolve())

    s = cfg["server"]
    for k in ("cache_dir", "scratch_dir"):
        if k in s:
            s[k] = _abs(s[k])
    # Allow the launcher to select a model without modifying the frozen config.
    s["model"] = os.environ.get("MODEL", s["model"])
    # Never place a paid OpenRouter credential in YAML or source control.
    # OPENROUTER_API_KEY is the only accepted source for this bundle.
    if "openrouter.ai" in s["base_url"]:
        s["api_key"] = os.environ.get("OPENROUTER_API_KEY", "")
        # Hugging Face incorporates the cache path into lock filenames. A cache
        # nested below a long Windows project path can exceed MAX_PATH before the
        # dataset is downloaded. Default to a short per-user persistent location.
        cache_override = os.environ.get("ACS_CACHE_DIR")
        if cache_override:
            s["cache_dir"] = str(
                Path(os.path.expandvars(os.path.expanduser(cache_override))).resolve()
            )
        elif os.name == "nt":
            local_data = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP")
            if local_data:
                s["cache_dir"] = str(Path(local_data) / "ACSHF")
        Path(s["cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def make_client(cfg):
    server = cfg["server"]
    if "openrouter.ai" in server["base_url"] and not server.get("api_key"):
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Export it before running the benchmark."
        )
    headers = {}
    if "openrouter.ai" in server["base_url"]:
        referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.environ.get("OPENROUTER_APP_TITLE", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-OpenRouter-Title"] = title
    return OpenAI(
        base_url=server["base_url"],
        api_key=server["api_key"],
        default_headers=headers or None,
        timeout=600.0,
    )


def chat(client, model, system, user, *, max_tokens, logprobs=False, retries=4,
         temperature=0.0, top_p=None, top_k=None, choices=None):
    """Deterministic by default. temperature/top_p/top_k exist only so the judge can
    reproduce the official BrowseComp-Plus evaluator's sampling settings; every probe
    and fold call must stay at temperature 0."""
    is_openrouter = "openrouter.ai" in str(getattr(client, "base_url", ""))
    if is_openrouter:
        # Qwen3.5-397B-A17B exposes optional reasoning on OpenRouter. Disable it
        # explicitly: this experiment measures the non-thinking model and does
        # not reserve or bill a hidden reasoning budget.
        provider = {
            "allow_fallbacks": False,
            "require_parameters": True,
        }
        # Optional, explicit pin only.  There is deliberately no guessed default:
        # a provider advertised for the model may not support this request's full
        # parameter combination.  OpenRouter chooses a compatible endpoint when
        # OPENROUTER_PROVIDER is unset.
        provider_name = os.environ.get("OPENROUTER_PROVIDER", "").strip()
        if provider_name:
            provider["only"] = [provider_name]
        extra = {
            "reasoning": {"enabled": False},
            # Do not let OpenRouter silently retry an error on another provider.
            # Keep retries in this client rather than allowing provider fallback.
            "provider": provider,
        }
    else:
        extra = {"chat_template_kwargs": {"enable_thinking": False}}
    if choices:
        # Constrain the output to exactly these strings so every option is scored
        # and none can fall outside the returned top-k. vLLM >= 0.11 removed
        # `guided_choice` in favour of structured_outputs={"choice": [...]}; on
        # such a server the first call raises and the retry below switches keys.
        # If neither key constrains, posterior_from_logprobs raises ProbeInvalid
        # rather than letting the run quietly fabricate probabilities again.
        if is_openrouter:
            # OpenRouter supports provider-normalized strict JSON Schema, whereas
            # guided_choice/structured_outputs are vLLM-specific request fields.
            # A string enum keeps the choice set finite. posterior_from_logprobs
            # scans past the opening JSON quote to the constrained letter token.
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ACS_choice",
                    "strict": True,
                    "schema": {"type": "string", "enum": list(choices)},
                },
            }
        else:
            extra["guided_choice"] = list(choices)
    if top_k is not None:          # vLLM takes top_k through extra_body, not the API
        extra["top_k"] = top_k
    last = None
    # Each constraint key needs its OWN retry budget. Sharing one budget means the
    # keys alternate and each gets retries/2 attempts, so two transient failures on
    # the supported key can end the call having spent its last attempt on the
    # unsupported one.
    attempt_count = retries if is_openrouter else (retries * 2 if choices else retries)
    ordinary_failures = 0
    rate_limit_failures = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": user}],
                # Preserve the caller's requested completion ceiling exactly.
                temperature=temperature, max_tokens=max_tokens,
                **({"top_p": top_p} if top_p is not None else {}),
                # The MCQ token is constrained to four candidates, so top-5 is
                # sufficient and remains compatible with providers that cap this
                # field at five. Open-answer confidence uses only each generated
                # token's own `logprob`; alternatives are ignored.
                logprobs=logprobs,
                top_logprobs=5 if logprobs else None,
                extra_body=extra)
            # OpenRouter documents `usage` as present on every non-streaming
            # response.  A response without it cannot be accounted honestly.
            # Treat that incomplete response exactly like the API failures
            # already handled by this retry loop; never turn it into zero cost.
            if is_openrouter and getattr(response, "usage", None) is None:
                raise RuntimeError(
                    "OpenRouter response omitted required usage metadata"
                )
            return response
        except Exception as e:  # noqa: BLE001
            last = e
            status_code = getattr(e, "status_code", None)
            is_rate_limit = (
                is_openrouter
                and (status_code == 429 or "Error code: 429" in str(e))
            )
            if is_rate_limit:
                rate_limit_failures += 1
                if rate_limit_failures >= OPENROUTER_429_ATTEMPTS:
                    break
                delay = min(
                    5 * (2 ** (rate_limit_failures - 1)),
                    OPENROUTER_429_MAX_DELAY_S,
                )
                print(
                    "OpenRouter upstream 429; "
                    f"retry {rate_limit_failures + 1}/"
                    f"{OPENROUTER_429_ATTEMPTS} in {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                continue

            ordinary_failures += 1
            if ordinary_failures >= attempt_count:
                break
            if choices and not is_openrouter:
                # Alternate the constraint key between retries: vLLM < 0.11 only
                # knows `guided_choice`, vLLM >= 0.11 only knows structured_outputs
                # and rejects the old one. Alternating covers both without sniffing
                # the version, and covers a transient error under either. Dropping
                # the constraint instead would silently reintroduce the fabricated-
                # probability bug, so it is never dropped.
                if "guided_choice" in extra:
                    extra.pop("guided_choice")
                    extra["structured_outputs"] = {"choice": list(choices)}
                else:
                    extra.pop("structured_outputs", None)
                    extra["guided_choice"] = list(choices)
            time.sleep(min(2 ** (ordinary_failures - 1), 15))
    raise RuntimeError(f"LLM call failed: {last}")


def usage(resp):
    token_usage = getattr(resp, "usage", None)
    if token_usage is None:
        raise RuntimeError("LLM response omitted required usage metadata")
    prompt_tokens = getattr(token_usage, "prompt_tokens", None)
    completion_tokens = getattr(token_usage, "completion_tokens", None)
    if prompt_tokens is None or completion_tokens is None:
        raise RuntimeError("LLM response returned incomplete usage metadata")
    return int(prompt_tokens) + int(completion_tokens)


# ---- measured probe -----------------------------------------------------------
class ProbeInvalid(RuntimeError):
    """The probe did not return a measured score for every candidate.

    Raised instead of substituting a floor. A letter absent from the returned
    top-k has no measured probability, and inventing one is not a measurement.
    """


# Two strings no model emits unprompted. Used to prove the constraint is applied.
_CONSTRAINT_SENTINELS = ["ZQXJ_ALPHA", "ZQXJ_BETA"]


def verify_constrained_decoding(client, model):
    """Prove the server actually APPLIES the decoding constraint. Call once per run.

    A complete set of A-D logprobs does not prove it. An unconstrained probe with
    an unconstrained top-k can contain all four letters anyway, so
    posterior_from_logprobs sees nothing missing, never raises, and returns a
    plausible normalized posterior built from unconstrained scores. That is
    exactly the fabricated-measurement failure the ProbeInvalid guard exists to
    stop, wearing a valid disguise.

    So ask for one of two strings the model would never produce on its own. If the
    reply is not one of them, the constraint is being ignored and every posterior
    this run would write is unconstrained.
    """
    # JSON-string providers may spend tokens on quotes and tokenizer fragments;
    # leave enough room for the complete sentinel before validating it.
    resp = chat(client, model, "", "Reply with exactly one of the allowed strings.",
                max_tokens=32, choices=_CONSTRAINT_SENTINELS)
    got = (resp.choices[0].message.content or "").strip()
    # OpenRouter's strict string enum is valid JSON, so its visible content can
    # be quoted ("ZQXJ_ALPHA"). Normalize only for this preflight comparison.
    try:
        parsed = json.loads(got)
        if isinstance(parsed, str):
            got = parsed
    except Exception:  # noqa: BLE001
        pass
    if got not in _CONSTRAINT_SENTINELS:
        raise ProbeInvalid(
            f"constrained decoding is NOT in effect on this server: asked for one of "
            f"{_CONSTRAINT_SENTINELS}, got {got!r}. Completeness checks cannot catch "
            f"this — an unconstrained top-k may contain all of A/B/C/D, so the "
            f"run would produce plausible but unconstrained posteriors. Refusing to "
            f"start.")
    # An endpoint can enforce the enum yet expose logprobs only for a wrapper or
    # EOS token. ACS needs one measured position containing every A-D
    # candidate, so verify that instrumentation contract before the paid run.
    measured = chat(
        client, model, "", "Reply with exactly one allowed letter.",
        max_tokens=32, logprobs=True, choices=LETTERS,
    )
    try:
        posterior_from_logprobs(measured)
    except ProbeInvalid as error:
        raise ProbeInvalid(
            "constrained decoding is active, but this endpoint does not expose "
            "complete A-D logprobs; it cannot run the measured MCQ controller. "
            f"Provider diagnostic: {error}"
        ) from error
    return True


def posterior_from_logprobs(resp):
    """Renormalize the first token's top-logprobs over A-D.

    Every letter must carry a measured logprob or this raises ProbeInvalid. That
    is guaranteed by passing choices=LETTERS to chat(), which constrains the probe
    to emit exactly one of A/B/C/D so all four are always scored. The raise is
    also the tripwire for constrained decoding silently going away — see chat().

    No missing-label floor exists. Incomplete provider output is an instrumentation
    failure, not a probability distribution, and therefore raises ProbeInvalid.
    """
    # A local guided-choice response starts directly with A-D. OpenRouter's
    # strict JSON string enum starts with a quote and then emits the letter.
    # Scan the complete generated sequence and use the first position that
    # measures all four candidates; never combine scores across positions.
    try:
        positions = resp.choices[0].logprobs.content
    except Exception:  # noqa: BLE001
        positions = []
    # Models really do emit "(A", "**A", "A)", "A." — strip punctuation only;
    # scores still must come from one measured token position.
    strip_chars = "()[]*\"'`.:,"
    logps = {letter: -math.inf for letter in LETTERS}
    # Provider wrappers may return positions before the constrained JSON letter.
    # Scan the complete sequence, but accept a position only when that *single
    # position* measures every A-D candidate.
    for position in positions:
        candidate = {letter: -math.inf for letter in LETTERS}
        for c in (position.top_logprobs or []):
            tok = (c.token or "").strip().strip(strip_chars).upper()
            if tok in candidate:
                candidate[tok] = max(candidate[tok], float(c.logprob))
        if all(value > -math.inf for value in candidate.values()):
            logps = candidate
            break
    if all(v == -math.inf for v in logps.values()):
        # Keep this diagnostic prompt-free. It exposes only generated token
        # strings, which is enough to distinguish provider truncation from a
        # tokenizer/constraint mismatch if a future endpoint changes.
        tail = []
        for index, position in list(enumerate(positions))[-8:]:
            tail.append({
                "position": index,
                "token": getattr(position, "token", None),
                "top_tokens": [
                    getattr(candidate, "token", None)
                    for candidate in (
                        getattr(position, "top_logprobs", None) or []
                    )
                ],
            })
        raise ProbeInvalid(
            "probe returned no single position containing logprobs for all "
            f"of A/B/C/D; positions={len(positions)}, tail={tail!r}")
    n_missing = sum(1 for v in logps.values() if v == -math.inf)
    if n_missing:
        raise ProbeInvalid(
            f"probe returned {n_missing} unscored letters of {len(LETTERS)}; "
            f"constrained decoding is not in effect on this server")
    mx = max(logps.values())
    e = {letter: math.exp(value - mx) for letter, value in logps.items()}
    z = sum(e.values())
    return {letter: e[letter] / z for letter in LETTERS}

def jsd(p, q):
    m = {k: 0.5 * (p[k] + q[k]) for k in p}
    def kl(a, b):
        return sum(a[k] * math.log((a[k] + 1e-12) / (b[k] + 1e-12)) for k in a)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---- answer extraction and scoring --------------------------------------------
def strip_think(text):
    """Drop reasoning blocks so grading sees the final answer, not scratchpad text."""
    if not text:
        return ""
    t = re.sub(r"<think>.*?</think>", " ", text, flags=re.S)
    return t.split("</think>")[-1].strip() if "</think>" in t else t.strip()


def _clean(s):
    s = s.lower()
    for pre in ("label:", "answer:", "relation:"):
        s = s.replace(pre, " ")
    return " ".join(s.replace('"', "").replace("'", "").split())


def f1_tokens(pred, gold):
    p, g = _clean(pred).split(), _clean(gold).split()
    if not p or not g:
        return 0.0
    # Counter intersection is the same bag overlap the loop-with-.count() computed,
    # in O(n) instead of O(n^2): 227ms -> 0.7ms on a 4096-token draft.
    from collections import Counter
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


# ---- io -----------------------------------------------------------------------
def read_jsonl(p):
    return [
        json.loads(line)
        for line in Path(p).open(encoding="utf-8")
        if line.strip()
    ]


def append_jsonl(p, rec):
    with Path(p).open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resume_ids(out_path, retry_failed=True):
    """Ids to skip. With retry_failed, failed records are retried on rerun."""
    done = set()
    if Path(out_path).exists():
        for line in Path(out_path).open(encoding="utf-8"):
            try:
                r = json.loads(line)
                if not (retry_failed and r.get("failed")):
                    done.add(r["id"])
            except Exception:  # noqa: BLE001
                pass
    return done


def dedupe_prefer_success(recs):
    by = {}
    for r in recs:
        if r["id"] not in by or by[r["id"]].get("failed"):
            by[r["id"]] = r
    return list(by.values())


def is_dev(qid):
    return int(hashlib.md5(str(qid).encode()).hexdigest(), 16) % 2 == 0


# ---- statistics ----------------------------------------------------------------
def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return float("nan")
    rs, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rs += (i + j + 1) / 2.0 * sum(label for _, label in pairs[i:j])
        i = j
    return (rs - pos * (pos + 1) / 2) / (pos * neg)


def ece(scores, labels, bins=10):
    total, err = len(scores), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, s in enumerate(scores)
               if lo <= s < hi or (b == bins - 1 and s == 1.0)]
        if idx:
            conf = sum(scores[i] for i in idx) / len(idx)
            acc = sum(labels[i] for i in idx) / len(idx)
            err += len(idx) / total * abs(conf - acc)
    return err
