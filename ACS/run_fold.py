"""ACS chunked reader and trajectory recorder.

The controller is the windowed confidence-and-stability rule over the
measured contextual answer distribution. ``rule: none`` records complete
trajectories for paired offline replay.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, re, time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from draft_cleaner import normalize_draft
from common import (append_jsonl, chat, f1_tokens, is_non_answer, load_config,
                    make_client, jsd, posterior_from_logprobs,
                    ProbeInvalid, read_jsonl, resume_ids, usage,
                    verify_constrained_decoding)

_WLOCK = threading.Lock()   # append_jsonl is not atomic across threads

# Stopping configuration. The ``online_stop`` config block may override these
# defaults. ``rule: none`` records full trajectories for paired offline replay.
STOP = {"rule": "windowed", "theta": 0.995, "eps": 0.05, "window": 3,
        "min_chunks": 2,
        "record_verbalized": False, "record_gate": False}

# Numeric self-reported confidence and the direct END/CONTINUE decision are
# recorded as distinct comparator signals.
VERB = ("QUESTION:\n{qblock}\n\nNOTES SO FAR:\n{notes}\n\nHow confident are you "
        "(0-100) that the notes are sufficient to answer correctly? "
        "Reply with ONLY a number.\nConfidence:")

VERBALIZED_PROMPT = (
    "QUESTION:\n{qblock}\n\nNOTES SO FAR:\n{notes}\n\n"
    "Decide whether the notes contain enough information to answer the question. "
    "ONLY when enough information is collected, return <next>end</next>. "
    "Otherwise return <next>continue</next>.\nDecision:")


def parse_gate_response(text):
    """Parse only an exact bare or XML-wrapped END/CONTINUE decision."""
    value = (text or "").strip()
    try:
        decoded = json.loads(value)
        if isinstance(decoded, str):
            value = decoded.strip()
    except Exception:  # noqa: BLE001 - non-JSON text is the normal case
        pass
    match = re.fullmatch(
        r"(?:<next>\s*(end|continue)\s*</next>|(end|continue))",
        value,
        flags=re.IGNORECASE,
    )
    if not match:
        # Some reasoning models obey the requested tagged decision first, then
        # append an explanation. The leading tag is still an unambiguous raw
        # END/CONTINUE decision; do not search later prose for a guess.
        match = re.match(
            r"^<next>\s*(end|continue)\s*</next>(?:\s+[\s\S]+)$",
            value,
            flags=re.IGNORECASE,
        )
    if not match:
        return None
    return (match.group(1) or (match.group(2) if match.lastindex >= 2 else None)).lower() == "end"

FOLD_SYS_MCQ = ("You maintain running NOTES that gather every piece of information "
                "relevant to answering a multiple-choice question about a long "
                "document, read chunk by chunk. Update the notes with relevant facts "
                "from the new chunk, keep prior facts, stay under {cap} characters, "
                "output ONLY the updated notes.")

FOLD_SYS_OPEN = (
    "You maintain compact running NOTES needed to answer an open-ended "
    "question about a long context read chunk by chunk. Update and rewrite "
    "the notes using relevant information from the new chunk while preserving "
    "useful evidence from earlier chunks. Adapt the working state to the "
    "question: preserve exact names, facts, numbers, relationships, unresolved "
    "candidates, and contradictions; maintain counts or calculations for "
    "aggregation and evidence chains for multi-step questions. Remove only "
    "irrelevant or superseded material. Do not narrate the reading process, "
    "make unsupported guesses, or treat missing evidence as a negative answer. "
    "Place the most important current state near the end, stay under {cap} "
    "characters, and output ONLY the updated notes."
)

FOLD_USER = (
    "QUESTION:\n{qblock}\n\n"
    "CURRENT NOTES:\n{notes}\n\n"
    "NEW CHUNK ({i}/{n}):\n{chunk}\n\n"
    "UPDATED NOTES:"
)


PROBE_MCQ = ("QUESTION:\n{qblock}\n\nNOTES SO FAR:\n{notes}\n\nBased only on the "
             "notes, answer with a single letter (A, B, C, or D).\nAnswer:")

PROBE_OPEN = (
    "QUESTION:\n{qblock}\n\n"
    "NOTES SO FAR:\n{notes}\n\n"
    "Based only on the notes, give your best current answer. "
    "Reply with ONLY the answer.\n"
    "Answer:"
)

PROTOCOL_ID = hashlib.sha256(json.dumps({
    "controller": "windowed_mean_adjacent_v3_all_open_states_confident",
    "notes_retention": "rolling_tail",
    "mcq_measurement": "strict_constrained_exact_candidates",
    "optional_signal_parsing": "strict_with_invalid_marker_v2",
    "non_answer_cleaner": "conservative_explicit_abstention_v2",
    "prompts": {
        "fold_mcq": FOLD_SYS_MCQ,
        "fold_open": FOLD_SYS_OPEN,
        "fold_user": FOLD_USER,
        "probe_mcq": PROBE_MCQ,
        "probe_open": PROBE_OPEN,
        "numeric_confidence": VERB,
        "asked_gate": VERBALIZED_PROMPT,
    },
}, sort_keys=True).encode()).hexdigest()[:16]

EXECUTION_FINGERPRINT_VERSION = 1


def execution_config(cfg, stop_cfg=None):
    """Canonical, non-secret description of every trajectory-affecting setting."""
    fold = cfg["fold"]
    stop = dict(STOP if stop_cfg is None else stop_cfg)
    return {
        "version": EXECUTION_FINGERPRINT_VERSION,
        "protocol_id": PROTOCOL_ID,
        "model": str(cfg["server"]["model"]),
        "base_url": str(cfg["server"].get("base_url") or "").rstrip("/"),
        "openrouter_provider": os.environ.get("OPENROUTER_PROVIDER", "").strip(),
        "fold": {
            key: int(fold[key])
            for key in ("chunk_chars", "notes_chars", "notes_tokens", "draft_tokens")
        },
        "online_stop": {
            "rule": str(stop["rule"]),
            "theta": float(stop["theta"]),
            "eps": float(stop["eps"]),
            "window": int(stop["window"]),
            "min_chunks": int(stop["min_chunks"]),
            "record_verbalized": bool(stop["record_verbalized"]),
            "record_gate": bool(stop["record_gate"]),
        },
    }


def execution_fingerprint(spec):
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_execution_resume(existing, expected_spec, expected_fingerprint):
    """Refuse to reuse a successful row produced under any different setting."""
    sample_id = str(existing.get("id"))
    if existing.get("execution_fingerprint") != expected_fingerprint:
        raise RuntimeError(
            f"Output ID {sample_id} lacks the expected execution fingerprint; "
            "use a new output path"
        )
    if existing.get("execution_config") != expected_spec:
        raise RuntimeError(
            f"Output ID {sample_id} uses different execution settings; "
            "use a new output path"
        )


def compatible_disjoint_ruler_cohort(existing, current_rows):
    """True only for the documented two-file RULER-HotpotQA append workflow."""
    current_benchmarks = {
        str(row.get("benchmark")) for row in current_rows
        if row.get("benchmark") is not None
    }
    return (
        current_benchmarks == {"ruler_hotpotqa"}
        and existing.get("benchmark") == "ruler_hotpotqa"
    )

def qblock(row):
    if row["task_shape"] == "mcq":
        c = row["choices"]
        return (f"{row['question']}\nA) {c['A']}\nB) {c['B']}\n"
                f"C) {c['C']}\nD) {c['D']}")
    return row["question"]


def online_stop_decision(steps, task_shape, stop_cfg=None):
    """Evaluate the frozen windowed controller at the latest completed step."""
    cfg = dict(STOP)
    if stop_cfg is not None:
        cfg.update({key: value for key, value in stop_cfg.items() if key in cfg})
    current = steps[-1]
    # per-step divergence (recorded under every rule, incl. "none", for offline analysis)
    if len(steps) == 1:
        divergence = None
    elif task_shape == "mcq":
        divergence = jsd(steps[-2]["posterior"], current["posterior"])
    else:
        divergence = 1.0 - f1_tokens(
            steps[-2]["draft_norm"], current["draft_norm"]
        )
    confidence = float(current["confidence"])

    if cfg["rule"] == "none":                        # full-trajectory recording mode
        current.update({"divergence": divergence, "should_stop": False})
        return False, "", {"rule": "none"}

    # This predicate is shared with offline replay through semantic regression
    # tests, ensuring that online and offline decisions use identical rules.
    w = int(cfg["window"])
    t = len(steps)
    # W is a number of complete answer states, so W=3 cannot stop before the
    # third chunk. The statistic then contains exactly W-1 adjacent changes.
    enough = t >= max(int(cfg["min_chunks"]), w)
    if task_shape == "mcq":
        win = [s["posterior"] for s in steps[max(0, t - w):]]
        stat = (sum(jsd(win[i], win[i + 1]) for i in range(len(win) - 1)) / (len(win) - 1)
                if len(win) >= 2 else 1.0)
        should_stop = bool(enough and confidence >= cfg["theta"] and stat <= cfg["eps"])
        evid = {"rule": "windowed_paper", "confidence": confidence, "mean_jsd": stat,
                "window": w, "theta": cfg["theta"], "eps": cfg["eps"]}
    else:
        # open-ended: same two tests on the draft. An abstention is never a belief
        # (is_non_answer), the draft confidence must clear theta, and the mean
        # 1 - token-F1 between consecutive normalized drafts in the window must be
        # within eps -- exact string equality is the eps -> 0 limit of this test.
        recent_steps = steps[max(0, t - w):]
        win = [s["draft_norm"] for s in recent_steps]
        stat = (sum(1.0 - f1_tokens(win[i], win[i + 1]) for i in range(len(win) - 1))
                / (len(win) - 1) if len(win) >= 2 else 1.0)
        valid = bool(enough and len(recent_steps) == w and all(
            not is_non_answer(s.get("draft", ""))
            and float(s.get("draft_conf", 0.0)) >= cfg["theta"]
            for s in recent_steps
        ))
        should_stop = bool(valid and stat <= cfg["eps"])
        evid = {"rule": "windowed_paper_open", "confidence": current.get("draft_conf"),
                "instability": stat, "valid_recent_drafts": valid,
                "window": w, "theta": cfg["theta"], "eps": cfg["eps"]}
    current.update({"divergence": divergence, "should_stop": should_stop})
    return should_stop, ("windowed" if should_stop else ""), evid


def run_question(client, cfg, row, execution_spec=None, execution_id=None):
    question_t0 = time.time()
    f = cfg["fold"]
    model = cfg["server"]["model"]
    execution_spec = execution_spec or execution_config(cfg)
    execution_id = execution_id or execution_fingerprint(execution_spec)
    qb = qblock(row)
    shape = row["task_shape"]
    chunks = [row["context"][i:i + f["chunk_chars"]]
              for i in range(0, len(row["context"]), f["chunk_chars"])]
    if not chunks:
        raise ValueError(f"{row['id']}: empty context")
    sys_t = (FOLD_SYS_MCQ if shape == "mcq" else FOLD_SYS_OPEN).format(cap=f["notes_chars"])
    probe_t = PROBE_MCQ if shape == "mcq" else PROBE_OPEN

    cum = probe_tok = verb_tok = gate_tok = 0

    notes, steps, stop_trace = "(empty)", [], []
    stop_reason, stop_evidence = "full_context", {}
    for t, chunk in enumerate(chunks, 1):
        step_t0 = time.time()
        fr = chat(client, model, sys_t,
                  FOLD_USER.format(qblock=qb, notes=notes, chunk=chunk,
                                   i=t, n=len(chunks)),
                  max_tokens=f["notes_tokens"])
        cum += usage(fr)
        new = (fr.choices[0].message.content or "").strip()
        if new:
            notes = (new if len(new) <= f["notes_chars"]
                     else new[-f["notes_chars"]:])
        step = {"t": t, "notes": notes, "notes_chars": len(notes),
                "raw_notes_chars": len(new),
                "notes_overflow": len(new) > f["notes_chars"]}
        if shape == "mcq":
            measurement_error = None
            for probe_attempt in range(1, 5):
                pr = chat(
                    client, model, "",
                    probe_t.format(qblock=qb, notes=notes),
                    max_tokens=32, logprobs=True,
                    choices=["A", "B", "C", "D"],
                )
                call_tokens = usage(pr)
                cum += call_tokens
                probe_tok += call_tokens
                try:
                    step["posterior"] = posterior_from_logprobs(pr)
                    step["probe_attempts"] = probe_attempt
                    break
                except ProbeInvalid as error:
                    measurement_error = error
            else:
                raise ProbeInvalid(
                    f"{row['id']} chunk {t}: incomplete A-D measurement after "
                    f"4 identical attempts: {measurement_error}"
                ) from measurement_error
            choice = max(step["posterior"], key=step["posterior"].get)
            step["answer"] = choice
            step["answer_norm"] = choice
            step["confidence"] = max(step["posterior"].values())
        else:
            # Provider adapters occasionally omit generated-token logprobs while
            # returning text. Repeating the same deterministic measurement is safe;
            # silently replacing a missing measurement with confidence=0 is not.
            lps = []
            txt = ""
            for attempt in range(1, 5):
                pr = chat(client, model, "", probe_t.format(qblock=qb, notes=notes),
                          max_tokens=f["draft_tokens"], logprobs=True)
                call_tokens = usage(pr)
                cum += call_tokens
                probe_tok += call_tokens
                txt = (pr.choices[0].message.content or "").strip()
                try:
                    content = pr.choices[0].logprobs.content
                    lps = [float(x.logprob) for x in (content or [])]
                except (AttributeError, TypeError):
                    lps = []
                if txt and lps:
                    break
            if not txt or not lps:
                raise ProbeInvalid(
                    f"{row['id']} chunk {t}: open-answer probe omitted "
                    "answer text or generated-token logprobs after 4 attempts"
                )
            step["draft"] = txt
            step["probe_attempts"] = attempt
            step["draft_norm"] = normalize_draft(txt)
            step["draft_conf"] = math.exp(sum(lps) / len(lps)) if lps else 0.0
            step["draft_logprobs"] = [round(x, 4) for x in lps]
            step["answer"] = txt
            step["answer_norm"] = step["draft_norm"]
            step["confidence"] = step["draft_conf"]
        if STOP["record_verbalized"]:      # numeric asked-confidence comparator
            numeric = None
            for verbal_attempt in range(1, 5):
                vr = chat(
                    client, model, "", VERB.format(qblock=qb, notes=notes),
                    max_tokens=32,
                )
                call_tokens = usage(vr)
                cum += call_tokens
                verb_tok += call_tokens
                match = re.fullmatch(
                    r"\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*",
                    vr.choices[0].message.content or "",
                )
                if match and 0 <= float(match.group(1)) <= 100:
                    numeric = float(match.group(1))
                    break
            step["verbalized"] = numeric
            step["verbalized_valid"] = numeric is not None
            step["verbalized_attempts"] = verbal_attempt
        if STOP["record_gate"]:            # asked END/CONTINUE gate ablation
            gate_value = None
            gate_raw = ""
            for gate_attempt in range(1, 5):
                gr = chat(
                    client, model, "",
                    VERBALIZED_PROMPT.format(qblock=qb, notes=notes),
                    max_tokens=32,
                )
                call_tokens = usage(gr)
                cum += call_tokens
                gate_tok += call_tokens
                gate_raw = gr.choices[0].message.content or ""
                gate_value = parse_gate_response(gate_raw)
                if gate_value is not None:
                    break
            step["verbalized_stop"] = gate_value
            step["verbalized_gate_valid"] = gate_value is not None
            step["verbalized_gate_attempts"] = gate_attempt
            step["verbalized_gate_raw"] = gate_raw
        step["cum_tokens"] = cum
        # Split counters so each replayed policy is priced for only the calls it issues.
        # Lumping charged measured stopping for the verbalized comparator and vice versa.
        step["cum_probe_tokens"] = probe_tok
        step["cum_verb_tokens"] = verb_tok
        step["cum_gate_tokens"] = gate_tok
        step["cum_wall_s"] = round(time.time() - question_t0, 3)
        steps.append(step)

        should_stop, reason, evidence = online_stop_decision(steps, shape)
        step["step_latency_s"] = round(time.time() - step_t0, 6)
        step["cum_latency_s"] = round(time.time() - question_t0, 6)
        stop_trace.append({
            "t": t,
            "should_stop": should_stop,
            "reason": reason,
            "evidence": evidence,
        })
        if should_stop:
            stop_reason, stop_evidence = reason, evidence
            break

    # Pass through EVERY source field except the raw context. A hardcoded key list
    # silently dropped build_id (analyze.py's cross-build guard) and evidence_char_starts
    # (the oracle stopping evaluation), which is only discoverable after a full rerun.
    rec = {k: v for k, v in row.items() if k != "context"}
    source_chars_read = min(len(row["context"]), len(steps) * f["chunk_chars"])
    rec.update({"n_chunks": len(chunks), "chunks_read": len(steps),
                "method": "ACS", "chunk_chars": f["chunk_chars"],
                "source_chars_read": source_chars_read,
                "stopped_early": len(steps) < len(chunks),
                "stop_reason": stop_reason, "stop_evidence": stop_evidence,
                "stop_trace": stop_trace,
                # Only stamp online_stop_config when a runtime rule could TRUNCATE the
                # trajectory. Under rule "none" nothing is truncated, so the file stays a
                # legitimate full-trajectory input for analyze.py policies.
                **({"recording_mode": "full_trajectory"} if STOP["rule"] == "none"
                   else {"online_stop_config": dict(STOP)}),
                "stop_rule_used": STOP["rule"],
                "probe_template_version": "v2",
                "protocol_id": PROTOCOL_ID,
                "execution_config": execution_spec,
                "execution_fingerprint": execution_id,
                # Which weights produced this trajectory. Without it, two runs are
                # indistinguishable after the fact and a mislabeled model is unfalsifiable.
                "model": model,
                "base_url": cfg["server"].get("base_url"),
                "required_calls_per_chunk": 2,
                "recorded_optional_signals": {
                    "numeric_confidence": bool(STOP["record_verbalized"]),
                    "asked_gate": bool(STOP["record_gate"]),
                },
                "notes_retention": "rolling_tail", "steps": steps,
                "final_notes": notes,
                "wall_s": round(time.time() - question_t0, 3)})
    if "needle_position_chars" in row:
        rec["needle_seen"] = int(row["needle_position_chars"]) < source_chars_read
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0,
                    help="override fold.workers; 1 = strictly sequential")
    ap.add_argument("--rule", choices=["none", "windowed"], default=None,
                    help="override online_stop.rule (none = full-trajectory recording)")
    ap.add_argument("--record-verbalized", dest="record_verbalized",
                    action="store_true", default=None,
                    help="log numeric asked-confidence each step (analyze.py comparator)")
    ap.add_argument("--record-gate", dest="record_gate", action="store_true", default=None,
                    help="also log the separately prompted END/CONTINUE decision")
    a = ap.parse_args()
    cfg = load_config(a.config)
    STOP.update({k: v for k, v in (cfg.get("online_stop") or {}).items() if k in STOP})
    if a.rule:                       # CLI beats config, so no file edits between runs
        STOP["rule"] = a.rule
    if a.record_verbalized:
        STOP["record_verbalized"] = True
    if a.record_gate:
        STOP["record_gate"] = True
    expected_execution = execution_config(cfg)
    expected_fingerprint = execution_fingerprint(expected_execution)
    all_rows = read_jsonl(a.data)
    if len({str(row["id"]) for row in all_rows}) != len(all_rows):
        raise RuntimeError("Prepared input contains duplicate sample IDs")
    source_by_id = {str(row["id"]): row for row in all_rows}
    rows = all_rows
    if a.limit:
        rows = rows[: a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if Path(a.out).exists():
        for existing in read_jsonl(a.out):
            if existing.get("failed"):
                continue
            sample_id = str(existing.get("id"))
            source = source_by_id.get(sample_id)
            if source is None:
                # RULER-HotpotQA is intentionally prepared as two disjoint cohorts
                # and the README appends both to one output. Permit only that
                # explicitly tagged benchmark; every other cross-input reuse fails.
                if not compatible_disjoint_ruler_cohort(existing, all_rows):
                    raise RuntimeError(
                        f"Output contains ID {sample_id!r} outside the prepared input; "
                        "use a new output path"
                    )
                validate_execution_resume(
                    existing, expected_execution, expected_fingerprint
                )
                continue
            validate_execution_resume(
                existing, expected_execution, expected_fingerprint
            )
            if existing.get("model") != cfg["server"]["model"]:
                raise RuntimeError(
                    f"Output ID {sample_id} uses model={existing.get('model')!r}; "
                    "use a new output path"
                )
            if existing.get("protocol_id") != PROTOCOL_ID:
                raise RuntimeError(
                    f"Output ID {sample_id} uses another prompt/controller protocol; "
                    "use a new output path"
                )
            if int(existing.get("chunk_chars", -1)) != int(cfg["fold"]["chunk_chars"]):
                raise RuntimeError(
                    f"Output ID {sample_id} uses another chunk size; use a new output path"
                )
            if existing.get("stop_rule_used") != STOP["rule"]:
                raise RuntimeError(
                    f"Output ID {sample_id} uses stop rule "
                    f"{existing.get('stop_rule_used')!r}, expected {STOP['rule']!r}"
                )
            if source.get("build_id") is not None and (
                existing.get("build_id") != source.get("build_id")
            ):
                raise RuntimeError(
                    f"Output ID {sample_id} belongs to another data build; "
                    "use a new output path"
                )
    done = resume_ids(a.out)
    rows = [r for r in rows if r["id"] not in done]
    print(f"online_stop={STOP}")
    print(f"{len(done)} done, {len(rows)} to run")
    # Questions are independent, so they run concurrently against one vLLM server.
    # Chunks remain sequential inside each question.
    W = max(1, int(a.workers or cfg["fold"].get("workers", 1)))
    print(f"workers={W}")
    client = make_client(cfg)
    if any(row.get("task_shape") == "mcq" for row in rows):
        verify_constrained_decoding(client, cfg["server"]["model"])
        print("MCQ constrained-decoding preflight: PASS", flush=True)
    t0 = time.time()
    n_done = 0

    def work(row):
        try:
            return row, run_question(
                client, cfg, row, expected_execution, expected_fingerprint
            ), None
        except Exception as error:     # noqa: BLE001 - durable failure, retried next pass
            return row, {"id": row["id"], "failed": True, "error": str(error)[:300],
                         "steps": []}, error

    with ThreadPoolExecutor(max_workers=W) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for fut in as_completed(futs):
            row, rec, err = fut.result()
            with _WLOCK:                          # save before printing: printed == durable
                append_jsonl(a.out, rec)
            n_done += 1
            if err is not None:
                print(f"[{n_done}/{len(rows)}] {row['id']} FAILED: {str(err)[:120]}", flush=True)
                continue
            final = rec["steps"][-1]
            print(
                f"[{n_done}/{len(rows)}] {row['id']} "
                f"read={rec['chunks_read']}/{rec['n_chunks']} "
                f"answer={final['answer']!r} "
                f"confidence={final['confidence']:.4f} "
                f"reason={rec['stop_reason']} "
                f"tokens={final['cum_tokens']:,} "
                f"latency={rec['wall_s']:.1f}s "
                f"total={time.time()-t0:,.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
