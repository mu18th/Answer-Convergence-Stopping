"""Paper-aligned analysis suite.

``policies`` replays the reported stopping policies on full trajectories and
``evidence`` computes the evidence-relative metrics used in the paper.
"""
from __future__ import annotations
import argparse
import math
import random
import statistics
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from browsecomp_plus_judge import (
    OFFICIAL_JUDGE_MODEL,
    BrowseCompPlusJudge,
    JudgeItem,
    JudgeParseError,
    JudgeSettings,
)
from run_fold import STOP as RUNTIME_STOP
from run_fold import online_stop_decision
from common import (auroc, dedupe_prefer_success, ece, f1_tokens,
                    is_non_answer, jsd, load_config, read_jsonl, strip_think,)


def step_answer(tr, t):
    if tr["task_shape"] == "mcq":
        p = tr["steps"][t - 1]["posterior"]
        return max(p, key=p.get), max(p.values())
    s = tr["steps"][t - 1]
    return s["draft"], s.get("draft_conf", 0.0)


# Binary correctness for accuracy and calibration labels. Exact match is right
# for MCQ and numeric golds. A free-text token-F1 gold is a full sentence,
# so a 0.999 bar would binarize nearly every row to 0. The default is 0.999;
# --correct-at changes only F1 tasks.
# This module-level value is set once by the command-line entry point.
CORRECT_AT_F1 = 0.999

def is_correct(tr, score):
    thr = CORRECT_AT_F1 if str(tr.get("answer_type", "")) == "f1" else 0.999
    return int(score >= thr)


# BrowseComp-Plus grades with an LLM judge on semantic equivalence, not substring
# matching. Off by default so nothing changes silently; --judge turns it on and it
# only affects answer_type == "string" (the browsecomp_plus arm).
JUDGE: BrowseCompPlusJudge | None = None


def _enable_judge(a, cfg):
    global JUDGE
    if not getattr(a, "judge", False):
        return
    from common import make_client
    official = getattr(a, "judge_official", False)
    requested_model = getattr(a, "judge_model", None)
    settings = (
        JudgeSettings(model=OFFICIAL_JUDGE_MODEL)
        if official
        else JudgeSettings(
            model=requested_model or cfg["server"]["model"],
            temperature=0.0,
            top_p=None,
            top_k=None,
            max_tokens=512,
        )
    )
    path = Path(getattr(a, "judge_cache", None) or
                "results/.judge_cache_browsecomp_plus.json")
    JUDGE = BrowseCompPlusJudge(make_client(cfg), settings, path)
    sampling = {
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        **({"top_p": settings.top_p} if settings.top_p is not None else {}),
        **({"top_k": settings.top_k} if settings.top_k is not None else {}),
    }
    proto = "OFFICIAL sampling" if official else "deterministic"
    print(f"judge: {settings.model} via {cfg['server']['base_url']} "
          f"[{proto}: " + ", ".join(f"{k}={v}" for k, v in sorted(sampling.items()))
          + f"] ({JUDGE.cache_size} cached)")


def _check_build(trajs, label="trajectories"):
    """Results files are append-and-resume keyed on question id alone, so a rebuilt
    dataset lands its new rows NEXT TO trajectories from the previous construction
    and every metric silently averages two different experiments. Refuse that."""
    builds = {t.get("build_id") for t in trajs}
    if len(builds) > 1:
        counts = {b: sum(1 for t in trajs if t.get("build_id") == b) for b in builds}
        raise SystemExit(
            f"REFUSING TO ANALYSE: {label} span {len(builds)} dataset builds "
            f"{ {(k or '<pre-build_id>'): v for k, v in counts.items()} }.\n"
            f"These came from different context constructions and cannot be pooled.\n"
            f"Delete the stale results file and rerun, or split it by build_id.")
    return builds.pop() if builds else None


def score_answer(tr, ans):
    if tr["task_shape"] == "mcq":
        return float(ans == tr["answer"])
    at = str(tr.get("answer_type", ""))
    if at == "string" and JUDGE is not None:
        outcome = JUDGE.judge(JudgeItem(
            question=tr.get("question", ""),
            response=ans,
            correct_answer=tr["answer"],
        ))
        if outcome.parsed.parse_error:
            # An unreadable judge response is not a valid incorrect verdict.
            raise JudgeParseError(
                f"trajectory {tr.get('id')!r}: grader verdict unreadable "
                f"(no correct: yes/no). Raw: {outcome.raw_response[:200]!r}")
        return float(outcome.correct)
    if at == "substring":              # RULER string_match_all, NVIDIA/RULER
        golds = tr["answer"] if isinstance(tr["answer"], list) else [tr["answer"]]
        golds = [str(g) for g in golds]
        # Empty substring references make this metric undefined.
        if not golds or any(not g.strip() for g in golds):
            raise ValueError(
                f"trajectory {tr.get('id')!r} has an empty substring gold: "
                f"{tr['answer']!r}")
        low = strip_think(ans).lower()
        return sum(g.lower() in low for g in golds) / len(golds)
    if at == "f1":
        return f1_tokens(ans, tr["answer"])
    if at == "numeric_exact":
        import re
        m = re.search(r"-?\d+", ans or "")
        # Normalize JSON numeric references before exact comparison.
        return float(bool(m) and m.group() == str(tr["answer"]).strip())
    if at == "string":
        raise RuntimeError(
            "answer_type='string' requires semantic judging; rerun analysis "
            "with --judge (and the documented judge model)"
        )
    raise ValueError(
        f"unsupported answer_type {at!r} for trajectory {tr.get('id')!r}"
    )


# ---- stopping policies ---------------------------------------------------------
def stop_full(tr, **_):
    return len(tr["steps"])


def stop_fixed(tr, frac, **_):
    return max(1, math.ceil(frac * len(tr["steps"])))


def stop_random(tr, rng, **_):
    return rng.randint(1, len(tr["steps"]))


def captured_saving(stop, evidence, total):
    """Oracle-available saving captured safely by one stop decision."""
    return max(0, total - stop) if stop >= evidence else 0


def stop_verbalized(tr, v, **_):
    for s in tr["steps"]:
        value = s.get("verbalized")
        if isinstance(value, (int, float)) and value >= v:
            return s["t"]
    return len(tr["steps"])


def stop_verbalized_gate(tr, **_):
    """Replay the separately recorded END/CONTINUE asked-gate ablation."""
    for s in tr["steps"]:
        if s.get("verbalized_stop") is True:
            return s["t"]
    return len(tr["steps"])


def _open_instability(s, lo, t):
    """Mean dissimilarity of consecutive drafts in the window — the continuous
    analogue of the MCQ branch's mean JSD, so eps means the same thing on both
    task shapes. 0 = drafts identical, 1 = no tokens in common.
    """
    win = [s[i]["draft_norm"] for i in range(lo, t)]
    if len(win) < 2:
        return 1.0
    return sum(1.0 - f1_tokens(win[i], win[i + 1])
               for i in range(len(win) - 1)) / (len(win) - 1)


# Loaded from config by cmd_policies. W is the number of complete answer states;
# W=3 therefore uses two adjacent changes and cannot stop before step three.
STABILITY_W = 3


def stop_measured(tr, theta, eps, w=None, statistic="peak", **_):
    """Gate on confidence and require the last ``w`` probes to be stable."""
    w = w or STABILITY_W
    if tr["task_shape"] != "mcq":
        s = tr["steps"]
        for t in range(1, len(s) + 1):
            # W means W complete observations, not "up to W".  Without this
            # guard the confidence-gated W3 rule could stop after only two
            # chunks, which is precisely the premature-stop failure it is meant
            # to address.
            if t < w:
                continue
            # Never treat an abstention as convergence.
            if any(is_non_answer(s[i].get("draft", ""))
                   for i in range(t - w, t)):
                continue
            if any(
                s[i].get("draft_conf", 0.0) < theta
                for i in range(t - w, t)
            ):
                continue
            if _open_instability(s, max(0, t - w), t) <= eps:
                return t
        return len(s)
    ps = [s["posterior"] for s in tr["steps"]]
    for t in range(1, len(ps) + 1):
        if t < w:
            continue
        p = ps[t - 1]
        conf = {"peak": max(p.values()),
                "margin": sorted(p.values())[-1] - sorted(p.values())[-2],
                "entropy": 1 + sum(v * math.log(v + 1e-12) for v in p.values()) / math.log(4)
                }[statistic]
        if conf < theta:
            continue
        lo = max(0, t - w)
        win = ps[lo:t]
        mj = (sum(jsd(win[i], win[i+1]) for i in range(len(win)-1)) / (len(win)-1)
              if len(win) >= 2 else 1.0)
        if mj <= eps:
            return t
    return len(ps)


# Average repeated draws to estimate the expected random-stop score and cost.
RANDOM_DRAWS = 200


def evaluate_random(trajs, draws=None, uses=()):
    """Expected score/cost of stop-uniformly-at-random, averaged over `draws`."""
    draws = draws or RANDOM_DRAWS
    if not trajs:
        nan = float("nan")
        return {"score": nan, "tokens": nan, "auroc": nan, "ece": nan, "score_sd": nan}
    per_draw = []
    token_total = 0.0
    for d in range(draws):
        rng = random.Random(d)
        scs = []
        for tr in trajs:
            t = stop_random(tr, rng)
            ans, _ = step_answer(tr, t)
            scs.append(score_answer(tr, ans))
            # Average cost over the same draws used for accuracy.
            token_total += policy_tokens(tr, t, uses)
        per_draw.append(sum(scs) / len(scs))
    mean = sum(per_draw) / len(per_draw)
    var = sum((x - mean) ** 2 for x in per_draw) / len(per_draw)
    return {"score": mean,
            "tokens": token_total / (draws * len(trajs)),
            "auroc": float("nan"), "ece": float("nan"), "score_sd": var ** 0.5}


def policy_tokens(tr, t, uses=("probe",)):
    """Token cost of stopping at step t under a policy that consumes only `uses`.

    Every policy pays the folding cost and one answer probe at its stopping step.
    A policy additionally pays for each gating signal it consumes.
    """
    s = tr["steps"][t - 1]
    required = ("cum_tokens", "cum_probe_tokens", "cum_verb_tokens", "cum_gate_tokens")
    missing = [field for field in required if field not in s]
    if missing:
        raise RuntimeError(f"trajectory lacks split token counters: {missing}")
    probe = s.get("cum_probe_tokens", 0) or 0
    verb = s.get("cum_verb_tokens", 0) or 0
    gate = s.get("cum_gate_tokens", 0) or 0
    # These counters are all included in cum_tokens.  Subtract every optional
    # signal before pricing a logical policy, then add back only what that
    # policy consumes.
    fold = s["cum_tokens"] - probe - verb - gate
    prev = tr["steps"][t - 2] if t >= 2 else None
    one_probe = probe - ((prev.get("cum_probe_tokens", 0) or 0) if prev else 0)
    total = fold + (probe if "probe" in uses else one_probe)
    if "verb" in uses:
        total += verb
    if "gate" in uses:
        total += gate
    return total


def token_accounting(trajs):
    if not trajs:
        return "split (per-policy)"
    required = {"cum_tokens", "cum_probe_tokens", "cum_verb_tokens", "cum_gate_tokens"}
    if any(not required.issubset(step) for tr in trajs for step in tr["steps"]):
        raise RuntimeError("trajectory lacks required split token counters")
    return "split (per-policy)"


# The ONLINE gate is the shipped system; the offline rules below are analysis
# instruments. Import the runtime decision rather than restating it, so the two
# cannot drift.
ONLINE_STOP_CFG: dict | None = None


def _online_cfg_for(_tr, cfg):
    """Return the frozen runtime configuration for a trajectory."""
    return dict(cfg)


def stop_online(tr, cfg=None, **_):
    """Replay the RUNTIME stop rule on a full trajectory, step by step.

    This is what the deployed system does. Any offline number meant to describe
    the shipped method must come from here, not from stop_measured.
    """
    scfg = _online_cfg_for(tr, cfg or ONLINE_STOP_CFG or {})
    steps = []
    for original in tr["steps"]:
        step = dict(original)
        if tr["task_shape"] == "mcq":
            step["confidence"] = max(step["posterior"].values())
        steps.append(step)
    # online_stop_decision is the shared runtime function and reads the
    # run_fold.STOP module global. Temporarily load the frozen replay settings
    # into that exact global instead of reimplementing the stopping rule here.
    previous = dict(RUNTIME_STOP)
    RUNTIME_STOP.update({k: v for k, v in scfg.items() if k in RUNTIME_STOP})
    try:
        for t in range(1, len(steps) + 1):
            stop, _reason, _ev = online_stop_decision(
                steps[:t], tr["task_shape"], scfg)
            if stop:
                return t
        return len(steps)
    finally:
        RUNTIME_STOP.clear()
        RUNTIME_STOP.update(previous)


def evaluate(trajs, stopper, uses=("probe",), **kw):
    # A filtered or split input can be empty; every metric below divides by n.
    if not trajs:
        nan = float("nan")
        return {"score": nan, "tokens": nan, "auroc": nan, "ece": nan}
    scs, toks, confs = [], [], []
    for tr in trajs:
        t = stopper(tr, **kw)
        ans, conf = step_answer(tr, t)
        scs.append(score_answer(tr, ans))
        toks.append(policy_tokens(tr, t, uses))
        confs.append(conf)
    n = len(trajs)
    labels = [is_correct(tr, s) for tr, s in zip(trajs, scs)]
    return {"score": sum(scs)/n, "tokens": sum(toks)/n,
            "auroc": auroc(confs, labels), "ece": ece(confs, labels)}


def print_longbench_domain_breakdown(
    trajs, fixed_cfg, *, has_verbalized=False, has_verbalized_gate=False,
    verbalized_at=99.5,
):
    """Report LongBench-v2 composition and policy results by source domain.

    This is descriptive stratification, not parameter tuning. Every policy is
    replayed independently inside each domain using the same frozen constants.
    """
    grouped = {}
    for tr in trajs:
        grouped.setdefault(str(tr.get("domain", "unknown")), []).append(tr)
    if len(grouped) <= 1:
        return

    def display_name(domain):
        if domain == "Code Repository Understanding":
            return "CodeQA / Code Repository"
        return domain

    print("\nLongBench-v2 composition by domain:")
    print(
        f"{'domain':38s} {'n':>5s} {'share':>7s} {'easy':>6s} {'hard':>6s} "
        f"{'mean chars':>13s} {'median chars':>13s} {'mean chunks':>12s}"
    )
    for domain in sorted(grouped):
        group = grouped[domain]
        chars = [int(tr.get("n_chars", 0) or 0) for tr in group]
        easy = sum(str(tr.get("difficulty", "")).lower() == "easy" for tr in group)
        hard = sum(str(tr.get("difficulty", "")).lower() == "hard" for tr in group)
        print(
            f"{display_name(domain):38.38s} {len(group):5d} "
            f"{len(group)/len(trajs):7.1%} {easy:6d} {hard:6d} "
            f"{statistics.mean(chars):13,.0f} {statistics.median(chars):13,.0f} "
            f"{statistics.mean(len(tr['steps']) for tr in group):12.2f}"
        )

    policy_specs = [
        ("full", stop_full, (), {}),
        ("random", None, (), {}),
    ]
    if has_verbalized:
        policy_specs.append(
            (f"verbal{verbalized_at:g}", stop_verbalized, ("verb",),
             {"v": verbalized_at})
        )
    if has_verbalized_gate:
        policy_specs.append(
            ("END_gate", stop_verbalized_gate, ("gate",), {})
        )
    policy_specs.append(
        ("ACS", stop_online, ("probe",), {"cfg": fixed_cfg})
    )

    print("\nAccuracy by LongBench-v2 domain (raw posterior):")
    header = f"{'domain':38s} {'n':>5s}" + "".join(
        f" {name:>11s}" for name, *_ in policy_specs
    )
    print(header)
    macro = {name: [] for name, *_ in policy_specs}
    for domain in sorted(grouped):
        group = grouped[domain]
        values = []
        for name, stopper, uses, kwargs in policy_specs:
            if name == "random":
                result = evaluate_random(group, uses=uses)
            else:
                result = evaluate(
                    group, stopper, uses=uses,
                    **kwargs
                )
            values.append(result["score"])
            macro[name].append(result["score"])
        print(
            f"{display_name(domain):38.38s} {len(group):5d}" +
            "".join(f" {value:11.3f}" for value in values)
        )
    print(
        f"{'MACRO DOMAIN MEAN':38s} {len(grouped):5d}" +
        "".join(
            f" {statistics.mean(macro[name]):11.3f}"
            for name, *_ in policy_specs
        )
    )

    print("\nACS cost by LongBench-v2 domain:")
    print(
        f"{'domain':38s} {'n':>5s} {'full tok':>12s} {'stop tok':>12s} "
        f"{'saving':>9s} {'chunks':>13s} {'read frac':>10s} {'regret':>9s}"
    )
    macro_rows = []
    for domain in sorted(grouped):
        group = grouped[domain]
        full = evaluate(group, stop_full, uses=())
        stopped = evaluate(
            group, stop_online, uses=("probe",), cfg=fixed_cfg,
        )
        stop_steps = [
            stop_online(tr, cfg=fixed_cfg)
            for tr in group
        ]
        full_steps = [len(tr["steps"]) for tr in group]
        read_fraction = statistics.mean(
            stop / full_t for stop, full_t in zip(stop_steps, full_steps)
        )
        saving = 1.0 - stopped["tokens"] / full["tokens"]
        regret = full["score"] - stopped["score"]
        macro_rows.append((full["tokens"], stopped["tokens"], saving,
                           statistics.mean(stop_steps),
                           statistics.mean(full_steps), read_fraction, regret))
        print(
            f"{display_name(domain):38.38s} {len(group):5d} "
            f"{full['tokens']:12,.0f} {stopped['tokens']:12,.0f} "
            f"{saving:9.1%} "
            f"{statistics.mean(stop_steps):5.1f}/{statistics.mean(full_steps):5.1f} "
            f"{read_fraction:10.1%} {regret:+9.3f}"
        )
    print(
        f"{'MACRO DOMAIN MEAN':38s} {len(grouped):5d} "
        f"{statistics.mean(row[0] for row in macro_rows):12,.0f} "
        f"{statistics.mean(row[1] for row in macro_rows):12,.0f} "
        f"{statistics.mean(row[2] for row in macro_rows):9.1%} "
        f"{statistics.mean(row[3] for row in macro_rows):5.1f}/"
        f"{statistics.mean(row[4] for row in macro_rows):5.1f} "
        f"{statistics.mean(row[5] for row in macro_rows):10.1%} "
        f"{statistics.mean(row[6] for row in macro_rows):+9.3f}"
    )


def cmd_policies(a):
    global STABILITY_W
    cfg = load_config(a.config)
    _enable_judge(a, cfg)
    STABILITY_W = int(cfg["stopping"].get("stability_window", 3))
    # The shipped rule's config, so stop_online can replay exactly what runs.
    globals()["ONLINE_STOP_CFG"] = cfg.get("online_stop")
    trajs = [
        t for t in dedupe_prefer_success(read_jsonl(a.traj)) if t.get("steps")
    ]
    _check_build(trajs, "fold trajectories")
    if any(t.get("online_stop_config") for t in trajs):
        raise SystemExit(
            "REFUSING TO SIMULATE OFFLINE POLICIES ON RUNTIME-STOPPED TRAJECTORIES.\n"
            "They are truncated at the actual stop step. Score runtime output "
            "directly, and keep a separate full-trajectory file for offline policy "
            "replay and threshold sweeps."
        )
    is_mcq = bool(trajs) and trajs[0]["task_shape"] == "mcq"
    # Default analysis for a completed trajectory file: tuning-free ACS
    # versus full reading on EVERY supplied sample. The trajectory-recording
    # config necessarily says rule=none; replay changes only that mode switch to
    # the deployed paper rule and freezes every threshold from the same config.
    def run_tuning_free_analysis():
        fixed_cfg = dict(cfg.get("online_stop") or {})
        fixed_cfg["rule"] = "windowed"
        required = ("theta", "eps", "window", "min_chunks")
        missing = [key for key in required if key not in fixed_cfg]
        if missing:
            raise SystemExit(
                f"Tuning-free replay requires online_stop keys: {missing}"
            )
        print(f"{len(trajs)} trajectories | evaluation: ALL samples")
        print("mode: tuning-free ACS (no dev split, no parameter search)")
        print(
            "frozen controller: "
            f"rule={fixed_cfg['rule']} theta={fixed_cfg['theta']} "
            f"eps={fixed_cfg['eps']} window={fixed_cfg['window']} "
            f"min_chunks={fixed_cfg['min_chunks']}"
        )
        print(f"token accounting: {token_accounting(trajs)}\n")

        # A missing/invalid comparator response is a conservative CONTINUE.
        # Presence, rather than successful parsing at every step, determines
        # whether the separately recorded comparator baseline can be replayed.
        has_verbalized = all(
            "verbalized" in step for tr in trajs for step in tr["steps"]
        )
        has_verbalized_gate = all(
            "verbalized_stop" in step for tr in trajs for step in tr["steps"]
        )
        rows = []
        if is_mcq:
            raw_full = evaluate(
                trajs, stop_full, uses=()
            )
            raw_random = evaluate_random(
                trajs, uses=()
            )
            raw_online = evaluate(
                trajs, stop_online, uses=("probe",), cfg=fixed_cfg,
            )
            rows.extend([
                ("full reading", raw_full, raw_full),
                ("random stop", raw_random, raw_full),
            ])
            if has_verbalized:
                verbal_raw = evaluate(
                    trajs, stop_verbalized, uses=("verb",), v=a.verbalized_at,
                )
                rows.append((f"verbalized gate @{a.verbalized_at:g}", verbal_raw, raw_full))
            if has_verbalized_gate:
                gate_raw = evaluate(
                    trajs, stop_verbalized_gate, uses=("gate",),
                )
                rows.append(("END gate", gate_raw, raw_full))
            rows.append(("ACS, fixed", raw_online, raw_full))
        else:
            full_result = evaluate(trajs, stop_full, uses=())
            # Open-answer ACS rejects non-answers and requires
            # both measured draft confidence and W-step draft stability.
            ACS_open = evaluate(
                trajs, stop_online, uses=("probe",), cfg=fixed_cfg
            )
            rows.extend([
                ("full reading", full_result, full_result),
                ("random stop", evaluate_random(trajs), full_result),
            ])
            if has_verbalized:
                verbal = evaluate(
                    trajs, stop_verbalized, uses=("verb",), v=a.verbalized_at
                )
                rows.append((f"verbalized gate @{a.verbalized_at:g}", verbal, full_result))
            if has_verbalized_gate:
                gate = evaluate(
                    trajs, stop_verbalized_gate, uses=("gate",)
                )
                rows.append(("END gate", gate, full_result))
            rows.append(("ACS, fixed", ACS_open, full_result))

        print(
            f"{'policy':32s} {'correct':>9s} {'accuracy':>10s} "
            f"{'regret':>9s} {'mean tokens':>13s} {'saving':>9s}"
        )
        for name, result, reference in rows:
            saving = (
                1.0 - result["tokens"] / reference["tokens"]
                if reference["tokens"] else float("nan")
            )
            regret = reference["score"] - result["score"]
            print(
                f"{name:32s} "
                f"{round(result['score'] * len(trajs)):>4d}/{len(trajs):<4d} "
                f"{result['score']:10.3f} {regret:+9.3f} "
                f"{result['tokens']:13,.0f} "
                f"{saving:9.1%}"
            )
        if not has_verbalized:
            print("\nnumeric verbalized row omitted: per-step values are missing")
        if not has_verbalized_gate:
            print("\nasked END-gate rows omitted: per-step gate values are missing")
        print()

        if is_mcq:
            print_longbench_domain_breakdown(
                trajs, fixed_cfg,
                has_verbalized=has_verbalized,
                has_verbalized_gate=has_verbalized_gate,
                verbalized_at=a.verbalized_at,
            )

        print("per-sample paired replay:")
        for tr in trajs:
            full_t = len(tr["steps"])
            if not is_mcq:
                gated_t = stop_online(tr, cfg=fixed_cfg)
                full_answer, _ = step_answer(tr, full_t)
                gated_answer, _ = step_answer(tr, gated_t)
                print(
                    f"{tr['id']} "
                    f"full={full_answer}({bool(score_answer(tr, full_answer))}) "
                    f"ACS={gated_t}/{full_t}:{gated_answer}"
                    f"({bool(score_answer(tr, gated_answer))})"
                )
                continue

            raw_t = stop_online(tr, cfg=fixed_cfg)
            raw_full_answer, _ = step_answer(tr, full_t)
            raw_stop_answer, _ = step_answer(tr, raw_t)
            fields = [
                f"{tr['id']} gold={tr['answer']}",
                f"raw full={raw_full_answer}"
                f"({bool(score_answer(tr, raw_full_answer))})",
                f"raw stop={raw_t}/{full_t}:{raw_stop_answer}"
                f"({bool(score_answer(tr, raw_stop_answer))})",
            ]
            if has_verbalized:
                verbal_t = stop_verbalized(tr, a.verbalized_at)
                verbal_raw_answer, _ = step_answer(tr, verbal_t)
                fields.append(
                    f"verbal@{a.verbalized_at:g}={verbal_t}/{full_t}:{verbal_raw_answer}"
                    f"({bool(score_answer(tr, verbal_raw_answer))})"
                )
            if has_verbalized_gate:
                gate_t = stop_verbalized_gate(tr)
                gate_raw_answer, _ = step_answer(tr, gate_t)
                fields.append(
                    f"asked-gate={gate_t}/{full_t}:{gate_raw_answer}"
                    f"({bool(score_answer(tr, gate_raw_answer))})"
                )
            print(" | ".join(fields))
        return

    return run_tuning_free_analysis()


def _first_evidence_chunk(tr):
    """1-based index of the chunk where evidence first appears, or None."""
    starts = tr.get("evidence_char_starts") or []
    if not starts:
        return None
    return min(starts) // int(tr.get("chunk_chars") or 24000) + 1


def cmd_evidence(a):
    """Did the policy stop BEFORE it could have seen the evidence?

    Separates the two failure modes a score alone cannot: stopping too early
    (evidence never read) versus reading the evidence and still answering wrong.
    Meaningful whenever the prepared trajectory contains a defensible evidence
    position (the S-NIAH needle or a located BrowseComp answer-bearing document).
    """
    global STABILITY_W
    cfg = load_config(a.config)
    _enable_judge(a, cfg)
    STABILITY_W = int(cfg["stopping"].get("stability_window", 3))
    # The shipped rule's config, so stop_online can replay exactly what runs.
    globals()["ONLINE_STOP_CFG"] = cfg.get("online_stop")
    trajs = [
        t for t in dedupe_prefer_success(read_jsonl(a.traj)) if t.get("steps")
    ]
    _check_build(trajs, "fold trajectories")
    have = [t for t in trajs if t.get("evidence_char_starts")]
    if not have:
        print("no evidence_char_starts in these trajectories.\n"
              "Rebuild with the current prepare_data.py browsecomp_plus, then rerun run_fold.")
        return
    rows = have
    fixed_stop = dict(cfg.get("online_stop") or {})
    fixed_theta = float(fixed_stop["theta"])
    fixed_eps = float(fixed_stop["eps"])
    es = [_first_evidence_chunk(t) for t in rows]
    ns = [len(t["steps"]) for t in rows]
    print(f"{len(rows)} questions with known evidence positions "
          f"(frozen theta={fixed_theta} eps={fixed_eps}; no split or tuning)\n")
    print(f"first evidence chunk : min={min(es)} median={sorted(es)[len(es)//2]} max={max(es)}")
    print(f"chunks per question  : min={min(ns)} median={sorted(ns)[len(ns)//2]} max={max(ns)}")
    print(f"evidence depth       : {sum(e/n for e, n in zip(es, ns))/len(es):.2f} "
          f"of the document on average (0.5 = uniformly placed)\n")

    print(f"{'policy':22s} {'P(stop<ev)':>10s} {'acc|early':>10s} {'acc|late':>9s} "
          f"{'n_early':>8s} {'n_late':>7s} {'overread':>8s} {'acc':>6s} {'regret':>7s} {'savecap':>8s}")
    # Oracle stop = halt AT the first evidence chunk: the earliest any rule could stop
    # having read the evidence. Ceiling for every stopping policy (handoff Sec 5).
    oracle_saved = sum(n - e for e, n in zip(es, ns))
    _oacc = []
    for tr, e in zip(rows, es):
        ans, _ = step_answer(tr, e)
        _oacc.append(score_answer(tr, ans))
    oracle_acc = sum(_oacc) / len(_oacc)
    policy_specs = [("fixed at 25%", stop_fixed, {"frac": 0.25}),
                    ("random stop", "uniform-expectation", {})]
    if all(
        s.get("verbalized_valid") is True
        for tr in rows for s in tr["steps"]
    ):
        policy_specs.append((f"verbalized gate @{a.verbalized_at:g}", stop_verbalized,
                             {"v": a.verbalized_at}))
    # Include the separately prompted END/CONTINUE comparator only when its
    # per-step decisions are recorded.
    if all(
        s.get("verbalized_gate_valid") is True
        for tr in rows for s in tr["steps"]
    ):
        policy_specs.append(("END gate", stop_verbalized_gate, {}))
    existing_cfg = dict(fixed_stop)
    existing_cfg["rule"] = "windowed"
    policy_specs.extend([
        ("ACS, fixed", stop_online, {"cfg": existing_cfg}),
        ("full reading", stop_full, {}),
        ("oracle stop", None, {}),
    ])
    for name, fn, kw in policy_specs:
        early_score = late_score = early_mass = late_mass = 0.0
        score_total = over_total = saved = 0.0
        for tr, e, nn in zip(rows, es, ns):
            # Random stopping is reported as its exact expectation under a
            # discrete uniform stop over chunks, not as one noisy Monte Carlo draw.
            stops = range(1, nn + 1) if fn == "uniform-expectation" else [
                e if fn is None else fn(tr, **kw)
            ]
            weight = 1.0 / len(stops)
            for s in stops:
                ans, _ = step_answer(tr, s)
                sc = score_answer(tr, ans)
                score_total += weight * sc
                if s < e:
                    early_mass += weight
                    early_score += weight * sc
                else:
                    late_mass += weight
                    late_score += weight * sc
                    over_total += weight * (s - e)
                    # Safety-adjusted capture: an unsafe early stop captures no
                    # oracle-available saving, regardless of how many chunks it skipped.
                    saved += weight * captured_saving(s, e, nn)
        n = len(rows)
        ae = early_score / early_mass if early_mass else float("nan")
        al = late_score / late_mass if late_mass else float("nan")
        acc = score_total / n
        regret = oracle_acc - acc
        savecap = (saved / oracle_saved) if oracle_saved else float("nan")
        mo = over_total / late_mass if late_mass else float("nan")
        print(f"{name:22s} {early_mass/n:10.1%} {ae:10.3f} {al:9.3f} "
              f"{early_mass:8.1f} {late_mass:7.1f} {mo:8.2f} "
              f"{acc:6.3f} {regret:+7.3f} {savecap:8.1%}")
    print("\nacc|early = accuracy when the policy stopped before the evidence chunk.\n"
          "A high P(stop<ev) with acc|late >> acc|early means the stopping signal is\n"
          "premature, not that the reasoning is weak.\n"
          "overread = mean chunks consumed after the evidence chunk (late stops only).\n"
          "regret = oracle-stop accuracy minus this policy's accuracy (handoff Sec 5).\n"
          "savecap = safety-adjusted captured saving: premature stops receive zero credit.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("policies")
    p1.add_argument("--traj", required=True)
    p1.add_argument("--config", required=True)
    p1.add_argument("--correct-at", type=float, default=0.999)
    p1.add_argument("--verbalized-at", type=float, default=99.5,
                    help="numeric self-reported confidence threshold on the 0--100 scale")
    p1.add_argument("--judge-official", action="store_true",
                    help="use the official BrowseComp-Plus evaluator sampling "
                         "(temp 0.7, top_p 0.8, top_k 20, 4096 tokens) instead of "
                         "deterministic decoding")
    p1.add_argument("--judge", action="store_true",
                    help="grade answer_type=string with the BrowseComp-Plus semantic judge")
    p1.add_argument("--judge-model", default=None,
                    help="explicit deterministic semantic-judge model; defaults to "
                         "the trajectory model unless --judge-official is used")
    p4 = sub.add_parser("evidence")
    p4.add_argument("--traj", required=True)
    p4.add_argument("--config", required=True)
    p4.add_argument("--verbalized-at", type=float, default=99.5,
                    help="numeric self-reported confidence threshold on the 0--100 scale")
    p4.add_argument("--judge-official", action="store_true")
    p4.add_argument("--judge", action="store_true")
    p4.add_argument("--judge-model", default=None)
    a = ap.parse_args()
    CORRECT_AT_F1 = getattr(a, "correct_at", 0.999)   # f1 tasks only; see is_correct
    try:
        {"policies": cmd_policies, "evidence": cmd_evidence}[a.cmd](a)
    finally:
        if JUDGE is not None:
            JUDGE.flush()
