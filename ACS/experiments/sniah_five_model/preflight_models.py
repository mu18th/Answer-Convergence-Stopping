"""Verify every OpenRouter model supports the exact S-NIAH call shapes."""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from common import chat, load_config, make_client
from run_five_models import BASE_CONFIG, MODELS


def checked_text(response, label: str) -> str:
    text = (response.choices[0].message.content or "").strip()
    if not text:
        raise RuntimeError(f"{label}: empty completion")
    return text


def check_model(model: str) -> None:
    cfg = load_config(str(BASE_CONFIG))
    cfg["server"]["model"] = model
    client = make_client(cfg)

    fold = chat(
        client, model,
        "Maintain short notes. Return only the updated notes.",
        "QUESTION: What number is hidden?\nCURRENT NOTES: (empty)\n"
        "NEW CHUNK: The hidden number is 314159.\nUPDATED NOTES:",
        max_tokens=128,
    )
    checked_text(fold, "fold")

    probe = chat(
        client, model, "",
        "QUESTION: What number is hidden?\nNOTES SO FAR: The hidden number is "
        "314159.\nReply with ONLY the answer.\nAnswer:",
        max_tokens=32, logprobs=True,
    )
    checked_text(probe, "answer probe")
    positions = getattr(getattr(probe.choices[0], "logprobs", None), "content", None)
    if not positions:
        raise RuntimeError("answer probe: endpoint returned no generated-token logprobs")
    if any(getattr(position, "logprob", None) is None for position in positions):
        raise RuntimeError("answer probe: a generated token is missing its logprob")

    confidence = chat(
        client, model, "",
        "QUESTION: What number is hidden?\nNOTES SO FAR: The hidden number is "
        "314159.\nHow confident are you (0-100) that the notes are sufficient "
        "to answer correctly? Reply with ONLY a number.\nConfidence:",
        max_tokens=32,
    )
    confidence_text = checked_text(confidence, "numeric confidence")
    numeric = re.fullmatch(r"\s*(\d{1,3}(?:\.\d+)?)\s*%?\s*", confidence_text)
    if not numeric or not 0 <= float(numeric.group(1)) <= 100:
        raise RuntimeError(
            f"numeric confidence: invalid response {confidence_text!r}"
        )

    gate = chat(
        client, model, "",
        "QUESTION: What number is hidden?\nNOTES SO FAR: The hidden number is "
        "314159.\nDecide whether the notes contain enough information. Return "
        "<next>end</next> or <next>continue</next>.\nDecision:",
        max_tokens=32,
    )
    gate_text = checked_text(gate, "END gate").lower()
    if not re.fullmatch(
        r"(?:<next>\s*(end|continue)\s*</next>|(end|continue))", gate_text
    ):
        raise RuntimeError(f"END gate: invalid response {gate_text!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise SystemExit("Set OPENROUTER_API_KEY first.")
    selected = MODELS if args.model == "all" else {args.model: MODELS[args.model]}
    failures = []
    for name, model in selected.items():
        print(f"PREFLIGHT {name}: {model}", flush=True)
        try:
            check_model(model)
        except Exception as exc:  # report all requested models before failing
            failures.append((name, repr(exc)))
            print(f"  FAIL: {exc}", flush=True)
        else:
            print("  PASS: fold + generated-token logprobs + confidence + gate", flush=True)
    if failures:
        details = "\n".join(f"  {name}: {error}" for name, error in failures)
        raise SystemExit("OpenRouter compatibility preflight failed:\n" + details)


if __name__ == "__main__":
    main()
