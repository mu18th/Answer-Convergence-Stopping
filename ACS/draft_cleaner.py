"""Auditable normalization and non-answer handling for open-answer drafts."""

from __future__ import annotations

import re


_NON_ANSWER = re.compile(
    r"\b("
    r"do(es)? not (contain|provide|specify|mention|include)"
    r"|not enough (information|details|context)"
    r"|insufficient (information|details|context)"
    r"|cannot (be )?(determin|answer|identif|establish)"
    r"|can(no|')t (be )?(determin|answer|identif)"
    r"|unable to (determin|answer|identif)"
    r"|is not (explicitly )?(mentioned|specified|provided|stated|given)"
    r"|no (relevant |such )?(information|details|record|mention)"
    r"|unknown from the (notes|context|information)"
    r"|i (do not|don't) know"
    r"|the answer is (unknown|undetermined)"
    r")",
    re.I,
)


def normalize_draft(text: str) -> str:
    """Normalize answer-label formatting without changing answer content."""
    value, previous = (text or "").strip(), None
    while value != previous:
        previous = value
        value = re.sub(
            r"^\s*(the\s+)?(final\s+)?answer\s*(is)?\s*[:\-]?\s*",
            "", value, flags=re.I,
        )
        value = re.sub(r"^[\s'\"`*(\[]+|[\s'\"`*.)\]]+$", "", value)
    return " ".join(value.lower().split())


def is_non_answer(text: str) -> bool:
    """Return true only for empty drafts or explicit abstention language.

    The list is deliberately conservative because a broad semantic classifier
    could suppress a valid answer.
    """
    value = (text or "").strip()
    normalized = normalize_draft(value)
    short_abstentions = {
        "unknown",
        "undetermined",
        "not known",
        "not available",
        "n/a",
    }
    return (
        not normalized
        or normalized in short_abstentions
        or bool(_NON_ANSWER.search(value))
    )
