from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypedDict

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from openai import OpenAI
from common import chat

# OpenRouter's canonical identifier for the evaluator weights.
OFFICIAL_JUDGE_MODEL: Final = "qwen/qwen3-32b"
OFFICIAL_EVALUATOR_COMMIT: Final = "046949032b0328319cc9a02663a759ec601d9402"
PROTOCOL_ID: Final = f"browsecomp-plus-official_{OFFICIAL_EVALUATOR_COMMIT[:7]}"
GRADER_TEMPLATE: Final = Path(__file__).with_name(
    "browsecomp_plus_grader_prompt.txt"
).read_text().strip()
_GRADER_TEMPLATE_HASH: Final = hashlib.sha256(
    GRADER_TEMPLATE.encode()).hexdigest()[:16]


class JudgeParseError(RuntimeError):
    """The grader's verdict could not be read.

    Distinct from "the answer was wrong". `parsed.correct` is False in this case
    only because there is nothing to say otherwise, and scoring it 0 counts a
    grader malfunction as a wrong answer by the system under test. The two arms
    produce different response styles, so those malfunctions are not guaranteed to
    fall symmetrically.
    """


@dataclass(frozen=True, slots=True)
class JudgeSettings:
    model: str = OFFICIAL_JUDGE_MODEL
    temperature: float = 0.7
    top_p: float | None = 0.8
    top_k: int | None = 20
    max_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class JudgeItem:
    question: str
    response: str
    correct_answer: str


@dataclass(frozen=True, slots=True)
class JudgeParsed:
    extracted_final_answer: str | None
    reasoning: str | None
    correct: bool
    confidence: float | None
    parse_error: bool


@dataclass(frozen=True, slots=True)
class JudgeOutcome:
    parsed: JudgeParsed
    raw_response: str
    cached: bool

    @property
    def correct(self) -> bool:
        return self.parsed.correct


class CachedJudge(TypedDict):
    protocol_id: str
    judge_model: str
    extracted_final_answer: str | None
    reasoning: str | None
    correct: bool
    confidence: float | None
    parse_error: bool
    raw_response: str


def _first_match(patterns: tuple[str, ...], text: str, flags: int) -> str | None:
    for pattern in patterns:
        matched = re.search(pattern, text, flags)
        if matched:
            return matched.group(1).strip()
    return None


def parse_judge_response(text: str) -> JudgeParsed:
    extracted = _first_match((
        r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
        r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
        r"extracted_final_answer:\s*(.*?)(?=\n|$)",
    ), text, re.IGNORECASE | re.DOTALL)
    reasoning = _first_match((
        r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
        r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
    ), text, re.IGNORECASE | re.DOTALL)
    # The word boundary prevents ``correct:`` from matching inside
    # ``incorrect:`` and inverting the parsed verdict.
    correct_text = _first_match((
        r"\*\*correct:\*\*\s*(yes|no)",
        r"\*\*correct\*\*:\s*(yes|no)",
        r"\bcorrect:\s*(yes|no)",
    ), text, re.IGNORECASE)
    confidence_text = _first_match((
        r"\*\*confidence:\*\*\s*(\d+(?:\.\d+)?)\s*%?",
        r"\*\*confidence\*\*:\s*(\d+(?:\.\d+)?)\s*%?",
        r"confidence:\s*(\d+(?:\.\d+)?)\s*%?",
    ), text, re.IGNORECASE)
    confidence = min(float(confidence_text), 100.0) if confidence_text else None
    return JudgeParsed(
        extracted_final_answer=extracted,
        reasoning=reasoning,
        correct=correct_text is not None and correct_text.lower() == "yes",
        confidence=confidence,
        parse_error=correct_text is None,
    )


class BrowseCompPlusJudge:
    def __init__(
        self,
        client: OpenAI,
        settings: JudgeSettings,
        cache_path: Path,
    ) -> None:
        self._client = client
        self._settings = settings
        self._cache_path = cache_path
        self._cache: dict[str, CachedJudge] = {}
        self._pending = 0
        if cache_path.exists():
            self._cache = json.loads(cache_path.read_text(encoding="utf-8"))

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _key(self, item: JudgeItem) -> str:
        settings = self._settings
        material = "\x00".join((
            PROTOCOL_ID,
            # The grader prompt is part of the measurement, so its hash is part
            # of the cache key.
            _GRADER_TEMPLATE_HASH,
            settings.model,
            str(settings.temperature),
            str(settings.top_p),
            str(settings.top_k),
            str(settings.max_tokens),
            item.question,
            item.response,
            item.correct_answer,
        ))
        return hashlib.sha256(material.encode()).hexdigest()

    def judge(self, item: JudgeItem) -> JudgeOutcome:
        if not item.response.strip():
            parsed = JudgeParsed(None, None, False, None, False)
            return JudgeOutcome(parsed, "", False)

        key = self._key(item)
        cached = self._cache.get(key)
        if cached is not None:
            parsed = JudgeParsed(
                extracted_final_answer=cached["extracted_final_answer"],
                reasoning=cached["reasoning"],
                correct=cached["correct"],
                confidence=cached["confidence"],
                parse_error=cached["parse_error"],
            )
            return JudgeOutcome(parsed, cached["raw_response"], True)

        settings = self._settings
        prompt = GRADER_TEMPLATE.format(
            question=item.question,
            response=item.response,
            correct_answer=item.correct_answer,
        )
        raw_response = ""
        parsed = parse_judge_response(raw_response)
        for parse_attempt in range(1, 5):
            result = chat(
                self._client, settings.model, "", prompt,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
                top_p=settings.top_p,
                top_k=settings.top_k,
            )
            raw_response = result.choices[0].message.content or ""
            parsed = parse_judge_response(raw_response)
            if not parsed.parse_error:
                break
            if parse_attempt < 4:
                time.sleep(min(2 ** (parse_attempt - 1), 8))
        if parsed.parse_error:
            raise JudgeParseError(
                "grader verdict unreadable after 4 identical attempts; "
                f"raw tail={raw_response[-200:]!r}"
            )
        self._cache[key] = CachedJudge(
            protocol_id=PROTOCOL_ID,
            judge_model=settings.model,
            extracted_final_answer=parsed.extracted_final_answer,
            reasoning=parsed.reasoning,
            correct=parsed.correct,
            confidence=parsed.confidence,
            parse_error=parsed.parse_error,
            raw_response=raw_response,
        )
        self._pending += 1
        # Each entry carries a full raw_response (up to max_tokens), so rewriting the
        # whole cache after every verdict is quadratic in bytes written and grows
        # ~16 KB per judgement. Flush periodically instead; a crash costs at most
        # _FLUSH_EVERY verdicts, which are recomputable. flush() forces a final write.
        if self._pending >= self._FLUSH_EVERY:
            self.flush()
        return JudgeOutcome(parsed, raw_response, False)

    _FLUSH_EVERY: Final = 25

    def flush(self) -> None:
        """Persist the cache. Call once when a judging pass finishes."""
        if not self._pending:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._cache_path.with_suffix(self._cache_path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._cache, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self._cache_path)   # atomic: a crash mid-write cannot corrupt it
        self._pending = 0
