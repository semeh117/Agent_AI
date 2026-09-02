"""Shared infrastructure for Agent 2 CV and LinkedIn job parsers.

This module contains only cross-cutting parser contracts and utilities. The
CV-specific and job-specific extraction rules live in separate modules.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Optional

from pydantic import BaseModel

from config import get_parser_llm

CV_CACHE_VERSION = "agent2-cv-parser-v9-hybrid"
JOB_CACHE_VERSION = "agent2-job-parser-v12"
MAX_CV_SKILLS = 100
MAX_JOB_SKILLS = 45
MAX_RAW_CV_SKILLS = 200
MAX_RAW_JOB_SKILLS = 120
MIN_JOB_DESCRIPTION_CHARACTERS = 200


class Agent2ParserError(RuntimeError):
    """Raised when Agent 2 cannot obtain trustworthy structured extraction."""

def _cache_text(version: str, source_text: str) -> str:
    return f"{version}\n---\n{source_text}"


def _canonical_skill_key(value: str) -> str:
    """Normalize superficial variants without introducing domain categories."""

    text = str(value or "").casefold().replace("&", " and ")
    tokens = re.findall(r"[a-z0-9]+(?:\+\+|#)?", text)
    # Connectors vary in CVs (input/output vs input and output) but carry no
    # identity. Technology punctuation such as C++ and C# is retained above.
    tokens = [token for token in tokens if token not in {"and", "or"}]
    return " ".join(tokens)


def _skill_positions(value: str, source_text: str) -> list[int]:
    """Find literal skill occurrences without matching AI inside words like paid."""

    skill = str(value or "").strip()
    if not skill:
        return []
    pattern = re.compile(rf"(?<!\w){re.escape(skill)}(?!\w)", re.IGNORECASE)
    return [match.start() for match in pattern.finditer(source_text)]

#defense against hallucinations
def _ground_atomic_skills(
    skills: list[str],
    source_text: str,
    *,
    max_count: int,
) -> list[str]:
    """Deduplicate and retain concise skills explicitly present in the source."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in skills:
        skill = str(value or "").strip().strip("-•,;:. ")
        key = _canonical_skill_key(skill)
        if not skill or not key or key in seen:
            continue
        # Reject sentences and runaway generated combinations. Normal skill
        # names and short technical phrases remain well below these limits.
        if len(skill) > 70 or len(skill.split()) > 8:
            continue
        if not _skill_positions(skill, source_text):
            continue
        cleaned.append(skill)
        seen.add(key)
        if len(cleaned) >= max_count:
            break
    return cleaned

def _source_excerpt(value: str, source_text: str, radius: int = 120) -> str:
    """Return a compact source excerpt around the first exact occurrence."""

    index = source_text.casefold().find(value.casefold())
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(source_text), index + len(value) + radius)
    return " ".join(source_text[start:end].split())


def _evidence_map(skills: list[str], source_text: str) -> dict[str, str]:
    return {skill: _source_excerpt(skill, source_text) for skill in skills}

def _normalize_education(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    if "phd" in text or "doctor" in text:
        return "PhD"
    if "master" in text or "Engineering degree" in text:
        return "Master"
    if "bachelor" in text:
        return "Bachelor"
    if "high school" in text or "secondary" in text:
        return "High School"
    return None

def _invoke_structured_with_retry(
    *,
    schema: type[BaseModel],
    prompt: str,
    label: str,
    llm: Any = None,
    attempts: int = 2,
    validator: Optional[Callable[[BaseModel], None]] = None,
) -> BaseModel:
    if attempts < 1:
        raise ValueError("attempts must be at least 1.")
    model = llm or get_parser_llm(temperature=0.0)
    structured = model.with_structured_output(schema)
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        retry_note = ""
        if attempt > 1:
            previous_error = errors[-1] if errors else "invalid response"
            retry_note = (
                "\nIMPORTANT RETRY: The previous response was malformed. Return "
                "one short schema-valid object. Never repeat or combine skills. "
                f"Correct this problem: {previous_error[:220]}"
            )
        try:
            result = structured.invoke(prompt + retry_note)
            parsed = result if isinstance(result, schema) else schema.model_validate(result)
            if validator is not None:
                validator(parsed)
            return parsed
        except Exception as exc:
            errors.append(f"attempt {attempt}: {str(exc)[:350]}")
    raise Agent2ParserError(
        f"{label} parsing failed after {attempts} attempts: " + " | ".join(errors)
    )
