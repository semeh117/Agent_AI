"""CV-specific structured parsing for Agent 2."""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.agent2_parser_common import (
    Agent2ParserError,
    CV_CACHE_VERSION,
    MAX_CV_SKILLS,
    MAX_RAW_CV_SKILLS,
    _cache_text,
    _canonical_skill_key,
    _evidence_map,
    _ground_atomic_skills,
    _invoke_structured_with_retry,
    _normalize_education,
    _skill_positions,
)
from core.cv_parser import CVInfo
from core.extraction_cache import get_cached, set_cached


_CV_SECTION_HEADINGS = {
    "curriculum vitae",
    "resume",
    "professional summary",
    "profile summary",
    "profile",
    "technical skills",
    "skills",
    "professional experience",
    "work experience",
    "experience",
    "projects",
    "education",
    "certifications",
    "languages",
}

class Agent2CVInfo(CVInfo):
    """CVInfo plus deterministic source evidence for Agent 2 review."""

    skill_evidence: dict[str, str] = Field(default_factory=dict)

class _CVExtraction(BaseModel):
    full_name: Optional[str] = None
    skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_RAW_CV_SKILLS,
        description=(
            "Atomic technical names explicitly present in the CV. Extract list "
            "items after category headings, never the category heading itself."
        ),
    )
    contextual_skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_RAW_CV_SKILLS,
        description=(
            "Atomic technical skills explicitly present in summaries, work "
            "experience, projects, or certifications rather than a dedicated "
            "skills list."
        ),
    )
    job_titles: list[str] = Field(default_factory=list, max_length=20)
    experience_years: Optional[float] = Field(default=None, ge=0)
    education: list[str] = Field(default_factory=list, max_length=20)
    highest_education_level: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    mail: Optional[str] = None
    github: Optional[str] = None

def _looks_like_cv_fragment(skill: str) -> bool:
    """Reject obvious prose/metric fragments without using domain categories."""

    text = str(skill or "").strip()
    lowered = text.casefold()
    if re.search(r"\b\d+(?:\.\d+)?%", text):
        return True
    if lowered.startswith(("and ", "or ")):
        return True
    if lowered.endswith((" deployed", " per", " late")):
        return True
    # A period followed by another word or a colon in the middle normally
    # means the model joined two neighboring CV phrases into one item.
    if re.search(r"[.!?]\s+[A-Za-z]", text) or re.search(r"\w:\s+\w", text):
        return True
    if re.match(r"^\d+(?:st|nd|rd|th)\s+year\b", lowered):
        return True
    if lowered.startswith("integrated preparatory cycle"):
        return True
    return False


def _clean_cv_skills(skills: list[str], source_text: str) -> list[str]:
    """Ground CV skills and remove unmistakable extraction fragments."""

    grounded = _ground_atomic_skills(
        skills,
        source_text,
        max_count=MAX_RAW_CV_SKILLS,
    )
    filtered = [
        skill
        for skill in grounded
        if not _looks_like_cv_fragment(skill)
    ]

    # Models commonly return both ``MLflow (familiar)`` and ``MLflow`` or
    # ``AudioCNN (log-mel spectrograms)`` and ``AudioCNN``. Treat a trailing
    # parenthetical as evidence/context, not a separate skill identity, and
    # retain the shortest exact source form for clean embedding input.
    unique: list[str] = []
    positions: dict[str, int] = {}
    for skill in filtered:
        base = re.sub(r"\s*\([^()]*\)\s*$", "", skill).strip()
        identity = _canonical_skill_key(base or skill)
        if identity in {"rest api", "rest apis"}:
            identity = "rest"
        if identity not in positions:
            positions[identity] = len(unique)
            unique.append(skill)
        elif len(skill) < len(unique[positions[identity]]):
            unique[positions[identity]] = skill
    return unique[:MAX_CV_SKILLS]


def _explicit_cv_project_stack_items(source_text: str) -> list[str]:
    """Extract concise technologies from project stack lines separated by dots.

    Docling preserves lines such as ``Python · PyTorch · Scikit-learn``. These
    are explicit skill evidence and are more reliable than asking the LLM to
    rediscover the same names from surrounding project prose.
    """

    items: list[str] = []
    for line in source_text.splitlines():
        if "·" not in line:
            continue
        for part in line.split("·"):
            candidate = re.sub(
                r"\s+(?:in progress|completed|deployed)$",
                "",
                part.strip(),
                flags=re.IGNORECASE,
            ).strip()
            if candidate:
                items.append(candidate)
    return _clean_cv_skills(items, source_text)


def _looks_like_contextual_cv_skill(skill: str) -> bool:
    """Retain reusable technical methods while rejecting project features.

    This rule is domain-independent. Product features and domain entities are
    ordinary noun phrases; technical methods usually expose an acronym, a
    product-like spelling, a version, or an established method suffix.
    Explicit Skills-table and project-stack items bypass this filter.
    """

    text = str(skill or "").strip()
    words = re.findall(r"[A-Za-z0-9]+(?:\+\+|#)?", text)
    if not words or len(words) > 6 or " with " in text.casefold():
        return False
    if any(any(character.isdigit() for character in word) for word in words):
        return True
    if any(re.search(r"[a-z][A-Z]", word) for word in words):
        return True
    if any(len(word) >= 2 and word.isupper() for word in words):
        return True
    if len(words) == 1 and words[0][:1].isupper():
        return True
    suffixes = (
        "api",
        "apis",
        "architecture",
        "architectures",
        "caching",
        "calling",
        "chaining",
        "chunking",
        "classification",
        "clustering",
        "embeddings",
        "engineering",
        "evaluation",
        "extraction",
        "filtering",
        "fine-tuning",
        "forest",
        "framework",
        "frameworks",
        "generation",
        "harness",
        "harnesses",
        "health checks",
        "ingestion",
        "isolation",
        "monitoring",
        "observability",
        "orchestration",
        "pipeline",
        "pipelines",
        "preprocessing",
        "processing",
        "prompting",
        "regression",
        "removal",
        "reranker",
        "routing",
        "schema",
        "schemas",
        "search",
        "segmentation",
        "simulation",
        "vector store",
    )
    lowered = text.casefold()
    return any(lowered.endswith(suffix) for suffix in suffixes)


def _deterministic_markdown_name(source_text: str) -> Optional[str]:
    """Recover a candidate name from Docling's leading Markdown heading."""

    for line in source_text.splitlines()[:12]:
        match = re.match(r"^#{1,3}\s+(.+?)\s*$", line.strip())
        if not match:
            continue
        candidate = match.group(1).strip()
        key = candidate.casefold()
        words = candidate.split()
        if key in _CV_SECTION_HEADINGS or not 2 <= len(words) <= 6:
            return None
        if re.search(r"[\d@|:/]", candidate):
            return None
        return candidate
    return None


def _deterministic_plain_name(source_text: str) -> Optional[str]:
    """Recover a name from the first plain-text column without guessing."""

    for raw_line in source_text.splitlines()[:8]:
        line = raw_line.strip().lstrip("#").strip()
        if not line:
            continue
        # PDF text commonly separates the name and headline using several
        # spaces even when both are rendered on the same visual line.
        candidate = re.split(r"\s{2,}|\s+[|•]\s+", line, maxsplit=1)[0].strip()
        words = candidate.split()
        if (
            candidate.casefold() in _CV_SECTION_HEADINGS
            or not 2 <= len(words) <= 5
            or re.search(r"[\d@:/]", candidate)
            or not all(re.fullmatch(r"[A-Za-zÀ-ÖØ-öø-ÿ.'-]+", word) for word in words)
        ):
            continue
        return candidate
    return None


def _deterministic_education_level(source_text: str) -> Optional[str]:
    """Map explicit degree wording to the highest normalized level."""

    text = source_text.casefold()
    patterns = (
        ("PhD", r"\b(?:ph\.?\s*d\.?|doctor(?:ate|al)?)\b"),
        ("Master", r"\b(?:m\.?\s*s\.?|m\.?\s*sc\.?|master(?:'s)?|mba)\b"),
        (
            "Bachelor",
            r"\b(?:b\.?\s*s\.?|b\.?\s*sc\.?|b\.?\s*a\.?|bachelor(?:'s)?)\b",
        ),
        ("High School", r"\b(?:high school|secondary school)\b"),
    )
    for level, pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return level
    return None


def _category_headings(skills: list[str], source_text: str) -> list[str]:
    """Return extracted values that are actually colon-terminated headings."""

    headings = []
    for skill in skills:
        pattern = rf"(?im)^\s*{re.escape(str(skill).strip())}\s*:"
        if re.search(pattern, source_text):
            headings.append(str(skill).strip())
    return headings

def _explicit_cv_skill_items(source_text: str) -> list[str]:
    """Extract items from plain-text or Markdown technical-skills sections."""

    match = re.search(
        r"(?is)(?:^|\n)\s*#{0,6}\s*technical skills\s*\n(.*?)"
        r"(?=\n\s*#{0,6}\s*(?:education|work experience|experience|projects|certifications|languages)\s*\n|\Z)",
        source_text,
    )
    if not match:
        return []
    items: list[str] = []
    for line in match.group(1).splitlines():
        stripped = line.strip()
        values = ""
        if stripped.startswith("|") and stripped.endswith("|"):
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            category = _canonical_skill_key(cells[0]) if cells else ""
            if category in {"language", "languages"}:
                continue
            if len(cells) >= 2 and not all(set(cell) <= {"-", ":"} for cell in cells):
                values = cells[-1]
        elif ":" in stripped:
            _, values = stripped.split(":", 1)
        if not values:
            continue
        items.extend(part.strip() for part in values.split(","))
    return _clean_cv_skills(items, source_text)


_MONTH_NUMBERS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
_MONTH_PATTERN = "|".join(
    sorted(_MONTH_NUMBERS, key=len, reverse=True)
)
_CV_DATE_RANGE = re.compile(
    rf"\b({_MONTH_PATTERN})\.?\s+(\d{{4}})\s*"
    rf"(?:-|–|—|to)\s*"
    rf"(?:(present|current)|({_MONTH_PATTERN})\.?\s+(\d{{4}}))\b",
    re.IGNORECASE,
)


def _cv_experience_region(source_text: str) -> str:
    """Return the explicit work-experience section when one is identifiable."""

    start_match = re.search(
        r"(?im)^\s*#{0,6}\s*(?:professional\s+|work\s+)?experience\s*$",
        source_text,
    )
    if not start_match:
        return ""
    tail = source_text[start_match.end():]
    end_match = re.search(
        r"(?im)^\s*#{0,6}\s*(?:education|technical skills|skills|projects|certifications|languages)\s*$",
        tail,
    )
    return tail[: end_match.start()] if end_match else tail


def _deterministic_experience_years(source_text: str) -> Optional[float]:
    """Calculate inclusive, non-overlapping month ranges from Experience."""

    region = _cv_experience_region(source_text)
    if not region:
        return None

    intervals: list[tuple[int, int]] = []
    today = date.today()
    for match in _CV_DATE_RANGE.finditer(region):
        start_month = _MONTH_NUMBERS[match.group(1).casefold()]
        start_year = int(match.group(2))
        if match.group(3):
            end_month, end_year = today.month, today.year
        else:
            end_month = _MONTH_NUMBERS[match.group(4).casefold()]
            end_year = int(match.group(5))
        start = start_year * 12 + start_month - 1
        end = end_year * 12 + end_month - 1
        if end >= start:
            intervals.append((start, end))

    if not intervals:
        return None
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    months = sum(end - start + 1 for start, end in merged)
    return round(months / 12.0, 2)

def extract_cv_info_agent2(
    cv_text: str,
    llm: Any = None,
    use_cache: bool = True,
    *,
    layout_text: Optional[str] = None,
    cache_identity: Optional[str] = None,
) -> CVInfo:
    """Parse one CV using sequential text plus optional Docling structure.

    ``cv_text`` is the stable PyPDF view used for metadata and the LLM call.
    ``layout_text`` is Docling Markdown used to recover structured skill lists.
    Both inputs are grounded into one unchanged ``CVInfo`` result.
    """

    source = str(cv_text or "").strip()
    if not source:
        raise ValueError("CV text cannot be empty.")
    layout_source = str(layout_text or "").strip() or source
    grounding_source = source
    if layout_source != source:
        grounding_source = f"{source}\n\n--- LAYOUT VIEW ---\n{layout_source}"
    identity = (
        f"pdf-sha256:{cache_identity.strip()}"
        if cache_identity and cache_identity.strip()
        else grounding_source
    )
    cache_key = _cache_text(CV_CACHE_VERSION, identity)
    if use_cache:
        cached = get_cached("agent2_cv", cache_key, Agent2CVInfo)
        if cached is not None:
            print(
                "  [CACHE HIT] Agent 2 CV already parsed — "
                f"{len(cached.skills)} skills."
            )
            return cached

    prompt = f"""Extract a CV into the supplied schema.

Rules:
- Read the complete CV.
- skills: explicit technical skills copied from dedicated Skills/Technology
  lists. Extract every item after each category heading.
- contextual_skills: explicit technical skills copied from the profile summary,
  work experience, projects, and certifications. Do not leave this empty when
  those sections name technologies or technical methods.
- Do not return project features, business/domain entities, dataset subjects,
  UI features, outcomes, or input data as skills. For example, an illness being
  classified, a team-form feature, or an outfit-rating feature is not a skill.
- Scan every work-experience bullet individually. Skills such as prompt
  engineering, retrieval-augmented generation, fine-tuning, function calling,
  APIs, evaluation methods, and deployment methods count only when those exact
  concepts occur in the CV; do not infer them from the candidate's role.
- Combined skills and contextual_skills: at most {MAX_CV_SKILLS} unique items.
- Every skill must be one atomic name copied from the CV, normally 1-5 words.
- For a line such as "Cloud: AWS, Azure, GCP", output AWS, Azure, and GCP.
  Never output "Cloud". Apply this to every colon-separated category list.
- Category headings are not skills. Do not infer related technologies.
- Deduplicate skills case-insensitively.
- job_titles: positions actually held, not the profile headline.
- Calculate non-overlapping professional experience from dated work entries.
  For "Present", use {date.today().isoformat()}.
- highest_education_level must be High School, Bachelor, Master, PhD, or null.
- Never invent contact details.
- Return one concise schema-valid object; no commentary.

CV:
---
{source}
---"""

    def validate_cv(parsed: BaseModel) -> None:
        all_extracted_skills = parsed.skills + parsed.contextual_skills
        headings = _category_headings(all_extracted_skills, grounding_source)
        if headings:
            raise ValueError(
                "category headings were returned as skills: "
                + ", ".join(headings[:8])
                + "; extract the individual items after each colon instead"
            )
        if not _clean_cv_skills(all_extracted_skills, grounding_source):
            raise ValueError("response contained no grounded technical skills")

    extraction = _invoke_structured_with_retry(
        schema=_CVExtraction,
        prompt=prompt,
        label="CV",
        llm=llm,
        validator=validate_cv,
    )
    data = extraction.model_dump()
    explicit_skills = _clean_cv_skills(
        _explicit_cv_skill_items(source)
        + _explicit_cv_project_stack_items(source)
        + _explicit_cv_skill_items(layout_source)
        + _explicit_cv_project_stack_items(layout_source),
        grounding_source,
    )
    explicit_keys = {_canonical_skill_key(skill) for skill in explicit_skills}
    contextual_skills = _clean_cv_skills(
        data.pop("contextual_skills", []) + data["skills"],
        grounding_source,
    )
    contextual_skills = [
        skill
        for skill in contextual_skills
        if _canonical_skill_key(skill) in explicit_keys
        or _looks_like_contextual_cv_skill(skill)
    ]
    data["skills"] = _clean_cv_skills(
        explicit_skills + contextual_skills,
        grounding_source,
    )
    deterministic_experience = _deterministic_experience_years(source)
    if deterministic_experience is not None:
        data["experience_years"] = deterministic_experience
    deterministic_name = (
        _deterministic_plain_name(source)
        or _deterministic_markdown_name(layout_source)
    )
    if deterministic_name is not None:
        data["full_name"] = deterministic_name
    elif str(data.get("full_name") or "").strip().casefold() in _CV_SECTION_HEADINGS:
        data["full_name"] = None
    data["highest_education_level"] = (
        _deterministic_education_level(source)
        or _normalize_education(data["highest_education_level"])
    )
    for field in ("phone", "linkedin", "mail", "github"):
        value = data.get(field)
        if value and str(value).casefold() not in grounding_source.casefold():
            data[field] = None
    data["skill_evidence"] = _evidence_map(data["skills"], grounding_source)
    result = Agent2CVInfo.model_validate(data)
    if not result.skills:
        raise Agent2ParserError(
            "CV parsing produced no grounded technical skills; refusing an empty profile."
        )
    if use_cache:
        set_cached("agent2_cv", cache_key, result)
    return result
