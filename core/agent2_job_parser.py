"""LinkedIn-job-specific structured parsing for Agent 2."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from core.agent2_parser_common import (
    Agent2ParserError,
    JOB_CACHE_VERSION,
    MAX_JOB_SKILLS,
    MAX_RAW_JOB_SKILLS,
    MIN_JOB_DESCRIPTION_CHARACTERS,
    _cache_text,
    _canonical_skill_key,
    _evidence_map,
    _ground_atomic_skills,
    _invoke_structured_with_retry,
    _skill_positions,
)
from core.extraction_cache import get_cached, set_cached
from core.job_parser import JobRequirements

class Agent2JobRequirements(JobRequirements):
    """Job requirements with preferred skills kept outside cosine scoring."""

    preferred_skills: list[str] = Field(default_factory=list)
    required_skill_groups: list[list[str]] = Field(default_factory=list)
    preferred_education_level: Optional[str] = None
    required_skill_evidence: dict[str, str] = Field(default_factory=dict)
    preferred_skill_evidence: dict[str, str] = Field(default_factory=dict)

class _JobExtraction(BaseModel):
    required_skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_RAW_JOB_SKILLS,
        description=(
            "Unique atomic mandatory/core technical skills copied from the job; "
            "never sentences, generated combinations, or optional skills."
        ),
    )
    responsibility_skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_RAW_JOB_SKILLS,
        description=(
            "Concrete technical skills and methods explicitly used in the "
            "role's core responsibilities. Exclude generic action words."
        ),
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        max_length=MAX_RAW_JOB_SKILLS,
        description=(
            "Atomic technical skills explicitly marked preferred, optional, "
            "bonus, plus, or nice-to-have."
        ),
    )
    job_title: Optional[str] = None
    seniority_level: Optional[str] = None
    required_experience_years: Optional[float] = Field(default=None, ge=0)
    required_education_level: Optional[str] = None
    preferred_education_level: Optional[str] = None


_VAGUE_STANDALONE_JOB_SKILLS = {
    "analysis",
    "adapting",
    "building",
    "coding",
    "collaborating",
    "curating",
    "curation",
    "designing",
    "deployment",
    "developing",
    "driving",
    "engineering",
    "evaluation",
    "experimentation",
    "filtering",
    "implementing",
    "ideation",
    "inference",
    "integrations",
    "integrating",
    "investigating",
    "learning",
    "leveraging",
    "maintain",
    "orchestration",
    "planning",
    "semantic",
    "productionizing",
    "reasoning",
    "scaling",
    "serving",
    "sourcing",
    "training",
    "validation",
    "optimization",
}

_NON_TECHNICAL_JOB_SKILLS = {
    "ability to obtain sponsorship",
    "accelerated development",
    "ai driven solutions",
    "business outcomes",
    "best practices",
    "cost",
    "customer success",
    "data",
    "data teams",
    "employee productivity",
    "engineering team",
    "efficiency",
    "establish",
    "gtm",
    "growth",
    "high quality code",
    "identify",
    "internal adoption rate",
    "internal team enablement",
    "internal team velocity",
    "marketing",
    "models",
    "modular composable services",
    "operational cost savings",
    "operations",
    "other machine learning technologies",
    "product",
    "product lifecycle",
    "prompts",
    "reduce operational costs",
    "rapid iteration",
    "reliable internal tools",
    "reliability",
    "sales",
    "scalability",
    "security clearance",
    "technical architecture",
    "team learning sessions",
    "translate",
    "engineering standards",
    "maintainability",
    "agents",
    "skills",
    "travel",
    "u s citizenship",
    "us citizenship",
    "write maintainable",
    "peer reviews",
    "performance",
    "document",
    "code review discipline",
    "collaborative cross functional team",
    "diverse ideas",
    "engineering teams",
    "enterprise environment",
    "failing fast",
    "flexible hybrid working environment",
    "innovative ideas",
    "large analytical datasets",
    "libraries and apis",
    "minimum viable products",
    "new approaches",
    "packaging standards",
    "production grade coding practices",
    "problem solving skills",
    "production readiness reviews",
    "code reviews",
    "similar ranking",
    "fast paced",
    "projects with clearly defined scope",
    "technical and data exploration expertise",
    "timely well managed deliverables",
    "written and verbal communication skills",
    "vector other databases",
    "technology consulting",
    "federal health space",
    "translate business requirements into technical specifications",
}

_NON_SKILL_PHRASE_MARKERS = (
    "adoption rate",
    "business outcome",
    "citizenship",
    "cost saving",
    "employee productivity",
    "operational cost",
    "security clearance",
    "team velocity",
    "travel requirement",
    "adoption of",
    "business problem",
    "communication skill",
    "cross functional team",
    "decision making",
    "deliverable",
    "engineering team",
    "enterprise environment",
    "failing fast",
    "innovative idea",
    "minimum viable product",
    "new approach",
    "project with",
    "problem solving",
    "stakeholder",
    "communication of",
    "working environment",
    "agile environment",
    "compliance",
    "diverse experience",
    "use case",
)

_GENERIC_ACTION_PREFIX = re.compile(
    r"^(?:achiev(?:e|ing)|assist(?:s|ing)?|automat(?:e|ing)|build(?:s|ing)?|"
    r"break(?:s|ing)?|collaborat(?:e|ing)|contribut(?:e|ing)|coordinat(?:e|ing)|design(?:s|ing)?|deploy(?:s|ing)?|"
    r"develop(?:s|ing)?|driv(?:e|ing)|enhanc(?:e|ing)|ensur(?:e|ing)|"
    r"evaluat(?:e|ing)|formulat(?:e|ing)|ideat(?:e|ing|ion)|implement(?:s|ing)?|increas(?:e|ing)|integrat(?:e|ing)|interact(?:s|ing)?|"
    r"leverag(?:e|ing)|optimiz(?:e|ing)|participat(?:e|ing)|partner(?:s|ing)?|productioniz(?:e|ing)|"
    r"reduc(?:e|ing)|remain(?:s|ing)?|scal(?:e|ing))\b"
)

_SKILL_WRAPPER_PREFIXES = (
    "deep experience leveraging ",
    "hands-on experience with ",
    "hands-on work with ",
    "knowledge of ",
    "exposure to ",
    "public presence on ",
    "strong proficiency with ",
    "experience with ",
    "experience in ",
    "familiarity with ",
    "proficiency with ",
    "proficiency in ",
)

_OPTIONAL_MARKERS = (
    "a plus",
    "is a plus",
    "preferred",
    "optional",
    "bonus",
    "nice to have",
    "nice-to-have",
)

_REQUIREMENT_LANGUAGE = re.compile(
    r"\b(?:required|requirements?|must|need(?:ed)?|expect(?:ed)?|"
    r"experience|expertise|proficien(?:cy|t)|knowledge|familiarity|"
    r"hands-on|mastery|proven|ability|strong|solid)\b",
    re.IGNORECASE,
)


_JOB_SECTION_HEADINGS: tuple[tuple[str, str], ...] = (
    # Longer/specific headings are intentionally listed before their suffixes.
    ("roles & responsibilities", "responsibility"),
    ("role/responsibilities", "responsibility"),
    ("scope of responsibilities", "responsibility"),
    ("primary responsibilities", "responsibility"),
    ("key responsibilities", "responsibility"),
    ("your responsibilities", "responsibility"),
    ("what you'll do", "responsibility"),
    ("what you’ll do", "responsibility"),
    ("what you will do", "responsibility"),
    ("what you'll be doing", "responsibility"),
    ("what you’ll be doing", "responsibility"),
    ("about the opportunity", "responsibility"),
    ("role overview", "responsibility"),
    ("job description", "responsibility"),
    ("about the role", "responsibility"),
    ("responsibilities", "responsibility"),
    ("must have technical/functional skills", "required"),
    ("must-have technical/functional skills", "required"),
    ("required skills & experience", "required"),
    ("skills & experience", "required"),
    ("minimum qualifications", "required"),
    ("required qualifications", "required"),
    ("basic qualifications", "required"),
    ("required skills", "required"),
    ("about you", "required"),
    ("who you are", "required"),
    ("what you bring", "required"),
    ("what you'll need", "required"),
    ("what you’ll need", "required"),
    ("what we are looking for", "required"),
    ("what we're looking for", "required"),
    ("your technical toolkit", "required"),
    ("technical toolkit", "required"),
    ("qualifications", "required"),
    ("requirements", "required"),
    ("preferred qualifications", "preferred"),
    ("desired qualifications", "preferred"),
    ("preferred skills", "preferred"),
    ("desired skills", "preferred"),
    ("strong candidates have", "preferred"),
    ("nice to have", "preferred"),
    ("nice-to-have", "preferred"),
    ("bonus qualifications", "preferred"),
    ("the following skills and tools are preferred, but not required", "preferred"),
    ("how we define success", "ignored"),
    ("personal attributes", "ignored"),
    ("how we work", "ignored"),
    ("what we offer", "ignored"),
    ("eeo and accommodations", "ignored"),
    ("key metrics", "ignored"),
    ("company description", "ignored"),
    ("about the company", "ignored"),
    ("about us", "ignored"),
    ("benefits", "ignored"),
    ("compensation", "ignored"),
    ("equal opportunity", "ignored"),
)

_GENERIC_SINGLE_WORD_HEADINGS = {
    "benefits",
    "compensation",
    "qualifications",
    "requirements",
    "responsibilities",
}


@dataclass(frozen=True)
class _SectionHeading:
    start: int
    end: int
    label: str
    kind: str


@dataclass(frozen=True)
class _JobRegions:
    responsibility: str
    required: str
    preferred: str
    ignored: str
    headings: tuple[str, ...]
    has_explicit_required_heading: bool

def _is_vague_job_skill(skill: str) -> bool:
    """Reject context-free action nouns while retaining qualified phrases."""

    return skill.casefold().strip() in _VAGUE_STANDALONE_JOB_SKILLS


def _looks_non_technical_job_skill(skill: str) -> bool:
    """Reject eligibility, departments, metrics, and business outcome labels."""

    key = _canonical_skill_key(skill)
    if re.match(r"^\d+(?:\.\d+)?\+?\s+years?\b", key):
        return True
    if key in {
        _canonical_skill_key(value) for value in _NON_TECHNICAL_JOB_SKILLS
    }:
        return True
    if _GENERIC_ACTION_PREFIX.match(key):
        return True
    return any(marker in key for marker in _NON_SKILL_PHRASE_MARKERS)


def _looks_like_education(skill: str) -> bool:
    lowered = skill.casefold()
    return any(
        marker in lowered
        for marker in (
            "bachelor",
            "master's degree",
            "master’s degree",
            "doctoral degree",
            "doctorate",
            "phd",
            "high school diploma",
        )
    )


def _clean_job_skills(
    skills: list[str], source_text: str, *, max_count: int
) -> list[str]:
    expanded: list[str] = []
    for value in skills:
        skill = str(value or "").strip()
        lowered = skill.casefold()
        for prefix in _SKILL_WRAPPER_PREFIXES:
            if lowered.startswith(prefix):
                skill = skill[len(prefix) :].strip()
                lowered = skill.casefold()
                break

        if (
            _is_vague_job_skill(skill)
            or _looks_non_technical_job_skill(skill)
            or _looks_like_education(skill)
        ):
            continue

        examples_match = re.match(
            r"^(.*?)\s+such as\s+(.+)$",
            skill,
            flags=re.IGNORECASE,
        )
        if examples_match:
            base, examples = examples_match.groups()
            if base.strip():
                expanded.append(base.strip())
            for part in re.split(
                r"\s*,\s*|\s+(?:and/or|and|or)\s+",
                examples,
                flags=re.IGNORECASE,
            ):
                candidate = part.strip(" .")
                if candidate:
                    expanded.append(candidate)
            continue

        match = re.match(
            r"^(.*?)\s*\(([^()]*)\)(?:\s+.*)?$",
            skill,
        )
        if match:
            base, parenthetical = match.groups()
            base = re.sub(r"\s+experience$", "", base, flags=re.IGNORECASE).strip()
            if base:
                expanded.append(base)
            # Parentheses commonly contain named examples. Split commas and
            # table-style bars, while discarding non-skill continuations.
            if "," in parenthetical or "|" in parenthetical:
                for part in re.split(r"\s*[,|]\s*", parenthetical):
                    candidate = re.sub(
                        r"^(?:and|or)\s+", "", part.strip(), flags=re.IGNORECASE
                    ).strip(" .)")
                    if candidate.casefold() not in {
                        "",
                        "equivalent",
                        "etc",
                        "etc.",
                    }:
                        expanded.append(candidate)
            continue

        if "," in skill:
            for part in skill.split(","):
                candidate = re.sub(
                    r"^(?:and|or)\s+",
                    "",
                    part.strip(),
                    flags=re.IGNORECASE,
                )
                if candidate:
                    expanded.append(candidate)
            continue

        if " & " in skill:
            parts = [part.strip() for part in skill.split(" & ")]
            named_part = any(
                re.search(r"[a-z][A-Z]", part)
                or re.search(r"\b[A-Z]{2,}\b", part)
                for part in parts
            )
            if named_part and all(parts):
                expanded.extend(parts)
                continue

        # Split compact alternative technology names while preserving common
        # compound practices whose slash is part of one established concept.
        protected_slashes = (
            "ci/cd",
            "input/output",
            "client/server",
            "and/or",
            "retry/repair",
        )
        if (
            1 <= skill.count("/") <= 3
            and not any(marker in lowered for marker in protected_slashes)
        ):
            parts = [part.strip() for part in skill.split("/")]
            if all(parts) and all(re.search(r"[A-Za-z]", part) for part in parts):
                expanded.extend(parts)
                continue

        skill = re.sub(r"\s+experience$", "", skill, flags=re.IGNORECASE).strip()
        expanded.append(skill)

    grounded = _ground_atomic_skills(expanded, source_text, max_count=max_count * 2)
    cleaned: list[str] = []
    seen: set[str] = set()
    for skill in grounded:
        if (
            _is_vague_job_skill(skill)
            or _looks_non_technical_job_skill(skill)
            or _looks_like_education(skill)
        ):
            continue
        key = _canonical_skill_key(skill)
        if not key or key in seen:
            continue
        cleaned.append(skill)
        seen.add(key)
        if len(cleaned) >= max_count:
            break

    # Prefer a qualified atomic phrase when the model returns both it and a
    # substring from the same source occurrence (agent coordination versus
    # multi-agent coordination). Single-token concepts are not collapsed.
    specific: list[str] = []
    for skill in cleaned:
        key = _canonical_skill_key(skill)
        key_words = key.split()
        skip = False
        replace_indexes: list[int] = []
        for index, existing in enumerate(specific):
            existing_key = _canonical_skill_key(existing)
            existing_words = existing_key.split()
            if len(key_words) >= 2 and f" {key} " in f" {existing_key} ":
                skip = True
                break
            if (
                len(existing_words) >= 2
                and f" {existing_key} " in f" {key} "
            ):
                replace_indexes.append(index)
        if skip:
            continue
        for index in reversed(replace_indexes):
            specific.pop(index)
        specific.append(skill)
    return specific[:max_count]


def _required_alternative_groups(
    required_skills: list[str], required_region: str
) -> list[list[str]]:
    """Recover small, local alternative/example groups from requirements.

    Groups are metadata for scoring, not embedding inputs. Only explicit
    alternatives (``one or more of``, ``A or B``, ``A/B``) and examples after
    ``such as`` qualify. Plain ``including`` is deliberately excluded because
    it often introduces mandatory components or experience subsets, as in
    ``8 years in ML, including 2 years in Generative AI``.
    """

    max_group_size = 8
    grounded_skills = _ground_atomic_skills(
        required_skills,
        required_region,
        max_count=MAX_JOB_SKILLS,
    )
    if len(grounded_skills) < 2:
        return []

    # LinkedIn sometimes flattens bullets. These qualification lead-ins are
    # therefore treated as boundaries in addition to real newlines and normal
    # sentence punctuation.
    segment_boundary = re.compile(
        r"(?:\r?\n+|[.;]+|,\s+plus\b|"
        r"(?=\s+(?:[-•]\s*)?(?:strong\b|hands-on\b|real-world\b|"
        r"familiarity\b|proficiency\b|solid foundation\b|working knowledge\b|"
        r"experience\s+(?:with|in|designing|deploying|building)\b|"
        r"knowledge\b|expertise\b|excellent\b|comfort\b|"
        r"bachelor(?:['’]s)?\b|master(?:['’]s)?\b)))",
        flags=re.IGNORECASE,
    )
    segments = [
        segment.strip(" \t:-–—•")
        for segment in segment_boundary.split(required_region)
        if segment.strip(" \t:-–—•")
    ]

    candidates: list[tuple[int, int, list[str]]] = []
    seen_candidates: set[tuple[str, ...]] = set()

    def occurrences(text: str) -> list[tuple[int, int, str]]:
        found: list[tuple[int, int, str]] = []
        for skill in grounded_skills:
            for position in _skill_positions(skill, text):
                found.append((position, position + len(skill), skill))
        found.sort(key=lambda item: (item[0], -(item[1] - item[0])))

        unique: list[tuple[int, int, str]] = []
        identities: set[str] = set()
        for item in found:
            identity = _canonical_skill_key(item[2])
            if identity and identity not in identities:
                unique.append(item)
                identities.add(identity)
        return unique

    def add_candidate(
        values: list[str], *, priority: int, source_order: int
    ) -> None:
        group = _ground_atomic_skills(
            values,
            required_region,
            max_count=max_group_size + 1,
        )
        if not 2 <= len(group) <= max_group_size:
            return
        identity = tuple(sorted(_canonical_skill_key(skill) for skill in group))
        if identity in seen_candidates:
            return
        candidates.append((priority, source_order, group))
        seen_candidates.add(identity)

    for source_order, segment in enumerate(segments):
        segment_occurrences = occurrences(segment)
        if len(segment_occurrences) < 2:
            continue

        # The marker itself is sufficient evidence of alternatives.
        for marker in re.finditer(
            r"\bone or more of\s*:?\s*", segment, flags=re.IGNORECASE
        ):
            add_candidate(
                [
                    skill
                    for start, _end, skill in segment_occurrences
                    if start >= marker.end()
                ],
                priority=4,
                source_order=source_order,
            )

        # ``such as`` expresses one broad capability with named examples. Keep
        # the closest preceding parsed capability and the local examples.
        for marker in re.finditer(r"\bsuch as\s+", segment, flags=re.IGNORECASE):
            preceding = [
                skill
                for start, _end, skill in segment_occurrences
                if start < marker.start()
            ]
            examples = [
                skill
                for start, _end, skill in segment_occurrences
                if start >= marker.end()
            ]
            add_candidate(
                ([preceding[-1]] if preceding else []) + examples,
                priority=2,
                source_order=source_order,
            )

        # An explicit ``or`` makes the skills in that local clause alternatives.
        # Splitting on ``, plus`` prevents a following requirement from joining
        # the same group (AWS/Azure/GCP, plus Docker/Kubernetes).
        for clause in re.split(r",\s+plus\b|\bplus\b", segment, flags=re.IGNORECASE):
            if re.search(r"\b(?:or|and/or)\b", clause, flags=re.IGNORECASE):
                add_candidate(
                    [skill for _start, _end, skill in occurrences(clause)],
                    priority=3,
                    source_order=source_order,
                )

        # Slash-separated atomic names are local alternatives. Established
        # compound labels such as CI/CD remain one parsed skill and therefore
        # never appear as two adjacent occurrences here.
        for left, right in zip(segment_occurrences, segment_occurrences[1:]):
            separator = segment[left[1] : right[0]]
            if re.fullmatch(r"\s*/\s*", separator):
                add_candidate(
                    [left[2], right[2]],
                    priority=1,
                    source_order=source_order,
                )

    # Prefer explicit alternatives over examples and slash fallbacks. A skill
    # may belong to only one accepted group, preventing nested suffix groups.
    accepted: list[tuple[int, list[str]]] = []
    claimed: set[str] = set()
    for _priority, source_order, group in sorted(
        candidates,
        key=lambda item: (-item[0], item[1], len(item[2])),
    ):
        identities = {_canonical_skill_key(skill) for skill in group}
        if identities & claimed:
            continue
        accepted.append((source_order, group))
        claimed.update(identities)

    return [group for _order, group in sorted(accepted, key=lambda item: item[0])]


def _sentence_around(description: str, position: int) -> str:
    """Return only the sentence containing an occurrence.

    Local sentence boundaries prevent "React is a plus" from incorrectly
    making a following sentence such as "Experience with CI/CD" optional.
    """

    left_boundaries = [
        description.rfind(marker, 0, position)
        for marker in (".", "!", "?", ";", "\n", "(", ")")
    ]
    start = max(left_boundaries) + 1
    right_boundaries = [
        index
        for marker in (".", "!", "?", ";", "\n", "(", ")")
        if (index := description.find(marker, position)) >= 0
    ]
    end = min(right_boundaries) if right_boundaries else len(description)
    return description[start:end].casefold()


def _has_optional_occurrence(skill: str, description: str) -> bool:
    return any(
        any(marker in _sentence_around(description, position) for marker in _OPTIONAL_MARKERS)
        for position in _skill_positions(skill, description)
    )


def _has_nonoptional_occurrence(skill: str, description: str) -> bool:
    return any(
        not any(
            marker in _sentence_around(description, position)
            for marker in _OPTIONAL_MARKERS
        )
        for position in _skill_positions(skill, description)
    )


def _has_explicit_requirement_occurrence(skill: str, description: str) -> bool:
    """Require qualification wording around skills outside required sections."""

    return any(
        _REQUIREMENT_LANGUAGE.search(_sentence_around(description, position))
        is not None
        and not any(
            marker in _sentence_around(description, position)
            for marker in _OPTIONAL_MARKERS
        )
        for position in _skill_positions(skill, description)
    )


def _deterministic_required_experience_years(required_region: str) -> Optional[float]:
    """Return the largest mandatory experience duration in qualifications.

    Job posts often state an overall minimum and a smaller sub-specialty
    duration (for example five years Python and 1.5 years GenAI). The overall
    requirement is therefore the largest non-optional experience duration in
    the mandatory region.
    """

    durations: list[float] = []
    pattern = re.compile(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*years?\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(required_region):
        sentence = _sentence_around(required_region, match.start())
        if "experience" not in sentence:
            continue
        if any(marker in sentence for marker in _OPTIONAL_MARKERS):
            continue
        durations.append(float(match.group(1)))
    return max(durations) if durations else None


def _find_job_section_headings(description: str) -> list[_SectionHeading]:
    """Locate flattened LinkedIn headings and discard overlapping suffix matches."""

    candidates: list[_SectionHeading] = []
    for label, kind in _JOB_SECTION_HEADINGS:
        words = [re.escape(word) for word in label.split()]
        label_pattern = r"\s+".join(words)
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){label_pattern}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        for match in pattern.finditer(description):
            # Flattened LinkedIn descriptions contain ordinary sentences such
            # as "customer requirements" and "these benefits". A generic
            # one-word section label must retain heading-style capitalization;
            # longer labels remain case-insensitive for localized formatting.
            if (
                label in _GENERIC_SINGLE_WORD_HEADINGS
                and not match.group(0)[:1].isupper()
            ):
                continue
            candidates.append(
                _SectionHeading(match.start(), match.end(), label, kind)
            )

    accepted: list[_SectionHeading] = []
    for candidate in sorted(
        candidates, key=lambda item: (item.start, -(item.end - item.start))
    ):
        if any(
            candidate.start < existing.end and existing.start < candidate.end
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: item.start)


def _join_regions(parts: list[str]) -> str:
    return "\n".join(part.strip(" \t\r\n:.-") for part in parts if part.strip())


def _job_regions(description: str) -> _JobRegions:
    """Split LinkedIn prose using every recognized heading range in source order."""

    headings = _find_job_section_headings(description)
    parts: dict[str, list[str]] = {
        "responsibility": [],
        "required": [],
        "preferred": [],
        "ignored": [],
    }
    for index, heading in enumerate(headings):
        end = headings[index + 1].start if index + 1 < len(headings) else len(description)
        parts[heading.kind].append(description[heading.end:end])

    # Some postings contain no explicit qualifications heading. In that case,
    # retain the pre-preferred/non-boilerplate body as the conservative required
    # region rather than silently returning no requirements.
    has_explicit_required_heading = bool(parts["required"])
    if not has_explicit_required_heading:
        stop_positions = [
            heading.start
            for heading in headings
            if heading.kind in {"preferred", "ignored"}
        ]
        fallback_end = min(stop_positions) if stop_positions else len(description)
        parts["required"].append(description[:fallback_end])

    return _JobRegions(
        responsibility=_join_regions(parts["responsibility"]),
        required=_join_regions(parts["required"]),
        preferred=_join_regions(parts["preferred"]),
        ignored=_join_regions(parts["ignored"]),
        headings=tuple(heading.label for heading in headings),
        has_explicit_required_heading=has_explicit_required_heading,
    )


def _education_patterns(level: str) -> tuple[str, ...]:
    return {
        "Bachelor": (
            r"\bbachelor(?:['’]?s)?\b",
            r"\bb\.?s\.?(?=\s|,|$)",
            r"\bb\.?a\.?(?=\s|,|$)",
        ),
        "Master": (
            r"\bmaster(?:['’]?s)?\b",
            r"\bm\.?s\.?(?=\s|,|$)",
            r"\bm\.?a\.?(?=\s|,|$)",
        ),
        "PhD": (r"\bph\.?d\.?\b", r"\bdoctor(?:al|ate)?\b"),
        "High School": (r"\bhigh school\b", r"\bsecondary school\b"),
    }.get(level, ())


_EDUCATION_ORDER = ("High School", "Bachelor", "Master", "PhD")


def _education_levels(
    text: str, *, optional_occurrences: Optional[bool] = None
) -> list[str]:
    """Return every degree level mentioned, optionally filtered by wording."""

    levels: list[str] = []
    for level in _EDUCATION_ORDER:
        found = False
        for pattern in _education_patterns(level):
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                sentence = _sentence_around(text, match.start())
                is_optional = any(marker in sentence for marker in _OPTIONAL_MARKERS)
                if optional_occurrences is None or is_optional == optional_occurrences:
                    found = True
                    break
            if found:
                break
        if found:
            levels.append(level)
    return levels


def _lowest_education_level(levels: list[str]) -> Optional[str]:
    present = set(levels)
    return next((level for level in _EDUCATION_ORDER if level in present), None)

def _description_truncation_reason(description: str) -> Optional[str]:
    compact = " ".join(str(description or "").split())
    if len(compact) < MIN_JOB_DESCRIPTION_CHARACTERS:
        return (
            f"only {len(compact)} characters; minimum is "
            f"{MIN_JOB_DESCRIPTION_CHARACTERS}"
        )
    if len(compact) < 500 and re.search(
        r"\b(?:show|see|read)\s+more\s*$", compact, flags=re.IGNORECASE
    ):
        return f"ends with a continuation marker at only {len(compact)} characters"
    return None

def extract_job_requirements_agent2(
    job_title: str,
    job_description: str,
    llm: Any = None,
    use_cache: bool = True,
) -> JobRequirements:
    """Parse one LinkedIn description into atomic, grounded requirements."""

    title = str(job_title or "").strip()
    description = str(job_description or "").strip()
    if not title or not description:
        raise ValueError("Job title and description are required.")
    truncation_reason = _description_truncation_reason(description)
    if truncation_reason:
        raise Agent2ParserError(
            "LinkedIn job description appears truncated or too short "
            f"({truncation_reason}); refusing to score incomplete requirements."
        )
    combined_source = f"{title}\n{description}"
    cache_key = _cache_text(JOB_CACHE_VERSION, combined_source)
    if use_cache:
        cached = get_cached("agent2_job", cache_key, Agent2JobRequirements)
        if cached is not None:
            print(
                "  [CACHE HIT] Agent 2 job already parsed — "
                f"{len(cached.required_skills)} skills."
            )
            return cached

    regions = _job_regions(description)
    prompt = f"""You are a strict information-extraction system. Extract technical skills, experience, education, and seniority from the supplied LinkedIn job excerpts into the provided schema.

Your output must contain only one schema-valid object. Do not include commentary, explanations, markdown, or additional fields.

## 1. SECTION BOUNDARIES ARE AUTHORITATIVE

The excerpts were extracted from explicit LinkedIn section headings. Treat their classifications as ground truth.

* MANDATORY QUALIFICATIONS → source for `required_skills`, mandatory experience, required education, and explicitly stated seniority.
* CORE RESPONSIBILITIES → source only for `responsibility_skills`.
* PREFERRED/OPTIONAL → source for `preferred_skills` and preferred education.

Never promote an item from PREFERRED/OPTIONAL into a required field.

Do not infer that something is mandatory merely because it appears in responsibilities.

## 2. REQUIRED SKILLS

For `required_skills`, extract every concrete technical skill explicitly named in the MANDATORY QUALIFICATIONS excerpt.

Include:

* programming languages
* libraries
* frameworks
* tools
* platforms
* infrastructure technologies
* technical systems/concepts
* concrete technical methods or techniques

When a mandatory statement names a broad technical concept and specific examples, extract BOTH the concept and every named example as separate atomic items.

Examples:

* `cloud infrastructure (AWS, GCP)` → `cloud infrastructure`, `AWS`, `GCP`
* `frameworks such as LangChain and LangGraph` → `frameworks`, `LangChain`, `LangGraph`
* `Python and PyTorch` → `Python`, `PyTorch`
* `PyTorch, TensorFlow, or Hugging Face` → `PyTorch`, `TensorFlow`, `Hugging Face`

This rule also applies to constructions such as:

* "such as"
* "including"
* "e.g."
* parenthetical examples
* comma-separated lists
* "and"/"or" technology lists

Do not collapse explicitly named technologies into an umbrella category.

Before returning the result, scan the entire MANDATORY QUALIFICATIONS excerpt again and ensure every explicitly required language, library, framework, tool, platform, infrastructure technology, technical concept, and technical method has been considered.

If there are no explicit required technical skills, return `[]`.

## 3. RESPONSIBILITY SKILLS

For `responsibility_skills`, use only the CORE RESPONSIBILITIES excerpt.

Extract a technical skill only when its local sentence explicitly establishes candidate possession as a requirement using language equivalent to:

* required
* must have
* expected
* need experience with/in
* experience with/in
* proficiency in
* knowledge of
* expertise in

A technology or activity merely used, performed, developed, built, maintained, deployed, designed, or operated by the role is NOT sufficient evidence that it is a pre-existing candidate requirement.

Do not infer skills from job duties.

## 4. PREFERRED SKILLS

For `preferred_skills`, extract concrete technical skills from the PREFERRED/OPTIONAL excerpt when they are explicitly presented as:

* preferred
* optional
* bonus
* a plus
* nice-to-have
* desired

Never place these items in `required_skills` or `responsibility_skills`.

Apply the same atomic extraction and example-expansion rules used for required skills.

## 5. SKILL NORMALIZATION

Each skill must:

* represent one atomic technical concept/name
* normally contain 1–5 words
* remain faithful to wording explicitly present in the posting

Split technology lists and parenthetical examples into separate items.

Do NOT invent:

* synonyms
* aliases not present in the posting
* inferred technologies
* combinations
* embellished variations such as `ML-powered X`

Deduplicate case-insensitively while preserving the first-occurring spelling.

Exclude:

* soft skills
* departments or teams
* business outcomes
* performance/business metrics
* benefits
* eligibility conditions
* generic action words

Generic terms such as `adapting`, `filtering`, `deployment`, `serving`, `engineering`, and `curation` are not skills by themselves.

However, preserve a technically qualified concept when explicitly stated, such as `model serving`.

## 6. EXPERIENCE

Extract the overall mandatory minimum years of experience explicitly required by the posting.

Rules:

* Do not infer years from the job title or seniority.
* Do not use a smaller duration that applies only to a sub-specialty when a broader overall minimum is explicitly stated.
* If multiple durations exist, identify the duration representing the overall mandatory experience requirement.
* Do not treat preferred experience as mandatory.
* If no mandatory minimum is explicitly stated, use the schema's null/empty representation.

Example:
`5+ years of software engineering experience, including 2+ years with ML systems`
→ overall mandatory experience = `5`, not `2`.

## 7. EDUCATION

For `required_education_level`, extract only a mandatory degree level.

For `preferred_education_level`, extract only a degree explicitly marked preferred or optional.

Allowed normalized values:

* `High School`
* `Bachelor`
* `Master`
* `PhD`
* `null`

Do not populate `required_education_level` from a Preferred/Desired Qualifications section.

When multiple degree levels are alternatives satisfying the same requirement, return the lowest qualifying level.

Example:
`Bachelor's, Master's, or Doctorate degree`
→ `Bachelor`

Do not infer a degree level that is not explicitly supported.

## 8. SENIORITY

Extract seniority only when the supplied posting explicitly supports a seniority level.

Do not infer seniority solely from:

* years of experience
* responsibilities
* compensation
* presumed scope
* industry conventions

If no seniority level is explicitly supported, return `null`.

## 9. FINAL VALIDATION

Before producing the object, silently verify:

1. Every required skill comes from MANDATORY QUALIFICATIONS.
2. Every responsibility skill satisfies the explicit-requirement test.
3. Every preferred skill remains preferred.
4. Explicit technology/example lists have been split into atomic items.
5. Broad technical concepts explicitly named alongside examples have also been preserved.
6. No technical skill was invented or inferred.
7. No preferred qualification leaked into a required field.
8. Skills are deduplicated case-insensitively.
9. Experience represents the overall mandatory minimum rather than a sub-specialty minimum.
10. Education obeys the mandatory/preferred distinction and allowed normalization.
11. Seniority is explicit rather than inferred.
12. The final output conforms exactly to the supplied schema.

## INPUT

JOB TITLE:
{title}

## CORE RESPONSIBILITIES EXCERPT:

## {regions.responsibility or "(No separate responsibilities section identified.)"}

## MANDATORY QUALIFICATIONS EXCERPT:

## {regions.required}

## PREFERRED/OPTIONAL EXCERPT:

## {regions.preferred or "(No separate preferred section identified.)"}

Return only the schema-valid object.

---"""

    def validate_job(parsed: BaseModel) -> None:
        mandatory = _clean_job_skills(
            list(parsed.required_skills),
            description,
            max_count=MAX_JOB_SKILLS,
        )
        mandatory = [
            skill for skill in mandatory if _skill_positions(skill, regions.required)
        ]
        if (
            regions.has_explicit_required_heading
            and re.search(
                r"\b(?:technologies\s+such\s+as|including)\b",
                regions.required,
                flags=re.IGNORECASE,
            )
            and len(mandatory) < 6
        ):
            raise ValueError(
                "too few mandatory technical skills were extracted from an "
                "explicit qualifications section containing a technology list"
            )

    extraction = _invoke_structured_with_retry(
        schema=_JobExtraction,
        prompt=prompt,
        label=f"job '{title}'",
        llm=llm,
        validator=validate_job,
    )
    data = extraction.model_dump()
    extracted_skills = _clean_job_skills(
        data["required_skills"]
        + data.pop("responsibility_skills", [])
        + data["preferred_skills"],
        description,
        max_count=MAX_JOB_SKILLS * 2,
    )
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    for skill in extracted_skills:
        in_required_section = (
            _has_nonoptional_occurrence(skill, regions.required)
            if regions.has_explicit_required_heading
            else _has_explicit_requirement_occurrence(skill, regions.required)
        )
        in_responsibility_section = _has_explicit_requirement_occurrence(
            skill, regions.responsibility
        )
        in_preferred_section = bool(_skill_positions(skill, regions.preferred))
        inline_optional = _has_optional_occurrence(skill, description)
        if in_required_section or in_responsibility_section:
            required_skills.append(skill)
        elif in_preferred_section or inline_optional:
            preferred_skills.append(skill)

    required_skills = required_skills[:MAX_JOB_SKILLS]
    required_keys = {_canonical_skill_key(skill) for skill in required_skills}
    preferred_skills = [
        skill
        for skill in preferred_skills
        if _canonical_skill_key(skill) not in required_keys
    ][:MAX_JOB_SKILLS]
    data["required_skills"] = required_skills
    data["preferred_skills"] = preferred_skills
    data["required_skill_groups"] = _required_alternative_groups(
        required_skills,
        regions.required,
    )
    deterministic_experience = _deterministic_required_experience_years(
        regions.required
    )
    data["required_experience_years"] = deterministic_experience
    required_education = _lowest_education_level(
        _education_levels(regions.required, optional_occurrences=False)
    )
    preferred_levels = _education_levels(regions.preferred)
    preferred_levels.extend(
        _education_levels(description, optional_occurrences=True)
    )
    preferred_education = _lowest_education_level(preferred_levels)
    data["required_education_level"] = required_education
    data["preferred_education_level"] = preferred_education
    data["required_skill_evidence"] = _evidence_map(required_skills, description)
    data["preferred_skill_evidence"] = _evidence_map(preferred_skills, description)
    result = Agent2JobRequirements.model_validate(data)
    if use_cache:
        set_cached("agent2_job", cache_key, result)
    return result
