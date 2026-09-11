"""LinkedIn-job-specific structured parsing for Agent 2."""

from __future__ import annotations

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from next_chapter.parsing.agent2_parser_common import (
    Agent2ParserError,
    JOB_CACHE_VERSION,
    MAX_JOB_SKILLS,
    MAX_RAW_JOB_SKILLS,
    MIN_JOB_DESCRIPTION_CHARACTERS,
    _cache_text,
    _canonical_skill_key,
    _evidence_map,
    _invoke_structured_with_retry,
    _skill_positions,
)
from next_chapter.parsing.agent2_job_rules import (
    _clean_job_skills,
    _description_truncation_reason,
    _deterministic_required_experience_years,
    _education_levels,
    _find_job_section_headings,
    _has_explicit_requirement_occurrence,
    _has_nonoptional_occurrence,
    _has_optional_occurrence,
    _job_regions,
    _lowest_education_level,
    _required_alternative_groups,
    _sentence_around,
)
from next_chapter.parsing.job_parser import JobRequirements
from next_chapter.storage.extraction_cache import get_cached, set_cached


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
