"""CV-specific structured parsing for Agent 2."""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from next_chapter.parsing.agent2_parser_common import (
    Agent2ParserError,
    CV_CACHE_VERSION,
    MAX_CV_SKILLS,
    MAX_RAW_CV_SKILLS,
    _cache_text,
    _canonical_skill_key,
    _evidence_map,
    _invoke_structured_with_retry,
    _normalize_education,
)
from next_chapter.parsing.agent2_cv_rules import (
    _CV_SECTION_HEADINGS,
    _category_headings,
    _clean_cv_job_titles,
    _clean_cv_skills,
    _cv_headline,
    _deterministic_education_level,
    _deterministic_experience_years,
    _deterministic_markdown_name,
    _deterministic_plain_name,
    _explicit_cv_project_stack_items,
    _explicit_cv_skill_items,
    _looks_like_contextual_cv_skill,
)
from next_chapter.parsing.cv_parser import CVInfo
from next_chapter.storage.extraction_cache import get_cached, set_cached


class Agent2CVInfo(CVInfo):
    """CVInfo plus deterministic source evidence for Agent 2 review."""

    skill_evidence: dict[str, str] = Field(default_factory=dict)
    # Profile headline printed under the name ("AI & Machine Learning
    # Engineer"). It is the candidate's target role, not a position held, so
    # it is kept apart from job_titles and used for the LinkedIn query when a
    # student or career changer has no professional title yet.
    headline: Optional[str] = None


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


def extract_cv_info_agent2(
    cv_text: str,
    llm: Any = None,
    use_cache: bool = True,
    *,
    layout_text: Optional[str] = None,
    cache_identity: Optional[str] = None,
) -> CVInfo:
    """Parse one CV using source text plus optional structured fixture text.

    ``cv_text`` is used for metadata and the LLM call. ``layout_text`` remains
    available for replaying older evaluation fixtures. Production uploads use
    the PyPDF text for both extraction and grounding.
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
    data["job_titles"] = _clean_cv_job_titles(data.get("job_titles", []), source)
    data["headline"] = _cv_headline(source)
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
