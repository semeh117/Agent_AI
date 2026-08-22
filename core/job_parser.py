"""
job_parser.py
--------------
Extracts required skills from a real job posting's description text
(via Himalayas API — full descriptions, not truncated). Mirrors
cv_parser.py's pattern: LLM structured extraction with a grounding
check against hallucination.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from config import get_parser_llm
from core.extraction_cache import get_cached, set_cached


class JobRequirements(BaseModel):
    required_skills: List[str] = Field(
        description="Technical skills, tools, languages, frameworks explicitly "
                    "mentioned as required or relevant in the posting text."
    )
    job_title: Optional[str] = Field(default=None)
    seniority_level: Optional[str] = Field(
        default=None, description="e.g. 'Junior', 'Principal', 'Senior' if stated or implied by title."
    )
    required_experience_years: Optional[float] = Field(
        default=None,
        description="Minimum years of professional experience required, if explicitly "
                    "stated (e.g. '3+ years experience' -> 3.0). Null if not mentioned."
    )
    required_education_level: Optional[str] = Field(
        default=None,
        description="Minimum education level required, if stated. One of: "
                    "'High School', 'Bachelor', 'Master', 'PhD'. Null if not mentioned."
    )


def extract_job_requirements(job_title: str, job_description: str, llm=None ,use_cache :bool=True) -> JobRequirements:
    """
    Extract required skills from a job posting.

    Args:
        job_title: the job's title, as a plain string
        job_description: the full job description text, as a plain string
        llm: optional pre-built LLM instance (for model comparison scripts),
             defaults to config.py's configured model otherwise.

    Returns:
        JobRequirements with extracted skills, title, and seniority level
    """
    cache_key_text = f"{job_title}\n--\n{job_description}"
    if use_cache:
        cached = get_cached("job",cache_key_text ,JobRequirements)
        if cached is not None:
            print(f" [CACHE HIT] Job posting already parsed before — skipping LLM call "
                  f"({len(cached.required_skills)} skills, no wait).")
            return cached
    if llm is None:
        llm = get_parser_llm(temperature=0.0)

    structured_llm = llm.with_structured_output(JobRequirements)

    prompt = f"""You are an expert technical recruiter and job-description analyst.
Extract every explicitly required technical skill from the ENTIRE job
posting below, along with seniority level, required experience, and
required education.

Read the entire job description from beginning to end before producing
your answer. Do not stop after the first paragraph, requirements
section, or technology list.

## WHERE TO LOOK
Extract technical skills mentioned anywhere in the posting, including:
- Requirements and qualifications
- Preferred qualifications, ONLY if explicitly presented as required for
  the role (see REQUIRED VS. PREFERRED below)
- Responsibilities and duties, when they describe core work the
  candidate is expected to perform
- Project descriptions, technology stacks, tools, platforms, and
  methodologies mentioned in the context of work the candidate will do

## WHAT COUNTS AS A TECHNICAL SKILL
Extract only concrete, explicitly named technical skills: programming
languages (Python, Go), tools/technologies (Docker, Kubernetes, Git),
platforms/cloud (AWS, Azure), databases (PostgreSQL, SQL), protocols
(TCP/IP, HTTP), frameworks/libraries (React, TensorFlow), and technical
methodologies/concepts (Machine Learning, CI/CD).

Do NOT extract: soft skills (communication, teamwork, leadership),
generic traits (motivated, detail-oriented), language fluency
requirements, or anything inferred from the role/industry rather than
explicitly named in the text. Do not guess — a skill must be explicitly
mentioned or clearly named.

## CRITICAL: SPLIT CONJOINED SKILLS
Treat every distinct technical skill as separate, even when several
appear together in one phrase, sentence, bullet, or heading:
- "Unsupervised & Deep Learning Experience" -> "Unsupervised Learning"
  AND "Deep Learning"
- "relational and NoSQL/Graph databases" -> "relational databases" AND
  "NoSQL databases" AND "Graph databases"
- "Research & develop Machine Learning models" -> "Machine Learning"
- "Python and SQL" -> "Python" AND "SQL"
Also split lists separated by: and, or, &, /, commas, semicolons,
parentheses, or other conjunctions/punctuation. Do not treat a compound
phrase as one skill merely because it appears as a single requirement.

## ATOMIC SKILL NAMES
Each skill must be short, clean, atomic, and technically meaningful —
not a sentence or requirement description.
Bad: "Experience developing production machine learning models using Python"
Good: "Python", "Machine Learning"
Preserve the standard technical name; don't expand an acronym unless the
posting itself provides the expanded form.

## REQUIRED VS. PREFERRED — determines what goes in the final list
Include a skill in required_skills when the posting clearly makes it
mandatory: "required", "must have", "must", "mandatory", "minimum
qualifications", "qualifications", "requirements", "you will need",
"essential", or equivalent wording establishing necessity. A skill
described in Responsibilities also counts as required when it clearly
describes core work the candidate will perform.

Do NOT include a skill merely because it appears somewhere in the
posting. Do NOT include skills marked as explicitly optional/preferred
("nice to have", "preferred", "bonus", "plus") — these are excluded from
required_skills entirely. If wording is ambiguous, classify
conservatively (exclude rather than include).

## DEDUPLICATION
If the same skill appears multiple times, include it only once. Treat
capitalization differences as the same skill ("python" = "Python").
Do not merge genuinely distinct technologies just because they're
related (SQL and PostgreSQL stay separate; Machine Learning and Deep
Learning stay separate).

## SENIORITY LEVEL
Extract seniority_level from the job title or explicit wording in the
text (e.g. "Junior", "Mid-level", "Senior", "Staff", "Principal"). Null
if genuinely not indicated anywhere.

## YEARS OF EXPERIENCE
Extract required_experience_years as the MINIMUM years explicitly
stated or clearly implied by a range:
- "3+ years of experience" -> 3.0
- "At least 5 years" -> 5.0
- "2-4 years" -> 2.0
- "Minimum of 2 years" -> 2.0
If the posting frames a number as "preferred" or "a plus" rather than
required, still extract that number — this field only tracks whether a
number was stated at all, regardless of how firmly required it is
elsewhere in the text. If multiple experience requirements appear, use
the overall required minimum for the role when one can be clearly
identified. Do NOT infer years from seniority labels alone (e.g. do not
assume "Senior" implies "5.0" if no number is actually stated).
Return null ONLY if no number appears anywhere in the text.

## EDUCATION
Extract required_education_level (one of: High School, Bachelor,
Master, PhD) when a specific degree level is named anywhere in the
text — including when an alternative is also offered:
- "Bachelor's degree required" -> Bachelor
- "Master's degree in Computer Science" -> Master
- "PhD required" -> PhD
- "Bachelor's or equivalent experience" -> Bachelor (a level IS named,
  even though an alternative is also allowed)
Do NOT infer a level from job seniority or title alone. Return null
ONLY if no degree level is named anywhere in the text.

## BEFORE FINALIZING
Mentally verify you've reviewed the ENTIRE job description and split
all conjoined technical skills before producing your final answer.

JOB TITLE: {job_title}

JOB DESCRIPTION:
---
{job_description}
---"""

    result = structured_llm.invoke(prompt)

    # Grounding check — same pattern as cv_parser.py, filters out any
    # hallucinated skill not actually present in the source text.
    combined_text = (job_title + " " + job_description).lower()
    result.required_skills = [
        s for s in result.required_skills if s.lower() in combined_text
    ]

    if use_cache:
        set_cached("job",cache_key_text,result)

    return result
