"""
job_parser.py
--------------
Extracts required skills from a real job posting's description text
(via Adzuna). Mirrors cv_parser.py's pattern: LLM structured extraction
with a grounding check against hallucination.

Note: Adzuna descriptions are truncated snippets, not full postings —
the prompt is written to extract only what's genuinely present rather
than over-inferring from incomplete text.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from config import get_llm


class JobRequirements(BaseModel):
    required_skills: List[str] = Field(
        description="Technical skills, tools, languages, frameworks explicitly "
                    "mentioned as required or relevant in the posting text."
    )
    job_title: Optional[str] = Field(default=None)
    seniority_level: Optional[str] = Field(
        default=None, description="e.g. 'Junior', 'Principal', 'Senior' if stated or implied by title."
    )


def extract_job_requirements(job_title: str, job_description: str, llm=None) -> JobRequirements:
    """
    llm: optional pre-built LLM instance (for model comparison scripts),
    defaults to config.py's configured model otherwise.
    """
    if llm is None:
        llm = get_llm(temperature=0.0)

    structured_llm = llm.with_structured_output(JobRequirements)

    prompt = f"""Extract the required technical skills from this job posting.

IMPORTANT: This description is a TRUNCATED SNIPPET (often cut off mid-sentence
at the end) — not the full posting. Only extract skills that are explicitly
and literally mentioned in the text below. Do NOT infer or guess additional
skills that would typically be needed for this type of role but aren't
actually written here.

JOB TITLE: {job_title}

JOB DESCRIPTION (snippet):
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

    return result