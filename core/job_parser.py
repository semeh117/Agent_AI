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


def extract_job_requirements(job_title: str, job_description: str, llm=None) -> JobRequirements:
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
    if llm is None:
        llm = get_llm(temperature=0.0)

    structured_llm = llm.with_structured_output(JobRequirements)

    prompt = f"""Extract EVERY required technical skill from this job posting.
Read the ENTIRE description carefully, from start to finish — do not stop
after the first paragraph or the first list of requirements.

CRITICAL — watch for compound/conjoined phrases where TWO skills are
combined into one sentence. You must split these into separate skills:
- "Unsupervised & Deep Learning Experience" -> "Unsupervised Learning" AND
  "Deep Learning" (two separate skills, not one combined phrase)
- "relational and NoSQL/Graph databases" -> "relational databases" AND
  "NoSQL databases" AND "Graph databases" (three separate skills)
- "Research & develop Machine Learning models" -> "Machine Learning" is a
  skill here, even though it's phrased as an action/responsibility rather
  than a bullet-point requirement.

Rules:
- Extract ONLY concrete technical skills: programming languages, tools,
  frameworks, platforms, databases, protocols, methodologies (e.g. "Go",
  "AWS", "SQL", "Machine Learning", "TCP/IP").
- Each skill must be a SHORT, clean, atomic name — NOT a sentence.
- Do NOT include soft skills (communication, teamwork, critical thinking,
  problem-solving, curiosity) or language fluency requirements.
- A skill can appear anywhere in the text — in a "Responsibilities"
  section describing what the candidate will DO, not just in a
  "Requirements" list. Extract from both.
- Only extract skills explicitly and literally mentioned or clearly
  named in the text. Do not guess additional skills merely typical for
  this type of role.
-PRIORITIZATION: Determine if a skill is "required" (must-have, mandatory, core to the
- Also extract required_experience_years (a number, e.g. "3+ years" -> 3.0)
  and required_education_level (one of: High School, Bachelor, Master, PhD)
  if explicitly stated in the posting. Leave null if not mentioned.
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

    return result