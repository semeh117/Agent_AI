"""
skill_matcher_llm.py
--------------
LLM-based semantic skill matching: judges whether a candidate's existing,
verified skill list satisfies each job-required skill — including cases
where the candidate has a specific tool that implies a broader requirement
(e.g. "PyTorch" satisfies "Deep Learning"; "Terraform" satisfies
"Infrastructure as Code"). Works across ANY field, since it's the LLM's
general knowledge doing the judgment, not a hardcoded per-domain dictionary.

Lower hallucination risk than raw-text extraction: the model is reasoning
over two already-extracted, already-verified lists — it can't invent a
skill that isn't in either list, it can only judge equivalence between
things that are already real.
"""

from typing import List
from pydantic import BaseModel, Field
from config import get_llm


class SkillMatch(BaseModel):
    job_skill: str
    matched: bool = Field(description="True if the candidate's skills satisfy this requirement, directly or via an equivalent/related tool.")
    matched_via: str = Field(default="", description="Which of the candidate's own skills justifies this match, if matched=True. Empty if not matched.")


class SkillMatchResult(BaseModel):
    matches: List[SkillMatch]


def match_skills_llm(cv_skills: List[str], job_skills: List[str], llm=None) -> SkillMatchResult:
    if llm is None:
        llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SkillMatchResult)

    prompt = f"""A candidate has these skills:
{", ".join(cv_skills)}

A job requires these skills:
{", ".join(job_skills)}

For EACH required job skill, decide if the candidate's skill list satisfies
it — either directly (exact/near match) OR because the candidate has a
SPECIFIC tool/technique that is a well-known real-world instance of that
requirement (e.g. "PyTorch" satisfies "Deep Learning"; "Terraform"
satisfies "Infrastructure as Code"; "Snyk" satisfies "Security Scanning").

Be honest and conservative: only mark matched=True if there's a genuine,
well-established real-world relationship — not a loose or creative guess."""

    return structured_llm.invoke(prompt)