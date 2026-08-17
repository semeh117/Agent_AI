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

    prompt = f"""You are an expert technical skills-matching evaluator. Determine
whether a candidate's skills satisfy EACH required skill in a job
posting.

CANDIDATE SKILLS:
{", ".join(cv_skills)}

JOB REQUIRED SKILLS:
{", ".join(job_skills)}

Evaluate EVERY job-required skill independently. For each one, decide
if the candidate satisfies it through:

1. DIRECT MATCH — the candidate has the same skill or a clear, standard
   near-equivalent (e.g. "ML" <-> "Machine Learning", "NLP" <-> "Natural
   Language Processing", "K8s" <-> "Kubernetes").
2. ESTABLISHED TECHNICAL RELATIONSHIP — the candidate has a specific
   technology/tool/framework/platform/technique that is a well-known
   real-world implementation or instance of the required skill (e.g.
   "PyTorch" satisfies "Deep Learning"; "Terraform" satisfies
   "Infrastructure as Code"; "Snyk" satisfies "Security Scanning").

Be STRICT and evidence-based. Mark matched=True only when there's a
genuine, well-established technical relationship.

ACCEPT: exact matches, standard abbreviations, clear synonymous
terminology, a specific technology that's a recognized implementation
of the required technology/concept, a specific framework/tool that
directly demonstrates the required capability.

REJECT — do NOT match based on: broad similarity, related but different
technologies, skills merely commonly used together, skills that "could
potentially" be transferable, industry/domain similarity, job-title
similarity, assumptions about what someone "would probably know", a
parent technology when the required skill is a distinct child
technology (unless the relationship is explicitly well-established), or
a child technology when the required skill is a broader concept (unless
that child is a recognized real-world instance of it).

For example, do NOT automatically conclude: Python -> Machine Learning,
JavaScript -> React, AWS -> Kubernetes, SQL -> PostgreSQL, Docker ->
Kubernetes. These may be related, but relatedness alone does not
establish that the candidate possesses the required skill.

DIRECTION MATTERS: evaluate whether the CANDIDATE skill demonstrates
the REQUIRED job skill, not the reverse. Candidate "PyTorch" -> Required
"Deep Learning" = matched. Candidate "Deep Learning" -> Required
"PyTorch" = NOT matched, unless the candidate also explicitly lists
PyTorch. Similarly: candidate "PostgreSQL" -> required "SQL" = matched
(PostgreSQL is a relational database system that uses SQL); candidate
"SQL" -> required "PostgreSQL" = NOT matched.

Do not normalize genuinely distinct technologies into one another
merely because they belong to the same ecosystem.

For matched_via, when matched=True, name the specific candidate skill
that establishes the match, and briefly indicate whether it's a direct
match or an instance-of relationship (e.g. "PyTorch (instance of Deep
Learning)" or "Python (direct match)"). Leave matched_via empty when
matched=False.

Before finalizing: evaluate EVERY required job skill, never omit one;
do not count the same candidate skill as evidence for unrelated
requirements; require a genuine technical relationship for indirect
matches; prefer matched=False when the relationship is ambiguous or
debatable; never invent candidate skills; never infer knowledge solely
from job title, seniority, or industry."""

    return structured_llm.invoke(prompt)