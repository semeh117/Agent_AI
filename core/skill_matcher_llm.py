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

import logging
from typing import List
from pydantic import BaseModel, Field
from config import get_parser_llm


logger = logging.getLogger(__name__)

# Long structured arrays are the most common cause of truncated/malformed JSON
# from smaller free models. Ten decisions fit comfortably inside the configured
# completion budget while keeping the number of calls reasonable.
SKILLS_PER_CALL = 10
MAX_PARSE_ATTEMPTS = 2


class SkillMatch(BaseModel):
    job_skill: str
    matched: bool = Field(description="True if the candidate's skills satisfy this requirement, directly or via an equivalent/related tool.")
    matched_via: str = Field(default="", description="Which of the candidate's own skills justifies this match, if matched=True. Empty if not matched.")


class SkillMatchResult(BaseModel):
    matches: List[SkillMatch]


def match_skills_llm(cv_skills: List[str], job_skills: List[str], llm=None) -> SkillMatchResult:
    if llm is None:
        llm = get_parser_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(SkillMatchResult)

    prompt_template = """You are an expert technical skills-matching evaluator. Determine
whether a candidate's skills satisfy EACH required skill in a job
posting.

CANDIDATE SKILLS:
{candidate_skills}

JOB REQUIRED SKILLS:
{job_skills}

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
from job title, seniority, or industry.

Return exactly one decision for every required skill in the supplied
order. Keep matched_via short."""

    combined_matches: List[SkillMatch] = []
    for start in range(0, len(job_skills), SKILLS_PER_CALL):
        chunk = job_skills[start:start + SKILLS_PER_CALL]
        prompt = prompt_template.format(
            candidate_skills=", ".join(cv_skills),
            job_skills=", ".join(chunk),
        )

        parsed = None
        last_error = None
        for attempt in range(1, MAX_PARSE_ATTEMPTS + 1):
            try:
                parsed = structured_llm.invoke(prompt)
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Skill matching JSON failed for chunk %s-%s (attempt %s/%s): %s",
                    start + 1,
                    start + len(chunk),
                    attempt,
                    MAX_PARSE_ATTEMPTS,
                    exc,
                )

        # Preserve the original required-skill spelling/order and never let an
        # omitted, duplicated, or malformed model item disappear from scoring.
        # If both attempts fail, the conservative result is "missing", not a
        # crashed end-to-end workflow or an invented match.
        returned_by_name = {}
        if parsed is not None:
            for match in parsed.matches:
                returned_by_name.setdefault(match.job_skill.strip().casefold(), match)
        elif last_error is not None:
            logger.error(
                "Skill matching chunk failed after retries; marking %s requirement(s) missing.",
                len(chunk),
            )

        for required_skill in chunk:
            returned = returned_by_name.get(required_skill.strip().casefold())
            combined_matches.append(
                SkillMatch(
                    job_skill=required_skill,
                    matched=bool(returned and returned.matched),
                    matched_via=(returned.matched_via if returned and returned.matched else ""),
                )
            )

    return SkillMatchResult(matches=combined_matches)
