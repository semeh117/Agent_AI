"""
matcher.py
--------------
Deterministic compatibility scoring: CV vs job requirements.
Skills matching uses LLM-based semantic judgment (skill_matcher_llm.py)
instead of exact string matching or raw embedding similarity — this
generalizes across ANY field (AI/ML, DevOps, security, etc.) since the
LLM's own knowledge judges equivalence, no per-domain dictionary needed.
"""

from typing import List, Dict, Optional
from core.skill_matcher_llm import match_skills_llm

EDUCATION_RANK = {
    "high school": 0,
    "bachelor": 1,
    "master": 2,
    "phd": 3,
}


def _skills_match(cv_skills: List[str], job_skills: List[str]) -> Dict:
    if not job_skills:
        return {"score": 0.5, "matching": [], "missing": [],
                "note": "Job posting listed no explicit required skills — "
                        "inconclusive, not a verified match."}

    cv_skills_lower = [s.lower() for s in cv_skills]

    # Pass 1: exact/substring match — deterministic, no LLM involved,
    # and immune to the run-to-run inconsistency a small LLM can show
    # even on obvious cases (e.g. "Docker" vs "Docker").
    matching = []
    remaining_job_skills = []
    for job_skill in job_skills:
        job_skill_lower = job_skill.lower()
        exact_hit = next(
            (cv_s for cv_s, cv_s_lower in zip(cv_skills, cv_skills_lower)
             if job_skill_lower == cv_s_lower or job_skill_lower in cv_s_lower or cv_s_lower in job_skill_lower),
            None,
        )
        if exact_hit:
            matching.append({"job_skill": job_skill, "matched_via": exact_hit})
        else:
            remaining_job_skills.append(job_skill)

    # Pass 2: only the skills that DIDN'T exact-match go to the LLM for
    # semantic/equivalence judgment (e.g. "PyTorch" satisfies "Deep Learning").
    missing = []
    if remaining_job_skills:
        result = match_skills_llm(cv_skills, remaining_job_skills)
        for m in result.matches:
            if m.matched:
                matching.append({"job_skill": m.job_skill, "matched_via": m.matched_via})
            else:
                missing.append(m.job_skill)

    score = len(matching) / len(job_skills)
    return {"score": score, "matching": matching, "missing": missing}


def _experience_match(cv_years: Optional[float], required_years: Optional[float]) -> Dict:
    if required_years is None or required_years == 0:
        return {"score": 1.0, "note": "No experience requirement stated."}
    if cv_years is None:
        cv_years = 0.0
    score = min(cv_years / required_years, 1.0)
    return {"score": score, "candidate_years": cv_years, "required_years": required_years}


def _education_match(cv_highest_level: Optional[str], required_level: Optional[str]) -> Dict:
    if not required_level:
        return {"score": 1.0, "note": "No education requirement stated."}
    required_rank = EDUCATION_RANK.get(required_level.strip().lower())
    if required_rank is None or not cv_highest_level:
        return {"score": 0.5, "note": "Could not determine education levels."}
    candidate_rank = EDUCATION_RANK.get(cv_highest_level.strip().lower())
    if candidate_rank is None:
        return {"score": 0.5, "note": f"Unrecognized candidate level '{cv_highest_level}'."}
    if candidate_rank >= required_rank:
        return {"score": 1.0}
    elif candidate_rank == required_rank - 1:
        return {"score": 0.5, "note": "One level below requirement."}
    return {"score": 0.0, "note": "Below requirement."}


def calculate_compatibility(cv_info, job_requirements) -> Dict:
    skills_result = _skills_match(cv_info.skills, job_requirements.required_skills)
    experience_result = _experience_match(cv_info.experience_years, job_requirements.required_experience_years)
    education_result = _education_match(cv_info.highest_education_level, job_requirements.required_education_level)

    final_score = (
        skills_result["score"] * 0.5
        + experience_result["score"] * 0.3
        + education_result["score"] * 0.2
    )

    return {
        "score_percent": round(final_score * 100, 1),
        "skills": skills_result,
        "experience": experience_result,
        "education": education_result,
    }