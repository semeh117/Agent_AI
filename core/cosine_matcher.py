"""Weighted matching for the LinkedIn job agent.

Skills are compared semantically with embedding cosine similarity.
Experience and education are evaluated deterministically. The final score is:

    skills * 0.5 + experience * 0.3 + education * 0.2
"""

from __future__ import annotations

from math import fsum, sqrt
from typing import Any, Iterable, Optional


SKILLS_WEIGHT = 0.5
EXPERIENCE_WEIGHT = 0.3
EDUCATION_WEIGHT = 0.2

EDUCATION_RANK = {
    "high school": 0,
    "bachelor": 1,
    "master": 2,
    "phd": 3,
}


def _clean_values(values: Iterable[Any]) -> list[str]:
    """Return unique, non-empty string values while preserving order."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


def build_candidate_text(cv_info: Any) -> str:
    """Build the candidate's skills-only text for embedding."""

    skills = _clean_values(getattr(cv_info, "skills", []) or [])
    return f"Candidate skills: {', '.join(skills)}." if skills else ""


def build_job_text(job: dict[str, Any], requirements: Any) -> str:
    """Build the job's required-skills-only text for embedding."""

    del job  # Signature stays aligned with the job/requirements pair API.
    skills = _clean_values(getattr(requirements, "required_skills", []) or [])
    return f"Required skills: {', '.join(skills)}." if skills else ""


def cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    """Return cosine similarity in [-1, 1], or zero for a zero vector."""

    a = [float(value) for value in vector_a]
    b = [float(value) for value in vector_b]
    if len(a) != len(b):
        raise ValueError(
            "Embedding dimensions must match; "
            f"received {len(a)} and {len(b)} values."
        )

    denominator = sqrt(fsum(value * value for value in a)) * sqrt(
        fsum(value * value for value in b)
    )
    if denominator == 0.0:
        return 0.0

    similarity = fsum(left * right for left, right in zip(a, b)) / denominator
    return max(-1.0, min(1.0, similarity))


def calculate_experience_score(
    candidate_years: Optional[float],
    required_years: Optional[float],
) -> float:
    """Return experience eligibility in [0, 1]."""

    if required_years is None or required_years <= 0:
        return 1.0
    candidate = max(float(candidate_years or 0.0), 0.0)
    return min(candidate / float(required_years), 1.0)


def calculate_education_score(
    candidate_level: Optional[str],
    required_level: Optional[str],
) -> float:
    """Return education eligibility in [0, 1]."""

    if not required_level:
        return 1.0
    if not candidate_level:
        return 0.5

    candidate_rank = EDUCATION_RANK.get(candidate_level.strip().casefold())
    required_rank = EDUCATION_RANK.get(required_level.strip().casefold())
    if candidate_rank is None or required_rank is None:
        return 0.5
    if candidate_rank >= required_rank:
        return 1.0
    if candidate_rank == required_rank - 1:
        return 0.5
    return 0.0


def rank_jobs_by_cosine(
    cv_info: Any,
    parsed_jobs: list[tuple[dict[str, Any], Any]],
    embeddings: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Score and rank parsed jobs by skills, experience, and education."""

    candidate_text = build_candidate_text(cv_info)
    if not candidate_text:
        raise ValueError("The candidate has no skills to embed.")

    if embeddings is None:
        from config import get_embeddings

        embeddings = get_embeddings()

    candidate_vector = embeddings.embed_query(candidate_text)
    candidate_skills = _clean_values(getattr(cv_info, "skills", []) or [])
    candidate_skill_keys = {skill.casefold() for skill in candidate_skills}
    candidate_experience = getattr(cv_info, "experience_years", None)
    candidate_education = getattr(cv_info, "highest_education_level", None)

    prepared: list[dict[str, Any]] = []
    valid_texts: list[str] = []
    valid_indexes: list[int] = []

    for job, requirements in parsed_jobs:
        required_skills = _clean_values(
            getattr(requirements, "required_skills", []) or []
        )
        exact_matching_skills = [
            skill for skill in required_skills
            if skill.casefold() in candidate_skill_keys
        ]
        missing_skills = [
            skill for skill in required_skills
            if skill.casefold() not in candidate_skill_keys
        ]
        job_text = build_job_text(job, requirements)
        required_experience = getattr(
            requirements, "required_experience_years", None
        )
        required_education = getattr(
            requirements, "required_education_level", None
        )
        experience_match = calculate_experience_score(
            candidate_experience, required_experience
        )
        education_match = calculate_education_score(
            candidate_education, required_education
        )
        final_match = (
            experience_match * EXPERIENCE_WEIGHT
            + education_match * EDUCATION_WEIGHT
        )

        item = {
            "job_title": (
                getattr(requirements, "job_title", None)
                or job.get("title")
                or ""
            ),
            "company": job.get("company", ""),
            "url": job.get("url", ""),
            "description": job.get("description", ""),
            "skills_similarity": 0.0,
            "skills_score": 0.0,
            "experience_match": experience_match,
            "experience_score": round(experience_match * 100, 1),
            "education_match": education_match,
            "education_score": round(education_match * 100, 1),
            "final_match": final_match,
            "final_score": round(final_match * 100, 1),
            # Compatibility shape consumed by the shared cover-letter and
            # Gmail/Telegram rendering pipelines.
            "score_percent": round(final_match * 100, 1),
            "skills_detail": {
                "matching": [
                    {"job_skill": skill, "matched_via": skill}
                    for skill in exact_matching_skills
                ],
                "missing": missing_skills,
            },
            "inconclusive": not bool(job_text),
            "required_skills": required_skills,
            "required_experience_years": required_experience,
            "candidate_experience_years": candidate_experience,
            "required_education": required_education,
            "candidate_education": candidate_education,
        }
        prepared.append(item)

        if job_text:
            valid_indexes.append(len(prepared) - 1)
            valid_texts.append(job_text)

    if valid_texts:
        job_vectors = embeddings.embed_documents(valid_texts)
        if len(job_vectors) != len(valid_texts):
            raise ValueError("The embedding model returned an unexpected vector count.")

        for index, vector in zip(valid_indexes, job_vectors):
            skills_similarity = max(
                0.0, cosine_similarity(candidate_vector, vector)
            )
            final_match = (
                skills_similarity * SKILLS_WEIGHT
                + prepared[index]["experience_match"] * EXPERIENCE_WEIGHT
                + prepared[index]["education_match"] * EDUCATION_WEIGHT
            )
            prepared[index]["skills_similarity"] = skills_similarity
            prepared[index]["skills_score"] = round(skills_similarity * 100, 1)
            prepared[index]["final_match"] = final_match
            prepared[index]["final_score"] = round(final_match * 100, 1)
            prepared[index]["score_percent"] = round(final_match * 100, 1)

    return sorted(
        prepared,
        key=lambda item: (item["inconclusive"], -item["final_match"]),
    )
