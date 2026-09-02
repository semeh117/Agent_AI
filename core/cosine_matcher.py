"""Deterministic, explainable cosine matching for Agent 2.

``calculate_compatibility_cosine`` mirrors ``core.matcher``: it receives one
parsed CV and one parsed job, then returns the same compatibility dictionary.
Skills are matched individually with a normalized-literal pass followed by
embedding cosine similarity only for the unmatched job skills.
"""

from __future__ import annotations

from functools import lru_cache
from math import fsum, sqrt
import re
from typing import Any, Iterable, Optional

from core.esco_normalizer import NormalizedSkill, get_esco_normalizer
from core.matcher import _education_match, _experience_match


SKILLS_WEIGHT = 0.5
EXPERIENCE_WEIGHT = 0.3
EDUCATION_WEIGHT = 0.2
DEFAULT_COSINE_THRESHOLD = 0.59
MAX_ALTERNATIVE_GROUP_SIZE = 8


def _clean_values(values: Iterable[Any]) -> list[str]:
    """Return unique, non-empty strings while preserving their order."""

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        key = text.casefold()
        if text and key not in seen:
            cleaned.append(text)
            seen.add(key)
    return cleaned


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


def _skill_tokens(value: str) -> list[str]:
    """Tokenize skill names while preserving forms such as C++ and C#."""

    return re.findall(r"[a-z0-9]+(?:\+\+|#)?", value.casefold())


_ALIAS_GROUP_VALUES = (
    ("Anthropic Claude", "Claude"),
    (
        "retrieval-augmented generation",
        "RAG",
        "RAG systems",
        "RAG pipelines",
        "RAG techniques",
        "RAG patterns",
    ),
    ("large language model", "large language models", "LLM", "LLMs"),
    ("LLM API", "LLM APIs", "LLM API integration", "LLM API integrations"),
    (
        "continuous integration continuous delivery",
        "CI/CD",
        "CI CD",
        "CI/CD practices",
    ),
    ("generative AI", "GenAI"),
    ("machine learning", "ML"),
    ("artificial intelligence", "AI"),
    ("Hugging Face Transformers", "HuggingFace Transformers"),
    ("OpenAI", "OpenAI API"),
    ("Git", "Git/GitHub"),
    ("Google Cloud Platform", "GCP"),
    ("Amazon Web Services", "AWS"),
)


def _surface_skill_key(value: str) -> str:
    return "".join(_skill_tokens(value))


_ALIAS_LOOKUP = {
    key: group[0]
    for values in _ALIAS_GROUP_VALUES
    for group in [tuple(_surface_skill_key(value) for value in values)]
    for key in group
}


# Reviewed one-way evidence relationships. These do not make products
# equivalent to one another: ChromaDB can prove the broader capability
# ``vector databases``, but it never proves Pinecone specifically.
_CAPABILITY_EVIDENCE_VALUES = {
    "machine learning models": (
        "PyTorch",
        "TensorFlow",
        "Scikit-learn",
        "XGBoost",
        "Random Forest",
        "Logistic Regression",
    ),
    "machine learning models and algorithms": (
        "PyTorch",
        "TensorFlow",
        "Scikit-learn",
        "XGBoost",
        "Random Forest",
        "Logistic Regression",
    ),
    "vector databases": (
        "ChromaDB",
        "Pinecone",
        "Weaviate",
        "pgvector",
    ),
    "version control software": (
        "Git",
        "Git/GitHub",
        "Mercurial",
        "CVS",
        "TFS",
        "Subversion",
    ),
}


def _skill_identity_key(value: str) -> str:
    """Return one identity for exact spellings and reviewed abbreviations."""

    surface = _surface_skill_key(value)
    return _ALIAS_LOOKUP.get(surface, surface)


_CAPABILITY_EVIDENCE = {
    _surface_skill_key(requirement): {
        _skill_identity_key(evidence) for evidence in evidence_values
    }
    for requirement, evidence_values in _CAPABILITY_EVIDENCE_VALUES.items()
}


def _capability_evidence_hit(
    job_skill: str, candidates: Iterable[NormalizedSkill]
) -> Optional[NormalizedSkill]:
    accepted = _CAPABILITY_EVIDENCE.get(_surface_skill_key(job_skill), set())
    return next(
        (
            candidate
            for candidate in candidates
            if _skill_identity_key(candidate.original) in accepted
        ),
        None,
    )


def _literal_skill_match(job_skill: str, candidate_skill: str) -> bool:
    """Match only the same normalized name or a reviewed abbreviation.

    A token contained in a longer phrase is not necessarily the same skill:
    ``Claude`` does not prove ``Claude Code`` and ML ``regression`` is not a
    ``regression suite``.  Those relationships must reach the semantic pass
    instead of being accepted as certain literal matches.  Removing separators
    still treats superficial forms such as ``CrewAI``/``Crew AI`` and
    ``Docker-Compose``/``Docker Compose`` as the same name.
    """

    job_key = _skill_identity_key(job_skill)
    candidate_key = _skill_identity_key(candidate_skill)
    return bool(job_key) and job_key == candidate_key


def _looks_like_named_technology(value: str) -> bool:
    """Conservatively identify product/framework names from their spelling.

    This is deliberately domain-independent: acronyms, versioned names,
    CamelCase names, and short title-cased product names receive the same
    treatment whether they belong to AI, cybersecurity, or electronics.
    """

    text = str(value or "").strip()
    words = re.findall(r"[A-Za-z0-9]+(?:\+\+|#)?", text)
    if not words:
        return False
    if any(any(character.isdigit() for character in word) for word in words):
        return True
    if any(re.search(r"[a-z][A-Z]", word) for word in words):
        return True
    # Acronyms inside a general capability phrase (RAG techniques, vector
    # databases, AI applications) must remain eligible for semantic matching.
    # A standalone acronym such as AWS or a title-cased product such as Claude
    # Code remains protected from cosine-based product substitution.
    general_suffixes = {
        "applications",
        "concepts",
        "databases",
        "deployments",
        "development",
        "embeddings",
        "evaluation",
        "frameworks",
        "graphs",
        "infrastructure",
        "intelligence",
        "learning",
        "models",
        "monitoring",
        "orchestration",
        "patterns",
        "pipelines",
        "platforms",
        "practices",
        "retrieval",
        "search",
        "services",
        "systems",
        "techniques",
        "testing",
        "workflows",
    }
    if words[-1].casefold() in general_suffixes:
        return False
    if any(len(word) >= 2 and word.isupper() for word in words):
        return True
    if all(word[:1].isupper() for word in words):
        return True
    return False


def _known_semantic_conflict(job_skill: str, candidate_skill: str) -> bool:
    """Block reviewed false friends that share words but not capabilities."""

    job_key = _surface_skill_key(job_skill)
    candidate_key = _surface_skill_key(candidate_skill)
    return (
        _skill_identity_key(job_skill) == _skill_identity_key("RAG")
        and candidate_key == "ragas"
    ) or (
        job_key == "regression" and "regressionsuite" in candidate_key
    ) or (
        job_key == "versioncontrol" and "promptversion" in candidate_key
    ) or (
        job_key == "claudecode" and candidate_key == "claude"
    )


def _semantic_skill_pair_allowed(job_skill: str, candidate_skill: str) -> bool:
    """Say whether cosine may infer equivalence for this pair.

    Named technologies must match literally or through a reviewed alias.
    Cosine remains available for general competencies and methods.
    """

    if _literal_skill_match(job_skill, candidate_skill):
        return True
    if _looks_like_named_technology(job_skill):
        return False
    return not _known_semantic_conflict(job_skill, candidate_skill)


@lru_cache(maxsize=1)
def _default_esco_normalizer() -> Optional[Any]:
    """Load ESCO once; keep Agent 2 operational if its local CSV is absent."""

    try:
        return get_esco_normalizer()
    except (FileNotFoundError, OSError, ValueError):
        return None


def _prepare_skills(
    values: Iterable[Any],
    normalizer: Optional[Any],
    *,
    deduplicate_concepts: bool,
) -> list[NormalizedSkill]:
    """Normalize and deduplicate skills by ESCO identity or original text."""

    prepared: list[NormalizedSkill] = []
    seen: set[str] = set()
    for original in _clean_values(values):
        normalized = (
            normalizer.normalize(original)
            if normalizer is not None
            else NormalizedSkill(original=original, matching_text=original)
        )
        identity = (
            normalized.identity_key
            if deduplicate_concepts
            else f"text:{_surface_skill_key(normalized.original)}"
        )
        if not normalized.original or identity in seen:
            continue
        prepared.append(normalized)
        seen.add(identity)
    return prepared


def _skills_match_cosine(
    candidate_skills: Iterable[Any],
    required_skills: Iterable[Any],
    threshold: float,
    embeddings: Optional[Any] = None,
    esco_normalizer: Optional[Any] = None,
    use_esco: bool = True,
) -> dict[str, Any]:
    """Match required skills using the same two-pass pattern as matcher.py."""

    if not -1.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between -1.0 and 1.0.")

    if use_esco and esco_normalizer is None:
        esco_normalizer = _default_esco_normalizer()
    if not use_esco:
        esco_normalizer = None

    candidates = _prepare_skills(
        candidate_skills,
        esco_normalizer,
        deduplicate_concepts=True,
    )
    required = _prepare_skills(
        required_skills,
        esco_normalizer,
        deduplicate_concepts=False,
    )
    normalization = {
        "enabled": esco_normalizer is not None,
        "candidate_mapped": sum(skill.mapped for skill in candidates),
        "required_mapped": sum(skill.mapped for skill in required),
        "candidate_total": len(candidates),
        "required_total": len(required),
        "note": "ESCO coverage only; these counts are not candidate-job matches.",
    }
    if not required:
        return {
            "score": 0.5,
            "matching": [],
            "missing": [],
            "normalization": normalization,
            "note": (
                "Job posting listed no explicit required skills — "
                "inconclusive, not a verified match."
            ),
        }

    matching: list[dict[str, str]] = []
    remaining: list[NormalizedSkill] = []

    # Pass 1: cheap normalized equality. Partial-name relationships are left
    # for cosine instead of being treated as guaranteed exact matches.
    for job_skill in required:
        literal_hit = next(
            (
                candidate
                for candidate in candidates
                if _literal_skill_match(job_skill.original, candidate.original)
            ),
            None,
        )
        esco_hit = next(
            (
                candidate
                for candidate in candidates
                if job_skill.concept_uri
                and candidate.concept_uri == job_skill.concept_uri
            ),
            None,
        )
        capability_hit = _capability_evidence_hit(job_skill.original, candidates)
        if literal_hit:
            matching.append(
                {
                    "job_skill": job_skill.original,
                    "matched_via": literal_hit.original,
                }
            )
        elif esco_hit:
            matching.append(
                {
                    "job_skill": job_skill.original,
                    "matched_via": (
                        f"{esco_hit.original} "
                        f"(ESCO: {job_skill.preferred_label})"
                    ),
                }
            )
        elif capability_hit:
            matching.append(
                {
                    "job_skill": job_skill.original,
                    "matched_via": (
                        f"{capability_hit.original} (capability evidence)"
                    ),
                }
            )
        else:
            remaining.append(job_skill)

    # Pass 2: embed individual skill strings, never aggregate sentences.
    missing: list[str] = []
    if remaining and candidates:
        if embeddings is None:
            from config import get_embeddings

            embeddings = get_embeddings()

        texts = [skill.matching_text for skill in candidates + remaining]
        vectors = embeddings.embed_documents(texts)
        if len(vectors) != len(texts):
            raise ValueError("The embedding model returned an unexpected vector count.")

        candidate_vectors = vectors[: len(candidates)]
        required_vectors = vectors[len(candidates) :]
        for job_skill, job_vector in zip(remaining, required_vectors):
            similarities = [
                (
                    cosine_similarity(job_vector, candidate_vector)
                    if _semantic_skill_pair_allowed(
                        job_skill.original, candidate_skill.original
                    )
                    else -1.0
                )
                for candidate_skill, candidate_vector in zip(
                    candidates, candidate_vectors
                )
            ]
            best_index = max(range(len(similarities)), key=similarities.__getitem__)
            # Stabilize threshold comparisons against insignificant float noise.
            best_similarity = round(similarities[best_index], 6)
            if best_similarity >= threshold:
                matched_skill = candidates[best_index]
                matching.append(
                    {
                        "job_skill": job_skill.original,
                        "matched_via": (
                            f"{matched_skill.original} "
                            f"(cosine={best_similarity:.2f})"
                        ),
                    }
                )
            else:
                missing.append(job_skill.original)
    else:
        missing.extend(skill.original for skill in remaining)

    return {
        "score": len(matching) / len(required),
        "matching": matching,
        "missing": missing,
        "normalization": normalization,
    }


def _apply_required_skill_groups(
    skills_result: dict[str, Any],
    groups: Iterable[Iterable[Any]],
) -> dict[str, Any]:
    """Score explicit alternative lists once instead of once per option."""

    cleaned_groups: list[list[str]] = []
    claimed: set[str] = set()
    for values in groups:
        group = _clean_values(values)
        keys = [_skill_identity_key(value) for value in group]
        if (
            len(group) < 2
            or len(group) > MAX_ALTERNATIVE_GROUP_SIZE
            or any(not key or key in claimed for key in keys)
        ):
            continue
        cleaned_groups.append(group)
        claimed.update(keys)
    if not cleaned_groups:
        return skills_result

    flat_matching = list(skills_result["matching"])
    flat_missing = list(skills_result["missing"])
    grouped_matching: list[dict[str, str]] = [
        match
        for match in flat_matching
        if _skill_identity_key(match["job_skill"]) not in claimed
    ]
    grouped_missing: list[str] = [
        skill
        for skill in flat_missing
        if _skill_identity_key(skill) not in claimed
    ]

    matched_count = len(grouped_matching)
    for group in cleaned_groups:
        group_keys = {_skill_identity_key(skill) for skill in group}
        hit = next(
            (
                match
                for match in flat_matching
                if _skill_identity_key(match["job_skill"]) in group_keys
            ),
            None,
        )
        label = "one of: " + " | ".join(group)
        if hit is not None:
            grouped_matching.append(
                {
                    "job_skill": label,
                    "matched_via": (
                        f"{hit['job_skill']} <- {hit['matched_via']}"
                    ),
                }
            )
            matched_count += 1
        else:
            grouped_missing.append(label)

    scored_total = len(grouped_matching) + len(grouped_missing)
    result = {
        **skills_result,
        "score": matched_count / scored_total if scored_total else 0.5,
        "matching": grouped_matching,
        "missing": grouped_missing,
    }
    result["normalization"] = {
        **skills_result["normalization"],
        "raw_required_total": (
            len(flat_matching) + len(flat_missing)
        ),
        "scored_requirement_total": scored_total,
        "alternative_group_count": len(cleaned_groups),
    }
    return result


def _calculate_compatibility_cosine(
    cv_info: Any,
    job_requirements: Any,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    embeddings: Optional[Any] = None,
    esco_normalizer: Optional[Any] = None,
    use_esco: bool = True,
) -> dict[str, Any]:
    skills_result = _skills_match_cosine(
        getattr(cv_info, "skills", []) or [],
        getattr(job_requirements, "required_skills", []) or [],
        threshold,
        embeddings,
        esco_normalizer=esco_normalizer,
        use_esco=use_esco,
    )
    skills_result = _apply_required_skill_groups(
        skills_result,
        getattr(job_requirements, "required_skill_groups", []) or [],
    )
    experience_result = _experience_match(
        getattr(cv_info, "experience_years", None),
        getattr(job_requirements, "required_experience_years", None),
    )
    education_result = _education_match(
        getattr(cv_info, "highest_education_level", None),
        getattr(job_requirements, "required_education_level", None),
    )

    final_score = (
        skills_result["score"] * SKILLS_WEIGHT
        + experience_result["score"] * EXPERIENCE_WEIGHT
        + education_result["score"] * EDUCATION_WEIGHT
    )
    return {
        "score_percent": round(final_score * 100, 1),
        "skills": skills_result,
        "experience": experience_result,
        "education": education_result,
    }


def calculate_compatibility_cosine(
    cv_info: Any,
    job_requirements: Any,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    use_esco: bool = True,
) -> dict[str, Any]:
    """Score one candidate against one parsed job using cosine similarity."""

    return _calculate_compatibility_cosine(
        cv_info,
        job_requirements,
        threshold,
        use_esco=use_esco,
    )


def rank_jobs_by_cosine(
    cv_info: Any,
    parsed_jobs: list[tuple[dict[str, Any], Any]],
    embeddings: Optional[Any] = None,
    threshold: float = DEFAULT_COSINE_THRESHOLD,
    use_esco: bool = True,
) -> list[dict[str, Any]]:
    """Compatibility wrapper used by the current LinkedIn ranking pipeline.

    Each job is scored independently through the one-job matcher. The optional
    embedding object is shared only to avoid repeatedly loading the same model.
    """

    if embeddings is None:
        # Share one lazily-created model across jobs, while still avoiding any
        # model load when every required skill is resolved by the exact pass.
        class _LazyEmbeddings:
            instance = None

            def embed_documents(self, texts):
                if self.instance is None:
                    from config import get_embeddings

                    self.instance = get_embeddings()
                return self.instance.embed_documents(texts)

        embeddings = _LazyEmbeddings()

    prepared: list[dict[str, Any]] = []
    esco_normalizer = _default_esco_normalizer() if use_esco else None
    for job, requirements in parsed_jobs:
        result = _calculate_compatibility_cosine(
            cv_info,
            requirements,
            threshold=threshold,
            embeddings=embeddings,
            esco_normalizer=esco_normalizer,
            use_esco=use_esco,
        )
        skills = result["skills"]
        experience = result["experience"]
        education = result["education"]
        inconclusive = "note" in skills and not getattr(
            requirements, "required_skills", []
        )

        prepared.append(
            {
                "job_title": (
                    getattr(requirements, "job_title", None)
                    or job.get("title")
                    or ""
                ),
                "company": job.get("company", ""),
                "url": job.get("url", ""),
                "description": job.get("description", ""),
                "skills_similarity": skills["score"],
                "skills_coverage": skills["score"],
                "skills_score": round(skills["score"] * 100, 1),
                "experience_match": experience["score"],
                "experience_score": round(experience["score"] * 100, 1),
                "experience_detail": experience,
                "education_match": education["score"],
                "education_score": round(education["score"] * 100, 1),
                "education_detail": education,
                "final_match": result["score_percent"] / 100.0,
                "final_score": result["score_percent"],
                "score_percent": result["score_percent"],
                "skills_detail": {
                    "matching": skills["matching"],
                    "missing": skills["missing"],
                    "normalization": skills["normalization"],
                },
                "inconclusive": inconclusive,
                "required_skills": _clean_values(
                    getattr(requirements, "required_skills", []) or []
                ),
                "required_skill_groups": [
                    _clean_values(group)
                    for group in (
                        getattr(requirements, "required_skill_groups", []) or []
                    )
                    if len(_clean_values(group)) >= 2
                ],
                "required_experience_years": getattr(
                    requirements, "required_experience_years", None
                ),
                "candidate_experience_years": getattr(
                    cv_info, "experience_years", None
                ),
                "required_education": getattr(
                    requirements, "required_education_level", None
                ),
                "candidate_education": getattr(
                    cv_info, "highest_education_level", None
                ),
            }
        )

    return sorted(
        prepared,
        key=lambda item: (item["inconclusive"], -item["final_score"]),
    )
