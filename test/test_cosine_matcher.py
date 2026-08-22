"""Tests for the LinkedIn agent's cosine-similarity matcher.

The tests use a fake embedding provider, so they are deterministic and do
not download a sentence-transformer model or require API credentials.
"""

import os
import sys
from math import isclose
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cosine_matcher import (
    build_candidate_text,
    build_job_text,
    calculate_education_score,
    calculate_experience_score,
    cosine_similarity,
    rank_jobs_by_cosine,
)


def candidate(**overrides):
    values = {
        "job_titles": ["Machine Learning Engineer", "Data Scientist"],
        "skills": ["Python", "PyTorch", "Docker"],
        "experience_years": 2.5,
        "highest_education_level": "Master",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def requirements(**overrides):
    values = {
        "job_title": None,
        "seniority_level": None,
        "required_skills": [],
        "required_experience_years": None,
        "required_education_level": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeEmbeddings:
    """Return configured vectors while recording embedding calls."""

    def __init__(self, candidate_vector, job_vectors):
        self.candidate_vector = candidate_vector
        self.job_vectors = job_vectors
        self.query_calls = []
        self.document_calls = []

    def embed_query(self, text):
        self.query_calls.append(text)
        return self.candidate_vector

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return self.job_vectors[: len(texts)]


def test_build_candidate_text_uses_only_unique_skills():
    cv = candidate(
        job_titles=["ML Engineer", "ML Engineer", "Data Scientist"],
        skills=["Python", "python", "Docker", ""],
    )

    text = build_candidate_text(cv)

    assert text == "Candidate skills: Python, Docker."
    assert "ML Engineer" not in text


def test_build_job_text_uses_parsed_requirements():
    job = {"title": "Fallback title"}
    parsed = requirements(
        job_title="AI Engineer",
        seniority_level="Senior",
        required_skills=["Python", "Docker", "python"],
    )

    text = build_job_text(job, parsed)

    assert text == "Required skills: Python, Docker."
    assert "AI Engineer" not in text
    assert "Senior" not in text


def test_build_job_text_does_not_add_scraped_title():
    text = build_job_text(
        {"title": "Backend Engineer"},
        requirements(required_skills=["Python"]),
    )

    assert text == "Required skills: Python."
    assert "Backend Engineer" not in text


def test_experience_score_uses_candidate_required_ratio():
    assert isclose(calculate_experience_score(0.17, 5.0), 0.034)
    assert calculate_experience_score(6.0, 5.0) == 1.0
    assert calculate_experience_score(0.17, None) == 1.0


def test_education_score_uses_ordered_levels():
    assert calculate_education_score("Master", "Bachelor") == 1.0
    assert calculate_education_score("Bachelor", "Master") == 0.5
    assert calculate_education_score("High School", "Master") == 0.0
    assert calculate_education_score("Master", None) == 1.0


def test_cosine_similarity_for_identical_vectors():
    assert isclose(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)


def test_cosine_similarity_for_orthogonal_vectors():
    assert isclose(cosine_similarity([1, 0], [0, 1]), 0.0)


def test_cosine_similarity_for_zero_vector_is_zero():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_cosine_similarity_rejects_different_dimensions():
    try:
        cosine_similarity([1, 2], [1, 2, 3])
    except ValueError as exc:
        assert "dimensions must match" in str(exc)
    else:
        raise AssertionError("Different embedding dimensions must raise ValueError")


def test_rank_jobs_descending_and_embed_in_one_batch():
    jobs = [
        (
            {"title": "Accountant", "company": "Finance Co", "url": "u1"},
            requirements(job_title="Accountant", required_skills=["Excel"]),
        ),
        (
            {"title": "AI Engineer", "company": "AI Co", "url": "u2"},
            requirements(
                job_title="AI Engineer",
                required_skills=["Python", "PyTorch"],
                required_experience_years=3,
                required_education_level="Bachelor",
            ),
        ),
        (
            {"title": "Data Engineer", "company": "Data Co", "url": "u3"},
            requirements(
                job_title="Data Engineer",
                required_skills=["Python", "SQL"],
            ),
        ),
    ]
    embeddings = FakeEmbeddings(
        candidate_vector=[1.0, 0.0],
        # Scores in input order: 0%, 100%, 60%.
        job_vectors=[[0.0, 1.0], [1.0, 0.0], [0.6, 0.8]],
    )

    ranked = rank_jobs_by_cosine(candidate(), jobs, embeddings=embeddings)

    assert [item["job_title"] for item in ranked] == [
        "AI Engineer",
        "Data Engineer",
        "Accountant",
    ]
    assert [item["skills_score"] for item in ranked] == [100.0, 60.0, 0.0]
    assert [item["final_score"] for item in ranked] == [95.0, 80.0, 50.0]
    assert len(embeddings.query_calls) == 1
    assert len(embeddings.document_calls) == 1
    assert len(embeddings.document_calls[0]) == 3

    top = ranked[0]
    assert top["candidate_experience_years"] == 2.5
    assert top["required_experience_years"] == 3
    assert top["candidate_education"] == "Master"
    assert top["required_education"] == "Bachelor"
    assert top["experience_score"] == 83.3
    assert top["education_score"] == 100.0


def test_job_without_semantic_information_is_inconclusive_and_last():
    jobs = [
        ({"title": "", "company": "Unknown", "url": "empty"}, requirements()),
        (
            {"title": "Python Developer", "company": "Acme", "url": "valid"},
            requirements(required_skills=["Python"]),
        ),
    ]
    embeddings = FakeEmbeddings(
        candidate_vector=[1.0, 0.0],
        job_vectors=[[1.0, 0.0]],
    )

    ranked = rank_jobs_by_cosine(candidate(), jobs, embeddings=embeddings)

    assert ranked[0]["url"] == "valid"
    assert ranked[0]["inconclusive"] is False
    assert ranked[1]["url"] == "empty"
    assert ranked[1]["inconclusive"] is True
    assert ranked[1]["skills_score"] == 0.0
    assert ranked[1]["final_score"] == 50.0
    assert len(embeddings.document_calls[0]) == 1


def test_candidate_without_skills_is_rejected():
    empty_candidate = candidate(job_titles=[], skills=[])
    embeddings = FakeEmbeddings([1.0], [])

    try:
        rank_jobs_by_cosine(empty_candidate, [], embeddings=embeddings)
    except ValueError as exc:
        assert "no skills" in str(exc)
    else:
        raise AssertionError("An empty candidate profile must raise ValueError")


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"[PASS] {name}")

    print(f"All {len(tests)} cosine matcher tests passed.")
