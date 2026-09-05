"""Deterministic tests for the one-job cosine compatibility matcher."""

import os
import sys
from math import isclose
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.cosine_matcher import (
    DEFAULT_COSINE_THRESHOLD,
    _apply_required_skill_groups,
    calculate_compatibility_cosine as calculate_compatibility_cosine_with_esco,
    cosine_similarity,
    rank_jobs_by_cosine as rank_jobs_by_cosine_with_esco,
)
from core.esco_normalizer import get_esco_normalizer
from pipeline.linkedin_cosine_pipeline import match_linkedin_jobs
from search.job_scraper import _build_search_url, _job_url_identity


def calculate_compatibility_cosine(*args, **kwargs):
    """Keep legacy matcher unit cases isolated; ESCO has dedicated cases below."""

    kwargs.setdefault("use_esco", False)
    return calculate_compatibility_cosine_with_esco(*args, **kwargs)


def rank_jobs_by_cosine(*args, **kwargs):
    kwargs.setdefault("use_esco", False)
    return rank_jobs_by_cosine_with_esco(*args, **kwargs)


def candidate(**overrides):
    values = {
        "skills": ["Python", "PyTorch", "Docker"],
        "experience_years": 2.5,
        "highest_education_level": "Master",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def requirements(**overrides):
    values = {
        "job_title": "AI Engineer",
        "required_skills": [],
        "required_experience_years": None,
        "required_education_level": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeEmbeddings:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [self.vectors[text] for text in texts]


def with_fake_embeddings(fake, function):
    original = config.get_embeddings
    config.get_embeddings = lambda: fake
    try:
        return function()
    finally:
        config.get_embeddings = original


def test_cosine_similarity_math_and_dimension_validation():
    assert isclose(cosine_similarity([1, 2], [1, 2]), 1.0)
    assert isclose(cosine_similarity([1, 0], [0, 1]), 0.0)
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    try:
        cosine_similarity([1], [1, 2])
    except ValueError as exc:
        assert "dimensions must match" in str(exc)
    else:
        raise AssertionError("Dimension mismatch must fail.")


def test_normalized_literal_matches_do_not_load_embeddings():
    original = config.get_embeddings
    config.get_embeddings = lambda: (_ for _ in ()).throw(
        AssertionError("Exact matches must not load embeddings.")
    )
    try:
        result = calculate_compatibility_cosine(
            candidate(skills=["Python", "Docker Compose"]),
            requirements(required_skills=["python", "Docker-Compose"]),
        )
    finally:
        config.get_embeddings = original

    assert result["skills"]["score"] == 1.0
    assert result["skills"]["missing"] == []
    assert result["skills"]["matching"][1]["matched_via"] == "Docker Compose"


def test_partial_names_and_contextual_phrases_are_not_literal_matches():
    fake = FakeEmbeddings(
        {
            "Claude": [1.0, 0.0],
            "regression suites": [1.0, 0.0],
            "Claude Code": [0.0, 1.0],
            "regression": [0.0, 1.0],
        }
    )
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["Claude", "regression suites"]),
            requirements(required_skills=["Claude Code", "regression"]),
        ),
    )

    assert result["skills"]["matching"] == []
    assert result["skills"]["missing"] == ["Claude Code", "regression"]


def test_reviewed_aliases_match_without_loading_embeddings():
    original = config.get_embeddings
    config.get_embeddings = lambda: (_ for _ in ()).throw(
        AssertionError("Reviewed aliases must resolve before embeddings.")
    )
    try:
        result = calculate_compatibility_cosine(
            candidate(
                skills=[
                    "Claude",
                    "retrieval-augmented generation",
                    "LLM API integrations",
                    "OpenAI API",
                    "Git/GitHub",
                ]
            ),
            requirements(
                required_skills=[
                    "Anthropic Claude",
                    "RAG systems",
                    "RAG techniques",
                    "RAG patterns",
                    "LLM APIs",
                    "OpenAI",
                    "Git",
                ]
            ),
        )
    finally:
        config.get_embeddings = original

    assert result["skills"]["score"] == 1.0
    assert result["skills"]["missing"] == []


def test_specific_tools_prove_broad_capabilities_only_one_way():
    original = config.get_embeddings
    config.get_embeddings = lambda: (_ for _ in ()).throw(
        AssertionError("Reviewed capability evidence must resolve before embeddings.")
    )
    try:
        result = calculate_compatibility_cosine(
            candidate(skills=["ChromaDB", "Git/GitHub", "Random Forest"]),
            requirements(
                required_skills=[
                    "vector databases",
                    "Version control software",
                    "Machine learning models",
                ]
            ),
        )
    finally:
        config.get_embeddings = original

    assert result["skills"]["score"] == 1.0
    assert result["skills"]["missing"] == []
    assert all(
        "capability evidence" in match["matched_via"]
        for match in result["skills"]["matching"]
    )


def test_capitalized_general_concept_can_reach_cosine():
    fake = FakeEmbeddings(
        {
            "all-MiniLM-L6-v2 embeddings": [1.0, 0.0],
            "Embeddings": [0.8, 0.6],
        }
    )
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["all-MiniLM-L6-v2 embeddings"]),
            requirements(required_skills=["Embeddings"]),
            threshold=0.59,
        ),
    )

    assert result["skills"]["matching"] == [
        {
            "job_skill": "Embeddings",
            "matched_via": "all-MiniLM-L6-v2 embeddings (cosine=0.80)",
        }
    ]


def test_esco_preferred_and_alternative_labels_share_one_concept():
    normalizer = get_esco_normalizer()
    preferred = normalizer.normalize("manage musical staff")
    alternative = normalizer.normalize("manage music staff")

    assert normalizer.concept_count > 10_000
    assert preferred.mapped is True
    assert alternative.mapped is True
    assert preferred.concept_uri == alternative.concept_uri
    assert alternative.preferred_label == "manage musical staff"
    assert normalizer.normalize("LangChain").mapped is False


def test_esco_concept_match_is_explainable_and_skips_embeddings():
    original = config.get_embeddings
    config.get_embeddings = lambda: (_ for _ in ()).throw(
        AssertionError("The same ESCO concept must match before embeddings.")
    )
    try:
        result = calculate_compatibility_cosine_with_esco(
            candidate(skills=["manage music staff"]),
            requirements(required_skills=["manage musical staff"]),
        )
    finally:
        config.get_embeddings = original

    assert result["skills"]["score"] == 1.0
    assert result["skills"]["missing"] == []
    assert result["skills"]["matching"] == [
        {
            "job_skill": "manage musical staff",
            "matched_via": "manage music staff (ESCO: manage musical staff)",
        }
    ]
    assert result["skills"]["normalization"]["candidate_mapped"] == 1
    assert result["skills"]["normalization"]["required_mapped"] == 1


def test_live_false_pairs_are_blocked_even_with_identical_embeddings():
    candidates = [
        "Llama",
        "RAGAS",
        "LLM evaluation",
        "AWS Bedrock",
        "DeepEval",
        "prompt versioning",
        "regression suites",
        "Claude",
    ]
    required = [
        "LlamaIndex",
        "RAG systems",
        "LLM APIs",
        "LLM pipeline",
        "AWS ECS",
        "DeepAgent",
        "version control",
        "regression",
        "Claude Code",
    ]
    pair_index = {
        "Llama": 0,
        "LlamaIndex": 0,
        "RAGAS": 1,
        "RAG systems": 1,
        "LLM evaluation": 2,
        "LLM APIs": 2,
        "LLM pipeline": 2,
        "AWS Bedrock": 3,
        "AWS ECS": 3,
        "DeepEval": 4,
        "DeepAgent": 4,
        "prompt versioning": 5,
        "version control": 5,
        "regression suites": 6,
        "regression": 6,
        "Claude": 7,
        "Claude Code": 7,
    }
    fake = FakeEmbeddings(
        {
            skill: [1.0 if position == pair_index[skill] else 0.0 for position in range(8)]
            for skill in candidates + required
        }
    )

    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=candidates),
            requirements(required_skills=required),
        ),
    )

    assert result["skills"]["matching"] == []
    assert result["skills"]["missing"] == required


def test_semantic_matching_embeds_each_skill_and_explains_matches():
    fake = FakeEmbeddings(
        {
            "Python": [0.0, 1.0],
            "PyTorch": [1.0, 0.0],
            "Deep Learning": [0.8, 0.6],
            "Kubernetes": [-1.0, 0.0],
        }
    )
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["Python", "PyTorch"]),
            requirements(required_skills=["Deep Learning", "Kubernetes"]),
            threshold=0.65,
        ),
    )

    assert fake.calls == [["Python", "PyTorch", "Deep Learning", "Kubernetes"]]
    assert result["skills"]["score"] == 0.5
    assert result["skills"]["matching"] == [
        {"job_skill": "Deep Learning", "matched_via": "PyTorch (cosine=0.80)"}
    ]
    assert result["skills"]["missing"] == ["Kubernetes"]


def test_calibrated_default_threshold_is_used():
    fake = FakeEmbeddings(
        {
            "Candidate skill": [1.0, 0.0],
            "Job requirement": [0.60, 0.80],
        }
    )
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["Candidate skill"]),
            requirements(required_skills=["Job requirement"]),
        ),
    )

    assert DEFAULT_COSINE_THRESHOLD == 0.59
    assert result["skills"]["matching"] == [
        {
            "job_skill": "Job requirement",
            "matched_via": "Candidate skill (cosine=0.60)",
        }
    ]


def test_short_skill_is_not_matched_inside_an_unrelated_word():
    fake = FakeEmbeddings({"LangChain": [1.0, 0.0], "AI": [0.0, 1.0]})
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["LangChain"]),
            requirements(required_skills=["AI"]),
        ),
    )

    assert result["skills"]["matching"] == []
    assert result["skills"]["missing"] == ["AI"]


def test_product_prefix_is_not_treated_as_an_exact_skill_match():
    fake = FakeEmbeddings({"Llama": [1.0, 0.0], "LlamaIndex": [0.0, 1.0]})
    result = with_fake_embeddings(
        fake,
        lambda: calculate_compatibility_cosine(
            candidate(skills=["Llama"]),
            requirements(required_skills=["LlamaIndex"]),
        ),
    )

    assert result["skills"]["matching"] == []
    assert result["skills"]["missing"] == ["LlamaIndex"]


def test_output_shape_and_weighted_score_match_core_matcher():
    result = calculate_compatibility_cosine(
        candidate(skills=["Python"], experience_years=2.5),
        requirements(
            required_skills=["Python"],
            required_experience_years=5.0,
            required_education_level="Bachelor",
        ),
    )

    assert set(result) == {"score_percent", "skills", "experience", "education"}
    assert result["skills"]["score"] == 1.0
    assert result["experience"]["score"] == 0.5
    assert result["education"]["score"] == 1.0
    assert result["score_percent"] == 85.0


def test_empty_required_skills_is_inconclusive_and_scores_one_half():
    result = calculate_compatibility_cosine(candidate(), requirements())

    assert result["skills"]["score"] == 0.5
    assert result["skills"]["matching"] == []
    assert result["skills"]["missing"] == []
    assert "inconclusive" in result["skills"]["note"]
    assert result["score_percent"] == 75.0


def test_fixed_inputs_produce_identical_results():
    fake = FakeEmbeddings({"PyTorch": [1.0, 0.0], "Deep Learning": [0.8, 0.6]})
    cv = candidate(skills=["PyTorch"])
    job = requirements(required_skills=["Deep Learning"])

    first = with_fake_embeddings(fake, lambda: calculate_compatibility_cosine(cv, job))
    second = with_fake_embeddings(fake, lambda: calculate_compatibility_cosine(cv, job))

    assert first == second


def test_ranking_wrapper_scores_each_job_with_the_same_shape():
    fake = FakeEmbeddings(
        {
            "Python": [1.0, 0.0],
            "Accounting": [0.0, 1.0],
        }
    )
    jobs = [
        (
            {"title": "Accountant", "company": "Fin", "url": "u1"},
            requirements(job_title="Accountant", required_skills=["Accounting"]),
        ),
        (
            {"title": "Python Engineer", "company": "AI", "url": "u2"},
            requirements(job_title="Python Engineer", required_skills=["Python"]),
        ),
    ]

    ranked = rank_jobs_by_cosine(
        candidate(skills=["Python"]), jobs, embeddings=fake, threshold=0.65
    )

    assert [item["url"] for item in ranked] == ["u2", "u1"]
    assert [item["skills_score"] for item in ranked] == [100.0, 0.0]
    assert ranked[0]["skills_detail"]["matching"][0]["job_skill"] == "Python"


def test_explicit_alternatives_are_scored_as_one_requirement_each():
    result = calculate_compatibility_cosine(
        candidate(skills=["PyTorch"]),
        requirements(
            required_skills=[
                "TensorFlow",
                "PyTorch",
                "Scikit-learn",
                "AWS",
                "Azure",
            ],
            required_skill_groups=[
                ["TensorFlow", "PyTorch", "Scikit-learn"],
                ["AWS", "Azure"],
            ],
        ),
    )

    assert result["skills"]["score"] == 0.5
    assert result["skills"]["matching"] == [
        {
            "job_skill": "one of: TensorFlow | PyTorch | Scikit-learn",
            "matched_via": "PyTorch <- PyTorch",
        }
    ]
    assert result["skills"]["missing"] == ["one of: AWS | Azure"]
    assert result["skills"]["normalization"]["raw_required_total"] == 5
    assert result["skills"]["normalization"]["scored_requirement_total"] == 2


def test_oversized_alternative_group_is_ignored():
    skills = [f"Skill {index}" for index in range(1, 10)]
    original = {
        "score": 1 / len(skills),
        "matching": [{"job_skill": skills[0], "matched_via": skills[0]}],
        "missing": skills[1:],
        "normalization": {
            "enabled": False,
            "candidate_mapped": 0,
            "required_mapped": 0,
            "candidate_total": 1,
            "required_total": len(skills),
        },
    }

    result = _apply_required_skill_groups(original, [skills])

    assert result == original


def test_linkedin_pipeline_deduplicates_by_job_id_not_title_company():
    scraped = [
        {"title": "AI Engineer", "company": "Example", "url": "https://linkedin.com/jobs/view/ai-engineer-123", "description": "d1"},
        {"title": "Wrong heading", "company": "Example", "url": "https://linkedin.com/jobs/view/other-slug-123?trk=x", "description": "d2"},
        {"title": "AI Engineer", "company": "Example", "url": "https://linkedin.com/jobs/view/ai-engineer-124", "description": "d3"},
        {"title": "Data Engineer", "company": "Third", "url": "https://linkedin.com/jobs/view/data-engineer-125", "description": "d4"},
    ]
    requested_limits = []

    def fake_search(**kwargs):
        requested_limits.append(kwargs["max_jobs"])
        return scraped

    def fake_parser(job_title, job_description, use_cache):
        return requirements(job_title=job_title, required_skills=["Python"])

    result = match_linkedin_jobs(
        candidate(skills=["Python"]),
        query="AI Engineer Python",
        max_jobs=3,
        search_fn=fake_search,
        parser_fn=fake_parser,
    )

    assert requested_limits == [5]
    assert result["scraped_count"] == 4
    assert result["unique_scraped_count"] == 3
    assert result["duplicate_count"] == 1
    assert result["parsed_count"] == 3
    assert [job["url"] for job in result["ranked_jobs"]] == [
        "https://linkedin.com/jobs/view/ai-engineer-123",
        "https://linkedin.com/jobs/view/ai-engineer-124",
        "https://linkedin.com/jobs/view/data-engineer-125",
    ]


def test_linkedin_search_url_can_request_last_30_days():
    url = _build_search_url(
        "AI Engineer Python",
        "Germany",
        posted_within_hours=30 * 24,
    )
    assert "keywords=AI+Engineer+Python" in url
    assert "location=Germany" in url
    assert "f_TPR=r2592000" in url
    assert _job_url_identity(
        "https://de.linkedin.com/jobs/view/role-12345?tracking=x"
    ) == _job_url_identity(
        "https://www.linkedin.com/jobs/view/different-slug-12345"
    )


def test_linkedin_pipeline_corrects_company_as_title_from_parser():
    def fake_search(**_kwargs):
        return [
            {
                "title": "Mind Maze",
                "company": "Mind Maze",
                "url": "https://linkedin.com/jobs/view/role-999",
                "description": "A complete role description.",
            }
        ]

    def fake_parser(**_kwargs):
        return requirements(job_title="AI Engineer", required_skills=["Python"])

    result = match_linkedin_jobs(
        candidate(skills=["Python"]),
        query="AI Engineer",
        max_jobs=1,
        search_fn=fake_search,
        parser_fn=fake_parser,
    )

    assert result["ranked_jobs"][0]["job_title"] == "AI Engineer"
    assert result["identity_corrections"][0]["original_title"] == "Mind Maze"


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
