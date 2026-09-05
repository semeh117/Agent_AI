"""Review captured CV and LinkedIn job parser fixtures without any API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.agent2_parser import (
    _looks_like_contextual_cv_skill,
    _required_alternative_groups,
)
from dev.replay_parser_fixtures import replay_cv, replay_jobs
from search.job_scraper import _clean_text


CV_FIXTURE = PROJECT_ROOT / "fixtures" / "cv_parser_fixture.json"
JOB_FIXTURE = PROJECT_ROOT / "fixtures" / "linkedin_job_parser_fixture.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Capture it first using the corresponding dev script."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _skill_audit(skills: list[str], source_text: str) -> tuple[list[str], list[str]]:
    source_lower = source_text.casefold()
    ungrounded = [skill for skill in skills if skill.casefold() not in source_lower]
    sentence_like = [
        skill for skill in skills if len(skill) > 55 or len(skill.split()) > 7
    ]
    return ungrounded, sentence_like


def review_cv_fixture() -> dict:
    fixture = _load(CV_FIXTURE)
    parsed = fixture["parsed_cv"]
    source_text = "\n".join(
        filter(None, [fixture.get("raw_text"), fixture.get("layout_text")])
    )
    ungrounded, sentence_like = _skill_audit(
        parsed.get("skills", []), source_text
    )
    return {
        "parser": f"{fixture['parser_provider']} / {fixture['parser_model']}",
        "candidate": parsed.get("full_name"),
        "skills_count": len(parsed.get("skills", [])),
        "skills": parsed.get("skills", []),
        "evidence_count": len(parsed.get("skill_evidence", {})),
        "ungrounded_skills": ungrounded,
        "sentence_like_skills": sentence_like,
        "experience_years": parsed.get("experience_years"),
        "education": parsed.get("highest_education_level"),
    }


def review_job_fixture() -> list[dict]:
    fixture = _load(JOB_FIXTURE)
    reviews = []
    for entry in fixture["jobs"]:
        raw = entry["linkedin_job"]
        parsed = entry.get("parsed_requirements")
        if parsed is None:
            reviews.append(
                {
                    "title": raw.get("title"),
                    "url": raw.get("url"),
                    "error": entry.get("parser_error"),
                }
            )
            continue
        skills = parsed.get("required_skills", [])
        combined_source = f"{raw.get('title', '')} {raw.get('description', '')}"
        ungrounded, sentence_like = _skill_audit(skills, combined_source)
        reviews.append(
            {
                "title": raw.get("title"),
                "company": raw.get("company"),
                "url": raw.get("url"),
                "description_characters": len(raw.get("description", "")),
                "required_skills": skills,
                "required_skill_groups": parsed.get("required_skill_groups", []),
                "preferred_skills": parsed.get("preferred_skills", []),
                "required_skill_evidence": parsed.get(
                    "required_skill_evidence", {}
                ),
                "ungrounded_skills": ungrounded,
                "sentence_like_skills": sentence_like,
                "seniority": parsed.get("seniority_level"),
                "experience_years": parsed.get("required_experience_years"),
                "required_education": parsed.get("required_education_level"),
                "preferred_education": parsed.get("preferred_education_level"),
            }
        )
    return reviews


def test_captured_parser_skills_are_grounded():
    cv_review = review_cv_fixture()
    job_reviews = review_job_fixture()

    assert cv_review["ungrounded_skills"] == []
    for review in job_reviews:
        if "error" in review:
            assert "truncated or too short" in str(review["error"]), review
            continue
        assert review["ungrounded_skills"] == [], review


def test_agent2_parser_postprocessing_rules():
    source = (
        "Experience with one or more of: TensorFlow, PyTorch, or Scikit-learn. "
        "Experience with AI technologies such as Gemini, Anthropic, OpenAI."
    )
    skills = [
        "TensorFlow",
        "PyTorch",
        "Scikit-learn",
        "AI technologies",
        "Gemini",
        "Anthropic",
        "OpenAI",
    ]
    groups = _required_alternative_groups(skills, source)

    assert ["TensorFlow", "PyTorch", "Scikit-learn"] in groups
    assert ["AI technologies", "Gemini", "Anthropic", "OpenAI"] in groups
    assert _looks_like_contextual_cv_skill("ICA artifact removal")
    assert _looks_like_contextual_cv_skill("Monte Carlo simulation")
    assert not _looks_like_contextual_cv_skill("Major Depressive Disorder")
    assert not _looks_like_contextual_cv_skill("outfit rating system")


def test_agent2_alternative_groups_are_local_and_explicit():
    source = (
        "8+ years in traditional ML, including 2+ years with Generative AI\n"
        "Strong experience with LLMs and prompt engineering\n"
        "Real-world experience with LangChain/LangGraph or similar\n"
        "Proficiency with AWS, Azure, or GCP, plus Docker/Kubernetes"
    )
    skills = [
        "Generative AI",
        "LLMs",
        "prompt engineering",
        "LangChain",
        "LangGraph",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
    ]

    groups = _required_alternative_groups(skills, source)

    assert ["LangChain", "LangGraph"] in groups
    assert ["AWS", "Azure", "GCP"] in groups
    assert ["Docker", "Kubernetes"] in groups
    assert all("Generative AI" not in group for group in groups)
    assert all(len(group) <= 8 for group in groups)


def test_linkedin_description_structure_is_preserved():
    raw = (
        "Required Qualifications\n\n"
        "  - Python and REST APIs  \n"
        "\n"
        "- AWS, Azure, or GCP\n"
    )

    assert _clean_text(raw) == (
        "Required Qualifications\n"
        "- Python and REST APIs\n"
        "- AWS, Azure, or GCP"
    )


def test_agent2_hybrid_cv_metadata_rules():
    from core.agent2_cv_parser import (
        _deterministic_education_level,
        _deterministic_experience_years,
        _deterministic_plain_name,
    )

    text = """Devon Cruz   AI Engineer   San Francisco, CA
PROFILE SUMMARY
AI Engineer with 6 years of experience.
EDUCATION
University of Washington   B.S. in Computer Science   Sep 2016 - Jun 2020
WORK EXPERIENCE
Perplexity   Senior AI Engineer   Sep 2022 - Present
Intercom   AI Engineer   Aug 2020 - Aug 2022
"""

    assert _deterministic_plain_name(text) == "Devon Cruz"
    assert _deterministic_education_level(text) == "Bachelor"
    experience = _deterministic_experience_years(text)
    assert experience is not None
    assert 6.0 <= experience <= 6.5


def test_agent2_hybrid_parser_corrects_metadata_and_keeps_methods():
    from core.agent2_cv_parser import extract_cv_info_agent2

    class FakeStructuredModel:
        def invoke(self, _prompt):
            return {
                "full_name": "PROFILE SUMMARY",
                "skills": ["Python"],
                "contextual_skills": [
                    "function calling",
                    "retrieval-augmented generation",
                ],
                "job_titles": ["Senior AI Engineer", "AI Engineer"],
                "experience_years": 10.0,
                "education": ["B.S. in Computer Science"],
                "highest_education_level": None,
                "mail": "aieng@gmail.com",
            }

    class FakeLLM:
        def with_structured_output(self, _schema):
            return FakeStructuredModel()

    source = """Devon Cruz   AI Engineer   aieng@gmail.com
PROFILE SUMMARY
AI Engineer specializing in retrieval-augmented generation and function calling.
TECHNICAL SKILLS
Languages: Python
EDUCATION
University of Washington   B.S. in Computer Science   Sep 2016 - Jun 2020
WORK EXPERIENCE
Perplexity   Senior AI Engineer   Sep 2022 - Present
Intercom   AI Engineer   Aug 2020 - Aug 2022
"""
    layout = """# Devon Cruz
## TECHNICAL SKILLS
| Languages | Python |
## EDUCATION
B.S. in Computer Science
## WORK EXPERIENCE
Senior AI Engineer
"""

    parsed = extract_cv_info_agent2(
        source,
        layout_text=layout,
        llm=FakeLLM(),
        use_cache=False,
    )

    assert parsed.full_name == "Devon Cruz"
    assert parsed.highest_education_level == "Bachelor"
    assert parsed.experience_years is not None
    assert 6.0 <= parsed.experience_years <= 6.5
    assert "function calling" in parsed.skills
    assert "retrieval-augmented generation" in parsed.skills


# ---------------------------------------------------------------------------
# Deterministic-layer regression tests replayed on the captured fixtures.
# The fixtures hold the live Qwen output; replaying it through the cleaning,
# region, grouping, experience and education rules proves each documented
# parsing defect stays fixed without any OpenRouter call.
# ---------------------------------------------------------------------------


def _replayed_jobs_by_company() -> dict[str, dict]:
    jobs = replay_jobs()
    by_company: dict[str, dict] = {}
    for job in jobs:
        assert "error" not in job, job
        by_company.setdefault(job["company"], job)
    return by_company


def test_replayed_cv_removes_project_features_and_duplicates():
    cv = replay_cv()
    skills = cv["skills"]
    lowered = {skill.casefold() for skill in skills}

    # Project/UI features and deployment adjectives are not skills.
    for feature in (
        "multi-select wardrobe with FilterChips",
        "occasion picker",
        "outfit rating system",
        "health checks",
        "environment isolation",
        "dark",
        "ChromaDB vector store (K=4)",
        "128-channel band-pass/notch filtering",
        "LR",
    ):
        assert feature.casefold() not in lowered, feature

    # Slash compounds and proficiency tags are normalized to atomic names.
    assert "SHAP" in skills and "LIME" in skills and "SHAP/LIME" not in skills
    assert "Git" in skills and "GitHub" in skills and "Git/GitHub" not in skills
    assert "MLflow" in skills and "MLflow (familiar)" not in skills

    # "<technology> <usage noun>" collapses onto the technology itself.
    assert "PyMuPDF" in skills and "PyMuPDF ingestion" not in skills
    assert "all-MiniLM-L6-v2" in skills and "all-MiniLM-L6-v2 embeddings" not in skills

    # Real technical methods and names from project prose are kept.
    for kept in (
        "Python", "LangChain", "RAG pipelines", "FAISS", "Docker", "FastAPI",
        "ICA artifact removal", "Logistic Regression", "Random Forest", "MFCC",
        "Gemini Vision API", "DeepEval", "Langfuse", "gRPC", "Kafka 4.x",
    ):
        assert kept in skills, kept
    assert len(skills) == len({s.casefold() for s in skills})


def test_replayed_cv_reports_honest_education_titles_and_headline():
    cv = replay_cv()
    # A 4th-year engineering cycle "graduating 2026 (expected)" has not
    # conferred an engineering (Master-level) diploma yet.
    assert cv["highest_education_level"] == "Bachelor"
    # Only the dated internship counts as professional experience (2 months).
    assert cv["experience_years"] == 0.17
    # The headline is a target role, and "Summer Internship" is an employment
    # type, so neither is a position held.
    assert cv["job_titles"] == []
    assert cv["headline"] == "AI & Machine Learning Engineer"

    from core.agent2_cv_parser import Agent2CVInfo
    from pipeline.linkedin_cosine_pipeline import build_linkedin_query

    query = build_linkedin_query(Agent2CVInfo.model_validate(cv), max_skills=2)
    assert query.startswith("AI & Machine Learning Engineer")


def test_engineering_programme_education_rules():
    from core.agent2_cv_parser import _deterministic_education_level

    assert _deterministic_education_level(
        "Ecole X - Engineering Cycle, Data Science\n2022 - 2026 (expected)"
    ) == "Bachelor"
    assert _deterministic_education_level(
        "Diplôme d'Ingénieur en Informatique, ENSI, 2018 - 2021"
    ) == "Master"
    assert _deterministic_education_level(
        "Engineering Degree in Computer Science (2015-2018)\nM.Sc. Data Science 2019"
    ) == "Master"
    assert _deterministic_education_level("B.Sc. Computer Science 2020") == "Bachelor"
    assert _deterministic_education_level("Team summary and plans") is None


def test_replayed_jobs_exclude_responsibility_duties_from_requirements():
    jobs = _replayed_jobs_by_company()

    hcl = jobs["HCLTech"]
    # HCL has no "Requirements" heading; "Level of Education" now anchors the
    # qualifications paragraph, so duties are no longer scored as skills.
    assert "testing" not in hcl["required_skills"]
    assert "Python" in hcl["required_skills"]
    assert "LangChain" in hcl["required_skills"]
    assert hcl["required_experience_years"] is None
    assert hcl["required_education_level"] == "Bachelor"

    tcs = jobs["Tata Consultancy Services"]
    # "Participate in hands-on technical evaluation, code reviews" is a duty.
    assert "hands-on technical evaluation" not in tcs["required_skills"]
    assert tcs["required_experience_years"] == 5.0  # overall minimum, not 1.5

    epri = jobs["EPRI"]
    for duty in (
        "telemetry",
        "quality analysis",
        "domain adaptation for EPRI\u2019s technical language",
        "Knowledge Graphs",  # named only in the role summary
    ):
        assert duty not in epri["required_skills"], duty
    assert "Kubernetes" in epri["required_skills"]
    assert epri["required_experience_years"] == 7.0
    # Unbalanced brackets copied from "(FAISS, Weaviate, Pinecone)" are gone.
    assert "FAISS" in epri["preferred_skills"] and "Pinecone" in epri["preferred_skills"]
    assert not any("(" in s or ")" in s for s in epri["preferred_skills"])

    tr = jobs["Thomson Reuters"]
    assert tr["required_experience_years"] == 3.0  # 6+ years is preferred only
    assert "PyTorch" in tr["preferred_skills"] and "PyTorch" not in tr["required_skills"]


def test_replayed_jobs_keep_alternative_groups_local():
    jobs = _replayed_jobs_by_company()

    def groups(company: str) -> list[list[str]]:
        return jobs[company]["required_skill_groups"]

    # Thomson Reuters: "such as RAG patterns, ReAct, LangChain etc for
    # performing document summarization, knowledge graphs, information
    # extraction, or analysis" -> the examples stop at "etc for"; the tasks
    # after it are separate requirements, not alternatives.
    tr_groups = groups("Thomson Reuters")
    assert ["Generative AI technologies", "RAG patterns", "ReAct", "LangChain"] in tr_groups
    assert all("document summarization" not in g and "knowledge graphs" not in g for g in tr_groups)

    # TCS: "RAG and vector search concepts including embeddings, chunking,
    # ..., and disambiguation flows" is a mandatory list, never one group.
    tcs_groups = groups("Tata Consultancy Services")
    assert all("embeddings" not in g and "chunking" not in g for g in tcs_groups)
    assert ["LangChain", "LangGraph", "LlamaIndex", "function calling", "custom orchestration"] in tcs_groups
    assert ["AWS", "GCP"] in tcs_groups

    # HCL: "Claude" inside "Claude Code" is the same mention, and "such as
    # GitHub Copilot, ..., or Microsoft Copilot" is exactly one group.
    hcl_groups = groups("HCLTech")
    assert [
        "GitHub Copilot", "Codex", "Claude Code", "LangChain", "ADK", "ChatGPT", "Microsoft Copilot",
    ] in hcl_groups
    assert ["cloud platforms", "AWS", "GCP"] in hcl_groups
    assert all(len(g) <= 8 for company in jobs for g in groups(company))


def test_job_region_fallback_never_scores_duties():
    from core.agent2_job_parser import _job_regions

    description = (
        "Acme is hiring. Role/Responsibilities Design and deploy LLM agents. "
        "Evaluate Claude, GPT and Gemini. Partner with product teams. "
        "Strong programming experience in Python and TypeScript. "
        "Hands-on experience with LangChain or LlamaIndex. "
        "Benefits Free lunch and Kubernetes clusters for everyone."
    )
    regions = _job_regions(description)
    assert not regions.has_explicit_required_heading
    assert "Python" in regions.required and "LangChain" in regions.required
    assert "Evaluate Claude" not in regions.required
    assert "Design and deploy" not in regions.required
    assert "Kubernetes" not in regions.required

    # Lowercase prose such as "the role of Generative AI Engineer" or
    # "customer requirements" is not a heading.
    prose = "You're a good fit for the role of AI Engineer if you meet customer requirements."
    assert _job_regions(prose).headings == ()


def main() -> int:
    cv_review = review_cv_fixture()
    job_reviews = review_job_fixture()

    print("CV PARSER REVIEW")
    print(json.dumps(cv_review, indent=2, ensure_ascii=False))
    print("\nLINKEDIN JOB PARSER REVIEW")
    for index, review in enumerate(job_reviews, start=1):
        print(f"\nJob {index}")
        print(json.dumps(review, indent=2, ensure_ascii=False))

    test_captured_parser_skills_are_grounded()
    test_agent2_parser_postprocessing_rules()
    test_agent2_alternative_groups_are_local_and_explicit()
    test_linkedin_description_structure_is_preserved()
    test_agent2_hybrid_cv_metadata_rules()
    test_agent2_hybrid_parser_corrects_metadata_and_keeps_methods()
    test_replayed_cv_removes_project_features_and_duplicates()
    test_replayed_cv_reports_honest_education_titles_and_headline()
    test_engineering_programme_education_rules()
    test_replayed_jobs_exclude_responsibility_duties_from_requirements()
    test_replayed_jobs_keep_alternative_groups_local()
    test_job_region_fallback_never_scores_duties()
    print("[PASS] Deterministic parser regression rules hold on the replayed fixtures.")
    suspicious = cv_review["sentence_like_skills"] or any(
        review.get("sentence_like_skills") for review in job_reviews
    )
    print("\n[PASS] Every successfully extracted skill is grounded in its source text.")
    if suspicious:
        print("[REVIEW] Sentence-like skills were found; inspect them above.")
    else:
        print("[PASS] No obviously sentence-like skill entries were found.")
    print(
        "[MANUAL REVIEW] Check whether mandatory skills were missed or optional "
        "skills were incorrectly classified as required."
    )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
