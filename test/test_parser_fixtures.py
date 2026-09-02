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
