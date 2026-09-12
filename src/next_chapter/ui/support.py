"""Small, testable adapters for the local Streamlit application."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from next_chapter.parsing.agent2_cv_parser import Agent2CVInfo, extract_cv_info_agent2
from next_chapter.parsing.agent2_document_extractor import extract_cv_document_agent2

from next_chapter.paths import PROJECT_ROOT


def parse_uploaded_cv(content: bytes) -> tuple[dict, list[str]]:
    if not content.startswith(b"%PDF-"):
        raise ValueError("Please upload a valid PDF document.")
    if len(content) > 10 * 1024 * 1024:
        raise ValueError("The CV must be smaller than 10 MB.")
    directory = PROJECT_ROOT / "runtime" / "uploads"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".pdf", delete=False) as handle:
        handle.write(content)
        path = Path(handle.name)
    try:
        document = extract_cv_document_agent2(path)
        profile = extract_cv_info_agent2(
            document.text,
            layout_text=document.layout_text,
            cache_identity=f"{document.extraction_version}:{document.content_hash}",
        )
        return profile.model_dump(), list(document.warnings)
    finally:
        path.unlink(missing_ok=True)


def provider_ready(role: str) -> bool:
    defaults = {"parser": "openrouter", "agent": "openrouter",
                "cover_letter": "gemini", "interview": "groq"}
    provider = os.getenv(f"{role.upper()}_PROVIDER", defaults[role]).lower()
    return provider == "ollama" or bool(os.getenv(f"{provider.upper()}_API_KEY", "").strip())


def sample_profile() -> dict:
    return Agent2CVInfo(
        full_name="Alex Morgan", mail="alex@example.com",
        headline="Python developer", skills=["Python", "FastAPI", "SQL", "Docker", "Git"],
        job_titles=["Backend Developer"], experience_years=2.0,
        education=["BSc Computer Science"], highest_education_level="Bachelor",
    ).model_dump()


def sample_result() -> dict:
    jobs = []
    for title, company, score, missing in [
        ("Python Backend Developer", "Northstar · sample company", 90.0, ["Kubernetes"]),
        ("Software Engineer", "Fieldwork · sample company", 80.0, ["Redis"]),
        ("Data Platform Engineer", "Common Ground · sample company", 65.0, ["Spark", "Airflow"]),
    ]:
        jobs.append({
            "job_title": title, "company": company, "url": "https://example.com",
            "description": "Illustrative role building Python services, APIs, and data integrations.",
            "final_score": score, "score_percent": score, "skills_score": score - 10,
            "experience_score": 100.0, "education_score": 100.0,
            "skills_detail": {"matching": [{"job_skill": "Python", "matched_via": "Python"},
                                           {"job_skill": "SQL", "matched_via": "SQL"}],
                              "missing": missing}, "inconclusive": False,
        })
    return {
        "workflow_id": "sample", "status": "sample", "ranked_jobs": jobs,
        "cv_info": Agent2CVInfo.model_validate(sample_profile()), "top_job": jobs[0],
        "match_result": {"parsed_count": 10, "skipped_count": 0},
        "cover_letter": "Dear hiring team,\n\nI am interested in the Python Backend Developer role. "
                        "My experience includes building Python APIs with FastAPI and SQL, "
                        "and packaging services with Docker. I would welcome a conversation "
                        "about contributing to your team.\n\nBest regards,\nAlex Morgan",
    }
