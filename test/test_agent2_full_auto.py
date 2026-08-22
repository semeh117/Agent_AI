"""Deterministic tests for Agent 2 cover-letter and delivery completion."""

from types import SimpleNamespace

import agent.agent2 as agent2_module
from agent.agent2 import complete_agent2_delivery
import core.cv_parser as cv_parser
import pipeline.cover_letter as cover_pipeline
import pipeline.send_results_email as email_pipeline
import pipeline.send_results_telegram as telegram_pipeline


def _job(title: str, score: float) -> dict:
    return {
        "job_title": title,
        "company": "Example",
        "url": f"https://example.com/{title}",
        "description": f"Full description for {title}",
        "final_score": score,
        "score_percent": score,
        "inconclusive": False,
        "skills_detail": {
            "matching": [{"job_skill": "Python", "matched_via": "Python"}],
            "missing": ["Kubernetes"],
        },
    }


def _base_result() -> dict:
    return {"output": "Ranked jobs.", "intermediate_steps": []}


def test_gmail_uses_true_top_job_and_writer_output():
    original_writer = cover_pipeline.generate_cover_letter
    original_gmail = email_pipeline.create_results_draft
    calls = {"writer": [], "gmail": []}

    cover_pipeline.generate_cover_letter = lambda cv, top: (
        calls["writer"].append(top) or "generated letter"
    )
    email_pipeline.create_results_draft = lambda cv, jobs, letter, to_email: (
        calls["gmail"].append((jobs, letter, to_email)) or {"id": "draft-1"}
    )
    try:
        result = complete_agent2_delivery(
            _base_result(),
            SimpleNamespace(mail="candidate@example.com"),
            {"ranked_jobs": [_job("Lower", 70.0), _job("Top", 90.0)]},
            delivery_channel="gmail",
        )
    finally:
        cover_pipeline.generate_cover_letter = original_writer
        email_pipeline.create_results_draft = original_gmail

    assert calls["writer"][0]["job_title"] == "Top"
    assert calls["gmail"][0][0][0]["job_title"] == "Top"
    assert calls["gmail"][0][1] == "generated letter"
    assert result["delivery"]["status"] == "completed"
    assert result["cover_letter"] == "generated letter"


def test_telegram_uses_same_ranked_payload_and_letter():
    original_writer = cover_pipeline.generate_cover_letter
    original_telegram = telegram_pipeline.create_results_telegram
    calls = []

    cover_pipeline.generate_cover_letter = lambda cv, top: "telegram letter"
    telegram_pipeline.create_results_telegram = lambda cv, jobs, letter: (
        calls.append((jobs, letter)) or {"messages": [{"ok": True}]}
    )
    try:
        result = complete_agent2_delivery(
            _base_result(),
            SimpleNamespace(mail=None),
            {"ranked_jobs": [_job("Top", 91.0)]},
            delivery_channel="telegram",
        )
    finally:
        cover_pipeline.generate_cover_letter = original_writer
        telegram_pipeline.create_results_telegram = original_telegram

    assert calls[0][0][0]["job_title"] == "Top"
    assert calls[0][1] == "telegram letter"
    assert result["delivery"]["channel"] == "telegram"
    assert result["delivery"]["status"] == "completed"


def test_delivery_is_skipped_when_there_are_no_ranked_jobs():
    result = complete_agent2_delivery(
        _base_result(),
        SimpleNamespace(mail="candidate@example.com"),
        {"ranked_jobs": []},
        delivery_channel="gmail",
    )
    assert result["cover_letter"] is None
    assert result["delivery"]["status"] == "skipped"


def test_pdf_entry_point_owns_cv_parsing_and_defaults_to_gmail():
    original_text_parser = cv_parser.extract_text_from_pdf
    original_cv_parser = cv_parser.extract_cv_info
    original_full_auto = agent2_module.run_agent2_full_auto
    calls = {}
    parsed_cv = SimpleNamespace(mail="candidate@example.com", skills=["Python"])

    cv_parser.extract_text_from_pdf = lambda source: (
        calls.__setitem__("source", source), "CV text"
    )[1]
    cv_parser.extract_cv_info = lambda text, use_cache=True: (
        calls.update({"text": text, "use_cache": use_cache}) or parsed_cv
    )
    agent2_module.run_agent2_full_auto = lambda cv, **kwargs: (
        calls.update({"cv": cv, **kwargs}) or {"output": "done"}
    )
    try:
        result = agent2_module.run_agent2_full_auto_from_pdf(
            "candidate.pdf",
            results_count=5,
            location="Berlin",
            use_cache=False,
        )
    finally:
        cv_parser.extract_text_from_pdf = original_text_parser
        cv_parser.extract_cv_info = original_cv_parser
        agent2_module.run_agent2_full_auto = original_full_auto

    assert calls["source"] == "candidate.pdf"
    assert calls["text"] == "CV text"
    assert calls["use_cache"] is False
    assert calls["cv"] is parsed_cv
    assert calls["delivery_channel"] == "gmail"
    assert calls["results_count"] == 5
    assert calls["location"] == "Berlin"
    assert result["cv_info"] is parsed_cv


if __name__ == "__main__":
    tests = sorted(
        (name, function)
        for name, function in globals().items()
        if name.startswith("test_") and callable(function)
    )
    for name, function in tests:
        function()
        print(f"[PASS] {name}")
    print(f"All {len(tests)} Agent 2 full-auto tests passed.")
