# test/test_ranked_evaluations.py
"""
test_ranked_evaluations.py
----------------------
Verifies get_ranked_evaluations(): the deterministic, de-duplicated ranking
used for the email / final answer. Two real-run problems it must fix:

  1. A job the LLM evaluated TWICE in one run (same URL re-typed) must only
     appear once in the ranked list.
  2. Inconclusive evaluations (no extractable requirements) must rank BELOW
     conclusive ones regardless of raw score — an inconclusive 75% is not a
     real match, a conclusive 55% is.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from types import SimpleNamespace
from search import job_search
import agent.tools.job_evaluator as je
import agent.tools.cover_letter as cl


def reset(evaluations):
    je._all_evaluations = list(evaluations)


def test_duplicate_same_url_deduped():
    reset([
        {"job_title": "FDE Teaching Expert", "company": "TripleTen", "url": "u1",
         "score_percent": 75.0, "matching_skills": [], "missing_skills": [], "inconclusive": True},
        {"job_title": "FDE Teaching Expert", "company": "TripleTen", "url": "u1",
         "score_percent": 75.0, "matching_skills": [], "missing_skills": [], "inconclusive": True},
        {"job_title": "AI Architect", "company": "Creative Chaos", "url": "u2",
         "score_percent": 55.5, "matching_skills": ["Python"], "missing_skills": ["K8s"], "inconclusive": False},
    ])
    ranked = je.get_ranked_evaluations()
    assert len(ranked) == 2, f"expected 2 unique jobs, got {len(ranked)}"


def test_url_less_duplicate_deduped_by_title_company():
    reset([
        {"job_title": "FDE Teaching Expert", "company": "TripleTen", "url": "",
         "score_percent": 75.0, "matching_skills": [], "missing_skills": [], "inconclusive": True},
        {"job_title": "FDE Teaching Expert", "company": "TripleTen", "url": "",
         "score_percent": 75.0, "matching_skills": [], "missing_skills": [], "inconclusive": True},
    ])
    assert len(je.get_ranked_evaluations()) == 1


def test_conclusive_ranks_above_inconclusive_regardless_of_score():
    reset([
        {"job_title": "Inconclusive 75", "company": "A", "url": "u1", "score_percent": 75.0,
         "matching_skills": [], "missing_skills": [], "inconclusive": True},
        {"job_title": "Conclusive 55", "company": "B", "url": "u2", "score_percent": 55.5,
         "matching_skills": ["Python"], "missing_skills": ["K8s"], "inconclusive": False},
    ])
    ranked = je.get_ranked_evaluations()
    assert ranked[0]["job_title"] == "Conclusive 55"
    assert ranked[1]["job_title"] == "Inconclusive 75"


def test_within_group_sorted_descending():
    reset([
        {"job_title": "C1", "company": "A", "url": "u1", "score_percent": 70.0,
         "matching_skills": ["x"], "missing_skills": [], "inconclusive": False},
        {"job_title": "C2", "company": "B", "url": "u2", "score_percent": 90.0,
         "matching_skills": ["x"], "missing_skills": [], "inconclusive": False},
        {"job_title": "I1", "company": "C", "url": "u3", "score_percent": 50.0,
         "matching_skills": [], "missing_skills": [], "inconclusive": True},
    ])
    titles = [r["job_title"] for r in je.get_ranked_evaluations()]
    assert titles == ["C2", "C1", "I1"]


def test_url_only_input_backfills_full_description_from_stored_search():
    """The context-overflow regression: the agent must be able to evaluate a
    job with just {{"url": "..."}} (no re-typed description), and the tool
    must score against the FULL stored description, not a truncated retype
    or an empty one."""
    stored_description = (
        "Full posting text " * 50 + " UNIQUE_MARKER REQUIREMENT: 5+ years Python"
    )
    job_search._last_search_results = [{
        "title": "AI Engineer", "company": "Acme", "url": "https://himalayas.app/jobs/x",
        "description": stored_description,
    }]
    je._current_cv_info = SimpleNamespace(mail=None)

    seen_desc = []
    def fake_extract(title, description, llm=None, use_cache=True):
        seen_desc.append(description)
        return {"skills": [], "title": title}

    def fake_compat(cv, reqs):
        return {
            "score_percent": 88.0,
            "skills": {"matching": [{"job_skill": "Python", "matched_via": "Python"}], "missing": []},
            "experience": {"score": 0.9},
            "education": {"score": 0.8},
        }

    original_extract = je.extract_job_requirements
    original_compat = je.calculate_compatibility
    je.extract_job_requirements = fake_extract
    je.calculate_compatibility = fake_compat
    try:
        je._all_evaluations = []
        out = json.loads(
            je.evaluate_job_match.func('{"url": "https://himalayas.app/jobs/x"}')
        )
    finally:
        je.extract_job_requirements = original_extract
        je.calculate_compatibility = original_compat
        je._current_cv_info = None

    assert out["job_title"] == "AI Engineer"
    assert out["company"] == "Acme"
    assert out["url"] == "https://himalayas.app/jobs/x"
    assert out["score_percent"] == 88.0
    assert seen_desc, "extract_job_requirements should have been called"
    assert seen_desc[0] == stored_description, (
        "must score against the FULL stored description, not an empty/truncated one"
    )


def test_observation_truncation_never_corrupts_stored_full_description():
    """The [truncated] summary shown to the LLM must be applied to COPIES of
    the jobs only. _search_jobs_core stores the returned job dicts in
    job_search._last_search_results by reference, and evaluate_job_match +
    the cover-letter step read the FULL description back from there — so
    mutating the description in place would silently degrade scoring."""
    from agent.tools import job_search_tool as jst

    full_desc = ("the quick brown fox jumps over the lazy dog. " * 300).strip()
    assert len(full_desc) > 1000, "fixture must exceed the summary cap"

    def fake_core(query, results_count):
        job = {"title": "DevOps", "company": "Acme", "url": "u9",
               "description": full_desc}
        job_search._record_returned([job])
        return {
            "jobs": [job],
            "query": query,
            "requested_count": results_count,
            "returned_count": 1,
            "filtered_seen_count": 0,
        }

    original_core = jst._search_jobs_core
    original_cap = jst.DESCRIPTION_SUMMARY_CHARS
    jst._search_jobs_core = fake_core
    jst.DESCRIPTION_SUMMARY_CHARS = 200  # force truncation regardless of the default
    job_search._last_search_results = []  # isolate from other tests
    job_search._session_returned_urls = set()
    try:
        obs = jst.search_jobs_for_agent.func("devops", 1)
    finally:
        jst._search_jobs_core = original_core
        jst.DESCRIPTION_SUMMARY_CHARS = original_cap

    assert "[truncated]" in obs, "observation must show a truncated summary"
    assert full_desc not in obs, "full text must not leak into the observation"
    stored = [j for j in job_search._last_search_results if j.get("url") == "u9"]
    assert stored, "the job must be recorded in _last_search_results"
    assert stored[0]["description"] == full_desc, (
        "stored description must remain FULL after the observation is shortened"
    )


def _install_eval_fakes():
    """Mock extract_job_requirements + calculate_compatibility so evaluation
    tests never touch the network or the real scoring internals. Returns a
    callable used to reset everything back."""
    def fake_extract(title, description, llm=None, use_cache=True):
        return {"skills": [], "title": title}

    def fake_compat(cv, reqs):
        return {
            "score_percent": 61.0,
            "skills": {"matching": [{"job_skill": "Python", "matched_via": "Python"}], "missing": []},
            "experience": {"score": 0.7},
            "education": {"score": 0.6},
        }

    original_extract = je.extract_job_requirements
    original_compat = je.calculate_compatibility
    original_cv = je._current_cv_info
    je.extract_job_requirements = fake_extract
    je.calculate_compatibility = fake_compat
    je._current_cv_info = SimpleNamespace(mail=None)

    def restore():
        je.extract_job_requirements = original_extract
        je.calculate_compatibility = original_compat
        je._current_cv_info = original_cv

    return restore


def test_batch_input_evaluates_each_entry():
    """The model reliably (wrongly) passes {"jobs": [...]} instead of one
    job per call. It must NOT error and loop — each entry gets evaluated."""
    job_search._last_search_results = [
        {"title": "Principal Architect, AI/ML", "company": "Zencore", "url": "u1",
         "description": "full desc principal"},
        {"title": "AI Architect", "company": "Creative Chaos", "url": "u2",
         "description": "full desc architect"},
        {"title": "Conversational AI Engineer", "company": "Bright Vision Technologies",
         "url": "u3", "description": "full desc conversational"},
    ]
    restore = _install_eval_fakes()
    try:
        je._all_evaluations = []
        out = json.loads(je.evaluate_job_match.func(json.dumps({"jobs": [
            {"url": "u1"}, {"url": "u2"}, {"url": "u3"},
        ]})))
    finally:
        restore()

    assert "results" in out, "batch input must return a results list"
    titles = {r["job_title"] for r in out["results"]}
    assert titles == {"Principal Architect, AI/ML", "AI Architect", "Conversational AI Engineer"}
    assert len(je._all_evaluations) == 3


def test_evaluation_shaped_batch_entries_resolve_by_title():
    """The exact repeated-wrong-call from the live run: the model passed its
    OWN evaluation output ({title, score_percent, inconclusive}, no url, no
    company) back into evaluate_job_match. With no company given, a UNIQUE
    title match against this run's stored postings must resolve the job so
    the call succeeds instead of looping."""
    job_search._last_search_results = [
        {"title": "Principal Architect, AI/ML", "company": "Zencore", "url": "u1",
         "description": "full desc principal"},
        {"title": "AI Architect", "company": "Creative Chaos", "url": "u2",
         "description": "full desc architect"},
        {"title": "Conversational AI Engineer", "company": "Bright Vision Technologies",
         "url": "u3", "description": "full desc conversational"},
    ]
    restore = _install_eval_fakes()
    try:
        je._all_evaluations = []
        out = json.loads(je.evaluate_job_match.func(json.dumps({"jobs": [
            {"title": "Principal Architect, AI/ML", "score_percent": 41.1, "inconclusive": False},
            {"title": "AI Architect", "score_percent": 55.8, "inconclusive": False},
            {"title": "Conversational AI Engineer", "score_percent": 67.2, "inconclusive": False},
        ]})))
    finally:
        restore()

    assert "results" in out and len(out["results"]) == 3, out
    assert all("error" not in r for r in out["results"]), (
        "title-only entries should resolve against stored postings"
    )
    assert len(je._all_evaluations) == 3


def test_cover_letter_accepts_url_only_by_looking_up_evaluation():
    """The model drifts the same way with write_cover_letter: it passes just
    {{"url": ...}}. The tool must look the evaluation up from this run's
    records instead of rejecting the call and looping."""
    job_search._last_search_results = [{
        "title": "AI Engineer", "company": "Acme", "url": "u1",
        "description": "full stored description",
    }]
    restore = _install_eval_fakes()
    original_generate = cl.generate_cover_letter
    cl.generate_cover_letter = lambda cv, top: f"letter for {top['job_title']}"
    try:
        je._all_evaluations = []
        cl._last_cover_letter = None
        cl._last_cover_letter_job = None
        je.evaluate_job_match.func('{"url": "u1"}')  # records the evaluation
        out = cl.write_cover_letter.func('{"url": "u1"}')
    finally:
        restore()
        cl.generate_cover_letter = original_generate

    assert "Error" not in out, out
    assert cl._last_cover_letter == "letter for AI Engineer"
    assert cl._last_cover_letter_job["job_title"] == "AI Engineer"
    assert cl._last_cover_letter_job["company"] == "Acme"
    assert cl._last_cover_letter_job["description"] == "full stored description"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All ranked-evaluation tests passed.")