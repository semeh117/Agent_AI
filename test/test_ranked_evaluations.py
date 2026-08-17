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

import agent.tools.job_evaluator as je


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  [PASS] {name}")
    print("All ranked-evaluation tests passed.")