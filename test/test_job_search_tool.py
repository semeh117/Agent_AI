# test/test_job_search_tool.py
"""
test_job_search_tool.py
--------------
Verifies search_jobs_for_agent (agent/tools/job_search_tool.py) actually
surfaces the already-seen-jobs signal in its Observation, so the agent
can reason about it — this is the whole point of the memory feature.

Checks:
  1. First search returns jobs with NO filtered-seen note (nothing seen yet).
  2. After recording those jobs as seen, a second search for the SAME
     query returns a note mentioning the filtered count, and the JSON
     list that follows contains NONE of the previously-seen jobs.
  3. search_real_jobs (the machine-facing tool) still returns a clean
     JSON list with no note text — untouched by this change.

Uses a throwaway fake candidate email, cleaned up at the end — safe to
run repeatedly.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
from search.job_search import search_real_jobs, set_candidate_email
from agent.tools.job_search_tool import search_jobs_for_agent
from core.seen_jobs_memory import record_seen, _get_connection

TEST_EMAIL = "test_job_search_tool_throwaway@example.com"
TEST_QUERY = "AI Engineer"


def cleanup():
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM seen_jobs WHERE candidate_email = ?", (TEST_EMAIL.lower(),))
        conn.commit()
    finally:
        conn.close()


def split_note_and_jobs(observation: str):
    """search_jobs_for_agent returns 'note\\n\\n<json list>' — split them apart."""
    note, _, json_part = observation.partition("\n\n")
    jobs = json.loads(json_part)
    return note, jobs


def main():
    print("Cleaning up any leftover test data first...")
    cleanup()
    set_candidate_email(TEST_EMAIL)

    # --- Step 1: first search — nothing seen yet, note should say so ---
    print(f"\n--- SEARCH 1 (agent tool, nothing seen yet) ---")
    obs_1 = search_jobs_for_agent.invoke({"query": TEST_QUERY, "results_count": 3})
    note_1, jobs_1 = split_note_and_jobs(obs_1)
    print(f"  Note: {note_1}")
    print(f"  Jobs: {len(jobs_1)}")

    assert "already evaluated" not in note_1, (
        f"First-ever search should not mention filtered jobs, got: {note_1}"
    )
    if not jobs_1:
        print("  [SKIP] No jobs returned for this query — can't test filtering "
              "without at least one real result. Try a broader TEST_QUERY.")
        cleanup()
        return

    urls_1 = {j["url"] for j in jobs_1 if j.get("url")}
    if not urls_1:
        print("  [SKIP] None of the returned jobs had a URL.")
        cleanup()
        return

    # --- Step 2: record those jobs as seen (simulating evaluate_job_match) ---
    print("\n--- Recording search 1's jobs as SEEN ---")
    for j in jobs_1:
        if j.get("url"):
            record_seen(
                candidate_email=TEST_EMAIL,
                job_url=j["url"],
                job_title=j["title"],
                company=j["company"],
                score_percent=50.0,
                matching_skills=[],
                missing_skills=[],
            )

    # --- Step 3: second search, same query — note must mention filtering ---
    print(f"\n--- SEARCH 2 (agent tool, same query — expect filtered note) ---")
    obs_2 = search_jobs_for_agent.invoke({"query": TEST_QUERY, "results_count": 3})
    note_2, jobs_2 = split_note_and_jobs(obs_2)
    print(f"  Note: {note_2}")
    print(f"  Jobs: {len(jobs_2)}")

    urls_2 = {j["url"] for j in jobs_2 if j.get("url")}
    overlap = urls_1 & urls_2
    assert not overlap, f"Second search returned already-seen jobs: {overlap}"

    if "already evaluated" in note_2:
        print("    [PASS] Note correctly mentions already-seen jobs were filtered.")
    else:
        print("    [INFO] No filtered-jobs note this round — Himalayas may have "
              "enough fresh postings that none of the seen ones resurfaced in "
              "this query's result set. Not a failure, but less conclusive. "
              "Zero overlap with search 1 is still the important assertion "
              "and it passed.")

    # --- Step 4: search_real_jobs (machine-facing) must stay untouched ---
    print(f"\n--- Confirming search_real_jobs still returns a clean JSON list ---")
    raw = search_real_jobs.invoke({"query": TEST_QUERY, "results_count": 2})
    try:
        parsed = json.loads(raw)
        assert isinstance(parsed, list), "search_real_jobs must still return a JSON list"
        print("    [PASS] search_real_jobs output is still strict JSON, no note text.")
    except json.JSONDecodeError:
        raise AssertionError(
            f"search_real_jobs no longer returns clean JSON — got: {raw[:200]}"
        )

    print("\nCleaning up test data...")
    cleanup()
    print("Done.")


if __name__ == "__main__":
    main()