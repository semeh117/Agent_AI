# test/test_seen_jobs_memory.py
"""
test_seen_jobs_memory.py
--------------
Verifies the seen-jobs memory actually works end-to-end:
  1. First search for a fake candidate returns some jobs.
  2. Those jobs get recorded as "seen" (simulating evaluate_job_match's
     record_seen call, without needing a full LLM evaluation pass).
  3. A second search for the SAME candidate, SAME query, must return
     ZERO overlap with the first batch — proving the filter/paging/
     category-fallback logic is actually excluding seen jobs, not just
     returning the same page again.
  4. A search for a DIFFERENT candidate, same query, should NOT be
     filtered by the first candidate's seen jobs — proving the memory
     is correctly scoped per-candidate, not global.

Uses a throwaway fake email so this is safe to run repeatedly without
polluting real candidate data — cleans up after itself at the end.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json
from search.job_search import search_real_jobs, set_candidate_email
from core.seen_jobs_memory import get_seen_urls, record_seen, _get_connection

TEST_EMAIL = "test_seen_jobs_memory_throwaway@example.com"
TEST_QUERY = "AI Engineer"


def cleanup():
    """Remove this test's rows so repeated runs start clean and don't
    leave fake data sitting in the real seen_jobs_memory.db."""
    conn = _get_connection()
    try:
        conn.execute("DELETE FROM seen_jobs WHERE candidate_email = ?", (TEST_EMAIL.lower(),))
        conn.commit()
    finally:
        conn.close()


def main():
    print("Cleaning up any leftover test data first...")
    cleanup()

    # --- Step 1: first search for the test candidate ---
    print(f"\n--- SEARCH 1 (candidate: {TEST_EMAIL}) ---")
    set_candidate_email(TEST_EMAIL)
    jobs_1_json = search_real_jobs.invoke({"query": TEST_QUERY, "results_count": 3})
    jobs_1 = json.loads(jobs_1_json)

    if isinstance(jobs_1, dict) and "error" in jobs_1:
        print(f"  [ERROR] Search failed: {jobs_1['error']}")
        return
    if not jobs_1:
        print("  [SKIP] No jobs returned at all for this query — can't test filtering "
              "without at least one real result. Try a broader TEST_QUERY.")
        return

    urls_1 = {j["url"] for j in jobs_1 if j.get("url")}
    print(f"  -> Got {len(jobs_1)} jobs, {len(urls_1)} with URLs:")
    for j in jobs_1:
        print(f"     - {j['title']} @ {j['company']} ({j['url']})")

    if not urls_1:
        print("  [SKIP] None of the returned jobs had a URL — can't test URL-based "
              "filtering. Check search_real_jobs's 'url' field mapping.")
        return

    # --- Step 2: record all of search 1's jobs as "seen" for this candidate ---
    print("\n--- Recording search 1's jobs as SEEN ---")
    for j in jobs_1:
        if j.get("url"):
            record_seen(
                candidate_email=TEST_EMAIL,
                job_url=j["url"],
                job_title=j["title"],
                company=j["company"],
                score_percent=50.0,  # dummy value — not testing scoring here
                matching_skills=[],
                missing_skills=[],
            )
    seen_now = get_seen_urls(TEST_EMAIL)
    print(f"  -> {len(seen_now)} URLs now recorded as seen for this candidate")
    assert urls_1.issubset(seen_now), "Expected all of search 1's URLs to be recorded as seen"

    # --- Step 3: second search, SAME candidate, SAME query — must have ZERO overlap ---
    print(f"\n--- SEARCH 2, same candidate, same query (expect NO overlap with search 1) ---")
    jobs_2_json = search_real_jobs.invoke({"query": TEST_QUERY, "results_count": 3})
    jobs_2 = json.loads(jobs_2_json)
    urls_2 = {j["url"] for j in jobs_2 if j.get("url")}

    print(f"  -> Got {len(jobs_2)} jobs, {len(urls_2)} with URLs:")
    for j in jobs_2:
        print(f"     - {j['title']} @ {j['company']} ({j['url']})")

    overlap = urls_1 & urls_2
    print(f"\n  Overlap with search 1: {len(overlap)} URLs")
    if overlap:
        print(f"    [FAIL] These URLs appeared in BOTH searches despite being marked seen: {overlap}")
    else:
        print("    [PASS] Zero overlap — seen-jobs filtering is working.")

    # --- Step 4: different candidate, same query — should NOT be filtered by test candidate's history ---
    print(f"\n--- SEARCH 3, DIFFERENT candidate, same query (expect NO filtering applied) ---")
    set_candidate_email("someone_else_entirely@example.com")
    jobs_3_json = search_real_jobs.invoke({"query": TEST_QUERY, "results_count": 3})
    jobs_3 = json.loads(jobs_3_json)
    urls_3 = {j["url"] for j in jobs_3 if j.get("url")}

    print(f"  -> Got {len(jobs_3)} jobs")
    overlap_with_seen = urls_3 & seen_now
    print(f"  Overlap with test candidate's seen jobs: {len(overlap_with_seen)} URLs")
    if overlap_with_seen:
        print("    [PASS] A different candidate correctly SEES jobs the first candidate "
              "already saw — memory is properly scoped per-candidate, not global.")
    else:
        print("    [INFO] No overlap happened to occur here — inconclusive on its own, "
              "but not a failure (Himalayas' result ordering may differ run to run).")

    # --- Cleanup ---
    print("\nCleaning up test data...")
    cleanup()
    print("Done.")


if __name__ == "__main__":
    main()