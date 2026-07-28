"""
job_matching_pipeline.py
--------------
End-to-end automated pipeline:
1. Extract CV info (cv_parser.py)
2. Build a search query from the candidate's profile
3. Search for real jobs (job_search.py -> Himalayas API)
4. For each job, extract required skills/experience/education (job_parser.py)
5. Score compatibility for each job (matcher.py)
6. Return jobs RANKED by score, highest match first

"""

import json
from cv_parser import extract_text_from_pdf, extract_cv_info
from job_search import search_real_jobs
from job_parser import extract_job_requirements
from matcher import calculate_compatibility


def build_search_query(cv_info) -> str:
    """
    Simple heuristic: most recent job title + top 3 skills.
    """
    title = cv_info.job_titles[0] if cv_info.job_titles else ""
    top_skills = " ".join(cv_info.skills[:3])
    return f"{title} {top_skills}".strip()


def run_job_matching(cv_path: str, results_count: int = 3, search_query: str = None) -> list:
    """
    Runs the full pipeline and returns a list of jobs sorted by
    compatibility score, highest first.
    """
    print("Step 1: Extracting CV info...")
    raw_text = extract_text_from_pdf(cv_path)
    cv_info = extract_cv_info(raw_text)
    print(f"  -> {cv_info.full_name}, {len(cv_info.skills)} skills, "
          f"{cv_info.experience_years} yrs, {cv_info.highest_education_level}\n")

    query = search_query or build_search_query(cv_info)
    print(f"Step 2: Search query -> '{query}'\n")

    print(f"Step 3: Searching Himalayas for {results_count} jobs...")
    jobs_json = search_real_jobs.invoke({"query": query, "results_count": results_count})
    jobs = json.loads(jobs_json)

# Guard against the search tool returning an error object instead of a list
    if isinstance(jobs, dict) and "error" in jobs:
       print(f"  [ERROR] Job search failed: {jobs['error']}")
       return []

    if not jobs:
        print("  No jobs found for this query.")
        return []

    print(f"  -> Found {len(jobs)} job postings\n")

    print("Step 4-5: Extracting requirements + scoring each job...")
    results = []
    for job in jobs:
        print(f"  Processing: {job['title']} @ {job['company']}...")
        job_req = extract_job_requirements(job["title"], job["description"])
        match = calculate_compatibility(cv_info, job_req)

        results.append({
            "job_title": job["title"],
            "company": job["company"],
            "url": job["url"],
            "score_percent": match["score_percent"],
            "skills_detail": match["skills"],
            "experience_detail": match["experience"],
            "education_detail": match["education"],
        })

    # Step 6: rank by score, highest first
    results.sort(key=lambda r: r["score_percent"], reverse=True)
    return results


if __name__ == "__main__":
    from pathlib import Path

    CV_FOLDER = "cv"
    pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))
    print("Available CVs:")
    for i, path in enumerate(pdf_files, start=1):
        print(f"  {i}. {path.name}")
    choice = input("\nWhich CV to use? (enter number): ").strip()
    selected_path = pdf_files[int(choice) - 1]

    manual_query = input("\nSearch query (leave blank to auto-generate from CV): ").strip()

    ranked_jobs = run_job_matching(
        str(selected_path),
        results_count=3,
        search_query=manual_query if manual_query else None,
    )
    print("\n" + "=" * 70)
    print("RANKED JOB MATCHES")
    print("=" * 70)
    for r in ranked_jobs:
        print(f"\n{r['job_title']} @ {r['company']} — {r['score_percent']}% match")
        print(f"  Skills:     {r['skills_detail']['score']*100:.0f}% "
              f"({len(r['skills_detail']['matching'])} matched, {len(r['skills_detail']['missing'])} missing)")
        print(f"  Experience: {r['experience_detail']['score']*100:.0f}%")
        print(f"  Education:  {r['education_detail']['score']*100:.0f}%")
        print(f"  URL: {r['url']}")