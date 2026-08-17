# test/test_job_cache_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import json, time
from core.job_parser import extract_job_requirements
from core.extraction_cache import clear_cache
from search.job_search_fixture import search_jobs_from_fixture

jobs = json.loads(search_jobs_from_fixture(results_count=1))
job = jobs[0]

print("Clearing job cache for a clean run...")
clear_cache("job")

print("\n--- RUN 1 (expect MISS) ---")
t0 = time.time()
req_1 = extract_job_requirements(job["title"], job["description"])
t1 = time.time()
print(f"  Time: {t1 - t0:.2f}s -> {len(req_1.required_skills)} skills")

print("\n--- RUN 2, SAME POSTING (expect HIT) ---")
t2 = time.time()
req_2 = extract_job_requirements(job["title"], job["description"])
t3 = time.time()
print(f"  Time: {t3 - t2:.2f}s -> {len(req_2.required_skills)} skills")

print(f"\nSpeedup: {(t1 - t0) / max(t3 - t2, 0.001):.1f}x")
print(f"Identical results: {req_1.model_dump() == req_2.model_dump()}")