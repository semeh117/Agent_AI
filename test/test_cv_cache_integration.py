# test/test_cv_cache_integration.py
"""
test_cv_cache_integration.py
--------------
Confirms extract_cv_info() is actually using the disk cache correctly:
run 1 should call the LLM (slow); run 2, on IDENTICAL text, should hit
the cache (fast, no LLM call) and return an IDENTICAL result.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import time
from pathlib import Path
from core.cv_parser import extract_text_from_pdf, extract_cv_info
from core.extraction_cache import clear_cache

CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))
print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]

raw_text = extract_text_from_pdf(str(selected_path))

print("\nClearing CV cache to guarantee a clean first run...")
clear_cache("cv")

print("\n--- RUN 1 (expect: cache MISS, real LLM call, slower) ---")
t0 = time.time()
info_1 = extract_cv_info(raw_text)
t1 = time.time()
print(f"  Time: {t1 - t0:.2f}s")
print(f"  -> {info_1.full_name}, {len(info_1.skills)} skills")

print("\n--- RUN 2, SAME TEXT (expect: cache HIT, no LLM call, much faster) ---")
t2 = time.time()
info_2 = extract_cv_info(raw_text)
t3 = time.time()
print(f"  Time: {t3 - t2:.2f}s")
print(f"  -> {info_2.full_name}, {len(info_2.skills)} skills")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Run 1 time: {t1 - t0:.2f}s")
print(f"Run 2 time: {t3 - t2:.2f}s")
print(f"Speedup: {(t1 - t0) / max(t3 - t2, 0.001):.1f}x faster on cache hit")
print(f"Results identical: {info_1.model_dump() == info_2.model_dump()}")

if t3 - t2 < 0.5 and info_1.model_dump() == info_2.model_dump():
    print("\n[PASS] Cache is working correctly.")
else:
    print("\n[CHECK] Run 2 wasn't fast enough or results differ — check "
          "extract_cv_info()'s get_cached()/set_cached() wiring.")