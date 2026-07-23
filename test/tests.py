# test_real_cv.py
import time
from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from cv_parser import extract_text_from_pdf, extract_cv_info

CV_FOLDER = "cv"  # update to your actual folder name

pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))

if not pdf_files:
    print(f"No PDF files found in '{CV_FOLDER}'. Check the folder path.")
    exit()

print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")

choice = input("\nWhich CV do you want to test? (enter number): ").strip()

try:
    selected_path = pdf_files[int(choice) - 1]
except (ValueError, IndexError):
    print("Invalid choice.")
    exit()

print(f"\nUsing: {selected_path.name}\n")

# --- Stage 1: PDF -> text (deterministic, should be near-instant) ---
t0 = time.time()
raw_text = extract_text_from_pdf(str(selected_path))
t1 = time.time()

print("=== RAW EXTRACTED TEXT ===")
print(raw_text)
print(f"\n(length: {len(raw_text)} characters)")
print(f"(PDF extraction took {t1 - t0:.2f}s)\n")

# --- Stage 2: text -> structured info (LLM call, the slow part) ---
t2 = time.time()
info = extract_cv_info(raw_text)
t3 = time.time()

print("=== STRUCTURED CV INFO ===")
print(info.model_dump_json(indent=2))
print(f"\n(LLM extraction took {t3 - t2:.2f}s)")
print(f"(Total pipeline time: {t3 - t0:.2f}s)")