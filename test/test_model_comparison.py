"""
test_model_comparison.py
--------------
Benchmarks multiple LLMs (via OpenRouter) on CV extraction quality + speed.
Uses the same extract_cv_info() logic from cv_parser.py, just swapping
the underlying model each run.
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from cv_parser import extract_text_from_pdf, extract_cv_info

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# --- Models to compare. Edit this list freely. ---
# Check exact current model strings at https://openrouter.ai/models
MODELS_TO_TEST = [
    "qwen/qwen-2.5-7b-instruct",
    "qwen/qwen-2.5-14b-instruct",
    "deepseek/deepseek-chat",
    "google/gemini-2.0-flash-exp",
    "meta-llama/llama-3.3-70b-instruct",
    "mistralai/mistral-nemo",
    
]


def build_openrouter_llm(model_name: str):
    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


# --- Pick a CV to test against ---
CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))

print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV to use for the comparison? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]
raw_text = extract_text_from_pdf(str(selected_path))
print(f"\nUsing '{selected_path.name}' ({len(raw_text)} chars) for all model tests.\n")


# --- Run each model, collect results ---
results = []

for model_name in MODELS_TO_TEST:
    print(f"Testing: {model_name} ...")
    try:
        llm = build_openrouter_llm(model_name)
        start = time.time()
        info = extract_cv_info(raw_text, llm=llm)
        elapsed = time.time() - start

        results.append({
            "model": model_name,
            "status": "OK",
            "time_s": round(elapsed, 2),
            "skills_found": len(info.skills),
            "name_extracted": bool(info.full_name),
            "experience_years": info.experience_years,
            "education_count": len(info.education),
        })

    except Exception as e:
        results.append({
            "model": model_name,
            "status": f"ERROR: {str(e)[:60]}",
            "time_s": None,
            "skills_found": None,
            "name_extracted": None,
            "experience_years": None,
            "education_count": None,
        })

    print(f"   -> {results[-1]['status']} ({results[-1]['time_s']}s)\n")


# --- Print comparison table ---
print("\n" + "=" * 100)
print(f"{'MODEL':<38}{'STATUS':<12}{'TIME(s)':<10}{'SKILLS':<9}{'NAME OK':<10}{'EXP.YRS':<10}{'EDU':<6}")
print("=" * 100)
for r in results:
    print(f"{r['model']:<38}{r['status']:<12}{str(r['time_s']):<10}"
          f"{str(r['skills_found']):<9}{str(r['name_extracted']):<10}"
          f"{str(r['experience_years']):<10}{str(r['education_count']):<6}")