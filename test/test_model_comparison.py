"""
test_model_comparison.py
--------------
Benchmarks LLMs across TWO providers (OpenRouter + Groq) on CV extraction
quality + speed, to find out if anything beats qwen-2.5-7b-instruct.
"""

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from core.cv_parser import extract_text_from_pdf, extract_cv_info

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MOONSHOTAI_API_KEY = os.getenv("MOONSHOTAI_API_KEY")

# --- Models to compare: (provider, model_name) ---
# Verify exact strings at https://openrouter.ai/models and
# https://console.groq.com/docs/models before running.
MODELS_TO_TEST = [
       ("Qwen 2.5 7B (local)", "ollama", "qwen2.5:7b-instruct"),

]


def build_llm(provider: str, model_name: str):
    if provider == "openrouter":
        return ChatOpenAI(
            model=model_name, temperature=0.0,
            api_key=OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    elif provider == "groq":
        return ChatOpenAI(
            model=model_name, temperature=0.0,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )
    raise ValueError(f"Unknown provider: {provider}")


# --- Pick a CV ---
CV_FOLDER = "cv"
pdf_files = sorted(Path(CV_FOLDER).glob("*.pdf"))

print("Available CVs:")
for i, path in enumerate(pdf_files, start=1):
    print(f"  {i}. {path.name}")
choice = input("\nWhich CV to use for the comparison? (enter number): ").strip()
selected_path = pdf_files[int(choice) - 1]
raw_text = extract_text_from_pdf(str(selected_path))
print(f"\nUsing '{selected_path.name}' ({len(raw_text)} chars) for all model tests.\n")


# --- Run each model ---
results = []
for provider, model_name in MODELS_TO_TEST:
    label = f"{provider}/{model_name}"
    print(f"Testing: {label} ...")
    try:
        llm = build_llm(provider, model_name)
        start = time.time()
        info = extract_cv_info(raw_text, llm=llm)
        elapsed = time.time() - start

        results.append({
            "model": label,
            "status": "OK",
            "time_s": round(elapsed, 2),
            "skills_found": len(info.skills),
            "name_extracted": bool(info.full_name),
            "experience_years": info.experience_years,
            "education_count": len(info.education),
        })
    except Exception as e:
        results.append({
            "model": label,
            "status": f"ERROR: {str(e)[:60]}",
            "time_s": None, "skills_found": None,
            "name_extracted": None, "experience_years": None,
            "education_count": None,
        })

    print(f"   -> {results[-1]['status']} ({results[-1]['time_s']}s)\n")


# --- Results table ---
print("\n" + "=" * 110)
print(f"{'MODEL':<45}{'STATUS':<12}{'TIME(s)':<10}{'SKILLS':<9}{'NAME OK':<10}{'EXP.YRS':<10}{'EDU':<6}")
print("=" * 110)
for r in results:
    print(f"{r['model']:<45}{r['status']:<12}{str(r['time_s']):<10}"
          f"{str(r['skills_found']):<9}{str(r['name_extracted']):<10}"
          f"{str(r['experience_years']):<10}{str(r['education_count']):<6}")