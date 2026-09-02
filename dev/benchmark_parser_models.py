"""Benchmark Agent 2 parser models on the same frozen CV and LinkedIn jobs.

This is a live API benchmark, not a unit test. It deliberately bypasses the
parser cache and never changes ``.env``. The raw inputs come from the existing
fixtures, so every model receives the same CV and job descriptions.

Usage:
    python -m dev.benchmark_parser_models
    python -m dev.benchmark_parser_models --models z-ai/glm-5.2:free
    python -m dev.benchmark_parser_models --self-test

The automatic winner is provisional: it uses a transparent gold checklist for
the current Devon Cruz / Eightpoint / Air fixture set. Always inspect the saved
per-model parsed outputs before making the production choice.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CV_FIXTURE = PROJECT_ROOT / "fixtures" / "cv_parser_fixture.json"
DEFAULT_JOB_FIXTURE = PROJECT_ROOT / "fixtures" / "linkedin_job_parser_fixture.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "fixtures" / "parser_model_benchmark.json"

DEFAULT_MODELS = (
    "z-ai/glm-5.2:free",
    "dots-studio/dots-3-note-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "liquid/lfm-2.5-2.6b:free",
)


# Gold concepts explicitly present in the frozen CV. Alias groups make the
# benchmark tolerant of punctuation and equally valid names such as RAG versus
# retrieval-augmented generation. These are not production skill categories.
CV_GOLD: dict[str, tuple[str, ...]] = {
    "prompt engineering": ("prompt engineering", "prompt-engineering"),
    "retrieval-augmented generation": (
        "retrieval-augmented generation",
        "retrieval augmented generation",
        "RAG",
    ),
    "agent orchestration": ("agent orchestration",),
    "REST APIs": ("REST APIs", "REST API"),
    "function calling": ("function calling",),
    "structured outputs": (
        "structured outputs",
        "structured-output JSON schemas",
        "structured-output orchestration",
    ),
    "agentic workflows": ("agentic workflows", "agentic search workflows"),
    "prompt iteration loops": ("prompt iteration loops",),
    "few-shot prompting": ("few-shot prompting",),
    "prompt chaining": ("prompt chaining", "multi-step prompt chaining"),
    "BM25": ("BM25",),
    "semantic search": ("semantic search",),
    "adaptive chunking": ("adaptive chunking",),
    "multi-agent systems": ("multi-agent search system", "multi-agent system"),
    "tool use": ("tool use",),
    "LLM evaluation": ("LLM evaluation framework", "LLM-as-judge evaluation"),
    "regression testing": ("regression suites", "regression testing"),
    "content moderation": ("content-moderation routing", "content moderation"),
    "semantic caching": ("semantic caching",),
    "token-budget streaming": ("token-budget streaming",),
    "fine-tuning": ("fine-tuning",),
    "LoRA": ("LoRA",),
    "drift monitoring": ("drift-monitoring", "drift monitoring"),
    "A/B testing": ("A/B test routing", "A/B testing"),
    "responsible AI": ("responsible-AI", "responsible AI"),
}


JOB_GOLD: dict[str, dict[str, Any]] = {
    "eightpoint": {
        "required": {
            "Python": ("Python",),
            "LLM application development": ("LLM application development",),
            "RAG": ("Retrieval-Augmented Generation", "RAG"),
            "vector databases": ("vector databases",),
            "embedding models": ("embedding models",),
            "LLM orchestration frameworks": ("LLM orchestration frameworks",),
            "LangChain": ("LangChain",),
            "LangGraph": ("LangGraph",),
            "fine-tuning": ("fine-tuning",),
            "LoRA": ("LoRA",),
            "PyTorch": ("PyTorch",),
            "TensorFlow": ("TensorFlow",),
            "dataset creation": ("dataset creation",),
            "AWS SageMaker": ("AWS SageMaker", "SageMaker"),
            "model/retrieval evaluation": (
                "evaluation of model and retrieval quality",
                "retrieval performance",
                "regression testing",
            ),
            "telemetry and logging": ("telemetry and logging",),
        },
        "preferred": {
            "FastAPI": ("FastAPI",),
            "HuggingFace": ("HuggingFace",),
            "MLOps": ("MLOps",),
            "containerization": ("containerization",),
            "CI/CD": ("CI/CD",),
            "Vertex AI": ("Vertex AI",),
            "ONNX": ("ONNX",),
        },
        "forbidden_required": (
            "designing",
            "building",
            "deploying",
            "leveraging",
        ),
        "experience": None,
        "education": None,
    },
    "air": {
        "required": {
            "Python": ("Python",),
            "machine learning systems": ("machine learning systems",),
            "LLM algorithms": ("LLM algorithms",),
            "coding agents": ("coding agents",),
            "cloud infrastructure": ("cloud infrastructure",),
            "AWS": ("AWS",),
            "GCP": ("GCP",),
            "Git": ("Git",),
            "multi-agent systems": ("multi-agent systems",),
            "tool use": ("tool use",),
            "memory": ("memory",),
            "routing": ("routing",),
            "automated evaluation": ("automated evaluation",),
            "data processing": ("data processing",),
        },
        "preferred": {
            "React": ("React",),
            "Next.js": ("Next.js", "NextJS"),
            "API-driven UI architectures": ("API-driven UI architectures",),
        },
        "forbidden_required": (
            "U.S. Citizenship",
            "Product",
            "GTM",
            "Operations",
            "Internal Team Velocity",
            "Operational Cost Savings",
            "Employee Productivity",
            "Internal Adoption Rate",
            "Reduce Operational Costs",
        ),
        "experience": 3.0,
        "education": "Bachelor",
    },
}


_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_URL_RE = re.compile(r"https?://\S+", re.I)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d\s().-]{7,}\d(?!\w)")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(value: Any) -> str:
    text = html.unescape(str(value or "")).casefold()
    text = text.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+(?:\+\+|#)?", text))


def _phrase_contains(container: str, phrase: str) -> bool:
    return f" {phrase} " in f" {container} "


def _concept_found(skills: Iterable[str], aliases: Iterable[str]) -> bool:
    normalized_skills = [_normalize(skill) for skill in skills]
    normalized_aliases = [_normalize(alias) for alias in aliases]
    for skill in normalized_skills:
        for alias in normalized_aliases:
            if not skill or not alias:
                continue
            if skill == alias or _phrase_contains(skill, alias):
                return True
    return False


def _applicable_concepts(
    concepts: dict[str, tuple[str, ...]], source_text: str
) -> dict[str, tuple[str, ...]]:
    source = _normalize(source_text)
    return {
        name: aliases
        for name, aliases in concepts.items()
        if any(_phrase_contains(source, _normalize(alias)) for alias in aliases)
    }


def _redact_cv(raw_text: str, parsed_cv: dict[str, Any]) -> str:
    text = raw_text
    full_name = str(parsed_cv.get("full_name") or "").strip()
    if full_name:
        text = re.sub(re.escape(full_name), "<CANDIDATE_NAME>", text, flags=re.I)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _URL_RE.sub("<PROFILE_URL>", text)
    text = _PHONE_RE.sub("<PHONE>", text)
    return text


def _evaluate_cv(parsed: dict[str, Any], source_text: str) -> dict[str, Any]:
    skills = [str(value) for value in parsed.get("skills", [])]
    applicable = _applicable_concepts(CV_GOLD, source_text)
    matched = [
        name for name, aliases in applicable.items() if _concept_found(skills, aliases)
    ]
    missing = [name for name in applicable if name not in matched]
    canonical = [_normalize(skill) for skill in skills]
    duplicate_count = len(canonical) - len(set(canonical))
    total = len(applicable)
    return {
        "skills_count": len(skills),
        "gold_total": total,
        "gold_matched": len(matched),
        "recall_percent": round(100 * len(matched) / total, 1) if total else None,
        "matched_concepts": matched,
        "missing_concepts": missing,
        "exact_duplicate_count": duplicate_count,
    }


def _evaluate_job(
    parsed: dict[str, Any], raw_job: dict[str, Any]
) -> dict[str, Any]:
    company_key = str(raw_job.get("company") or "").strip().casefold()
    expectation = JOB_GOLD.get(company_key)
    required = [str(value) for value in parsed.get("required_skills", [])]
    preferred = [str(value) for value in parsed.get("preferred_skills", [])]
    base = {
        "title": raw_job.get("title"),
        "company": raw_job.get("company"),
        "url": raw_job.get("url"),
        "description_characters": len(str(raw_job.get("description") or "")),
        "scored_against_gold": expectation is not None,
    }
    if expectation is None:
        return base

    required_gold = expectation["required"]
    preferred_gold = expectation["preferred"]
    required_matched = [
        name for name, aliases in required_gold.items() if _concept_found(required, aliases)
    ]
    required_missing = [name for name in required_gold if name not in required_matched]
    required_misclassified = [
        name
        for name, aliases in required_gold.items()
        if name in required_missing and _concept_found(preferred, aliases)
    ]
    preferred_matched = [
        name
        for name, aliases in preferred_gold.items()
        if _concept_found(preferred, aliases)
    ]
    preferred_missing = [name for name in preferred_gold if name not in preferred_matched]
    preferred_misclassified = [
        name
        for name, aliases in preferred_gold.items()
        if name in preferred_missing and _concept_found(required, aliases)
    ]

    required_keys = {_normalize(skill) for skill in required}
    forbidden = [
        value
        for value in expectation["forbidden_required"]
        if _normalize(value) in required_keys
    ]
    expected_experience = expectation["experience"]
    actual_experience = parsed.get("required_experience_years")
    experience_correct = (
        actual_experience is None
        if expected_experience is None
        else actual_experience is not None
        and abs(float(actual_experience) - float(expected_experience)) < 0.01
    )
    education_correct = parsed.get("required_education_level") == expectation["education"]

    base.update(
        {
            "required_total": len(required_gold),
            "required_matched": required_matched,
            "required_missing": required_missing,
            "required_misclassified_as_preferred": required_misclassified,
            "preferred_total": len(preferred_gold),
            "preferred_matched": preferred_matched,
            "preferred_missing": preferred_missing,
            "preferred_misclassified_as_required": preferred_misclassified,
            "forbidden_required_skills": forbidden,
            "experience_correct": experience_correct,
            "education_correct": education_correct,
        }
    )
    return base


def _failed_job_review(raw_job: dict[str, Any], error: Exception) -> dict[str, Any]:
    """Represent a failed gold-fixture parse as zero recall, not a skipped case."""
    company_key = str(raw_job.get("company") or "").strip().casefold()
    expectation = JOB_GOLD.get(company_key)
    base = {
        "title": raw_job.get("title"),
        "company": raw_job.get("company"),
        "url": raw_job.get("url"),
        "description_characters": len(str(raw_job.get("description") or "")),
        "scored_against_gold": expectation is not None,
        "parse_error": str(error),
    }
    if expectation is None:
        return base

    base.update(
        {
            "required_total": len(expectation["required"]),
            "required_matched": [],
            "required_missing": list(expectation["required"]),
            "required_misclassified_as_preferred": [],
            "preferred_total": len(expectation["preferred"]),
            "preferred_matched": [],
            "preferred_missing": list(expectation["preferred"]),
            "preferred_misclassified_as_required": [],
            "forbidden_required_skills": [],
            "experience_correct": False,
            "education_correct": False,
        }
    )
    return base


def _aggregate_score(
    cv_review: dict[str, Any],
    job_reviews: list[dict[str, Any]],
    successful_calls: int,
    attempted_calls: int,
) -> dict[str, Any]:
    scored_jobs = [review for review in job_reviews if review["scored_against_gold"]]
    required_total = sum(review["required_total"] for review in scored_jobs)
    required_matched = sum(len(review["required_matched"]) for review in scored_jobs)
    preferred_total = sum(review["preferred_total"] for review in scored_jobs)
    preferred_matched = sum(len(review["preferred_matched"]) for review in scored_jobs)
    eligibility_checks = 2 * len(scored_jobs)
    eligibility_correct = sum(
        int(review["experience_correct"]) + int(review["education_correct"])
        for review in scored_jobs
    )
    forbidden_count = sum(
        len(review["forbidden_required_skills"]) for review in scored_jobs
    )

    cv_recall = float(cv_review.get("recall_percent") or 0.0)
    required_recall = 100 * required_matched / required_total if required_total else 0.0
    preferred_recall = 100 * preferred_matched / preferred_total if preferred_total else 0.0
    eligibility_accuracy = (
        100 * eligibility_correct / eligibility_checks if eligibility_checks else 0.0
    )
    call_success = 100 * successful_calls / attempted_calls if attempted_calls else 0.0
    penalty = min(20.0, forbidden_count * 2.5)
    penalty += min(5.0, cv_review.get("exact_duplicate_count", 0) * 0.5)
    score = (
        0.30 * cv_recall
        + 0.35 * required_recall
        + 0.15 * preferred_recall
        + 0.10 * eligibility_accuracy
        + 0.10 * call_success
        - penalty
    )
    return {
        "quality_score": round(max(0.0, score), 1),
        "cv_recall_percent": round(cv_recall, 1),
        "job_required_recall_percent": round(required_recall, 1),
        "job_preferred_recall_percent": round(preferred_recall, 1),
        "eligibility_accuracy_percent": round(eligibility_accuracy, 1),
        "call_success_percent": round(call_success, 1),
        "forbidden_required_count": forbidden_count,
        "penalty_points": round(penalty, 1),
    }


def _run_model(
    model_id: str,
    cv_text: str,
    jobs: list[dict[str, Any]],
    delay_seconds: float,
) -> dict[str, Any]:
    from config import _build_llm
    from core.agent2_parser import (
        extract_cv_info_agent2,
        extract_job_requirements_agent2,
    )

    print(f"\n[{model_id}] starting")
    llm = _build_llm(
        provider="openrouter",
        model=model_id,
        temperature=0.0,
        role="parser",
    )
    started = time.perf_counter()
    errors: list[dict[str, str]] = []
    attempted_calls = 1 + len(jobs)
    successful_calls = 0

    cv_output: dict[str, Any] | None = None
    cv_review: dict[str, Any] = {
        "skills_count": 0,
        "gold_total": 0,
        "gold_matched": 0,
        "recall_percent": 0.0,
        "matched_concepts": [],
        "missing_concepts": list(CV_GOLD),
        "exact_duplicate_count": 0,
    }
    try:
        parsed_cv = extract_cv_info_agent2(cv_text, llm=llm, use_cache=False)
        cv_output = parsed_cv.model_dump()
        cv_review = _evaluate_cv(cv_output, cv_text)
        successful_calls += 1
    except Exception as exc:
        errors.append({"input": "CV", "error": str(exc)})

    job_outputs: list[dict[str, Any]] = []
    job_reviews: list[dict[str, Any]] = []
    for index, raw_job in enumerate(jobs, start=1):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        parsed_output = None
        try:
            parsed = extract_job_requirements_agent2(
                job_title=str(raw_job.get("title") or ""),
                job_description=str(raw_job.get("description") or ""),
                llm=llm,
                use_cache=False,
            )
            parsed_output = parsed.model_dump()
            successful_calls += 1
            review = _evaluate_job(parsed_output, raw_job)
        except Exception as exc:
            errors.append(
                {
                    "input": f"Job {index}: {raw_job.get('company', '')}",
                    "error": str(exc),
                }
            )
            review = _failed_job_review(raw_job, exc)
        job_outputs.append({"job": raw_job, "parsed": parsed_output})
        job_reviews.append(review)

    elapsed = time.perf_counter() - started
    metrics = _aggregate_score(
        cv_review, job_reviews, successful_calls, attempted_calls
    )
    print(
        f"[{model_id}] score={metrics['quality_score']:.1f} "
        f"success={metrics['call_success_percent']:.1f}% time={elapsed:.1f}s"
    )
    return {
        "model": model_id,
        "metrics": metrics,
        "elapsed_seconds": round(elapsed, 2),
        "errors": errors,
        "cv_review": cv_review,
        "job_reviews": job_reviews,
        "parsed_cv": cv_output,
        "parsed_jobs": job_outputs,
    }


def _print_summary(results: list[dict[str, Any]]) -> None:
    print("\nPARSER MODEL BENCHMARK")
    print(
        f"{'Model':53} {'Score':>6} {'CV':>6} {'Req':>6} "
        f"{'Pref':>6} {'Elig':>6} {'OK':>6} {'Sec':>8}"
    )
    print("-" * 105)
    for result in sorted(
        results, key=lambda item: item["metrics"]["quality_score"], reverse=True
    ):
        metrics = result["metrics"]
        print(
            f"{result['model'][:53]:53} "
            f"{metrics['quality_score']:6.1f} "
            f"{metrics['cv_recall_percent']:6.1f} "
            f"{metrics['job_required_recall_percent']:6.1f} "
            f"{metrics['job_preferred_recall_percent']:6.1f} "
            f"{metrics['eligibility_accuracy_percent']:6.1f} "
            f"{metrics['call_success_percent']:6.1f} "
            f"{result['elapsed_seconds']:8.1f}"
        )


def _run_self_tests() -> int:
    assert _normalize("Retrieval-Augmented Generation (RAG)") == (
        "retrieval augmented generation rag"
    )
    assert _concept_found(
        ["Retrieval-Augmented Generation (RAG)"],
        ("retrieval augmented generation",),
    )
    assert not _concept_found(["fine-tuning"], ("performance tuning",))
    redacted = _redact_cv(
        "Devon Cruz devon@example.com +1 415 555 1212",
        {"full_name": "Devon Cruz"},
    )
    assert "Devon Cruz" not in redacted
    assert "devon@example.com" not in redacted
    assert "415 555 1212" not in redacted
    print("[PASS] Parser benchmark offline self-tests passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare live parser models on frozen Agent 2 fixtures."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--cv-fixture", type=Path, default=DEFAULT_CV_FIXTURE)
    parser.add_argument("--job-fixture", type=Path, default=DEFAULT_JOB_FIXTURE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.5,
        help="Pause between free-endpoint calls to reduce rate-limit errors.",
    )
    parser.add_argument(
        "--no-redact",
        action="store_true",
        help="Send the original CV identity/contact fields. Not recommended.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run offline scoring/redaction checks without API calls.",
    )
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.self_test:
        return _run_self_tests()
    if args.delay_seconds < 0:
        raise ValueError("--delay-seconds cannot be negative.")

    # Importing config loads the project's .env before the explicit key check.
    import config as _project_config  # noqa: F401

    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is missing from the environment/.env.")

    cv_fixture = _load_json(args.cv_fixture)
    job_fixture = _load_json(args.job_fixture)
    cv_text = str(cv_fixture.get("raw_text") or "")
    if not cv_text:
        raise ValueError("CV fixture has no raw_text.")
    if not args.no_redact:
        cv_text = _redact_cv(cv_text, cv_fixture.get("parsed_cv") or {})

    jobs = [
        entry["linkedin_job"]
        for entry in job_fixture.get("jobs", [])
        if entry.get("linkedin_job", {}).get("title")
        and entry.get("linkedin_job", {}).get("description")
    ]
    if not jobs:
        raise ValueError("LinkedIn fixture contains no complete raw job inputs.")

    results = []
    for model_id in dict.fromkeys(args.models):
        try:
            results.append(
                _run_model(
                    model_id=model_id,
                    cv_text=cv_text,
                    jobs=jobs,
                    delay_seconds=args.delay_seconds,
                )
            )
        except Exception as exc:
            print(f"[{model_id}] fatal setup error: {exc}")
            results.append(
                {
                    "model": model_id,
                    "metrics": {
                        "quality_score": 0.0,
                        "cv_recall_percent": 0.0,
                        "job_required_recall_percent": 0.0,
                        "job_preferred_recall_percent": 0.0,
                        "eligibility_accuracy_percent": 0.0,
                        "call_success_percent": 0.0,
                        "forbidden_required_count": 0,
                        "penalty_points": 0.0,
                    },
                    "elapsed_seconds": 0.0,
                    "errors": [{"input": "setup", "error": str(exc)}],
                    "cv_review": {},
                    "job_reviews": [],
                    "parsed_cv": None,
                    "parsed_jobs": [],
                }
            )

    ranked = sorted(
        results,
        key=lambda item: (
            item["metrics"]["quality_score"],
            item["metrics"]["call_success_percent"],
            -item["elapsed_seconds"],
        ),
        reverse=True,
    )
    winner = ranked[0]["model"] if ranked else None
    report = {
        "benchmark_type": "agent2_parser_model_comparison",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_fixture": str(args.cv_fixture),
        "job_fixture": str(args.job_fixture),
        "personal_data_redacted": not args.no_redact,
        "cache_used": False,
        "models": list(dict.fromkeys(args.models)),
        "scoring": {
            "cv_recall_weight": 0.30,
            "job_required_recall_weight": 0.35,
            "job_preferred_recall_weight": 0.15,
            "experience_education_weight": 0.10,
            "successful_call_weight": 0.10,
            "forbidden_required_penalty": 2.5,
            "note": "Provisional winner; inspect parsed outputs before selection.",
        },
        "recommended_model": winner,
        "results": ranked,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(results)
    print(f"\nProvisional winner: {winner or 'none'}")
    print(f"Full comparison saved to: {args.out}")
    print("Inspect missing/misclassified skills before changing PARSER_MODEL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
